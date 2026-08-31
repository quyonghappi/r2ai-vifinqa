"""Qwen-backed plan generation with strict grounding before Pandas rendering."""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from normalization.schema import infer_column_metadata, unit_scale_to_vnd
from retrieval.full_corpus import TICKER_TOKEN_RE
from schema_linking.linker import SchemaLinkResult, is_numeric_data_column, parse_financial_number

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
MODEL_REVISION = "b56cc04415fac88c421533036e44149a5983dd2a"
MODEL_ELIGIBILITY = {
    "category": "pipeline component", "parameters": "7.61B", "license": "Apache-2.0",
    "released": "2024-09-19", "competition_cutoff": "2026-06-01",
    "pinned_revision": MODEL_REVISION,
}
MAX_PROMPT_OPERANDS = 32
MAX_NEW_TOKENS = 256
PROMPT_SAFETY_MARGIN = 256
DEFAULT_MAX_INPUT_TOKENS = 4096


class PromptBudgetError(RuntimeError):
    """The compact, grounded prompt cannot fit the model context window."""


@dataclass(frozen=True)
class BoundOperand:
    alias: str
    table_key: str
    row_index: int
    column_index: int


@dataclass
class QueryPlan:
    operands: list[BoundOperand]
    expression: str
    answer_unit: str | None
    rationale: str


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("model output contains no JSON object")
    return json.loads(match.group(0))


def _validate_expression(expression: str, aliases: set[str]) -> None:
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Add,
               ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Load)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"disallowed expression syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in aliases | {"abs"}:
            raise ValueError(f"unknown operand in expression: {node.id}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id == "abs"):
            raise ValueError("only abs(...) is allowed in expressions")


def validate_plan(payload: dict, linked: SchemaLinkResult) -> QueryPlan:
    allowed = {(x.table_key, x.row_index, x.column_index) for x in linked.operands}
    operands = [BoundOperand(**x) for x in payload.get("operands", [])]
    aliases = {x.alias for x in operands}
    if not operands or len(aliases) != len(operands) or not all(x.alias.isidentifier() for x in operands):
        raise ValueError("operands require unique Python identifier aliases")
    for operand in operands:
        if (operand.table_key, operand.row_index, operand.column_index) not in allowed:
            raise ValueError(f"invented or unresolved operand: {operand}")
    expression = payload.get("expression", "")
    _validate_expression(expression, aliases)
    return QueryPlan(operands, expression, payload.get("answer_unit"), payload.get("rationale", ""))


def _source_scale_to_vnd(table: dict, column_index: int, column_header: str) -> int | None:
    """Resolve a source unit from the selected column before trusting persisted metadata.

    Structured artifacts are versioned, but a resumable runtime can still carry column metadata
    produced by an older parser. Re-derive the selected column's header path from the source grid
    first; this keeps a stale scale from reversing or magnifying an otherwise grounded answer.
    """
    grid = table.get("grid") or []
    inferred = infer_column_metadata(grid)
    if column_index < len(inferred) and inferred[column_index].get("scale_to_vnd") is not None:
        return inferred[column_index]["scale_to_vnd"]
    column = next(
        (
            item for item in (table.get("column_metadata") or [])
            if item.get("column_index") == column_index
        ),
        {},
    )
    if column.get("scale_to_vnd") is not None:
        return column["scale_to_vnd"]
    scale = unit_scale_to_vnd(column_header)
    if scale is not None:
        return scale
    units = table.get("detected_units") or []
    # A table-wide hint is safe only when it is unambiguous.
    scales = {unit_scale_to_vnd(unit) for unit in units if unit_scale_to_vnd(unit) is not None}
    return scales.pop() if len(scales) == 1 else None


def generate_direct_lookup_plan(linked: SchemaLinkResult, tables_by_key: dict[str, dict]) -> QueryPlan:
    """Resolve a single grounded value without asking the model to choose evidence.

    Entity-scoped retrieval supplies the company candidate set.  This function applies the
    remaining report-variant, period, structural-cell, and per-column-unit constraints before
    rendering the simple arithmetic expression.
    """
    question = linked.question.lower()
    years = set(re.findall(r"\b20\d{2}\b", question))
    candidates = list(linked.operands)
    question_codes = set(TICKER_TOKEN_RE.findall(linked.question)) - {
        "BCTC", "CTCP", "TMCP", "TCT", "TNHH", "VND",
    }
    available_tickers = {
        str(tables_by_key[table_key].get("ticker", ""))
        for table_key in linked.table_keys
        if tables_by_key[table_key].get("ticker")
    }
    mentioned_tickers = question_codes & available_tickers
    if mentioned_tickers:
        candidates = [
            x for x in candidates
            if tables_by_key[x.table_key].get("ticker") in mentioned_tickers
        ]
        if not candidates:
            raise ValueError("schema linker retained no operands for the named ticker")
    if "công ty mẹ" in question or "công ty riêng" in question:
        candidates = [x for x in candidates if tables_by_key[x.table_key].get("variant") == "separate"]
    elif "hợp nhất" in question or "tập đoàn" in question:
        candidates = [x for x in candidates if tables_by_key[x.table_key].get("variant") == "consolidated"]

    candidates = [
        x for x in candidates
        if x.row_index > 0 and x.column_index > 0 and x.column_header.strip()
        and is_numeric_data_column(tables_by_key[x.table_key], x.column_index, x.column_header)
        and not any(token in x.column_header.lower() for token in ("mã số", "thuyết minh", "note", "biến động", "%"))
    ]
    if not candidates:
        raise ValueError("no numeric direct-lookup operand satisfies entity and structural constraints")

    # An explicit year is evidence, not a soft preference.  A comparative column in a
    # neighbouring report is still valid if its header names the requested year.
    if years:
        year_candidates = [
            x for x in candidates
            if str(tables_by_key[x.table_key].get("year", "")) in years
            or any(year in x.column_header for year in years)
        ]
        if not year_candidates:
            raise ValueError("no direct-lookup operand matches the requested year")
        candidates = year_candidates

    # Only filter when the table representation distinguishes opening from closing
    # balances.  Otherwise fail neither silently nor by forcing a guessed convention.
    requested_period = "cuối năm" if "cuối năm" in question else "đầu năm" if "đầu năm" in question else None
    if requested_period:
        period_candidates = [
            x for x in candidates
            if requested_period in f"{x.row_label} {x.column_header}".lower()
        ]
        opposite_period = "đầu năm" if requested_period == "cuối năm" else "cuối năm"
        has_explicit_period = any(
            requested_period in f"{x.row_label} {x.column_header}".lower()
            or opposite_period in f"{x.row_label} {x.column_header}".lower()
            for x in candidates
        )
        if has_explicit_period and not period_candidates:
            raise ValueError(f"no direct-lookup operand matches {requested_period}")
        if period_candidates:
            candidates = period_candidates

    def score(operand) -> tuple:
        table = tables_by_key[operand.table_key]
        header, row = operand.column_header.lower(), operand.row_label.lower()
        year_score = 20 if str(table.get("year", "")) in years else 0
        current_score = 10 if str(table.get("year", "")) in years and "năm nay" in header else 0
        period_score = 0  # period is filtered above when the source exposes it
        identity = " ".join(map(str, (table.get("table_identity", ""), table.get("caption_context", "")))).lower()
        narrative_penalty = 25 if any(token in identity for token in ("giải trình", "thuyết minh", "explan")) else 0
        return (operand.relevance_score * 100 + year_score + current_score + period_score - narrative_penalty,
                operand.table_key, -operand.row_index, -operand.column_index)

    operand = max(candidates, key=score)
    table = tables_by_key[operand.table_key]
    source_scale = _source_scale_to_vnd(table, operand.column_index, operand.column_header)
    target_scale = unit_scale_to_vnd(linked.requested_unit)
    if source_scale is None and target_scale is not None:
        raise ValueError("selected direct-lookup column has no unambiguous source unit")
    source_scale = source_scale or 1
    target_scale = target_scale or source_scale
    factor = source_scale / target_scale
    expression = "x" if factor == 1 else f"x * {factor!r}"
    return QueryPlan(
        [BoundOperand("x", operand.table_key, operand.row_index, operand.column_index)],
        expression,
        linked.requested_unit,
        "deterministic direct lookup with entity, period, and column-unit constraints",
    )


def build_evidence_frames(plan: QueryPlan, tables_by_key: dict[str, dict]) -> tuple[dict, dict]:
    frames, variables = {}, {}
    for index, key in enumerate(dict.fromkeys(x.table_key for x in plan.operands), 1):
        variable = f"df{index}"
        table = tables_by_key[key]
        rows = []
        for ri, row in enumerate(table["grid"]):
            for ci, text in enumerate(row):
                rows.append({"row_index": ri, "column_index": ci, "text": text,
                             "numeric_value": parse_financial_number(text)})
        frames[variable] = pd.DataFrame(rows)
        variables[key] = variable
    return frames, variables


def render_pandas_query(plan: QueryPlan, variables: dict[str, str]) -> str:
    rendered = plan.expression
    for operand in sorted(plan.operands, key=lambda x: -len(x.alias)):
        var = variables[operand.table_key]
        lookup = (f"float({var}.loc[({var}['row_index']=={operand.row_index}) & "
                  f"({var}['column_index']=={operand.column_index}), 'numeric_value'].iloc[0])")
        rendered = re.sub(rf"\b{re.escape(operand.alias)}\b", lookup, rendered)
    return rendered


def build_prompt(linked: SchemaLinkResult, feedback: str | None = None) -> str:
    contract = {
        "operands": [{"alias": "x", "table_key": "allowed key", "row_index": 0, "column_index": 1}],
        "expression": "x", "answer_unit": "requested unit or null", "rationale": "short",
    }
    return (
        "You generate a grounded numeric query plan for Vietnamese financial tables. "
        "Use ONLY allowed operands. Never invent tables, coordinates, columns, rows, or values. "
        "Expression may use aliases, numeric constants, + - * / ** and abs only. Return JSON only.\n"
        f"OUTPUT_SCHEMA={json.dumps(contract, ensure_ascii=False)}\n"
        f"CONTEXT={json.dumps(linked.prompt_payload(), ensure_ascii=False)}\n"
        f"REPAIR_FEEDBACK={feedback or 'none'}"
    )


class QwenAWQGenerator:
    """Lazy offline adapter; model_path must point to an attached Kaggle Dataset."""
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", local_files_only=True, low_cpu_mem_usage=True
        ).eval()

    def __call__(self, prompt: str) -> str:
        import torch
        prompt, inputs = self._fit_prompt(prompt)
        context_values = (
            getattr(self.model.config, "max_position_embeddings", 0),
            getattr(self.tokenizer, "model_max_length", 0),
        )
        valid_contexts = [
            int(value) for value in context_values
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        if not valid_contexts:
            raise PromptBudgetError("model exposes no finite context window")
        context_limit = min(valid_contexts)
        # Transformers 5 returns a BatchEncoding here, not a bare tensor.  Reading
        # ``.shape`` on it raises an empty AttributeError and hides the real cause.
        input_ids = inputs["input_ids"]
        input_tokens = int(input_ids.shape[-1])
        self.last_prompt_diagnostics = {
            "context_limit": context_limit,
            "input_tokens": input_tokens,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_operands": prompt.count('"table_key"'),
            "input_budget": self._input_token_budget(context_limit),
        }
        if input_tokens > self._input_token_budget(context_limit):
            raise PromptBudgetError(
                f"prompt has {input_tokens} tokens; input budget is {self._input_token_budget(context_limit)}"
            )
        # The embedding layer, unlike model.device, stays correct with device_map="auto".
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
        return self.tokenizer.decode(output[0][input_tokens:], skip_special_tokens=True)

    def _input_token_budget(self, context_limit: int) -> int:
        configured = int(os.environ.get("VIFINQA_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS))
        if configured <= 0:
            raise PromptBudgetError("VIFINQA_MAX_INPUT_TOKENS must be positive")
        return min(configured, context_limit - MAX_NEW_TOKENS - PROMPT_SAFETY_MARGIN)

    def _tokenize_prompt(self, prompt: str):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_tensors="pt",
        )

    def _fit_prompt(self, prompt: str):
        """Trim grounded operands to a tokenizer- and VRAM-safe input budget."""
        context_values = (getattr(self.model.config, "max_position_embeddings", 0),
                          getattr(self.tokenizer, "model_max_length", 0))
        valid_contexts = [int(value) for value in context_values if isinstance(value, int) and 0 < value < 1_000_000]
        if not valid_contexts:
            raise PromptBudgetError("model exposes no finite context window")
        context_limit = min(valid_contexts)
        budget = self._input_token_budget(context_limit)
        prefix, separator, tail = prompt.partition("CONTEXT=")
        if not separator or "\nREPAIR_FEEDBACK=" not in tail:
            inputs = self._tokenize_prompt(prompt)
            self.visible_operands = None
            return prompt, inputs
        context_text, feedback = tail.rsplit("\nREPAIR_FEEDBACK=", 1)
        context = json.loads(context_text)
        candidates = context.get("allowed_operands", [])
        chosen = []
        for candidate in candidates:
            proposal = dict(context, allowed_operands=chosen + [candidate])
            candidate_prompt = prefix + "CONTEXT=" + json.dumps(proposal, ensure_ascii=False, separators=(",", ":")) + "\nREPAIR_FEEDBACK=" + feedback
            if int(self._tokenize_prompt(candidate_prompt)["input_ids"].shape[-1]) > budget:
                break
            chosen.append(candidate)
        if candidates and not chosen:
            raise PromptBudgetError(f"first grounded operand exceeds {budget}-token input budget")
        compact_context = dict(context, allowed_operands=chosen)
        fitted_prompt = prefix + "CONTEXT=" + json.dumps(compact_context, ensure_ascii=False, separators=(",", ":")) + "\nREPAIR_FEEDBACK=" + feedback
        inputs = self._tokenize_prompt(fitted_prompt)
        self.visible_operands = {
            (x["table_key"], x["row_index"], x["column_index"]) for x in chosen
        }
        return fitted_prompt, inputs


def generate_plan(linked: SchemaLinkResult, complete: Callable[[str], str], feedback: str | None = None) -> QueryPlan:
    prompt_linked = SchemaLinkResult(
        question=linked.question,
        query_family=linked.query_family,
        requested_unit=linked.requested_unit,
        table_keys=linked.table_keys,
        operands=linked.operands[:MAX_PROMPT_OPERANDS],
    )
    payload = _extract_json(complete(build_prompt(prompt_linked, feedback)))
    visible_operands = getattr(complete, "visible_operands", None)
    if visible_operands is not None:
        prompt_linked = SchemaLinkResult(
            question=prompt_linked.question, query_family=prompt_linked.query_family,
            requested_unit=prompt_linked.requested_unit, table_keys=prompt_linked.table_keys,
            operands=[x for x in prompt_linked.operands if (x.table_key, x.row_index, x.column_index) in visible_operands],
        )
    return validate_plan(payload, prompt_linked)
