"""Execute grounded Pandas queries and expose every generation/repair attempt."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable

from query_generation.generator import (
    build_evidence_frames,
    generate_direct_lookup_plan,
    generate_plan,
    render_pandas_query,
)
from schema_linking.linker import SchemaLinkResult


def _is_oom_error(exc: Exception) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


@dataclass
class ExecutionAttempt:
    attempt: int
    pandas_query: str | None
    error_type: str | None
    error_message: str | None
    stage: str
    traceback: str | None = None
    prompt_diagnostics: dict | None = None


@dataclass
class ExecutionResult:
    success: bool
    answer: float | None
    pandas_query: str | None
    evidence_frames: dict
    attempts: list[ExecutionAttempt]
    plan: object | None


def execute_query(query: str, frames: dict) -> float:
    result = eval(query, {"__builtins__": {}, "float": float, "abs": abs}, frames)  # noqa: S307
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise TypeError(f"query result must be numeric, got {type(result).__name__}")
    return float(result)


def execute_with_repair(
    linked: SchemaLinkResult,
    tables_by_key: dict[str, dict],
    complete: Callable[[str], str],
    max_retries: int = 2,
) -> ExecutionResult:
    attempts, feedback = [], None
    for attempt_no in range(max_retries + 1):
        query = None
        stage = "generation_or_validation"
        try:
            plan = (
                generate_direct_lookup_plan(linked, tables_by_key)
                if linked.query_family == "direct_lookup"
                else generate_plan(linked, complete, feedback)
            )
            stage = "render"
            frames, variables = build_evidence_frames(plan, tables_by_key)
            query = render_pandas_query(plan, variables)
            stage = "execution"
            answer = execute_query(query, frames)
            attempts.append(ExecutionAttempt(
                attempt_no, query, None, None, stage,
                prompt_diagnostics=getattr(complete, "last_prompt_diagnostics", None),
            ))
            return ExecutionResult(True, answer, query, frames, attempts, plan)
        except Exception as exc:  # error is recorded and explicitly fed back on the next pass
            attempts.append(ExecutionAttempt(
                attempt_no, query, type(exc).__name__, str(exc), stage,
                traceback=traceback.format_exc(),
                prompt_diagnostics=getattr(complete, "last_prompt_diagnostics", None),
            ))
            if _is_oom_error(exc):
                break
            feedback = f"{type(exc).__name__}: {exc}"
    return ExecutionResult(False, None, query, {}, attempts, None)
