# Runtime generation model — Kaggle offline preparation

Approved pipeline component: `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`.

Pinned official revision: `b56cc04415fac88c421533036e44149a5983dd2a` (the official Qwen
weight-upload commit). Do not replace it with a moving `main` revision without a new recorded
experiment and eligibility review.

- Open weights and Apache-2.0 license.
- 7.61B parameters (below the competition's 14B limit).
- Released 2024-09-19 (before the 2026-06-01 cutoff).
- Official Qwen 4-bit AWQ repository; approximately 5.57 GB of weight shards.
- Development assistants such as Claude/ChatGPT are not called by runtime code.

Prepare once on an internet-enabled machine/Kaggle session, pinning a reviewed revision, then
publish the downloaded snapshot as a private Kaggle Dataset. Attach that Dataset to the inference
notebook and set `VIFINQA_QWEN_MODEL_PATH` to its read-only `/kaggle/input/...` directory. Official
inference uses `local_files_only=True`; it must succeed with Kaggle internet disabled.

Until the final notebook embeds every module directly, attach a versioned snapshot of this repo as
a second Kaggle Dataset and set `VIFINQA_CODE_PATH` to its root. This is a code artifact, not a
model/API dependency; its source remains the modules under `src/` and `eval/` in this repository.

The Kaggle Dataset must also contain pinned wheels required by the tested environment
(`transformers`, `accelerate`, and the AWQ backend if the selected Transformers build requires
one). Record their exact versions and the model snapshot commit in the final experiment report;
do not silently download a newer `main` revision during inference.

Preparation call:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    revision="b56cc04415fac88c421533036e44149a5983dd2a",
    local_dir="qwen25-coder-7b-instruct-awq",
)
```

Initial deterministic inference policy: batch size 1, `do_sample=False`, `max_new_tokens=768`,
and compact schema-linked context capped near 12K tokens. Prefer Kaggle T4 x2 for int4 inference;
the pipeline does not require both devices if the model fits on one.
