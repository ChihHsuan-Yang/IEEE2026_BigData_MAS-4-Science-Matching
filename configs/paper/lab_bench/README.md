# LAB-Bench Paper Configs

These configs adapt the four paper protocols to the public LAB-Bench text-only slices while keeping the Omni-MATH configs unchanged.

## Protocols

Each model family and slice contains the same four protocol directories:

- `baseline_llm`
- `single_agent_hint_llm`
- `per_hint_llm`
- `broadcast_hint_llm`

Each directory has a `config.yaml` and `benchmark.yaml`. The `benchmark.yaml` default points to the full generated slice, while launch scripts can override `--dataset_path` with a chunk or `first700_shuffled.jsonl`.

## Slices

- `llm_strict`: `CloningScenarios`, `ProtocolQA`, and `SeqQA`. This is the primary LLM-only validation slice because the task evidence is in the prompt.
- `text_no_tool`: the strict slice plus `DbQA`, `LitQA2`, and `SuppQA`. This is broader and should be described as closed-book/no-tool text-only validation, not as a tool-use or retrieval setting.

The generated dataset files and filtering rationale live in `data/lab-bench-public/README.md`.

## Model Families

- `gpt_oss_120b`: actors and constrained evaluator use `openai/gpt-oss-120b`.
- `gemma4_31b_eval_oss120b`: actors use `google/gemma-4-31B-it`, evaluator uses `openai/gpt-oss-120b`.
- `gemma4_e4b_eval_oss120b`: actors use `google/gemma-4-E4B-it`, evaluator uses `openai/gpt-oss-120b`.
- `llama31_70b_eval_oss120b`: actors use `meta-llama/Meta-Llama-3.1-70B-Instruct`, evaluator uses `openai/gpt-oss-120b`.

The evaluator is intentionally fixed to the OSS evaluator for the Gemma actor families so the second-family validation changes the reasoning actors without changing the constrained answer-comparison mechanism.

## Prompt Presets

LAB-Bench configs use `prompt_preset.path: configs/paper/prompt_presets/lab_bench_mc/<protocol>.yaml`. The loader resolves `prompt_ref` entries in each agent config at runtime.

The current Omni-MATH prompt wording has also been exported to `configs/paper/prompt_presets/omni_math/`, but the existing Omni-MATH paper configs remain inline/default so current math runs do not change.

## Output Layout

Full-run outputs default to:

```text
results/lab-bench-public/<slice>/<model_family>/<protocol>/all/
```

Smoke tests use:

```text
results/lab-bench-public/<slice>/smoke/<model_family>/<protocol>/
```

For large parallel runs, prefer one 15-question chunk per worker task. The strict 741-example slice has 50 such chunks.

```text
data/lab-bench-public/<slice>/chunks_15_shuffled/chunk_000.jsonl
results/lab-bench-public/<slice>/<model_family>/<protocol>/chunk_000/
```

For the planned first-pass 700-question run, use chunks `000` through `006`, or use `first700_shuffled.jsonl` if a single job is desired.

The 100-question chunk set is still available for local smoke tests and smaller debugging jobs.

## Smoke Commands

Run one question for one protocol:

```bash
AGENTVERSE_RATE_SOFT_LIMIT=2 agentverse-benchmark \
  --task configs/paper/lab_bench/llm_strict/gpt_oss_120b/baseline_llm \
  --dataset_path "$SCIAGENTTRACE_DATA_DIR/lab-bench-public/llm_strict/chunks_100_shuffled/chunk_000.jsonl" \
  --example_start 0 \
  --example_end 1 \
  --output_path "$SCIAGENTTRACE_RESULTS_DIR/lab-bench-public/llm_strict/smoke/gpt_oss_120b/baseline_llm" \
  --trace \
  --overwrite
```

After chunked runs finish, run the same outcome/cost and coupling analysis CLIs used for the Omni-MATH paper traces, for example `agentverse-trace-analysis` and `agentverse-strict-coupling-rate`, against the LAB-Bench result directory.

## Endpoint Note

Model availability depends on the OpenAI-compatible endpoint you point the runner at (see `.env.example`). Larger actor models have shown deployment-specific rate-limit and backoff behaviour under the broadcast protocol's longer prompts, so run a one-question smoke and a short worker ladder before launching full jobs. The smaller actor family is a practical fallback if the larger one is unavailable on your endpoint.
