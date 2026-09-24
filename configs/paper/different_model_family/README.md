# Second Actor-Family Bundle

Paper-facing replication bundles that hold the protocol fixed and vary the
actor model. The evaluator model is kept fixed at `openai/gpt-oss-120b` in every
bundle so that the answer-comparison mechanism is identical across families.

Actor families:

- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `google/gemma-3-27b-it`
- `google/gemma-4-31B-it`
- `google/gemma-4-E4B-it`

Each family has one directory per protocol:

```text
baseline_<family>            # Direct Baseline
single_agent_hint_<family>   # Single-Agent Iterative
per_hint_<family>            # Planner-Executor-Reviewer
broadcast_hint_<family>      # Broadcast Deliberation
```

## Running one bundle

Set the endpoint environment variables described in the repository `README.md`
and `.env.example`, then run from the repository root:

```bash
agentverse-benchmark \
  --task configs/paper/different_model_family/per_hint_llama31_70b_eval_oss120b \
  --dataset_path "$SCIAGENTTRACE_DATA_DIR/omni-math-2-filtered/tier_01.jsonl" \
  --output_path "$SCIAGENTTRACE_RESULTS_DIR/second_family/per_hint_llm" \
  --trace \
  --overwrite
```

Swap the last path component of `--task` to run the other three protocols.

## Scaling notes (site-independent)

- Concurrency is controlled by `AGENTVERSE_RATE_SOFT_LIMIT`. Choose it from a
  short smoke ladder against your own serving endpoint rather than copying a
  value; per-model rate and timeout behaviour differs between deployments.
- Run a one-question and then a single-tier smoke before launching a full
  dataset file.
- Larger actor models have shown deployment-specific rate-limit and timeout
  sensitivity under the long multi-agent prompt patterns used by `per_hint_*`
  and `broadcast_hint_*`.

Batch-scheduler submission is site-specific and is intentionally not included
in this release. Any scheduler that can invoke `agentverse-benchmark` with a
dataset shard and an output path is sufficient.
