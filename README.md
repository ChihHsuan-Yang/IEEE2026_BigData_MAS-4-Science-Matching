# SciAgentTrace — Protocol Execution and Trace Processing

Code accompanying **SciAgentTrace**, a matched multi-agent scientific-reasoning
trace corpus. This repository contains the two things needed to reproduce the
corpus end to end:

1. **Protocol execution** — the runtime that executes four multi-agent protocols
   over a benchmark of scientific and mathematical problems, emitting a
   structured execution trace for every problem it solves.
2. **Trace processing** — the scripts that normalize those raw traces into the
   released tables (per-problem outcomes, token and call accounting, wall-time,
   and aggregate metrics).

The released corpus is on the Hugging Face Hub:
**https://huggingface.co/datasets/AgentsSci/scientific-agent-protocol-traces**

Documentation site:
**https://huggingface.co/spaces/AgentsSci/scientific-agent-protocol-traces-site**

> **Paper status.** This release accompanies a paper under submission to
> IEEE BigData 2026. The citation block below is a placeholder and will be
> replaced with the proceedings reference if the paper is accepted.

---

## The four protocols

Every protocol solves the same problem with the same actor model and is scored
by the same constrained evaluator, so outcomes are matched per problem. The only
thing that varies is the interaction structure.

| Protocol | Config directory | `env_type` | Structure |
|---|---|---|---|
| **Direct Baseline** | `configs/paper/baseline_llm` | `task-single-llm-benchmark` | One solver call, one evaluator call. No revision. |
| **Single-Agent Iterative** | `configs/paper/single_agent_hint_llm` | `task-single-agent-reflect` | One solver revises its own answer across rounds under evaluator feedback. |
| **Planner-Executor-Reviewer (PER)** | `configs/paper/per_hint_llm` | `task-per` | Three roles in a loop: a planner proposes, an executor carries out, a reviewer critiques. |
| **Broadcast Deliberation** | `configs/paper/broadcast_hint_llm` | `task-broadcast-deliberation` | Several deliberators see one another's contributions each round and converge on a joint answer. |

The evaluator model is held fixed across protocols so the answer-comparison
mechanism never varies with the interaction structure being measured.

---

## Repository layout

```text
agentverse/                    Protocol runtime
  agents/tasksolving_agent/    Solver, critic, deliberator, evaluator, executor, manager roles
  environments/tasksolving_env/
    single_llm_benchmark.py    Direct Baseline
    single_agent_reflect.py    Single-Agent Iterative
    per.py                     Planner-Executor-Reviewer
    broadcast_deliberation.py  Broadcast Deliberation
    rules/                     Per-protocol turn/stop/feedback rules
  llms/                        OpenAI-compatible client: throttling, retry, token accounting
  evaluation/                  Answer extraction and equivalence checking
  memory/                      Chat-history and summary memory
  metrics/                     Trace logging, run aggregation, per-paper metric derivation
  output_parser/               Model-output parsers per agent role

agentverse_command/            Command-line entry points
  benchmark.py                 Run one protocol over a dataset  -> agentverse-benchmark
  trace_analysis.py            Outcome/cost + loop metrics      -> agentverse-trace-analysis
  trace_diagnostics.py         Supplemental process diagnostics
  rq1_outcome_cost.py          Outcome and cost report
  config_resolver.py           Resolve config directories and dataset loaders

dataloader/                    Benchmark readers (Omni-MATH, MGSM, GSM8K, generic JSONL)

configs/paper/                 Run configurations
  baseline_llm/ single_agent_hint_llm/ per_hint_llm/ broadcast_hint_llm/
                               Primary four-protocol bundle
  science_qa/                  Science-QA benchmark bundle, per actor family
  lab_bench/                   LAB-Bench text-only bundle, per actor family and slice
  different_model_family/      Second actor-family replication bundles
  prompt_presets/              Prompt and evaluator wording, shared across configs

scripts/                       Benchmark preparation and trace processing
  prepare_science_benchmarks.py  Build JEEBench / SciBench / MaScQA inputs
  prepare_lab_bench_llm.py       Build LAB-Bench text-only slices
  prepare_omni_math_test_sets.py, preprocess_omni_math.py
  generate_science_qa_configs.py, generate_lab_bench_configs.py
  aggregate_science_qa_runs.py, aggregate_science_qa_coupling.py,
  aggregate_lab_bench_chunks.py  Aggregate sharded runs into per-protocol tables
  dataset_release/
    prepare_hf_protocol_dataset_v2.py  Build the released table layout from raw traces
    build_hf_trace_manifest.py         Trace manifest + SHA-256 checksums
    stage_canonical_classic_run.py, merge_classic_recovery.py,
    audit_classic_recovery.py          Stage and audit a canonical run

tests/                         Unit tests for the trace-processing path
```

---

## Install

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the test suite:

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

The optional local Omni-Judge evaluator (only for `--extra_post_eval omni-judge`)
pulls in `torch` and `transformers`:

```bash
pip install -r requirements-omni-judge.txt
```

Run commands from the repository root, or install in editable mode
(`pip install -e .`) to get the `agentverse-*` console scripts on your `PATH`.
Where this README shows `agentverse-benchmark`, the equivalent without an
editable install is `python -m agentverse_command.benchmark`.

## Configure

Copy `.env.example` to `.env` and fill it in, or export the variables directly.
`.env` is git-ignored; never commit real keys.

The only two required variables are the endpoint:

```bash
export OPENAI_API_KEY=...                            # your key or access token
export OPENAI_BASE_URL=https://your-endpoint/v1      # any OpenAI-compatible server
```

Any OpenAI-compatible chat-completions endpoint works: the OpenAI API, a
self-hosted vLLM server, or an institutional inference gateway. The runs behind
the released corpus used open-weight models (`openai/gpt-oss-120b`,
`google/gemma-3-27b-it`, `google/gemma-4-31B-it`, `google/gemma-4-E4B-it`,
`meta-llama/Meta-Llama-3.1-70B-Instruct`) served behind such an endpoint.

`.env.example` documents the optional variables: gateway-specific model
aliasing and token refresh, split-backend routing, throttling and retry caps,
and the data/results directories.

Concurrency is set by `AGENTVERSE_RATE_SOFT_LIMIT`. The right value is a
property of your endpoint, not of this code — choose it from a short smoke
ladder rather than copying a number.

---

## Prepare benchmark inputs

The benchmarks are third-party datasets and are not redistributed here. Build
the JSONL inputs locally:

```bash
# Science QA: JEEBench, SciBench, MaScQA
python scripts/prepare_science_benchmarks.py --output-root "$SCIAGENTTRACE_DATA_DIR/science-benchmarks"

# LAB-Bench, text-only slices
python scripts/prepare_lab_bench_llm.py --output-root "$SCIAGENTTRACE_DATA_DIR/lab-bench-public"

# Omni-MATH
python scripts/preprocess_omni_math.py --help
```

Each writes JSONL rows carrying at least `input` and `answer`, plus source,
license, and difficulty metadata that the release tables later cite.

## Run a protocol

One protocol over one dataset, with tracing on:

```bash
agentverse-benchmark \
  --task configs/paper/per_hint_llm \
  --dataset_path "$SCIAGENTTRACE_DATA_DIR/omni-math-2-filtered/all.jsonl" \
  --output_path "$SCIAGENTTRACE_RESULTS_DIR/omni-math2/per_hint_llm" \
  --trace \
  --overwrite
```

Swap `--task` for the other three protocols:

```bash
configs/paper/baseline_llm            # Direct Baseline
configs/paper/single_agent_hint_llm   # Single-Agent Iterative
configs/paper/per_hint_llm            # Planner-Executor-Reviewer
configs/paper/broadcast_hint_llm      # Broadcast Deliberation
```

Useful flags:

| Flag | Effect |
|---|---|
| `--trace` | Write the full execution trace, not just the outcome |
| `--example_start N` / `--example_end M` | Run a shard of the dataset |
| `--extra_post_eval MODE` | Second evaluation layer: `exact`, `numeric-verifier`, `omni-judge`, `omni-verifier`, `omni-rule` |
| `--overwrite` | Replace an existing output directory |

Smoke-test one question before launching a full run:

```bash
AGENTVERSE_RATE_SOFT_LIMIT=2 agentverse-benchmark \
  --task configs/paper/baseline_llm \
  --dataset_path "$SCIAGENTTRACE_DATA_DIR/omni-math-2-filtered/all.jsonl" \
  --example_start 0 --example_end 1 \
  --output_path "$SCIAGENTTRACE_RESULTS_DIR/smoke/baseline_llm" \
  --trace --overwrite
```

Large runs are sharded by passing disjoint `--example_start`/`--example_end`
windows to parallel processes, each with its own `--output_path`. Any scheduler
that can invoke `agentverse-benchmark` is sufficient; no batch-scheduler scripts
are included, because they are site-specific.

### Running the full matched matrix

A matched cell is one (benchmark, actor model, protocol) triple. Loop the four
protocols over one benchmark and actor family:

```bash
for p in baseline_llm single_agent_hint_llm per_hint_llm broadcast_hint_llm; do
  agentverse-benchmark \
    --task "configs/paper/science_qa/gpt_oss_120b/$p" \
    --dataset_path "$SCIAGENTTRACE_DATA_DIR/science-benchmarks/jeebench/all.jsonl" \
    --output_path "$SCIAGENTTRACE_RESULTS_DIR/science-qa/gpt_oss_120b/$p" \
    --trace --overwrite
done
```

`configs/paper/different_model_family/` holds the same four protocols for the
other actor families.

## Regenerate the tables

After the runs finish, aggregate shards and build the release tables:

```bash
# 1. Aggregate sharded runs into per-protocol result files
python scripts/aggregate_science_qa_runs.py --help
python scripts/aggregate_lab_bench_chunks.py --help

# 2. Outcome and cost metrics per protocol
agentverse-trace-analysis --help

# 3. Build the released table layout from raw traces
python scripts/dataset_release/prepare_hf_protocol_dataset_v2.py \
  --repo-root . \
  --staging "$SCIAGENTTRACE_RESULTS_DIR/hf_staging" \
  --clean

# 4. Trace manifest and SHA-256 checksums
python scripts/dataset_release/build_hf_trace_manifest.py --help
```

Step 3 produces the registry, per-experiment tables, per-tier outcome files,
and sanitized traces that make up the released layout. It also runs a staging
safety scan for local paths and credential shapes and writes the result to
`docs/anonymization_report.md` inside the staging tree. Review that report
before publishing any staged output.

---

## Reproducibility notes

- **Determinism.** Configs run at `temperature: 0`, but decoding at temperature
  zero is not bit-reproducible across serving backends or batch compositions.
  Expect small outcome differences on re-run; the corpus is the record of the
  runs actually executed.
- **Matched design.** The comparison is valid because protocol varies while the
  problem, actor model, and evaluator are held fixed. Changing the evaluator
  between arms breaks the matching.
- **Coverage.** Not every cell of the matrix has every measurement. Coverage is
  recorded explicitly in the released tables rather than being imputed.
- **Benchmarks.** Source datasets are third-party and are not redistributed
  here; the preparation scripts fetch them and preserve their source, license,
  and identifier fields.

## Citation

Placeholder — to be replaced with the proceedings reference.

```bibtex
@misc{yang2026sciagenttrace,
  title        = {SciAgentTrace: A Matched Multi-Agent Scientific Reasoning Trace Corpus},
  author       = {Yang, Chih-Hsuan and Thakur, Rajeev},
  year         = {2026},
  note         = {Under submission to IEEE BigData 2026},
  howpublished = {\url{https://huggingface.co/datasets/AgentsSci/scientific-agent-protocol-traces}}
}
```

## Authors

- **Chih-Hsuan Yang**, Argonne National Laboratory — corresponding author,
  `bellayang@anl.gov`
- **Rajeev Thakur**, Argonne National Laboratory

## License

Apache License 2.0 — see [LICENSE](LICENSE).

This repository derives from [AgentVerse](https://github.com/OpenBMB/AgentVerse)
(Apache 2.0). The four protocol environments, the trace and metrics layer, and
the release-processing scripts are additions made for this work.

Benchmark datasets and model weights carry their own separate licenses.
