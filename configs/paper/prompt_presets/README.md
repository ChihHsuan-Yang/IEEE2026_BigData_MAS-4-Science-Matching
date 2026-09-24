# Paper Prompt Presets

Prompt presets make domain changes explicit without rewriting every protocol config.

- `omni_math/`: archived Omni-MATH prompt and evaluator wording exported from the current paper configs. The main Omni-MATH configs still keep their inline prompt text, so existing math runs remain the default behavior.
- `lab_bench_mc/`: LAB-Bench multiple-choice prompt and evaluator wording used by `configs/paper/lab_bench`.

Configs opt in with:

```yaml
prompt_preset:
  path: configs/paper/prompt_presets/lab_bench_mc/baseline_llm.yaml
```

Then agent prompt fields can use:

```yaml
prepend_prompt_template:
  prompt_ref: solver_prepend_prompt
```

The runtime resolves these references before building agents.
