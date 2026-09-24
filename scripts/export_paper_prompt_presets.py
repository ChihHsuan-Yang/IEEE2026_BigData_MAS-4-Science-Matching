#!/usr/bin/env python3
"""Export inline paper prompts into reusable prompt preset files."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_MODES = (
    "baseline_llm",
    "single_agent_hint_llm",
    "per_hint_llm",
    "broadcast_hint_llm",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", default="configs/paper")
    parser.add_argument("--output", default="configs/paper/prompt_presets/omni_math")
    args = parser.parse_args()

    config_root = Path(args.config_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for mode in DEFAULT_MODES:
        config_path = config_root / mode / "config.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        prompts = payload.get("prompts") or {}
        if not prompts:
            raise ValueError(f"No top-level prompts found in {config_path}")
        preset = {
            "metadata": {
                "source_config": str(config_path),
                "domain": "omni_math",
                "note": "Mirror of the current inline Omni-MATH prompt block. Existing Omni-MATH configs remain inline/default.",
            },
            "prompts": prompts,
        }
        out_path = output_root / f"{mode}.yaml"
        out_path.write_text(
            yaml.safe_dump(preset, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
