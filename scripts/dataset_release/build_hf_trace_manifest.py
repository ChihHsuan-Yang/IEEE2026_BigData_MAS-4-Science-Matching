"""Build Hugging Face trace manifests and SHA-256 checksums."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


TRACE_CORPORA = {
    "gpt_oss_120b": {
        "local_root": Path("results/trace/gpt_oss_120b"),
        "hf_root": Path("data/gpt_oss_120b"),
    },
    "gemma3_27b": {
        "local_root": Path("results/trace/gemma3_27b"),
        "hf_root": Path("data/gemma3_27b"),
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_release/huggingface/manifests"),
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Write manifest with empty sha256 values for a quick dry run.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "trace_manifest.csv"
    checksums_path = output_dir / "checksums.sha256"
    summary_path = output_dir / "summary.json"

    rows: list[dict[str, str | int]] = []
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpora": {},
        "total_files": 0,
        "total_size_bytes": 0,
    }

    with checksums_path.open("w", encoding="utf-8") as checksum_f:
        for corpus, spec in TRACE_CORPORA.items():
            local_root = repo_root / spec["local_root"]
            hf_root = spec["hf_root"]
            if not local_root.exists():
                raise FileNotFoundError(f"Missing trace root: {local_root}")

            file_count = 0
            size_bytes = 0
            for local_path in iter_files(local_root):
                rel = local_path.relative_to(local_root)
                hf_path = hf_root / rel
                size = local_path.stat().st_size
                digest = "" if args.skip_sha256 else sha256_file(local_path)
                rows.append(
                    {
                        "corpus": corpus,
                        "local_path": str(local_path.relative_to(repo_root)),
                        "hf_path": str(hf_path),
                        "size_bytes": size,
                        "sha256": digest,
                    }
                )
                if digest:
                    checksum_f.write(f"{digest}  {hf_path}\n")
                file_count += 1
                size_bytes += size

            summary["corpora"][corpus] = {
                "local_root": str(spec["local_root"]),
                "hf_root": str(hf_root),
                "files": file_count,
                "size_bytes": size_bytes,
            }
            summary["total_files"] = int(summary["total_files"]) + file_count
            summary["total_size_bytes"] = int(summary["total_size_bytes"]) + size_bytes

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["corpus", "local_path", "hf_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {checksums_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
