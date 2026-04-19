from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a generated synthetic grasp dataset and produce release statistics.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing train/test split folders with samples.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional report output directory. Defaults to <dataset-root>/audit.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def summarize_split(split_dir: Path) -> dict[str, object]:
    index_rows = load_jsonl(split_dir / "index.jsonl")
    sample_rows = load_jsonl(split_dir / "samples.jsonl")
    if not sample_rows:
        return {
            "split": split_dir.name,
            "episodes": 0,
            "samples": 0,
        }

    successes = [bool(row["label"]["success"]) for row in sample_rows]
    visible = [bool(row["projection"]["visible"]) for row in sample_rows]
    in_frame = [bool(row["projection"]["in_frame"]) for row in sample_rows]
    depths = [float(row["projection"]["camera_depth"]) for row in sample_rows if row["projection"]["camera_depth"] is not None]
    object_ids = {int(row["object_id"]) for row in sample_rows}
    episode_ids = {int(row["episode_index"]) for row in sample_rows}

    object_success_rates: dict[int, list[int]] = {}
    for row in sample_rows:
        object_success_rates.setdefault(int(row["object_id"]), []).append(int(bool(row["label"]["success"])))

    per_object_success = {
        str(object_id): float(sum(labels) / max(len(labels), 1))
        for object_id, labels in object_success_rates.items()
    }

    return {
        "split": split_dir.name,
        "episodes": len(index_rows) if index_rows else len(episode_ids),
        "samples": len(sample_rows),
        "unique_runtime_object_ids": len(object_ids),
        "success_rate": float(sum(successes) / len(successes)),
        "visible_rate": float(sum(visible) / len(visible)),
        "in_frame_rate": float(sum(in_frame) / len(in_frame)),
        "camera_depth_mean": float(statistics.fmean(depths)) if depths else None,
        "camera_depth_median": float(statistics.median(depths)) if depths else None,
        "per_object_success_rate": per_object_success,
    }


def write_markdown(report_path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Dataset Audit",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Splits audited: {', '.join(summary['splits'].keys())}",
        "",
    ]
    for split_name, split_summary in summary["splits"].items():
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- Episodes: {split_summary.get('episodes', 0)}",
                f"- Samples: {split_summary.get('samples', 0)}",
                f"- Success rate: {split_summary.get('success_rate')}",
                f"- Visible rate: {split_summary.get('visible_rate')}",
                f"- In-frame rate: {split_summary.get('in_frame_rate')}",
                f"- Mean camera depth: {split_summary.get('camera_depth_mean')}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.dataset_root / "audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, dict[str, object]] = {}
    for split_dir in sorted(args.dataset_root.iterdir()):
        if split_dir.is_dir() and (split_dir / "samples.jsonl").exists():
            split_summaries[split_dir.name] = summarize_split(split_dir)

    summary = {
        "dataset_root": str(args.dataset_root),
        "splits": split_summaries,
    }

    with (output_dir / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_markdown(output_dir / "dataset_audit.md", summary)
    print(f"Wrote dataset audit to {output_dir}")


if __name__ == "__main__":
    main()
