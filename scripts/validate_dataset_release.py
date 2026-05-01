from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = REPO_ROOT / "configs" / "object_split.yaml"
REQUIRED_SAMPLE_KEYS = {
    "sample_id",
    "split",
    "episode_index",
    "episode_seed",
    "view_index",
    "candidate_index",
    "object_id",
    "rgb_path",
    "depth_path",
    "camera",
    "candidate",
    "projection",
    "label",
}
REQUIRED_CAMERA_KEYS = {
    "width",
    "height",
    "eye",
    "target",
    "up",
    "fov",
    "near",
    "far",
    "intrinsics",
    "extrinsics",
    "view_matrix",
    "projection_matrix",
}
REQUIRED_CANDIDATE_KEYS = {
    "grasp_position_world",
    "approach_position_world",
    "lift_position_world",
    "yaw",
    "target_opening",
    "xy_offset",
}
REQUIRED_LABEL_KEYS = {
    "success",
    "physical_lift_success",
    "lifted_delta",
    "supported_after_close",
    "supported_after_lift",
    "contact_links_after_close",
    "contact_links_after_lift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a generated dataset as a release candidate."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--object-split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-train-episodes", type=int, default=600)
    parser.add_argument("--expected-test-episodes", type=int, default=180)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip full-file SHA256 manifest generation.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def normalize_asset_path(path_value: str) -> str:
    normalized = path_value.replace("\\", "/")
    for prefix in ("C:/PyBulletAssets/", "C:/Tez/recovered_repo/.venvfix/Lib/site-packages/pybullet_data/"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    marker = "/pybullet_data/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample_schema(sample: dict[str, Any], errors: list[str], context: str) -> None:
    missing = REQUIRED_SAMPLE_KEYS - sample.keys()
    if missing:
        errors.append(f"{context}: missing sample keys: {sorted(missing)}")
        return

    for section_name, required_keys in (
        ("camera", REQUIRED_CAMERA_KEYS),
        ("candidate", REQUIRED_CANDIDATE_KEYS),
        ("label", REQUIRED_LABEL_KEYS),
    ):
        section = sample.get(section_name)
        if not isinstance(section, dict):
            errors.append(f"{context}: {section_name} is not an object")
            continue
        missing_section = required_keys - section.keys()
        if missing_section:
            errors.append(
                f"{context}: missing {section_name} keys: {sorted(missing_section)}"
            )

    projection = sample.get("projection")
    if not isinstance(projection, dict):
        errors.append(f"{context}: projection is not an object")
    elif "visible" not in projection or "in_frame" not in projection:
        errors.append(f"{context}: projection missing visible/in_frame fields")


def validate_split(
    dataset_root: Path,
    split_name: str,
    expected_episodes: int,
    split_assets: set[str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    split_root = dataset_root / split_name
    index_path = split_root / "index.jsonl"
    samples_path = split_root / "samples.jsonl"
    if not split_root.exists():
        errors.append(f"{split_name}: split directory missing")
        return {"split": split_name, "episodes": 0, "samples": 0}
    if not index_path.exists():
        errors.append(f"{split_name}: index.jsonl missing")
        return {"split": split_name, "episodes": 0, "samples": 0}
    if not samples_path.exists():
        errors.append(f"{split_name}: samples.jsonl missing")
        return {"split": split_name, "episodes": 0, "samples": 0}

    index_rows = load_jsonl(index_path)
    sample_rows = load_jsonl(samples_path)
    episode_indices = [int(row["episode_index"]) for row in index_rows]
    expected_indices = list(range(expected_episodes))
    if episode_indices != expected_indices:
        errors.append(
            f"{split_name}: episode indices are not exactly 0..{expected_episodes - 1}"
        )

    sample_ids = [str(row.get("sample_id")) for row in sample_rows]
    duplicate_sample_ids = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
    if duplicate_sample_ids:
        errors.append(
            f"{split_name}: duplicate sample_id values, first examples: {duplicate_sample_ids[:10]}"
        )

    samples_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    success_count = 0
    visible_count = 0
    in_frame_count = 0
    asset_paths: set[str] = set()
    size_bins: Counter[str] = Counter()

    index_by_episode = {int(row["episode_index"]): row for row in index_rows}
    for sample_index, sample in enumerate(sample_rows):
        context = f"{split_name}:sample[{sample_index}]"
        validate_sample_schema(sample, errors, context)
        if sample.get("split") != split_name:
            errors.append(f"{context}: split mismatch")
        episode_index = int(sample.get("episode_index", -1))
        samples_by_episode[episode_index].append(sample)
        if episode_index not in index_by_episode:
            errors.append(f"{context}: references episode not in index: {episode_index}")

        rgb_path = split_root / str(sample.get("rgb_path", ""))
        depth_path = split_root / str(sample.get("depth_path", ""))
        if not rgb_path.exists():
            errors.append(f"{context}: missing rgb_path {rgb_path}")
        if not depth_path.exists():
            errors.append(f"{context}: missing depth_path {depth_path}")

        label = sample.get("label", {})
        projection = sample.get("projection", {})
        success_count += int(bool(label.get("success")))
        visible_count += int(bool(projection.get("visible")))
        in_frame_count += int(bool(projection.get("in_frame")))

    for row in index_rows:
        episode_index = int(row["episode_index"])
        relative_dir = str(row["relative_dir"])
        episode_dir = split_root / relative_dir
        metadata_path = episode_dir / "metadata.json"
        if not episode_dir.exists():
            errors.append(f"{split_name}: episode folder missing: {relative_dir}")
            continue
        if not metadata_path.exists():
            errors.append(f"{split_name}: metadata missing: {relative_dir}")
            continue

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if int(metadata.get("episode_index", -1)) != episode_index:
            errors.append(f"{split_name}: metadata episode mismatch in {relative_dir}")
        if metadata.get("split") != split_name:
            errors.append(f"{split_name}: metadata split mismatch in {relative_dir}")
        if int(metadata.get("total_samples", -1)) != len(samples_by_episode[episode_index]):
            errors.append(f"{split_name}: total_samples mismatch in {relative_dir}")

        for view in metadata.get("views", []):
            view_dir = split_root / str(view.get("relative_dir", ""))
            for file_name in ("rgb.png", "depth.npy", "segmentation.npy"):
                if not (view_dir / file_name).exists():
                    errors.append(f"{split_name}: missing {file_name} in {view_dir}")

        for spawned in metadata.get("spawned_objects", []):
            asset = normalize_asset_path(str(spawned.get("path", "")))
            asset_paths.add(asset)
            if split_assets and asset not in split_assets:
                errors.append(f"{split_name}: asset not in configured split: {asset}")

        for catalog_entry in metadata.get("catalog_entries", {}).values():
            if isinstance(catalog_entry, dict):
                size_bin = str(catalog_entry.get("size_bin", "unknown"))
                size_bins[size_bin] += 1

    orphan_episode_dirs = sorted(
        path.name
        for path in split_root.glob("episode_*")
        if path.is_dir() and path.name not in {str(row["relative_dir"]) for row in index_rows}
    )
    if orphan_episode_dirs:
        errors.append(
            f"{split_name}: orphan episode directories, first examples: {orphan_episode_dirs[:10]}"
        )

    sample_total = len(sample_rows)
    summary = {
        "split": split_name,
        "episodes": len(index_rows),
        "expected_episodes": expected_episodes,
        "samples": sample_total,
        "successes": success_count,
        "success_rate": success_count / sample_total if sample_total else None,
        "visible_rate": visible_count / sample_total if sample_total else None,
        "in_frame_rate": in_frame_count / sample_total if sample_total else None,
        "unique_asset_paths": len(asset_paths),
        "asset_paths": sorted(asset_paths),
        "size_bins": dict(sorted(size_bins.items())),
    }
    if summary["unique_asset_paths"] < 100:
        warnings.append(
            f"{split_name}: only {summary['unique_asset_paths']} unique assets appeared in generated episodes"
        )
    return summary


def build_checksum_manifest(dataset_root: Path, output_path: Path) -> dict[str, Any]:
    files = [
        path
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file() and "release_validation" not in path.parts and "audit" not in path.parts
    ]
    total_bytes = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for path in files:
            stat = path.stat()
            total_bytes += stat.st_size
            record = {
                "path": str(path.relative_to(dataset_root)).replace("\\", "/"),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "files": len(files),
        "bytes": total_bytes,
        "manifest_path": str(output_path),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    status = "PASS" if report["release_pass"] else "FAIL"
    lines = [
        "# Dataset Release Validation",
        "",
        f"- Status: {status}",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Errors: {len(report['errors'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
    ]
    for split_name, split in report["splits"].items():
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- Episodes: {split['episodes']} / {split['expected_episodes']}",
                f"- Samples: {split['samples']}",
                f"- Success rate: {split['success_rate']}",
                f"- Visible rate: {split['visible_rate']}",
                f"- In-frame rate: {split['in_frame_rate']}",
                f"- Unique asset paths: {split['unique_asset_paths']}",
                f"- Size bins: `{json.dumps(split['size_bins'], sort_keys=True)}`",
                "",
            ]
        )
    leakage = report["split_leakage"]
    lines.extend(
        [
            "## Split Integrity",
            "",
            f"- Train/test asset overlap: {leakage['overlap_count']}",
            "",
            "## Checksums",
            "",
            f"- Files: {report['checksums'].get('files')}",
            f"- Bytes: {report['checksums'].get('bytes')}",
            f"- Manifest: `{report['checksums'].get('manifest_path')}`",
            "",
        ]
    )
    if report["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"][:100])
        lines.append("")
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"][:100])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.dataset_root / "release_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.object_split.open("r", encoding="utf-8") as handle:
        split_config = yaml.safe_load(handle)
    configured_assets = {
        split_name: {str(path) for path in paths}
        for split_name, paths in split_config.items()
    }

    errors: list[str] = []
    warnings: list[str] = []
    train_summary = validate_split(
        args.dataset_root,
        "train",
        args.expected_train_episodes,
        configured_assets.get("train", set()),
        errors,
        warnings,
    )
    test_summary = validate_split(
        args.dataset_root,
        "test",
        args.expected_test_episodes,
        configured_assets.get("test", set()),
        errors,
        warnings,
    )

    train_assets = set(train_summary.get("asset_paths", []))
    test_assets = set(test_summary.get("asset_paths", []))
    overlap = sorted(train_assets & test_assets)
    if overlap:
        errors.append(f"train/test asset leakage, first examples: {overlap[:10]}")

    checksums = {"files": None, "bytes": None, "manifest_path": None}
    if not args.skip_checksums:
        checksums = build_checksum_manifest(
            args.dataset_root,
            output_dir / "checksums.sha256.jsonl",
        )

    report = {
        "dataset_root": str(args.dataset_root),
        "object_split": str(args.object_split),
        "release_pass": not errors,
        "splits": {
            "train": {key: value for key, value in train_summary.items() if key != "asset_paths"},
            "test": {key: value for key, value in test_summary.items() if key != "asset_paths"},
        },
        "split_leakage": {
            "overlap_count": len(overlap),
            "overlap_examples": overlap[:20],
        },
        "checksums": checksums,
        "errors": errors,
        "warnings": warnings,
    }

    with (output_dir / "release_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_markdown(output_dir / "RELEASE_REPORT.md", report)

    status = "PASS" if report["release_pass"] else "FAIL"
    print(f"Release validation {status}: {output_dir}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
