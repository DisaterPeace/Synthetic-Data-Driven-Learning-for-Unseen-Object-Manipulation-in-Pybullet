from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from synthetic_grasping import GraspingScene  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a larger train/test object split from PyBullet random URDFs.")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "configs" / "object_catalog" / "object_catalog.json",
        help="Optional audited object catalog. When present, only valid objects are used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs" / "object_split.yaml",
    )
    return parser.parse_args()


def list_random_urdfs() -> list[str]:
    root = GraspingScene.data_root() / "random_urdfs"
    urdfs: list[str] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        urdf_path = directory / f"{directory.name}.urdf"
        if urdf_path.exists():
            urdfs.append(str(urdf_path.relative_to(GraspingScene.data_root())).replace("\\", "/"))
    return urdfs


def load_catalog(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stratified_shuffle_split(
    entries: list[dict[str, object]],
    train_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    size_bins: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        size_bins.setdefault(str(entry.get("size_bin", "unknown")), []).append(entry)
    for bucket_entries in size_bins.values():
        rng.shuffle(bucket_entries)

    total_requested = train_size + test_size
    selected_entries: list[dict[str, object]] = []
    bin_names = sorted(size_bins.keys())
    while len(selected_entries) < total_requested:
        progress_made = False
        for bin_name in bin_names:
            bucket_entries = size_bins[bin_name]
            if bucket_entries and len(selected_entries) < total_requested:
                selected_entries.append(bucket_entries.pop())
                progress_made = True
        if not progress_made:
            break

    if len(selected_entries) < total_requested:
        raise ValueError(
            f"Requested {total_requested} filtered objects, but only found {len(selected_entries)}."
        )

    rng.shuffle(selected_entries)
    return {
        "train": [str(entry["relative_path"]) for entry in selected_entries[:train_size]],
        "test": [str(entry["relative_path"]) for entry in selected_entries[train_size:train_size + test_size]],
    }


def main() -> None:
    args = parse_args()
    if args.catalog.exists():
        catalog_entries = [entry for entry in load_catalog(args.catalog) if entry.get("valid", False)]
        split = stratified_shuffle_split(
            catalog_entries,
            train_size=args.train_size,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        urdfs = list_random_urdfs()
        requested = args.train_size + args.test_size
        if requested > len(urdfs):
            raise ValueError(f"Requested {requested} objects, but only found {len(urdfs)}.")
        rng = random.Random(args.seed)
        rng.shuffle(urdfs)
        split = {
            "train": urdfs[: args.train_size],
            "test": urdfs[args.train_size : args.train_size + args.test_size],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split, handle, sort_keys=False)

    print(
        f"Wrote object split with {len(split['train'])} train objects and "
        f"{len(split['test'])} test objects to {args.output}"
    )


if __name__ == "__main__":
    main()
