from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from synthetic_grasping import GraspingScene, SceneConfig  # noqa: E402


DEFAULT_SCALES = (0.28, 0.34, 0.40, 0.46, 0.52)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an audited object catalog for the PyBullet random URDF pool.")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="Object scales to probe for graspability and physical stability.",
    )
    parser.add_argument("--min-grasp-extent", type=float, default=0.018)
    parser.add_argument("--max-grasp-extent", type=float, default=0.075)
    parser.add_argument("--max-aspect-ratio", type=float, default=5.0)
    parser.add_argument("--settle-steps", type=int, default=360)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "object_catalog",
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


def evaluate_object_at_scale(
    scene: GraspingScene,
    urdf_relative_path: str,
    scale: float,
    settle_steps: int,
) -> dict[str, object]:
    client_id = scene.client_id
    assert client_id is not None
    object_path = scene.resolve_asset_path(urdf_relative_path)
    start_position = (0.63, 0.0, 0.86)
    object_id = None
    try:
        object_id = scene_spawn_object(scene, object_path, scale, start_position)
        initial_position = np.array(
            scene.get_object_pose(object_id)["position"],
            dtype=np.float32,
        )
        scene.step(settle_steps)
        final_pose = scene.get_object_pose(object_id)
        final_position = np.array(final_pose["position"], dtype=np.float32)
        aabb_min, aabb_max = scene.get_object_aabb(object_id)
        extents = (aabb_max - aabb_min).astype(np.float32)
        max_xy_extent = float(max(extents[0], extents[1]))
        min_xy_extent = float(max(min(extents[0], extents[1]), 1e-6))
        aspect_ratio = float(max(extents) / max(min(extents), 1e-6))
        settled_distance = float(np.linalg.norm(final_position - initial_position))
        supported = scene.is_object_supported_by_surface(object_id)
        finite = bool(np.isfinite(extents).all() and np.isfinite(final_position).all())
        return {
            "scale": scale,
            "valid_simulation": finite,
            "supported": supported,
            "final_position": final_position.tolist(),
            "settled_distance": settled_distance,
            "extents": extents.tolist(),
            "max_xy_extent": max_xy_extent,
            "min_xy_extent": min_xy_extent,
            "height": float(extents[2]),
            "aspect_ratio": aspect_ratio,
        }
    finally:
        if object_id is not None:
            import pybullet as p

            p.removeBody(object_id, physicsClientId=client_id)
            scene.spawned_object_ids = [oid for oid in scene.spawned_object_ids if oid != object_id]
            scene.step(2)


def scene_spawn_object(
    scene: GraspingScene,
    urdf_path: Path,
    scale: float,
    position: tuple[float, float, float],
) -> int:
    import pybullet as p

    assert scene.client_id is not None
    object_id = p.loadURDF(
        str(urdf_path),
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, 0.0)),
        useFixedBase=False,
        globalScaling=scale,
        physicsClientId=scene.client_id,
    )
    p.changeDynamics(
        object_id,
        -1,
        lateralFriction=2.0,
        spinningFriction=0.04,
        rollingFriction=0.01,
        restitution=0.0,
        physicsClientId=scene.client_id,
    )
    scene.spawned_object_ids.append(object_id)
    return object_id


def summarize_object_entry(
    relative_path: str,
    scale_evaluations: list[dict[str, object]],
    min_grasp_extent: float,
    max_grasp_extent: float,
    max_aspect_ratio: float,
) -> dict[str, object]:
    graspable_scales = []
    best_score = None
    best_scale_eval = None
    for evaluation in scale_evaluations:
        max_xy_extent = float(evaluation["max_xy_extent"])
        aspect_ratio = float(evaluation["aspect_ratio"])
        supported = bool(evaluation["supported"])
        valid_simulation = bool(evaluation["valid_simulation"])
        graspable = (
            valid_simulation
            and supported
            and min_grasp_extent <= max_xy_extent <= max_grasp_extent
            and aspect_ratio <= max_aspect_ratio
        )
        if graspable:
            graspable_scales.append(float(evaluation["scale"]))
            midpoint = 0.5 * (min_grasp_extent + max_grasp_extent)
            score = abs(max_xy_extent - midpoint) + 0.05 * aspect_ratio
            if best_score is None or score < best_score:
                best_score = score
                best_scale_eval = evaluation

    recommended_scale = float(best_scale_eval["scale"]) if best_scale_eval is not None else None
    entry = {
        "relative_path": relative_path,
        "valid": best_scale_eval is not None,
        "graspable_scales": graspable_scales,
        "recommended_scale": recommended_scale,
        "scale_evaluations": scale_evaluations,
    }
    if best_scale_eval is not None:
        entry["recommended_extents"] = list(best_scale_eval["extents"])
        entry["recommended_max_xy_extent"] = float(best_scale_eval["max_xy_extent"])
        entry["recommended_height"] = float(best_scale_eval["height"])
        entry["recommended_aspect_ratio"] = float(best_scale_eval["aspect_ratio"])
        entry["size_bin"] = size_bin(float(best_scale_eval["max_xy_extent"]))
    else:
        entry["recommended_extents"] = None
        entry["recommended_max_xy_extent"] = None
        entry["recommended_height"] = None
        entry["recommended_aspect_ratio"] = None
        entry["size_bin"] = "invalid"
    return entry


def size_bin(max_xy_extent: float) -> str:
    if max_xy_extent < 0.03:
        return "tiny"
    if max_xy_extent < 0.045:
        return "small"
    if max_xy_extent < 0.06:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene = GraspingScene(SceneConfig(gui=False, object_scale_range=(0.3, 0.3)))
    scene.setup()
    catalog_entries: list[dict[str, object]] = []
    try:
        urdf_paths = list_random_urdfs()
        if args.limit is not None:
            urdf_paths = urdf_paths[: args.limit]
        for index, relative_path in enumerate(urdf_paths):
            scale_evaluations = [
                evaluate_object_at_scale(
                    scene,
                    relative_path,
                    scale=scale,
                    settle_steps=args.settle_steps,
                )
                for scale in args.scales
            ]
            entry = summarize_object_entry(
                relative_path,
                scale_evaluations,
                min_grasp_extent=args.min_grasp_extent,
                max_grasp_extent=args.max_grasp_extent,
                max_aspect_ratio=args.max_aspect_ratio,
            )
            catalog_entries.append(entry)
            if (index + 1) % 50 == 0:
                valid_count = sum(1 for item in catalog_entries if item["valid"])
                print(f"[{index + 1}/{len(urdf_paths)}] valid={valid_count}")
    finally:
        scene.disconnect()

    valid_entries = [entry for entry in catalog_entries if entry["valid"]]
    invalid_entries = [entry for entry in catalog_entries if not entry["valid"]]
    size_histogram: dict[str, int] = {}
    for entry in valid_entries:
        size_histogram[entry["size_bin"]] = size_histogram.get(entry["size_bin"], 0) + 1

    summary = {
        "total_objects": len(catalog_entries),
        "valid_objects": len(valid_entries),
        "invalid_objects": len(invalid_entries),
        "valid_fraction": len(valid_entries) / max(len(catalog_entries), 1),
        "size_histogram": size_histogram,
        "scales_tested": list(args.scales),
        "filter_criteria": {
            "min_grasp_extent": args.min_grasp_extent,
            "max_grasp_extent": args.max_grasp_extent,
            "max_aspect_ratio": args.max_aspect_ratio,
        },
    }

    with (args.output_dir / "object_catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(catalog_entries, handle, indent=2)
    with (args.output_dir / "object_catalog_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"Wrote object catalog with {summary['valid_objects']} valid objects out of "
        f"{summary['total_objects']} to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
