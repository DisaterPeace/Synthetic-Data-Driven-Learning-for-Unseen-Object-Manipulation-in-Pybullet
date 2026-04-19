from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from synthetic_grasping import CameraConfig, GraspingScene, SceneConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a large synthetic RGB-D grasp dataset.")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--min-objects", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=7)
    parser.add_argument("--camera-views", type=int, default=2)
    parser.add_argument("--yaw-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "datasets" / "synthetic_grasping",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "configs" / "object_catalog" / "object_catalog.json",
        help="Optional audited object catalog used for recommended scales and richer metadata.",
    )
    return parser.parse_args()


def load_object_split(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_object_catalog(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        catalog_entries = json.load(handle)
    return {
        str(entry["relative_path"]): entry
        for entry in catalog_entries
        if entry.get("valid", False)
    }


def build_scene_config(rng: random.Random, width: int, height: int) -> SceneConfig:
    x_min = rng.uniform(0.56, 0.60)
    x_max = rng.uniform(0.70, 0.76)
    y_extent = rng.uniform(0.10, 0.18)
    min_scale = rng.uniform(0.28, 0.34)
    max_scale = rng.uniform(max(min_scale + 0.07, 0.40), 0.54)
    return SceneConfig(
        gui=False,
        object_xy_bounds=((x_min, x_max), (-y_extent, y_extent)),
        object_spawn_height=rng.uniform(0.74, 0.84),
        object_scale_range=(min_scale, max_scale),
        object_min_xy_separation=rng.uniform(0.045, 0.075),
        camera=CameraConfig(width=width, height=height),
        camera_distance_range=(0.72, 1.05),
        camera_azimuth_range_deg=(-55.0, 55.0),
        camera_elevation_range_deg=(30.0, 68.0),
        camera_target_jitter=(0.06, 0.06, 0.06),
        camera_fov_range=(48.0, 78.0),
        light_distance_range=(1.1, 2.4),
        grasp_lift_threshold=0.05,
    )


def save_observation(output_dir: Path, observation: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observation["rgb"]).save(output_dir / "rgb.png")
    np.save(output_dir / "depth.npy", observation["depth"])
    np.save(output_dir / "segmentation.npy", observation["segmentation"])


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector
    return vector / norm


def camera_intrinsics(camera: CameraConfig) -> list[list[float]]:
    focal = 0.5 * camera.width / math.tan(math.radians(camera.fov) * 0.5)
    cx = camera.width * 0.5
    cy = camera.height * 0.5
    return [
        [float(focal), 0.0, float(cx)],
        [0.0, float(focal), float(cy)],
        [0.0, 0.0, 1.0],
    ]


def camera_extrinsics(camera: CameraConfig) -> list[list[float]]:
    eye = np.array(camera.eye, dtype=np.float32)
    target = np.array(camera.target, dtype=np.float32)
    up = np.array(camera.up, dtype=np.float32)
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))

    rotation = np.stack([right, true_up, forward], axis=0)
    translation = -rotation @ eye
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    return extrinsic.tolist()


def camera_metadata(camera: CameraConfig) -> dict[str, object]:
    return {
        **asdict(camera),
        "intrinsics": camera_intrinsics(camera),
        "extrinsics": camera_extrinsics(camera),
        "view_matrix": camera.view_matrix(),
        "projection_matrix": camera.projection_matrix(),
    }


def normalize_catalog_key(path_value: str) -> str:
    normalized = path_value.replace("\\", "/")
    asset_prefix = "C:/PyBulletAssets/"
    if normalized.startswith(asset_prefix):
        return normalized[len(asset_prefix) :]
    return normalized


def project_world_point(camera: CameraConfig, point_world: list[float]) -> dict[str, object]:
    eye = np.array(camera.eye, dtype=np.float32)
    target = np.array(camera.target, dtype=np.float32)
    up = np.array(camera.up, dtype=np.float32)
    point = np.array(point_world, dtype=np.float32)

    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))
    delta = point - eye

    x_cam = float(np.dot(delta, right))
    y_cam = float(np.dot(delta, true_up))
    z_cam = float(np.dot(delta, forward))

    if z_cam <= 1e-6:
        return {
            "visible": False,
            "in_frame": False,
            "pixel_u": None,
            "pixel_v": None,
            "camera_depth": z_cam,
        }

    focal = 0.5 * camera.width / math.tan(math.radians(camera.fov) * 0.5)
    pixel_u = focal * (x_cam / z_cam) + camera.width * 0.5
    pixel_v = camera.height * 0.5 - focal * (y_cam / z_cam)
    in_frame = 0.0 <= pixel_u < camera.width and 0.0 <= pixel_v < camera.height

    return {
        "visible": True,
        "in_frame": bool(in_frame),
        "pixel_u": float(pixel_u),
        "pixel_v": float(pixel_v),
        "camera_depth": float(z_cam),
    }


def sample_depth_at_pixel(depth: np.ndarray, pixel_u: float | None, pixel_v: float | None) -> float | None:
    if pixel_u is None or pixel_v is None:
        return None
    u = int(round(pixel_u))
    v = int(round(pixel_v))
    if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
        return None
    return float(depth[v, u])


def main() -> None:
    args = parse_args()
    object_split = load_object_split(REPO_ROOT / "configs" / "object_split.yaml")
    object_catalog = load_object_catalog(args.catalog)
    available_objects = object_split[args.split]
    if not available_objects:
        raise ValueError(f"No objects available for split '{args.split}'.")

    split_root = args.output_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    index_path = split_root / "index.jsonl"
    samples_path = split_root / "samples.jsonl"
    dataset_manifest_path = args.output_root / "dataset_manifest.json"

    if not dataset_manifest_path.exists():
        dataset_manifest = {
            "version": "v1.0-paper-baseline",
            "task": "synthetic_rgbd_grasp_scoring",
            "input_modalities": ["rgb", "depth", "candidate_grasp"],
            "label_type": "physics_only_parallel_jaw_lift_success",
            "splits": list(object_split.keys()),
            "catalog_path": str(args.catalog) if args.catalog.exists() else None,
            "generator": "scripts/generate_dataset.py",
        }
        write_json(dataset_manifest_path, dataset_manifest)

    with index_path.open("a", encoding="utf-8") as index_handle, samples_path.open("a", encoding="utf-8") as samples_handle:
        for episode_idx in range(args.start_index, args.start_index + args.episodes):
            episode_seed = args.seed + episode_idx
            rng = random.Random(episode_seed)
            num_objects = min(len(available_objects), rng.randint(args.min_objects, args.max_objects))
            chosen_paths = rng.sample(available_objects, k=num_objects)
            chosen_objects: list[dict[str, object] | str] = []
            for path in chosen_paths:
                catalog_entry = object_catalog.get(path)
                if catalog_entry and catalog_entry.get("recommended_scale") is not None:
                    recommended_scale = float(catalog_entry["recommended_scale"])
                    scale = float(np.clip(recommended_scale + rng.uniform(-0.015, 0.015), 0.24, 0.60))
                    chosen_objects.append({"path": path, "scale": scale})
                else:
                    chosen_objects.append(path)

            scene = GraspingScene(
                build_scene_config(rng, width=args.image_width, height=args.image_height)
            )
            try:
                scene.setup()
                spawned = scene.spawn_objects(chosen_objects, seed=episode_seed)
                object_ids = [entry["object_id"] for entry in spawned]
                episode_dir = split_root / f"episode_{episode_idx:06d}"
                views_payload: list[dict[str, object]] = []
                view_observations: list[dict[str, object]] = []

                for view_idx in range(args.camera_views):
                    view_seed = episode_seed * 100 + view_idx
                    camera = scene.sample_camera_config(seed=view_seed, object_ids=object_ids)
                    observation = scene.capture_rgbd(camera=camera, seed=view_seed)
                    view_dir = episode_dir / f"view_{view_idx:02d}"
                    save_observation(view_dir, observation)
                    views_payload.append(
                        {
                            "view_index": view_idx,
                            "camera": camera_metadata(camera),
                            "light_direction": list(scene.last_light_direction or (1.0, 1.0, 1.0)),
                            "light_distance": scene.last_light_distance,
                            "relative_dir": str(view_dir.relative_to(split_root)).replace("\\", "/"),
                        }
                    )
                    view_observations.append(
                        {
                            "view_index": view_idx,
                            "camera": camera,
                            "observation": observation,
                            "relative_dir": str(view_dir.relative_to(split_root)).replace("\\", "/"),
                        }
                    )

                grasp_trials = scene.label_grasp_candidates(object_ids, yaw_count=args.yaw_count)
                success_count = sum(1 for trial in grasp_trials if trial["success"])
                sample_count = 0

                for candidate_index, trial in enumerate(grasp_trials):
                    for view_payload, view_data in zip(views_payload, view_observations):
                        projection = project_world_point(
                            view_data["camera"],
                            list(trial["grasp_position"]),
                        )
                        sample_record = {
                            "sample_id": f"{args.split}_{episode_idx:06d}_v{view_payload['view_index']:02d}_c{candidate_index:04d}",
                            "split": args.split,
                            "episode_index": episode_idx,
                            "episode_seed": episode_seed,
                            "view_index": view_payload["view_index"],
                            "candidate_index": candidate_index,
                            "object_id": trial["object_id"],
                            "rgb_path": f"{view_payload['relative_dir']}/rgb.png",
                            "depth_path": f"{view_payload['relative_dir']}/depth.npy",
                            "camera": view_payload["camera"],
                            "candidate": {
                                "grasp_position_world": list(trial["grasp_position"]),
                                "approach_position_world": list(trial["approach_position"]),
                                "lift_position_world": list(trial["lift_position"]),
                                "yaw": float(trial["yaw"]),
                                "target_opening": float(trial["target_opening"]),
                                "xy_offset": list(trial["xy_offset"]),
                            },
                            "projection": {
                                **projection,
                                "depth_at_pixel": sample_depth_at_pixel(
                                    view_data["observation"]["depth"],
                                    projection["pixel_u"],
                                    projection["pixel_v"],
                                ),
                            },
                            "label": {
                                "success": bool(trial["success"]),
                                "physical_lift_success": bool(trial["physical_lift_success"]),
                                "lifted_delta": float(trial["lifted_delta"]),
                                "supported_after_close": bool(trial["supported_after_close"]),
                                "supported_after_lift": bool(trial["supported_after_lift"]),
                                "contact_links_after_close": list(trial["contact_links_after_close"]),
                                "contact_links_after_lift": list(trial["contact_links_after_lift"]),
                            },
                        }
                        samples_handle.write(json.dumps(sample_record) + "\n")
                        sample_count += 1

                samples_handle.flush()
                metadata = {
                    "episode_index": episode_idx,
                    "episode_seed": episode_seed,
                    "split": args.split,
                    "num_objects": len(spawned),
                    "scene_config": asdict(scene.config),
                    "spawned_objects": spawned,
                    "catalog_entries": {
                        normalize_catalog_key(str(entry["path"])): object_catalog.get(
                            normalize_catalog_key(str(entry["path"]))
                        )
                        for entry in spawned
                    },
                    "views": views_payload,
                    "grasp_trials": grasp_trials,
                    "success_count": success_count,
                    "total_candidates": len(grasp_trials),
                    "total_samples": sample_count,
                }
                episode_dir.mkdir(parents=True, exist_ok=True)
                write_json(episode_dir / "metadata.json", metadata)

                summary = {
                    "episode_index": episode_idx,
                    "episode_seed": episode_seed,
                    "split": args.split,
                    "num_objects": len(spawned),
                    "success_count": success_count,
                    "total_candidates": len(grasp_trials),
                    "total_samples": sample_count,
                    "relative_dir": str(episode_dir.relative_to(split_root)).replace("\\", "/"),
                }
                index_handle.write(json.dumps(summary) + "\n")
                index_handle.flush()
                print(
                    f"[{episode_idx:06d}] objects={len(spawned)} "
                    f"successes={success_count}/{len(grasp_trials)} samples={sample_count}"
                )
            finally:
                scene.disconnect()


if __name__ == "__main__":
    main()
