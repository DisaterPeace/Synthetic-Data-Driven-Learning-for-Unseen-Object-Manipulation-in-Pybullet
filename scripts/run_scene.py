from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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

from synthetic_grasping import GraspingScene, SceneConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the initial PyBullet arm-table scene.")
    parser.add_argument("--gui", action="store_true", help="Launch PyBullet with the GUI enabled.")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-grasp-demo",
        action="store_true",
        help="Generate top-down grasp candidates and automatically label success.",
    )
    parser.add_argument(
        "--yaw-count",
        type=int,
        default=4,
        help="Number of top-down yaw candidates to try per object.",
    )
    parser.add_argument(
        "--gui-step-sleep",
        type=float,
        default=1.0,
        help="Playback multiplier for GUI stepping. Increase to slow the demo down.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If provided, saves RGB, depth, segmentation, and metadata.",
    )
    return parser.parse_args()


def load_object_split(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_outputs(output_dir: Path, observation: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observation["rgb"]).save(output_dir / "rgb.png")
    np.save(output_dir / "depth.npy", observation["depth"])
    np.save(output_dir / "segmentation.npy", observation["segmentation"])
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    object_split = load_object_split(REPO_ROOT / "configs" / "object_split.yaml")
    rng = random.Random(args.seed)
    chosen_paths = rng.sample(object_split[args.split], k=min(args.num_objects, len(object_split[args.split])))

    scene = GraspingScene(
        SceneConfig(
            gui=args.gui,
            gui_step_sleep=args.gui_step_sleep,
        )
    )
    scene.setup()
    spawned = scene.spawn_objects(chosen_paths, seed=args.seed)
    camera = scene.sample_camera_config(
        seed=args.seed,
        object_ids=[entry["object_id"] for entry in spawned],
    )
    observation = scene.capture_rgbd(camera=camera, seed=args.seed)
    grasp_trials: list[dict[str, object]] = []

    if args.run_grasp_demo:
        grasp_trials = scene.label_grasp_candidates(
            [entry["object_id"] for entry in spawned],
            yaw_count=args.yaw_count,
        )

    metadata = {
        "split": args.split,
        "seed": args.seed,
        "num_objects": len(spawned),
        "spawned_objects": spawned,
        "camera": asdict(camera),
        "light_direction": list(scene.last_light_direction or (1.0, 1.0, 1.0)),
        "light_distance": scene.last_light_distance,
        "grasp_trials": grasp_trials,
    }

    if args.output_dir is not None:
        save_outputs(args.output_dir, observation, metadata)

    print(f"Scene ready with {len(spawned)} objects from the '{args.split}' split.")
    if grasp_trials:
        success_count = sum(1 for trial in grasp_trials if trial["success"])
        print(
            f"Evaluated {len(grasp_trials)} grasp candidates. "
            f"Successful grasps: {success_count}."
        )
    if args.output_dir is not None:
        print(f"Saved RGB-D sample to: {args.output_dir}")

    if args.gui:
        print("Press Ctrl+C in this terminal when you are done inspecting the scene.")
        try:
            while True:
                scene.step(1)
        except KeyboardInterrupt:
            pass

    scene.disconnect()


if __name__ == "__main__":
    main()
