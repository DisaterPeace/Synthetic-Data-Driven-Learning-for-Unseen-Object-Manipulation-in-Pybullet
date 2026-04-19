from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from synthetic_grasping.training import (  # noqa: E402
    build_dataset,
    evaluate_binary_classifier,
    load_sample_records,
    standardize_features,
    train_logistic_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline grasp-success scorer from RGB-D candidate samples.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "datasets" / "synthetic_grasping",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--crop-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "grasp_scorer_baseline.npz",
    )
    return parser.parse_args()


def save_model(
    output_path: Path,
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    std: np.ndarray,
    metadata: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        weights=weights.astype(np.float32),
        bias=np.array([bias], dtype=np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        metadata_json=np.array([json.dumps(metadata)], dtype=object),
    )


def main() -> None:
    args = parse_args()
    train_root = args.dataset_root / args.train_split
    eval_root = args.dataset_root / args.eval_split
    train_samples_path = train_root / "samples.jsonl"
    eval_samples_path = eval_root / "samples.jsonl"

    if not train_samples_path.exists():
        raise FileNotFoundError(f"Training samples not found: {train_samples_path}")
    if not eval_samples_path.exists():
        raise FileNotFoundError(f"Evaluation samples not found: {eval_samples_path}")

    train_records = load_sample_records(train_samples_path)
    eval_records = load_sample_records(eval_samples_path)
    train_x_raw, train_y, _ = build_dataset(train_root, train_records, crop_size=args.crop_size)
    eval_x_raw, eval_y, _ = build_dataset(eval_root, eval_records, crop_size=args.crop_size)
    train_x, eval_x, mean, std = standardize_features(train_x_raw, eval_x_raw)

    weights, bias = train_logistic_regression(
        train_x,
        train_y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_metrics = evaluate_binary_classifier(train_x, train_y, weights, bias)
    eval_metrics = evaluate_binary_classifier(eval_x, eval_y, weights, bias)
    metadata = {
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "crop_size": args.crop_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_count": int(train_y.shape[0]),
        "eval_count": int(eval_y.shape[0]),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    save_model(args.output, weights, bias, mean, std, metadata)

    print(f"Saved model to {args.output}")
    print(f"Train metrics: {json.dumps(train_metrics)}")
    print(f"Eval metrics: {json.dumps(eval_metrics)}")


if __name__ == "__main__":
    main()
