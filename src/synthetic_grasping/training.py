from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    split: str
    label: int
    rgb_path: str
    depth_path: str
    candidate: dict[str, object]
    projection: dict[str, object]


def load_sample_records(samples_path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(
                SampleRecord(
                    sample_id=str(payload["sample_id"]),
                    split=str(payload["split"]),
                    label=int(bool(payload["label"]["success"])),
                    rgb_path=str(payload["rgb_path"]),
                    depth_path=str(payload["depth_path"]),
                    candidate=dict(payload["candidate"]),
                    projection=dict(payload["projection"]),
                )
            )
    return records


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def crop_depth_patch(
    depth: np.ndarray,
    pixel_u: float | None,
    pixel_v: float | None,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    patch = np.zeros((crop_size, crop_size), dtype=np.float32)
    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    if pixel_u is None or pixel_v is None:
        return patch, mask

    center_u = int(round(pixel_u))
    center_v = int(round(pixel_v))
    half = crop_size // 2

    for row_offset in range(-half, half):
        for col_offset in range(-half, half):
            src_v = center_v + row_offset
            src_u = center_u + col_offset
            dst_v = row_offset + half
            dst_u = col_offset + half
            if 0 <= src_v < depth.shape[0] and 0 <= src_u < depth.shape[1]:
                patch[dst_v, dst_u] = float(depth[src_v, src_u])
                mask[dst_v, dst_u] = 1.0
    return patch, mask


def build_feature_vector(
    dataset_root: Path,
    record: SampleRecord,
    crop_size: int,
) -> np.ndarray:
    depth = np.load(dataset_root / record.depth_path).astype(np.float32)
    patch, mask = crop_depth_patch(
        depth,
        record.projection.get("pixel_u"),
        record.projection.get("pixel_v"),
        crop_size,
    )

    valid_patch_values = patch[mask > 0.5]
    if valid_patch_values.size == 0:
        patch_mean = 0.0
        patch_std = 1.0
    else:
        patch_mean = float(np.mean(valid_patch_values))
        patch_std = float(np.std(valid_patch_values))
        if patch_std < 1e-6:
            patch_std = 1.0

    normalized_patch = (patch - patch_mean) / patch_std
    normalized_patch *= mask

    camera_width = float(record.projection.get("pixel_u") is not None and 1.0 or 1.0)
    del camera_width  # reserved for future explicit camera normalization

    pixel_u = record.projection.get("pixel_u")
    pixel_v = record.projection.get("pixel_v")
    camera_depth = float(record.projection.get("camera_depth") or 0.0)
    depth_at_pixel = record.projection.get("depth_at_pixel")
    depth_at_pixel_value = float(depth_at_pixel) if depth_at_pixel is not None else 0.0
    yaw = float(record.candidate["yaw"])
    target_opening = float(record.candidate["target_opening"])
    xy_offset = record.candidate.get("xy_offset", [0.0, 0.0])

    projection_features = np.array(
        [
            float(pixel_u or 0.0) / max(float(depth.shape[1] - 1), 1.0),
            float(pixel_v or 0.0) / max(float(depth.shape[0] - 1), 1.0),
            camera_depth,
            depth_at_pixel_value,
            float(record.projection.get("visible", False)),
            float(record.projection.get("in_frame", False)),
            float(np.sin(yaw)),
            float(np.cos(yaw)),
            target_opening,
            float(xy_offset[0]),
            float(xy_offset[1]),
            float(np.mean(mask)),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            normalized_patch.reshape(-1),
            mask.reshape(-1),
            projection_features,
        ]
    ).astype(np.float32)


def build_dataset(
    dataset_root: Path,
    records: Iterable[SampleRecord],
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    labels = []
    sample_ids: list[str] = []
    for record in records:
        features.append(build_feature_vector(dataset_root, record, crop_size=crop_size))
        labels.append(record.label)
        sample_ids.append(record.sample_id)
    return np.stack(features, axis=0), np.array(labels, dtype=np.float32), sample_ids


def standardize_features(
    train_features: np.ndarray,
    eval_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        (train_features - mean) / std,
        (eval_features - mean) / std,
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def train_logistic_regression(
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    bias = 0.0

    positive_count = float(np.sum(train_y))
    negative_count = float(train_y.shape[0] - positive_count)
    pos_weight = negative_count / max(positive_count, 1.0)

    for _ in range(epochs):
        logits = train_x @ weights + bias
        probabilities = sigmoid(logits)

        sample_weights = np.where(train_y > 0.5, pos_weight, 1.0).astype(np.float32)
        residual = (probabilities - train_y) * sample_weights / train_y.shape[0]
        gradient_w = train_x.T @ residual + weight_decay * weights
        gradient_b = float(np.sum(residual))

        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

    return weights, bias


def evaluate_binary_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> dict[str, float]:
    probabilities = sigmoid(features @ weights + bias)
    predictions = (probabilities >= 0.5).astype(np.float32)
    accuracy = float(np.mean(predictions == labels))

    true_positive = float(np.sum((predictions == 1.0) & (labels == 1.0)))
    false_positive = float(np.sum((predictions == 1.0) & (labels == 0.0)))
    false_negative = float(np.sum((predictions == 0.0) & (labels == 1.0)))

    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    logits = features @ weights + bias
    probabilities = sigmoid(logits)
    epsilon = 1e-6
    loss = -np.mean(
        labels * np.log(probabilities + epsilon) + (1.0 - labels) * np.log(1.0 - probabilities + epsilon)
    )

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(np.mean(labels)),
    }
