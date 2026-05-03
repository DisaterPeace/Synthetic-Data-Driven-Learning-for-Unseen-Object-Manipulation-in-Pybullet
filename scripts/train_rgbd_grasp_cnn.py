from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RGB-D crop + grasp-candidate CNN scorer."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print a JSON progress record every N training batches; set 0 to disable.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Read RGB-D files inside each epoch instead of preloading selected crops.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "rgbd_grasp_cnn.pt",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def sample_rows(
    rows: list[dict[str, Any]],
    max_samples: int,
    seed: int,
    balanced: bool,
) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)
    if not balanced:
        selected = list(rows)
        rng.shuffle(selected)
        return selected[:max_samples]

    positives = [row for row in rows if bool(row["label"]["success"])]
    negatives = [row for row in rows if not bool(row["label"]["success"])]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positive_target = min(len(positives), max_samples // 2)
    negative_target = max_samples - positive_target
    if negative_target > len(negatives):
        negative_target = len(negatives)
        positive_target = min(len(positives), max_samples - negative_target)

    selected = positives[:positive_target] + negatives[:negative_target]
    rng.shuffle(selected)
    return selected


class ArrayCache:
    def __init__(self, max_items: int = 96) -> None:
        self.max_items = max_items
        self._items: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, path: Path, loader) -> np.ndarray:
        if path in self._items:
            value = self._items.pop(path)
            self._items[path] = value
            return value
        value = loader(path)
        self._items[path] = value
        if len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return value


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_depth(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def crop_with_padding(array: np.ndarray, center_u: int, center_v: int, crop_size: int) -> np.ndarray:
    half = crop_size // 2
    if array.ndim == 2:
        padded = np.pad(array, ((half, half), (half, half)), mode="edge")
        return padded[center_v:center_v + crop_size, center_u:center_u + crop_size]
    padded = np.pad(array, ((half, half), (half, half), (0, 0)), mode="edge")
    return padded[center_v:center_v + crop_size, center_u:center_u + crop_size, :]


def candidate_features(row: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray:
    projection = row["projection"]
    candidate = row["candidate"]
    height, width = image_shape
    pixel_u = float(projection.get("pixel_u") or 0.0)
    pixel_v = float(projection.get("pixel_v") or 0.0)
    yaw = float(candidate["yaw"])
    xy_offset = candidate.get("xy_offset", [0.0, 0.0])
    return np.array(
        [
            pixel_u / max(width - 1, 1),
            pixel_v / max(height - 1, 1),
            float(projection.get("camera_depth") or 0.0),
            float(projection.get("depth_at_pixel") or 0.0),
            float(projection.get("visible", False)),
            float(projection.get("in_frame", False)),
            float(np.sin(yaw)),
            float(np.cos(yaw)),
            float(candidate["target_opening"]),
            float(xy_offset[0]),
            float(xy_offset[1]),
        ],
        dtype=np.float32,
    )


class RGBDGraspDataset(Dataset):
    def __init__(self, split_root: Path, rows: list[dict[str, Any]], crop_size: int) -> None:
        self.split_root = split_root
        self.rows = rows
        self.crop_size = crop_size
        self.rgb_cache = ArrayCache()
        self.depth_cache = ArrayCache()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        rgb = self.rgb_cache.get(self.split_root / row["rgb_path"], load_rgb)
        depth = self.depth_cache.get(self.split_root / row["depth_path"], load_depth)
        height, width = depth.shape

        pixel_u = int(round(float(row["projection"].get("pixel_u") or width * 0.5)))
        pixel_v = int(round(float(row["projection"].get("pixel_v") or height * 0.5)))
        pixel_u = int(np.clip(pixel_u, 0, width - 1))
        pixel_v = int(np.clip(pixel_v, 0, height - 1))

        rgb_crop = crop_with_padding(rgb, pixel_u, pixel_v, self.crop_size)
        depth_crop = crop_with_padding(depth, pixel_u, pixel_v, self.crop_size)
        depth_center = float(row["projection"].get("camera_depth") or np.mean(depth_crop))
        depth_crop = np.clip(depth_crop - depth_center, -0.25, 0.25) / 0.25
        depth_crop = depth_crop[:, :, None]

        image = np.concatenate([rgb_crop, depth_crop], axis=2)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        feature_tensor = torch.from_numpy(candidate_features(row, depth.shape)).float()
        label_tensor = torch.tensor(float(bool(row["label"]["success"])), dtype=torch.float32)
        return image_tensor, feature_tensor, label_tensor


def preload_dataset(dataset: RGBDGraspDataset, name: str) -> TensorDataset:
    images: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for index in range(len(dataset)):
        image, candidate, label = dataset[index]
        images.append(image)
        candidates.append(candidate)
        labels.append(label)
        if (index + 1) % 1000 == 0 or index + 1 == len(dataset):
            print(
                json.dumps(
                    {
                        "event": "preload",
                        "split": name,
                        "loaded": index + 1,
                        "total": len(dataset),
                    }
                ),
                flush=True,
            )
    return TensorDataset(
        torch.stack(images, dim=0),
        torch.stack(candidates, dim=0),
        torch.stack(labels, dim=0),
    )


class RGBDGraspCNN(nn.Module):
    def __init__(self, candidate_dim: int = 11) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 48),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(96 + 48, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(96, 1),
        )

    def forward(self, image: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image).flatten(1)
        candidate_features = self.candidate_encoder(candidate)
        return self.head(torch.cat([image_features, candidate_features], dim=1)).squeeze(1)


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    labels = labels.float()
    tp = torch.sum((predictions == 1.0) & (labels == 1.0)).item()
    fp = torch.sum((predictions == 1.0) & (labels == 0.0)).item()
    fn = torch.sum((predictions == 0.0) & (labels == 1.0)).item()
    accuracy = torch.mean((predictions == labels).float()).item()
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(torch.mean(labels).item()),
        "mean_probability": float(torch.mean(probabilities).item()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    losses: list[float] = []
    for images, candidates, labels in loader:
        images = images.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)
        logits = model(images, candidates)
        loss = criterion(logits, labels)
        losses.append(float(loss.item()))
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    metrics = binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    train_root = args.dataset_root / args.train_split
    eval_root = args.dataset_root / args.eval_split
    train_rows = sample_rows(
        load_jsonl(train_root / "samples.jsonl"),
        args.max_train_samples,
        seed=args.seed,
        balanced=True,
    )
    eval_rows = sample_rows(
        load_jsonl(eval_root / "samples.jsonl"),
        args.max_eval_samples,
        seed=args.seed + 1,
        balanced=False,
    )

    train_positive = sum(int(bool(row["label"]["success"])) for row in train_rows)
    train_negative = len(train_rows) - train_positive
    pos_weight = torch.tensor(
        [train_negative / max(train_positive, 1)],
        dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset: Dataset = RGBDGraspDataset(train_root, train_rows, crop_size=args.crop_size)
    eval_dataset: Dataset = RGBDGraspDataset(eval_root, eval_rows, crop_size=args.crop_size)
    if not args.streaming:
        train_dataset = preload_dataset(train_dataset, "train")
        eval_dataset = preload_dataset(eval_dataset, "eval")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = RGBDGraspCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "train_samples": len(train_rows),
                "eval_samples": len(eval_rows),
                "train_positive_rate": train_positive / max(len(train_rows), 1),
                "crop_size": args.crop_size,
                "streaming": args.streaming,
            }
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        total_batches = len(train_loader)
        for batch_index, (images, candidates, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, candidates)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if args.progress_interval > 0 and (
                batch_index % args.progress_interval == 0 or batch_index == total_batches
            ):
                print(
                    json.dumps(
                        {
                            "event": "train_batch",
                            "epoch": epoch,
                            "batch": batch_index,
                            "total_batches": total_batches,
                            "loss": float(np.mean(losses[-min(len(losses), args.progress_interval):])),
                        }
                    ),
                    flush=True,
                )

        train_metrics = evaluate(model, train_loader, criterion, device)
        eval_metrics = evaluate(model, eval_loader, criterion, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)) if losses else 0.0,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
        }
        history.append(epoch_record)
        print(json.dumps({"event": "epoch", **epoch_record}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state_dict": model.cpu().state_dict(),
        "metadata": {
            "dataset_root": str(args.dataset_root),
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "crop_size": args.crop_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "streaming": args.streaming,
            "history": history,
        },
    }
    torch.save(artifact, args.output)
    print(json.dumps({"event": "saved", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
