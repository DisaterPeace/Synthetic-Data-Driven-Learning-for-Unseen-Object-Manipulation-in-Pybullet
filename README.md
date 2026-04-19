# Synthetic Data-Driven Learning for Unseen Object Manipulation in PyBullet

Synthetic RGB-D grasp dataset generation and baseline learning pipeline built on PyBullet.

## Scope

This project currently supports:

1. Panda arm + tabletop scene setup in PyBullet
2. audited train/test object splits from the PyBullet `random_urdfs` pool
3. randomized RGB-D scene generation with clutter and camera variation
4. physics-only grasp candidate labeling
5. large-scale dataset export for unseen-object grasp scoring
6. a baseline grasp-success scorer trainer

## Repository Layout

- `src/synthetic_grasping/`
  Core simulator and training utilities
- `scripts/`
  Dataset generation, object auditing, training, and MCP registration scripts
- `configs/`
  Object catalog and filtered train/test split definitions
- `tools/`
  Local PyBullet MCP launcher
- `DATASET_SPEC.md`
  Dataset format and release policy

## Setup

This repo was developed on Windows with a local PyBullet build inside `.venv311`.

Key local requirements:

- Python 3.11
- PyBullet
- NumPy
- Pillow
- PyYAML

PyBullet asset resolution uses an ASCII-safe junction at `C:\PyBulletAssets` to avoid Unicode path issues on Windows.

## Quick Start

Run the scene in GUI mode:

```powershell
.\.venv311\Scripts\python.exe .\scripts\run_scene.py --gui --split train --num-objects 5 --seed 7
```

Run headless and save one RGB-D sample plus grasp labels:

```powershell
.\.venv311\Scripts\python.exe .\scripts\run_scene.py --split test --num-objects 4 --seed 21 --run-grasp-demo --yaw-count 2 --output-dir .\data\raw\sample_021
```

Build an audited object catalog:

```powershell
.\.venv311\Scripts\python.exe .\scripts\build_object_catalog.py
```

Generate a filtered train/test split from the audited catalog:

```powershell
.\.venv311\Scripts\python.exe .\scripts\generate_object_split.py --train-size 787 --test-size 200 --seed 7 --catalog .\configs\object_catalog\object_catalog.json
```

Generate a randomized dataset:

```powershell
.\.venv311\Scripts\python.exe .\scripts\generate_dataset.py --split train --episodes 1000 --min-objects 3 --max-objects 7 --camera-views 3 --yaw-count 3 --catalog .\configs\object_catalog\object_catalog.json
```

Audit a generated dataset:

```powershell
.\.venv311\Scripts\python.exe .\scripts\audit_dataset.py --dataset-root .\data\datasets\synthetic_grasping
```

Train the baseline grasp-success scorer:

```powershell
.\.venv311\Scripts\python.exe .\scripts\train_grasp_scorer.py --dataset-root .\data\datasets\synthetic_grasping --train-split train --eval-split test
```

## Dataset Outputs

The dataset generator writes:

- `episode_XXXXXX/view_YY/rgb.png`
- `episode_XXXXXX/view_YY/depth.npy`
- `episode_XXXXXX/view_YY/segmentation.npy`
- `episode_XXXXXX/metadata.json`
- `index.jsonl` with one line per episode
- `samples.jsonl` with one line per `RGB-D + grasp candidate + label` training sample
- `dataset_manifest.json` at the dataset root
- `audit/dataset_audit.json` and `audit/dataset_audit.md` after running the dataset audit

## PyBullet MCP

Register the local MCP server with:

```powershell
.\scripts\register_pybullet_mcp.ps1
```

After registration, restart the client session so the MCP tools are refreshed.

## Release Notes

The current pipeline is aimed at a paper-grade baseline release:

- filtered object pool
- explicit train/test object split
- standardized camera metadata
- physics-only labels
- dataset-level audit outputs

See [DATASET_SPEC.md](DATASET_SPEC.md) for the formal dataset definition.
