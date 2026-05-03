# Dataset Release Notes: `paper_release_v2`

This project uses a generated synthetic RGB-D grasping dataset for unseen-object manipulation experiments in PyBullet.

The dataset is intentionally not tracked in git. The repository tracks code, configuration, validation scripts, and documentation. The dataset should be published through a dataset host such as Hugging Face Datasets or Zenodo, then linked from this repository and any GitHub release.

## Recommended Hosting

- **Hugging Face Datasets** is recommended for iterative ML use, direct programmatic download, and large-folder upload workflows.
- **Zenodo** is recommended for thesis/paper archival because it can provide a DOI.
- **GitHub Releases** may be used for smaller derived artifacts or compressed shards, but raw dataset files should not be committed to the git repository.

## Local Dataset Path

Current verified local copy:

```text
C:\Tez_Workspace\data\datasets\paper_release_v2
```

## Dataset Summary

| Split | Episodes | Samples | Successful grasps | Success rate | Unique asset paths |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 600 | 134,100 | 20,802 | 15.51% | 770 |
| test | 180 | 40,410 | 6,327 | 15.66% | 198 |
| total | 780 | 174,510 | 27,129 | 15.55% | 968 |

## Validation Summary

Release validation status:

```text
PASS
Errors: 0
Warnings: 0
Train/test asset overlap: 0
Checksum entries: 7,813
```

Validation artifacts:

```text
paper_release_v2/audit/dataset_audit.json
paper_release_v2/audit/dataset_audit.md
paper_release_v2/release_validation/RELEASE_REPORT.md
paper_release_v2/release_validation/release_validation.json
paper_release_v2/release_validation/checksums.sha256.jsonl
```

## Dataset Format

Each split contains:

```text
index.jsonl
samples.jsonl
episode_XXXXXX/metadata.json
episode_XXXXXX/view_YY/rgb.png
episode_XXXXXX/view_YY/depth.npy
episode_XXXXXX/view_YY/segmentation.npy
```

Each training sample combines:

- RGB-D observation
- grasp candidate metadata
- projected image-space candidate location
- physics-only grasp-success label

## Baseline Results

Feature baseline, evaluated on unseen test objects:

| Metric | Value |
| --- | ---: |
| accuracy | 67.19% |
| precision | 26.86% |
| recall | 63.58% |
| F1 | 37.77% |

RGB-D CNN medium sanity run, evaluated on 10,000 unseen-object candidate samples:

| Metric | Value |
| --- | ---: |
| accuracy | 57.96% |
| precision | 24.33% |
| recall | 77.08% |
| F1 | 36.99% |

The RGB-D CNN confirms the image/depth training path works, but it is a first functional baseline rather than the final research model.

## Upload Checklist

Before publishing:

1. Keep the git repository focused on source code and documentation.
2. Upload `paper_release_v2` to the selected dataset host.
3. Preserve `checksums.sha256.jsonl` with the dataset.
4. Add the final public dataset URL and DOI, if available, to this file and the README.
5. Create a GitHub release that links to the dataset and includes the validation report.

