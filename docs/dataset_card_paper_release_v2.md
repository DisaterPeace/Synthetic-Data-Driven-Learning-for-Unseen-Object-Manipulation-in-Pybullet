# Dataset Card: Synthetic RGB-D Grasping `paper_release_v2`

## Dataset Summary

`paper_release_v2` is a synthetic RGB-D grasp-candidate dataset generated in PyBullet for unseen-object manipulation research.

Public download:

```text
https://github.com/DisaterPeace/Synthetic-Data-Driven-Learning-for-Unseen-Object-Manipulation-in-Pybullet/releases/tag/paper_release_v2
```

Archive SHA-256:

```text
0d7c04200b799d5f7ba6cf381db229b7f4e7c7e7ada706ed171221aff6609fca
```

The learning task is:

```text
Input: RGB-D observation + grasp candidate
Output: grasp success probability
```

The evaluation split contains objects that are not present in the training split.

## Intended Use

- Train and evaluate grasp-success scorers for parallel-jaw grasp candidates.
- Study synthetic data generation strategies for unseen-object robotic manipulation.
- Benchmark candidate-level and scene-level grasp selection methods in simulation.

## Dataset Statistics

| Split | Episodes | Samples | Successes | Success rate | Unique asset paths |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 600 | 134,100 | 20,802 | 15.51% | 770 |
| test | 180 | 40,410 | 6,327 | 15.66% | 198 |
| total | 780 | 174,510 | 27,129 | 15.55% | 968 |

## Data Fields

Each split contains:

```text
index.jsonl
samples.jsonl
episode_XXXXXX/metadata.json
episode_XXXXXX/view_YY/rgb.png
episode_XXXXXX/view_YY/depth.npy
episode_XXXXXX/view_YY/segmentation.npy
```

Each row in `samples.jsonl` includes:

- relative paths to RGB and depth observations
- grasp-candidate pose/opening/yaw metadata
- image projection metadata
- physics-derived success label

## Label Definition

Labels are generated with physics-only criteria:

1. the robot contacts the target object during grasp/lift,
2. the object is lifted above the configured threshold,
3. the object is no longer supported by the table/plane after lift.

No attachment shortcut or manual annotation is used.

## Validation

The release validation result is:

```text
Status: PASS
Errors: 0
Warnings: 0
Train/test asset overlap: 0
Checksum entries: 7,813
```

Validation files are included under:

```text
audit/
release_validation/
```

## Baselines

Feature baseline on unseen test objects:

| Metric | Value |
| --- | ---: |
| accuracy | 67.19% |
| precision | 26.86% |
| recall | 63.58% |
| F1 | 37.77% |

RGB-D CNN medium sanity run on 10,000 unseen-object candidate samples:

| Metric | Value |
| --- | ---: |
| accuracy | 57.96% |
| precision | 24.33% |
| recall | 77.08% |
| F1 | 36.99% |

## Limitations

- The dataset is synthetic and simulation-only.
- Candidate-level classification metrics are not the same as closed-loop robot grasp success.
- Successful candidates are sparse, so accuracy alone is not an adequate metric.
- The current RGB-D CNN is a functional baseline, not the final proposed model.

## Citation

Citation information should be updated after the dataset receives a public URL or DOI.
