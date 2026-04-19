# Synthetic RGB-D Grasp Dataset Spec

## Goal

This dataset targets unseen-object parallel-jaw grasp scoring from synthetic RGB-D data.

The primary learning problem is:

- input: `RGB-D observation + grasp candidate`
- output: `probability of grasp success`

## Split Policy

- Object identity is split between `train` and `test`.
- Split generation is performed after object auditing/filtering.
- Only audited objects marked `valid=true` are eligible for release splits.
- Train/test selection is stratified by object size bin when a catalog is available.

## Object Audit

Objects are filtered using an audited catalog built from the PyBullet `random_urdfs` pool.

Each object is evaluated across multiple scales for:

- simulation validity
- stable support on the tabletop
- graspable XY extent
- aspect ratio

The catalog stores:

- `recommended_scale`
- `recommended_extents`
- `recommended_max_xy_extent`
- `recommended_aspect_ratio`
- `size_bin`

## Scene Randomization

Each episode randomizes:

- number of objects
- object identities
- object poses
- object scales
- object frictions
- spawn region bounds
- camera pose
- camera field of view
- light direction and distance

## Observation Format

For each view:

- `rgb.png`
- `depth.npy`
- `segmentation.npy`

Camera metadata includes:

- image size
- `intrinsics`
- `extrinsics`
- `view_matrix`
- `projection_matrix`
- camera eye / target / up
- FOV, near, far

## Grasp Candidate Format

Each grasp sample stores:

- `grasp_position_world`
- `approach_position_world`
- `lift_position_world`
- `yaw`
- `target_opening`
- `xy_offset`

Projected image-space metadata includes:

- `pixel_u`
- `pixel_v`
- `camera_depth`
- `depth_at_pixel`
- `visible`
- `in_frame`

## Label Definition

Labels are physics-only.

A candidate is successful when:

- the object is contacted by the robot during grasp/lift, and
- the object is lifted above the configured threshold, and
- the object is no longer supported by the table/plane after lift

No attachment or kinematic shortcut is used in the label.

## Release Files

At the dataset root:

- `dataset_manifest.json`

Per split:

- `index.jsonl`
- `samples.jsonl`
- `episode_XXXXXX/...`

Optional audit outputs:

- `audit/dataset_audit.json`
- `audit/dataset_audit.md`

## Quality Checks Before Release

Before calling a release paper-ready, verify:

- object audit completed on the full candidate object pool
- split built from audited valid objects only
- dataset audit generated for both train and test
- random manual inspection of RGB-D samples and labels
- label balance is reported
- baseline train/test benchmark is reported
