from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import random
import time
from typing import Iterable

import numpy as np
import pybullet as p
import pybullet_data


PANDA_ARM_JOINTS = (0, 1, 2, 3, 4, 5, 6)
PANDA_FINGER_JOINTS = (9, 10)
PANDA_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
PANDA_LOWER_LIMITS = (-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671)
PANDA_UPPER_LIMITS = (2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671)
PANDA_JOINT_RANGES = tuple(upper - lower for lower, upper in zip(PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS))
PANDA_EE_LINK = 11
TOP_DOWN_GRASP_PITCH = np.pi
ASCII_PYBULLET_DATA_ROOT = Path("C:/PyBulletAssets")


@dataclass(slots=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    eye: tuple[float, float, float] = (1.05, 0.0, 1.15)
    target: tuple[float, float, float] = (0.55, 0.0, 0.62)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 60.0
    near: float = 0.02
    far: float = 2.0

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def view_matrix(self) -> list[float]:
        return p.computeViewMatrix(self.eye, self.target, self.up)

    def projection_matrix(self) -> list[float]:
        return p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far,
        )


@dataclass(slots=True)
class SceneConfig:
    gui: bool = False
    time_step: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    robot_urdf: str = "franka_panda/panda.urdf"
    table_urdf: str = "table/table.urdf"
    plane_urdf: str = "plane.urdf"
    robot_base_position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    table_base_position: tuple[float, float, float] = (0.55, 0.0, 0.0)
    object_xy_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (0.58, 0.72),
        (-0.14, 0.14),
    )
    object_spawn_height: float = 0.78
    object_scale_range: tuple[float, float] = (0.32, 0.48)
    object_min_xy_separation: float = 0.06
    settle_steps: int = 240
    gui_step_sleep: float = 1.0
    camera: CameraConfig = field(default_factory=CameraConfig)
    camera_distance_range: tuple[float, float] = (0.7, 1.0)
    camera_azimuth_range_deg: tuple[float, float] = (-40.0, 40.0)
    camera_elevation_range_deg: tuple[float, float] = (35.0, 65.0)
    camera_target_jitter: tuple[float, float, float] = (0.05, 0.05, 0.05)
    camera_fov_range: tuple[float, float] = (50.0, 75.0)
    light_distance_range: tuple[float, float] = (1.2, 2.2)
    grasp_lift_threshold: float = 0.05
    enable_gui_previews: bool = True
    live_camera_update_interval_steps: int = 12


class GraspingScene:
    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = config or SceneConfig()
        self.client_id: int | None = None
        self.robot_id: int | None = None
        self.table_id: int | None = None
        self.plane_id: int | None = None
        self.spawned_object_ids: list[int] = []
        self.last_camera_config: CameraConfig | None = None
        self.last_light_direction: tuple[float, float, float] | None = None
        self.last_light_distance: float | None = None
        self._simulation_step_counter = 0

    def connect(self) -> int:
        if self.client_id is not None:
            return self.client_id

        mode = p.GUI if self.config.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(str(self.data_root()), physicsClientId=self.client_id)
        p.setTimeStep(self.config.time_step, physicsClientId=self.client_id)
        p.setGravity(*self.config.gravity, physicsClientId=self.client_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=self.config.camera.target,
            physicsClientId=self.client_id,
        )
        if self.config.gui and self.config.enable_gui_previews:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                1,
                physicsClientId=self.client_id,
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                1,
                physicsClientId=self.client_id,
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                1,
                physicsClientId=self.client_id,
            )
        return self.client_id

    def setup(self) -> None:
        self.connect()
        assert self.client_id is not None

        plane_urdf = self.resolve_asset_path(self.config.plane_urdf)
        table_urdf = self.resolve_asset_path(self.config.table_urdf)
        robot_urdf = self.resolve_asset_path(self.config.robot_urdf)

        self.plane_id = p.loadURDF(
            str(plane_urdf),
            physicsClientId=self.client_id,
        )
        self.table_id = p.loadURDF(
            str(table_urdf),
            basePosition=self.config.table_base_position,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.robot_id = p.loadURDF(
            str(robot_urdf),
            basePosition=self.config.robot_base_position,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.reset_robot_home()
        self.configure_grasp_dynamics()
        self.step(self.config.settle_steps // 4)

    def reset_robot_home(self) -> None:
        assert self.client_id is not None
        assert self.robot_id is not None

        for joint_index, joint_value in zip(PANDA_ARM_JOINTS, PANDA_HOME):
            p.resetJointState(
                self.robot_id,
                joint_index,
                joint_value,
                physicsClientId=self.client_id,
            )
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=joint_value,
                force=240.0,
                physicsClientId=self.client_id,
            )

        for joint_index in PANDA_FINGER_JOINTS:
            p.resetJointState(
                self.robot_id,
                joint_index,
                0.04,
                physicsClientId=self.client_id,
            )
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                force=40.0,
                physicsClientId=self.client_id,
            )

    def spawn_objects(
        self,
        urdf_paths: Iterable[str | Path | dict[str, object]],
        seed: int = 0,
    ) -> list[dict[str, object]]:
        assert self.client_id is not None

        rng = random.Random(seed)
        spawned: list[dict[str, object]] = []
        used_positions: list[tuple[float, float]] = []

        for index, raw_path in enumerate(urdf_paths):
            scale_override = None
            if isinstance(raw_path, dict):
                urdf_path = self.resolve_asset_path(raw_path["path"])
                scale_override = raw_path.get("scale")
            else:
                urdf_path = self.resolve_asset_path(raw_path)
            x, y = self.sample_object_xy(rng, used_positions)
            z = self.config.object_spawn_height + 0.04 * index
            yaw = rng.uniform(-3.14159, 3.14159)
            if scale_override is None:
                scale = rng.uniform(*self.config.object_scale_range)
            else:
                scale = float(scale_override)
            orientation = p.getQuaternionFromEuler((0.0, 0.0, yaw))

            object_id = p.loadURDF(
                str(urdf_path),
                basePosition=(x, y, z),
                baseOrientation=orientation,
                useFixedBase=False,
                globalScaling=scale,
                physicsClientId=self.client_id,
            )
            self.spawned_object_ids.append(object_id)
            used_positions.append((x, y))
            p.changeDynamics(
                object_id,
                -1,
                lateralFriction=rng.uniform(1.2, 2.8),
                spinningFriction=rng.uniform(0.02, 0.08),
                rollingFriction=rng.uniform(0.005, 0.03),
                restitution=0.0,
                physicsClientId=self.client_id,
            )
            spawned.append(
                {
                    "object_id": object_id,
                    "path": str(urdf_path),
                    "position": [x, y, z],
                    "yaw": yaw,
                    "scale": scale,
                }
            )

        self.step(self.config.settle_steps)
        return spawned

    def sample_object_xy(
        self,
        rng: random.Random,
        used_positions: list[tuple[float, float]],
    ) -> tuple[float, float]:
        for _ in range(50):
            x = rng.uniform(*self.config.object_xy_bounds[0])
            y = rng.uniform(*self.config.object_xy_bounds[1])
            if all(
                math.dist((x, y), existing_xy) >= self.config.object_min_xy_separation
                for existing_xy in used_positions
            ):
                return x, y
        return (
            rng.uniform(*self.config.object_xy_bounds[0]),
            rng.uniform(*self.config.object_xy_bounds[1]),
        )

    def sample_camera_config(
        self,
        seed: int = 0,
        object_ids: Iterable[int] | None = None,
    ) -> CameraConfig:
        rng = random.Random(seed)
        focus_target = np.array(self.config.camera.target, dtype=np.float32)

        if object_ids:
            centers = []
            for object_id in object_ids:
                aabb_min, aabb_max = self.get_object_aabb(int(object_id))
                centers.append(0.5 * (aabb_min + aabb_max))
            if centers:
                focus_target = np.mean(np.stack(centers, axis=0), axis=0)

        jitter = np.array(
            [
                rng.uniform(-self.config.camera_target_jitter[0], self.config.camera_target_jitter[0]),
                rng.uniform(-self.config.camera_target_jitter[1], self.config.camera_target_jitter[1]),
                rng.uniform(-self.config.camera_target_jitter[2], self.config.camera_target_jitter[2]),
            ],
            dtype=np.float32,
        )
        target = focus_target + jitter

        distance = rng.uniform(*self.config.camera_distance_range)
        azimuth = math.radians(rng.uniform(*self.config.camera_azimuth_range_deg))
        elevation = math.radians(rng.uniform(*self.config.camera_elevation_range_deg))
        eye = (
            float(target[0] + distance * math.cos(elevation) * math.cos(azimuth)),
            float(target[1] + distance * math.cos(elevation) * math.sin(azimuth)),
            float(target[2] + distance * math.sin(elevation)),
        )
        return CameraConfig(
            width=self.config.camera.width,
            height=self.config.camera.height,
            eye=eye,
            target=(float(target[0]), float(target[1]), float(target[2])),
            up=(0.0, 0.0, 1.0),
            fov=rng.uniform(*self.config.camera_fov_range),
            near=self.config.camera.near,
            far=self.config.camera.far,
        )

    def capture_rgbd(
        self,
        camera: CameraConfig | None = None,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        assert self.client_id is not None
        camera = camera or self.config.camera
        self.last_camera_config = camera

        light_direction = (1.0, 1.0, 1.0)
        light_distance = 1.5
        if seed is not None:
            rng = random.Random(seed)
            light_direction = (
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(0.25, 1.0),
            )
            light_distance = rng.uniform(*self.config.light_distance_range)

        self.last_light_direction = light_direction
        self.last_light_distance = light_distance
        width, height, rgba, depth_buffer, segmentation = p.getCameraImage(
            width=camera.width,
            height=camera.height,
            viewMatrix=camera.view_matrix(),
            projectionMatrix=camera.projection_matrix(),
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            lightDirection=light_direction,
            lightDistance=light_distance,
            physicsClientId=self.client_id,
        )

        rgba_np = np.reshape(np.array(rgba, dtype=np.uint8), (height, width, 4))
        rgb = rgba_np[:, :, :3]
        depth_buffer_np = np.reshape(np.array(depth_buffer, dtype=np.float32), (height, width))
        depth_m = (camera.far * camera.near) / (
            camera.far - (camera.far - camera.near) * depth_buffer_np
        )
        segmentation_np = np.reshape(np.array(segmentation, dtype=np.int32), (height, width))

        return {
            "rgb": rgb,
            "depth": depth_m,
            "segmentation": segmentation_np,
        }

    def get_object_pose(self, object_id: int) -> dict[str, list[float]]:
        assert self.client_id is not None
        position, orientation = p.getBasePositionAndOrientation(
            object_id,
            physicsClientId=self.client_id,
        )
        return {
            "position": list(position),
            "orientation": list(orientation),
        }

    def get_object_aabb(self, object_id: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.client_id is not None
        aabb_min, aabb_max = p.getAABB(object_id, physicsClientId=self.client_id)
        return np.array(aabb_min, dtype=np.float32), np.array(aabb_max, dtype=np.float32)

    def save_state(self) -> int:
        assert self.client_id is not None
        return p.saveState(physicsClientId=self.client_id)

    def restore_state(self, state_id: int) -> None:
        assert self.client_id is not None
        p.restoreState(stateId=state_id, physicsClientId=self.client_id)
        self.reset_robot_home()

    def remove_state(self, state_id: int) -> None:
        assert self.client_id is not None
        p.removeState(stateUniqueId=state_id, physicsClientId=self.client_id)

    def open_gripper(self, target_opening: float = 0.04, steps: int = 120) -> None:
        self.set_gripper_opening(target_opening)
        self.step(steps)

    def close_gripper(self, target_opening: float = 0.0, steps: int = 180) -> None:
        self.set_gripper_opening(target_opening)
        self.step(steps)

    def set_gripper_opening(self, target_opening: float) -> None:
        assert self.client_id is not None
        assert self.robot_id is not None
        opening = float(np.clip(target_opening, 0.0, 0.04))
        for joint_index in PANDA_FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=opening,
                force=60.0,
                physicsClientId=self.client_id,
            )

    def configure_grasp_dynamics(self) -> None:
        assert self.client_id is not None
        assert self.robot_id is not None
        for link_index in (8, 9, 10):
            p.changeDynamics(
                self.robot_id,
                link_index,
                lateralFriction=5.0,
                spinningFriction=0.1,
                rollingFriction=0.05,
                restitution=0.0,
                physicsClientId=self.client_id,
            )

    def solve_arm_ik(
        self,
        target_position: Iterable[float],
        target_orientation: Iterable[float],
    ) -> list[float]:
        assert self.client_id is not None
        assert self.robot_id is not None
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            8,
            list(target_position),
            list(target_orientation),
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self.client_id,
        )
        return [float(value) for value in joint_positions[: len(PANDA_ARM_JOINTS)]]

    def interpolate_arm_joints(
        self,
        target_joints: Iterable[float],
        steps: int = 180,
    ) -> list[float]:
        assert self.client_id is not None
        assert self.robot_id is not None
        current_joints = [
            p.getJointState(self.robot_id, joint_index, physicsClientId=self.client_id)[0]
            for joint_index in PANDA_ARM_JOINTS
        ]
        target_joint_list = [float(value) for value in target_joints]

        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            for joint_offset, joint_index in enumerate(PANDA_ARM_JOINTS):
                joint_value = current_joints[joint_offset] + alpha * (
                    target_joint_list[joint_offset] - current_joints[joint_offset]
                )
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=joint_value,
                    force=240.0,
                    positionGain=0.08,
                    velocityGain=1.0,
                    maxVelocity=0.9,
                    physicsClientId=self.client_id,
                )
            self.step(1)

        for _ in range(30):
            for joint_offset, joint_index in enumerate(PANDA_ARM_JOINTS):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=target_joint_list[joint_offset],
                    force=240.0,
                    positionGain=0.08,
                    velocityGain=1.0,
                    maxVelocity=0.9,
                    physicsClientId=self.client_id,
                )
            self.step(1)

        return target_joint_list

    def move_end_effector(
        self,
        target_position: Iterable[float],
        target_orientation: Iterable[float],
        steps: int = 180,
    ) -> list[float]:
        joint_positions = self.solve_arm_ik(target_position, target_orientation)
        return self.interpolate_arm_joints(joint_positions, steps=steps)

    def generate_top_down_candidates(
        self,
        object_ids: Iterable[int],
        yaw_count: int = 4,
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        yaw_values = np.linspace(0.0, np.pi, num=yaw_count, endpoint=False)
        xy_offsets = (
            (0.0, 0.0),
            (0.008, 0.0),
            (-0.008, 0.0),
            (0.0, 0.008),
            (0.0, -0.008),
        )
        for object_id in object_ids:
            aabb_min, aabb_max = self.get_object_aabb(object_id)
            center = 0.5 * (aabb_min + aabb_max)
            extent = aabb_max - aabb_min
            top_z = float(aabb_max[2])
            for yaw in yaw_values:
                for x_offset, y_offset in xy_offsets:
                    x_position = float(center[0] + x_offset)
                    y_position = float(center[1] + y_offset)
                    max_xy_extent = float(max(extent[0], extent[1]))
                    opening = min(0.04, max(0.012, 0.5 * max_xy_extent))
                    candidates.append(
                        {
                            "object_id": object_id,
                            "yaw": float(yaw),
                            "extent": extent.tolist(),
                            "center": center.tolist(),
                            "xy_offset": [x_offset, y_offset],
                            "approach_position": [
                                x_position,
                                y_position,
                                top_z + 0.16,
                            ],
                            "grasp_position": [
                                x_position,
                                y_position,
                                top_z + max(0.006, float(extent[2]) * 0.18),
                            ],
                            "lift_position": [
                                x_position,
                                y_position,
                                top_z + 0.24,
                            ],
                            "target_opening": opening,
                        }
                    )
        return candidates

    def get_robot_object_contact_links(self, object_id: int) -> list[int]:
        assert self.client_id is not None
        assert self.robot_id is not None
        contact_points = p.getContactPoints(
            self.robot_id,
            object_id,
            physicsClientId=self.client_id,
        )
        return sorted({int(point[3]) for point in contact_points})

    def is_object_supported_by_surface(self, object_id: int) -> bool:
        assert self.client_id is not None
        assert self.table_id is not None
        assert self.plane_id is not None
        return bool(
            p.getContactPoints(object_id, self.table_id, physicsClientId=self.client_id)
            or p.getContactPoints(object_id, self.plane_id, physicsClientId=self.client_id)
        )

    def evaluate_grasp_candidate(self, candidate: dict[str, object]) -> dict[str, object]:
        assert self.client_id is not None
        object_id = int(candidate["object_id"])
        initial_pose = self.get_object_pose(object_id)
        initial_z = float(initial_pose["position"][2])
        target_orientation = p.getQuaternionFromEuler(
            [TOP_DOWN_GRASP_PITCH, 0.0, float(candidate["yaw"])]
        )

        self.open_gripper(float(candidate["target_opening"]), steps=120)
        self.move_end_effector(candidate["approach_position"], target_orientation, steps=120)
        self.move_end_effector(candidate["grasp_position"], target_orientation, steps=120)
        self.close_gripper(0.0, steps=180)
        self.step(30)
        contact_links_after_close = self.get_robot_object_contact_links(object_id)
        supported_after_close = self.is_object_supported_by_surface(object_id)
        self.move_end_effector(candidate["lift_position"], target_orientation, steps=240)
        self.step(120)

        final_pose = self.get_object_pose(object_id)
        final_z = float(final_pose["position"][2])
        lifted_delta = final_z - initial_z
        contact_links_after_lift = self.get_robot_object_contact_links(object_id)
        supported_after_lift = self.is_object_supported_by_surface(object_id)
        physical_lift_success = (
            lifted_delta > self.config.grasp_lift_threshold
            and not supported_after_lift
            and bool(contact_links_after_close or contact_links_after_lift)
        )

        return {
            "object_id": object_id,
            "yaw": float(candidate["yaw"]),
            "success": physical_lift_success,
            "physical_lift_success": physical_lift_success,
            "initial_position": initial_pose["position"],
            "final_position": final_pose["position"],
            "lifted_delta": lifted_delta,
            "contact_links_after_close": contact_links_after_close,
            "contact_links_after_lift": contact_links_after_lift,
            "supported_after_close": supported_after_close,
            "supported_after_lift": supported_after_lift,
            "approach_position": candidate["approach_position"],
            "grasp_position": candidate["grasp_position"],
            "lift_position": candidate["lift_position"],
            "target_opening": float(candidate["target_opening"]),
            "xy_offset": candidate["xy_offset"],
        }

    def label_grasp_candidates(
        self,
        object_ids: Iterable[int],
        yaw_count: int = 4,
    ) -> list[dict[str, object]]:
        state_id = self.save_state()
        results: list[dict[str, object]] = []
        try:
            for candidate in self.generate_top_down_candidates(object_ids, yaw_count=yaw_count):
                self.restore_state(state_id)
                results.append(self.evaluate_grasp_candidate(candidate))
        finally:
            self.remove_state(state_id)
            self.reset_robot_home()
        return results

    def step(self, steps: int = 1) -> None:
        assert self.client_id is not None
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            self._simulation_step_counter += 1
            if (
                self.config.gui
                and self.config.enable_gui_previews
                and self.config.live_camera_update_interval_steps > 0
                and self._simulation_step_counter % self.config.live_camera_update_interval_steps == 0
            ):
                self.capture_rgbd(camera=self.last_camera_config or self.config.camera)
            if self.config.gui:
                time.sleep(self.config.time_step * self.config.gui_step_sleep)

    def disconnect(self) -> None:
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
        self.client_id = None
        self.robot_id = None
        self.table_id = None
        self.plane_id = None
        self.spawned_object_ids = []
        self._simulation_step_counter = 0

    @staticmethod
    def data_root() -> Path:
        if ASCII_PYBULLET_DATA_ROOT.exists():
            return ASCII_PYBULLET_DATA_ROOT
        return Path(pybullet_data.getDataPath())

    @staticmethod
    def resolve_asset_path(raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path

        candidate = GraspingScene.data_root() / path
        if candidate.exists():
            return candidate

        raise FileNotFoundError(f"Could not resolve URDF asset: {raw_path}")
