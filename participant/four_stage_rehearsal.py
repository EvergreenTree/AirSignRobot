#!/usr/bin/env python3
"""Run and record an actuator-only rehearsal of all four EBiM Track 3 stages.

The workload uses the official Task 3 scene, the Robotiq mobile FR3 Duo, raw
Lula IK, articulation position targets, and the official mobile-base wheel
target computation. It records measured base, joint, gripper, TCP, and task
object state for deterministic browser replay.

This is intentionally a controller rehearsal, not a scored benchmark run.
It does not teleport task objects or robot links, and it never converts
actuator waypoint completion into an official stage or score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK3_ROOT = REPO_ROOT / "task3_isaacsim"
for module_dir in (
    REPO_ROOT / "scripts" / "scenes",
    REPO_ROOT / "scripts" / "evaluation" / "task3",
    REPO_ROOT / "task2_isaacsim" / "scripts",
    TASK3_ROOT / "scripts",
    TASK3_ROOT / "scripts" / "common",
):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import scene_robot_room_keyboard as room_scene  # noqa: E402
from gripper_profiles import (  # noqa: E402
    get_gripper_profile,
    get_profile_drive_gains,
)
from isaacsim_fr3duo_teleop_bridge_args import (  # noqa: E402
    add_common_bridge_args,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/EBiM_Challenge/airsign-four-stage"),
    )
    parser.add_argument(
        "--room-usd",
        type=Path,
        default=room_scene.asset_path("robot_room.usd"),
    )
    parser.add_argument("--robot-usd", type=Path, default=None)
    parser.add_argument("--robot-x", type=float, default=None)
    parser.add_argument("--robot-y", type=float, default=None)
    parser.add_argument("--robot-z", type=float, default=None)
    parser.add_argument("--robot-yaw", type=float, default=None)
    parser.add_argument(
        "--head-placement",
        type=room_scene.head_placement_arg,
        default="A",
    )
    parser.add_argument("--trace-hz", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--motion-steps", type=int, default=120)
    parser.add_argument("--gripper-steps", type=int, default=120)
    parser.add_argument("--feed-hold-seconds", type=float, default=3.25)
    parser.add_argument("--base-max-steps", type=int, default=6000)
    parser.add_argument("--base-max-speed", type=float, default=0.35)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render World steps. Replay JSON is recorded either way.",
    )
    add_common_bridge_args(parser)
    parser.set_defaults(
        headless=True,
        arm_teleop_gripper_open=None,
        arm_teleop_gripper_closed=None,
    )
    return parser


ARGS = build_arg_parser().parse_args()
PROFILE = get_gripper_profile("robotiq")
if ARGS.robot_usd is None:
    ARGS.robot_usd = PROFILE.robot_usd
if ARGS.arm_teleop_gripper_open is None:
    ARGS.arm_teleop_gripper_open = PROFILE.keyboard_positions[0]
if ARGS.arm_teleop_gripper_closed is None:
    ARGS.arm_teleop_gripper_closed = PROFILE.keyboard_positions[1]

from isaacsim import SimulationApp  # noqa: E402

SIMULATION_APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
    }
)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.robot_motion.motion_generation")
SIMULATION_APP.update()
enable_extension("isaacsim.ros2.bridge")
SIMULATION_APP.update()

import isaacsim_fr3duo_teleop_bridge_core as core  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from dual_arm_lula import create_raw_dual_arm_lula  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

ROBOT_PRIM_PATH = "/World/Robot"
TASK_OBJECT_NAMES = {
    "plate": "plate2",
    "cup": "cup",
    "bowl": "bowl2",
    "spoon": "spoon2",
}
OFFICIAL_STAGE_NAMES = {
    1: "Table Setup",
    2: "Feeding",
    3: "Bean Recovery",
    4: "Cleanup",
}


def rounded(values: Iterable[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def world_pose(
    robot: SingleArticulation,
) -> tuple[np.ndarray, np.ndarray]:
    position, orientation = robot.get_world_pose()
    return (
        np.asarray(position, dtype=np.float64),
        np.asarray(orientation, dtype=np.float64),
    )


def spine_position(robot: SingleArticulation) -> float:
    names = list(robot.dof_names)
    try:
        index = names.index("franka_spine_vertical_joint")
    except ValueError:
        return 0.0
    return float(robot.get_joint_positions()[index])


def yaw_from_wxyz(orientation: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in orientation)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def apply_targets(
    robot: SingleArticulation,
    target_by_name: dict[str, float],
) -> None:
    names = list(robot.dof_names)
    ordered = [
        (names.index(name), float(value))
        for name, value in target_by_name.items()
    ]
    robot.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=np.asarray(
                [value for _, value in ordered], dtype=np.float32
            ),
            joint_indices=np.asarray(
                [index for index, _ in ordered], dtype=np.int64
            ),
        )
    )


def find_prim_path(stage: Any, name: str) -> str:
    candidates = (
        f"/World/Environment/RobotRoom/Asset/{name}",
        f"/World/Environment/RobotRoom/Asset/root/{name}",
        f"/root/{name}",
    )
    for candidate in candidates:
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return candidate
    suffix = f"/{name}"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.endswith(suffix):
            return path
    raise RuntimeError(f"Could not find Task 3 prim named {name!r}")


def prim_pose(stage: Any, path: str) -> dict[str, list[float]]:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {path}")
    transform = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return {
        "position": rounded(translation),
        "orientation_wxyz": rounded(
            (
                rotation.GetReal(),
                imaginary[0],
                imaginary[1],
                imaginary[2],
            )
        ),
    }


def task_object_snapshot(
    stage: Any,
    object_paths: dict[str, str],
) -> dict[str, dict[str, list[float]]]:
    return {
        name: prim_pose(stage, path)
        for name, path in object_paths.items()
    }


def bean_summary(stage: Any, recovery_center: np.ndarray) -> dict[str, Any]:
    paths = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetName().startswith("Bean_")
    )
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    positions = np.asarray(
        [
            cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath(path)
            ).ExtractTranslation()
            for path in paths
        ],
        dtype=np.float64,
    )
    if positions.size == 0:
        return {
            "count": 0,
            "centroid": None,
            "inside_official_recovery_sphere": 0,
            "inside_fraction": None,
        }
    distances = np.linalg.norm(positions - recovery_center, axis=1)
    inside = int(np.sum(distances <= 0.2))
    return {
        "count": len(paths),
        "centroid": rounded(np.mean(positions, axis=0)),
        "inside_official_recovery_sphere": inside,
        "inside_fraction": round(inside / len(paths), 6),
    }


def max_object_displacement(
    before: dict[str, dict[str, list[float]]],
    after: dict[str, dict[str, list[float]]],
) -> float:
    return max(
        float(
            np.linalg.norm(
                np.asarray(after[name]["position"])
                - np.asarray(before[name]["position"])
            )
        )
        for name in before
    )


def discover_gripper_drivers(
    robot: SingleArticulation,
) -> dict[str, tuple[str, int]]:
    names = list(robot.dof_names)
    result: dict[str, tuple[str, int]] = {}
    for side in ("left", "right"):
        candidates = [
            (index, name)
            for index, name in enumerate(names)
            if name.startswith(f"{side}_")
            and "outer_knuckle_joint" in name
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one {side} Robotiq driver, found {candidates}"
            )
        index, name = candidates[0]
        result[side] = (name, index)
    return result


@dataclass
class TraceRecorder:
    world: World
    robot: SingleArticulation
    ik: Any
    stage: Any
    object_paths: dict[str, str]
    gripper_drivers: dict[str, tuple[str, int]]
    physics_hz: float
    trace_hz: int
    render: bool

    def __post_init__(self) -> None:
        ratio = self.physics_hz / self.trace_hz
        self.stride = int(round(ratio))
        if self.trace_hz < 1 or not math.isclose(
            ratio, self.stride, abs_tol=1e-9
        ):
            raise ValueError("--trace-hz must divide the physics rate")
        self.sim_step = 0
        self.frames: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.current_stage = 0
        self.current_phase = "initializing"

    def set_phase(
        self,
        stage_id: int,
        phase: str,
        *,
        intent: str,
    ) -> None:
        self.current_stage = stage_id
        self.current_phase = phase
        print(
            "FOUR_STAGE_REHEARSAL_PHASE "
            f"stage={stage_id} phase={phase} step={self.sim_step}",
            flush=True,
        )
        self.events.append(
            {
                "simulation_step": self.sim_step,
                "time_seconds": round(self.sim_step / self.physics_hz, 6),
                "stage_id": stage_id,
                "phase": phase,
                "intent": intent,
                "intent_is_measured_outcome": False,
            }
        )
        self.record(force=True)

    def step(self) -> None:
        self.world.step(render=self.render)
        self.sim_step += 1
        if self.sim_step % self.stride == 0:
            self.record()

    def record(self, *, force: bool = False) -> None:
        if (
            not force
            and self.frames
            and self.frames[-1]["simulation_step"] == self.sim_step
        ):
            return
        base_position, base_orientation = world_pose(self.robot)
        left, right = self.ik.current_end_effector_poses(
            base_position,
            base_orientation,
            spine_position(self.robot),
        )
        joint_positions = np.asarray(
            self.robot.get_joint_positions(), dtype=np.float64
        )
        driver_positions = {
            side: round(float(joint_positions[index]), 6)
            for side, (_name, index) in self.gripper_drivers.items()
        }
        self.frames.append(
            {
                "simulation_step": self.sim_step,
                "time_seconds": round(self.sim_step / self.physics_hz, 6),
                "stage_id": self.current_stage,
                "phase": self.current_phase,
                "base": {
                    "position": rounded(base_position),
                    "orientation_wxyz": rounded(base_orientation),
                },
                "tcp": {
                    "left": {
                        "position": rounded(left[0]),
                        "orientation_wxyz": rounded(left[1]),
                    },
                    "right": {
                        "position": rounded(right[0]),
                        "orientation_wxyz": rounded(right[1]),
                    },
                },
                "gripper_driver_positions": driver_positions,
                "joint_positions": rounded(joint_positions),
                "task_objects": task_object_snapshot(
                    self.stage, self.object_paths
                ),
            }
        )


def command_gripper(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    drivers: dict[str, tuple[str, int]],
    *,
    sides: tuple[str, ...],
    target: float,
    steps: int,
    stage_id: int,
    phase: str,
    intent: str,
    tolerance: float = 0.06,
) -> dict[str, Any]:
    recorder.set_phase(stage_id, phase, intent=intent)
    indices = np.asarray(
        [drivers[side][1] for side in sides], dtype=np.int64
    )
    targets = np.asarray([target] * len(sides), dtype=np.float32)
    controller = robot.get_articulation_controller()
    start_step = recorder.sim_step
    for _ in range(steps):
        controller.apply_action(
            ArticulationAction(
                joint_positions=targets,
                joint_indices=indices,
            )
        )
        recorder.step()
    measured = np.asarray(robot.get_joint_positions())[indices]
    errors = np.abs(measured - targets)
    return {
        "phase": phase,
        "sides": list(sides),
        "start_step": start_step,
        "end_step": recorder.sim_step,
        "target_position": round(float(target), 6),
        "measured_positions": {
            side: round(float(value), 6)
            for side, value in zip(sides, measured, strict=True)
        },
        "absolute_errors": {
            side: round(float(value), 6)
            for side, value in zip(sides, errors, strict=True)
        },
        "passed": bool(np.all(errors <= tolerance)),
        "pass_definition": "measured_driver_error_within_tolerance",
    }


def move_tcp_targets(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
    *,
    left_target: np.ndarray | None,
    right_target: np.ndarray | None,
    steps: int,
    settle_steps: int,
    stage_id: int,
    phase: str,
    intent: str,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    recorder.set_phase(stage_id, phase, intent=intent)
    base_position, base_orientation = world_pose(robot)
    initial_left, initial_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    target_left = np.asarray(
        initial_left[0] if left_target is None else left_target,
        dtype=np.float64,
    )
    target_right = np.asarray(
        initial_right[0] if right_target is None else right_target,
        dtype=np.float64,
    )
    start_step = recorder.sim_step
    left_successes = 0
    right_successes = 0
    for step in range(1, steps + 1):
        fraction = step / steps
        desired_left = (
            initial_left[0]
            + fraction * (target_left - initial_left[0])
        )
        desired_right = (
            initial_right[0]
            + fraction * (target_right - initial_right[0])
        )
        base_position, base_orientation = world_pose(robot)
        result = ik.solve(
            desired_left,
            desired_right,
            initial_left[1],
            initial_right[1],
            spine_position=spine_position(robot),
            base_position=base_position,
            base_orientation_wxyz=base_orientation,
        )
        left_successes += int(result.left_succeeded)
        right_successes += int(result.right_succeeded)
        apply_targets(robot, result.combined)
        recorder.step()
    for _ in range(settle_steps):
        base_position, base_orientation = world_pose(robot)
        result = ik.solve(
            target_left,
            target_right,
            initial_left[1],
            initial_right[1],
            spine_position=spine_position(robot),
            base_position=base_position,
            base_orientation_wxyz=base_orientation,
        )
        apply_targets(robot, result.combined)
        recorder.step()
    base_position, base_orientation = world_pose(robot)
    final_left, final_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    left_error = float(np.linalg.norm(final_left[0] - target_left))
    right_error = float(np.linalg.norm(final_right[0] - target_right))
    passed = bool(
        left_successes == steps
        and right_successes == steps
        and left_error <= tolerance
        and right_error <= tolerance
    )
    return {
        "phase": phase,
        "start_step": start_step,
        "end_step": recorder.sim_step,
        "motion_steps": steps,
        "left_solve_successes": left_successes,
        "right_solve_successes": right_successes,
        "target_positions": {
            "left": rounded(target_left),
            "right": rounded(target_right),
        },
        "final_positions": {
            "left": rounded(final_left[0]),
            "right": rounded(final_right[0]),
        },
        "errors_metres": {
            "left": round(left_error, 6),
            "right": round(right_error, 6),
        },
        "tolerance_metres": tolerance,
        "passed": passed,
        "pass_definition": "ik_solved_and_measured_tcp_error_within_tolerance",
    }


def move_tcp_relative(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
    *,
    left_delta: tuple[float, float, float],
    right_delta: tuple[float, float, float],
    steps: int,
    settle_steps: int,
    stage_id: int,
    phase: str,
    intent: str,
) -> dict[str, Any]:
    base_position, base_orientation = world_pose(robot)
    initial_left, initial_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    return move_tcp_targets(
        recorder,
        robot,
        ik,
        left_target=initial_left[0] + np.asarray(left_delta),
        right_target=initial_right[0] + np.asarray(right_delta),
        steps=steps,
        settle_steps=settle_steps,
        stage_id=stage_id,
        phase=phase,
        intent=intent,
    )


def hold_tcp_targets(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
    *,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_orientation: np.ndarray,
    right_orientation: np.ndarray,
    duration_seconds: float,
    stage_id: int,
    phase: str,
    intent: str,
) -> dict[str, Any]:
    recorder.set_phase(stage_id, phase, intent=intent)
    steps = int(math.ceil(duration_seconds * recorder.physics_hz))
    errors: list[float] = []
    successes = 0
    start_step = recorder.sim_step
    for _ in range(steps):
        base_position, base_orientation = world_pose(robot)
        result = ik.solve(
            left_target,
            right_target,
            left_orientation,
            right_orientation,
            spine_position=spine_position(robot),
            base_position=base_position,
            base_orientation_wxyz=base_orientation,
        )
        successes += int(result.left_succeeded and result.right_succeeded)
        apply_targets(robot, result.combined)
        recorder.step()
        base_position, base_orientation = world_pose(robot)
        measured_left, _measured_right = ik.current_end_effector_poses(
            base_position,
            base_orientation,
            spine_position(robot),
        )
        errors.append(
            float(np.linalg.norm(measured_left[0] - left_target))
        )
    actual_seconds = (recorder.sim_step - start_step) / recorder.physics_hz
    return {
        "phase": phase,
        "start_step": start_step,
        "end_step": recorder.sim_step,
        "commanded_duration_seconds": duration_seconds,
        "measured_simulation_duration_seconds": round(actual_seconds, 6),
        "ik_successes": successes,
        "steps": steps,
        "left_tcp_error_metres": {
            "mean": round(float(np.mean(errors)), 6),
            "maximum": round(float(np.max(errors)), 6),
        },
        "passed": bool(
            actual_seconds >= 3.0
            and successes == steps
            and max(errors) <= 0.03
        ),
        "pass_definition": (
            "simulation_hold_at_least_3_seconds_with_ik_success_and_"
            "measured_tcp_error_within_3cm"
        ),
    }


def drive_base_to(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    steering_ids: list[int],
    drive_ids: list[int],
    target_xy: np.ndarray,
    target_yaw: float,
    *,
    max_steps: int,
    max_speed: float,
    stage_id: int,
    phase: str,
    intent: str,
) -> dict[str, Any]:
    recorder.set_phase(stage_id, phase, intent=intent)
    controller = robot.get_articulation_controller()
    position_tolerance = 0.04
    yaw_tolerance = math.radians(3.0)
    settled_steps = 0
    start_step = recorder.sim_step
    steps = 0
    path_length = 0.0
    previous_position, _ = world_pose(robot)
    for steps in range(1, max_steps + 1):
        position, orientation = world_pose(robot)
        path_length += float(
            np.linalg.norm(position[:2] - previous_position[:2])
        )
        previous_position = position
        yaw = yaw_from_wxyz(orientation)
        world_error = np.asarray(target_xy, dtype=np.float64) - position[:2]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        body_error = np.array(
            (
                cos_yaw * world_error[0] + sin_yaw * world_error[1],
                -sin_yaw * world_error[0] + cos_yaw * world_error[1],
            )
        )
        yaw_error = wrap_to_pi(target_yaw - yaw)
        distance = float(np.linalg.norm(world_error))
        if distance <= position_tolerance and abs(yaw_error) <= yaw_tolerance:
            settled_steps += 1
            vx = vy = wz = 0.0
        else:
            settled_steps = 0
            body_command = 1.35 * body_error
            command_norm = float(np.linalg.norm(body_command))
            minimum_speed = min(0.14, max_speed)
            if 0.0 < command_norm < minimum_speed:
                body_command *= minimum_speed / command_norm
            command_norm = float(np.linalg.norm(body_command))
            if command_norm > max_speed:
                body_command *= max_speed / command_norm
            vx, vy = (float(value) for value in body_command)
            wz = float(np.clip(1.8 * yaw_error, -0.5, 0.5))
        steering_targets, drive_targets = core._compute_drive_targets(
            robot.get_joint_positions(),
            steering_ids,
            vx,
            vy,
            wz,
        )
        controller.apply_action(
            ArticulationAction(
                joint_positions=steering_targets,
                joint_indices=np.asarray(
                    steering_ids, dtype=np.int64
                ),
            )
        )
        controller.apply_action(
            ArticulationAction(
                joint_velocities=drive_targets,
                joint_indices=np.asarray(drive_ids, dtype=np.int64),
            )
        )
        recorder.step()
        if settled_steps >= 30:
            break
    position, orientation = world_pose(robot)
    final_yaw = yaw_from_wxyz(orientation)
    position_error = float(np.linalg.norm(target_xy - position[:2]))
    yaw_error = abs(wrap_to_pi(target_yaw - final_yaw))
    return {
        "phase": phase,
        "start_step": start_step,
        "end_step": recorder.sim_step,
        "steps": steps,
        "target_xy": rounded(target_xy),
        "final_xy": rounded(position[:2]),
        "position_error_metres": round(position_error, 6),
        "yaw_error_degrees": round(math.degrees(yaw_error), 6),
        "path_length_metres": round(path_length, 6),
        "position_tolerance_metres": position_tolerance,
        "yaw_tolerance_degrees": math.degrees(yaw_tolerance),
        "passed": bool(
            position_error <= position_tolerance
            and yaw_error <= yaw_tolerance
        ),
        "pass_definition": "closed_loop_base_pose_within_tolerance",
    }


def stage_record(
    stage_id: int,
    *,
    start_step: int,
    end_step: int,
    gates: list[dict[str, Any]],
    before_objects: dict[str, Any],
    after_objects: dict[str, Any],
    additional_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    motion_gate_passed = all(bool(gate["passed"]) for gate in gates)
    result = {
        "stage_id": stage_id,
        "stage_name": OFFICIAL_STAGE_NAMES[stage_id],
        "official_stage_max_score": 4,
        "classification": "actuator_rehearsal",
        "start_step": start_step,
        "end_step": end_step,
        "motion_gate_passed": motion_gate_passed,
        "motion_gates": gates,
        "task_objects_before": before_objects,
        "task_objects_after": after_objects,
        "maximum_task_object_displacement_metres": round(
            max_object_displacement(before_objects, after_objects), 6
        ),
        "official_stage_complete": False,
        "official_stage_score": None,
        "completion_reason": (
            "The run validates actuator waypoints only. It does not verify "
            "the rulebook's object and bean outcome predicates."
        ),
    }
    if additional_metrics:
        result["additional_metrics"] = additional_metrics
    return result


def run_all_stages(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
    steering_ids: list[int],
    drive_ids: list[int],
    gripper_drivers: dict[str, tuple[str, int]],
    stage: Any,
    object_paths: dict[str, str],
    head_path: str,
    recovery_center: np.ndarray,
    sink_center: np.ndarray,
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    base_origin, initial_orientation = world_pose(robot)
    nominal_yaw = yaw_from_wxyz(initial_orientation)
    forward_world = np.asarray(
        (math.cos(nominal_yaw), math.sin(nominal_yaw)),
        dtype=np.float64,
    )

    # Stage 1: short out-and-return transport motion rehearsal. The gripper
    # and lift cycle is actuator-measured; kitchen/dining traversal and actual
    # object transport are not inferred from it.
    stage_id = 1
    start_step = recorder.sim_step
    before = task_object_snapshot(stage, object_paths)
    gates = [
        drive_base_to(
            recorder,
            robot,
            steering_ids,
            drive_ids,
            base_origin[:2] + 0.40 * forward_world,
            nominal_yaw,
            max_steps=ARGS.base_max_steps,
            max_speed=ARGS.base_max_speed,
            stage_id=stage_id,
            phase="stage1_transport_outbound",
            intent="Execute the outbound leg of the table-setup transport rehearsal.",
        ),
        command_gripper(
            recorder,
            robot,
            gripper_drivers,
            sides=("left", "right"),
            target=PROFILE.keyboard_positions[0],
            steps=ARGS.gripper_steps,
            stage_id=stage_id,
            phase="stage1_open_grippers",
            intent="Open both grippers for the tableware pickup sequence.",
        ),
        command_gripper(
            recorder,
            robot,
            gripper_drivers,
            sides=("left", "right"),
            target=PROFILE.keyboard_positions[1],
            steps=ARGS.gripper_steps,
            stage_id=stage_id,
            phase="stage1_close_grippers",
            intent="Close both grippers for the transport grasp rehearsal.",
        ),
        move_tcp_relative(
            recorder,
            robot,
            ik,
            left_delta=(0.0, 0.0, 0.04),
            right_delta=(0.0, 0.0, 0.04),
            steps=ARGS.motion_steps,
            settle_steps=ARGS.settle_steps,
            stage_id=stage_id,
            phase="stage1_lift",
            intent="Lift the bimanual transport pose.",
        ),
        drive_base_to(
            recorder,
            robot,
            steering_ids,
            drive_ids,
            base_origin[:2],
            nominal_yaw,
            max_steps=ARGS.base_max_steps,
            max_speed=ARGS.base_max_speed,
            stage_id=stage_id,
            phase="stage1_transport_return",
            intent="Execute the return leg of the table-setup transport rehearsal.",
        ),
        move_tcp_relative(
            recorder,
            robot,
            ik,
            left_delta=(0.0, 0.0, -0.04),
            right_delta=(0.0, 0.0, -0.04),
            steps=ARGS.motion_steps,
            settle_steps=ARGS.settle_steps,
            stage_id=stage_id,
            phase="stage1_lower",
            intent="Lower the bimanual transport pose.",
        ),
        command_gripper(
            recorder,
            robot,
            gripper_drivers,
            sides=("left", "right"),
            target=PROFILE.keyboard_positions[0],
            steps=ARGS.gripper_steps,
            stage_id=stage_id,
            phase="stage1_release",
            intent="Release at the dining-side transport destination.",
        ),
    ]
    after = task_object_snapshot(stage, object_paths)
    stages.append(
        stage_record(
            stage_id,
            start_step=start_step,
            end_step=recorder.sim_step,
            gates=gates,
            before_objects=before,
            after_objects=after,
            additional_metrics={
                "rulebook_objects": list(TASK_OBJECT_NAMES),
                "kitchen_to_dining_traversal_executed": False,
                "assigned_target_poses_observed": False,
                "object_transport_verified": False,
            },
        )
    )

    # Stage 2: execute a short reachable TCP approach/insert/retract sequence
    # and hold for >3 simulated sec. The official head-relative feed target is
    # retained only as a reference and is not claimed as reached.
    stage_id = 2
    start_step = recorder.sim_step
    before = task_object_snapshot(stage, object_paths)
    head_position = np.asarray(
        prim_pose(stage, head_path)["position"], dtype=np.float64
    )
    feed_zone_reference = head_position + np.asarray((0.0, -0.10, 0.17))
    nav = drive_base_to(
        recorder,
        robot,
        steering_ids,
        drive_ids,
        base_origin[:2] + 0.20 * forward_world,
        nominal_yaw,
        max_steps=ARGS.base_max_steps,
        max_speed=ARGS.base_max_speed,
        stage_id=stage_id,
        phase="stage2_base_approach",
        intent="Execute a short closed-loop base approach for the feeding rehearsal.",
    )
    base_position, base_orientation = world_pose(robot)
    current_left, current_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    feed_approach = current_left[0] + np.asarray((0.0, 0.0, 0.04))
    approach = move_tcp_targets(
        recorder,
        robot,
        ik,
        left_target=feed_approach,
        right_target=current_right[0],
        steps=ARGS.motion_steps,
        settle_steps=ARGS.settle_steps,
        stage_id=stage_id,
        phase="stage2_feed_approach",
        intent="Move the left TCP to the start of the feeding corridor.",
    )
    base_position, base_orientation = world_pose(robot)
    _left_at_approach, right_at_approach = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    feed_insert = feed_approach + 0.04 * np.asarray(
        (forward_world[0], forward_world[1], 0.0)
    )
    insert = move_tcp_targets(
        recorder,
        robot,
        ik,
        left_target=feed_insert,
        right_target=right_at_approach[0],
        steps=ARGS.motion_steps,
        settle_steps=ARGS.settle_steps,
        stage_id=stage_id,
        phase="stage2_feed_insert",
        intent="Insert the left TCP 4 cm along the robot-forward direction.",
    )
    base_position, base_orientation = world_pose(robot)
    held_left, held_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    hold = hold_tcp_targets(
        recorder,
        robot,
        ik,
        left_target=feed_insert,
        right_target=held_right[0],
        left_orientation=held_left[1],
        right_orientation=held_right[1],
        duration_seconds=ARGS.feed_hold_seconds,
        stage_id=stage_id,
        phase="stage2_feed_hold",
        intent="Hold the feeding TCP pose for at least three simulated seconds.",
    )
    retract = move_tcp_targets(
        recorder,
        robot,
        ik,
        left_target=feed_approach,
        right_target=held_right[0],
        steps=ARGS.motion_steps,
        settle_steps=ARGS.settle_steps,
        stage_id=stage_id,
        phase="stage2_feed_retract",
        intent="Retract the left TCP from the feeding position.",
    )
    after = task_object_snapshot(stage, object_paths)
    stages.append(
        stage_record(
            stage_id,
            start_step=start_step,
            end_step=recorder.sim_step,
            gates=[nav, approach, insert, hold, retract],
            before_objects=before,
            after_objects=after,
            additional_metrics={
                "head_placement": ARGS.head_placement,
                "head_root_position": rounded(head_position),
                "official_feed_zone_reference": rounded(
                    feed_zone_reference
                ),
                "rehearsal_feed_approach_target": rounded(feed_approach),
                "rehearsal_feed_insert_target": rounded(feed_insert),
                "official_feed_zone_reached": False,
                "hold_seconds_measured": hold[
                    "measured_simulation_duration_seconds"
                ],
                "required_hold_seconds": 3.0,
                "spoon_grasp_verified": False,
                "beans_in_spoon_observed": False,
                "beans_returned_to_bowl_observed": False,
            },
        )
    )

    # Stage 3: execute a short closed-loop base approach plus a reachable
    # lift/forward/retract pour gesture. The official recovery sphere and bean
    # snapshot remain independent references, not controller outcomes.
    stage_id = 3
    start_step = recorder.sim_step
    before = task_object_snapshot(stage, object_paths)
    beans_before = bean_summary(stage, recovery_center)
    nav_corridor = drive_base_to(
        recorder,
        robot,
        steering_ids,
        drive_ids,
        base_origin[:2],
        nominal_yaw,
        max_steps=ARGS.base_max_steps,
        max_speed=ARGS.base_max_speed,
        stage_id=stage_id,
        phase="stage3_base_return",
        intent="Return to the rehearsal origin before bean recovery.",
    )
    nav_recovery = drive_base_to(
        recorder,
        robot,
        steering_ids,
        drive_ids,
        base_origin[:2] + 0.15 * forward_world,
        nominal_yaw,
        max_steps=ARGS.base_max_steps,
        max_speed=ARGS.base_max_speed,
        stage_id=stage_id,
        phase="stage3_base_approach",
        intent="Execute a short closed-loop approach for bean recovery.",
    )
    base_position, base_orientation = world_pose(robot)
    left_now, right_now = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    recovery_above = left_now[0] + np.asarray((0.0, 0.0, 0.04))
    move_above = move_tcp_targets(
        recorder,
        robot,
        ik,
        left_target=recovery_above,
        right_target=right_now[0],
        steps=ARGS.motion_steps,
        settle_steps=ARGS.settle_steps,
        stage_id=stage_id,
        phase="stage3_lift_for_pour",
        intent="Lift the left TCP for the recovery pour rehearsal.",
    )
    recovery_lower = recovery_above + 0.04 * np.asarray(
        (forward_world[0], forward_world[1], 0.0)
    )
    lower = move_tcp_targets(
        recorder,
        robot,
        ik,
        left_target=recovery_lower,
        right_target=right_now[0],
        steps=ARGS.motion_steps,
        settle_steps=ARGS.settle_steps,
        stage_id=stage_id,
        phase="stage3_pour_gesture",
        intent="Execute a 4 cm robot-forward recovery pour gesture.",
    )
    raise_again = move_tcp_targets(
        recorder,
        robot,
        ik,
        left_target=recovery_above,
        right_target=right_now[0],
        steps=ARGS.motion_steps,
        settle_steps=ARGS.settle_steps,
        stage_id=stage_id,
        phase="stage3_retract",
        intent="Retract from the recovery pour gesture.",
    )
    beans_after = bean_summary(stage, recovery_center)
    after = task_object_snapshot(stage, object_paths)
    stages.append(
        stage_record(
            stage_id,
            start_step=start_step,
            end_step=recorder.sim_step,
            gates=[
                nav_corridor,
                nav_recovery,
                move_above,
                lower,
                raise_again,
            ],
            before_objects=before,
            after_objects=after,
            additional_metrics={
                "official_recovery_sphere": {
                    "center": rounded(recovery_center),
                    "radius_metres": 0.2,
                },
                "rehearsal_above_target": rounded(recovery_above),
                "rehearsal_pour_target": rounded(recovery_lower),
                "official_recovery_sphere_reached": False,
                "beans_before": beans_before,
                "beans_after": beans_after,
                "recovered_mass_observed": False,
                "mass_sensor_available": False,
                "recovery_ratio": None,
            },
        )
    )

    # Stage 4: return to the rehearsal origin and execute four named
    # pick/lift/lower/release cycles. The official sink remains a reference;
    # object snapshots are the only admissible source for a later score.
    stage_id = 4
    start_step = recorder.sim_step
    before = task_object_snapshot(stage, object_paths)
    nav_sink = drive_base_to(
        recorder,
        robot,
        steering_ids,
        drive_ids,
        base_origin[:2],
        nominal_yaw,
        max_steps=ARGS.base_max_steps,
        max_speed=ARGS.base_max_speed,
        stage_id=stage_id,
        phase="stage4_base_return",
        intent="Return to the rehearsal origin for four cleanup cycles.",
    )
    gates = [nav_sink]
    object_cycles: list[dict[str, Any]] = []
    for object_name in ("plate", "cup", "bowl", "spoon"):
        close = command_gripper(
            recorder,
            robot,
            gripper_drivers,
            sides=("left",),
            target=PROFILE.keyboard_positions[1],
            steps=ARGS.gripper_steps,
            stage_id=stage_id,
            phase=f"stage4_{object_name}_close",
            intent=f"Close the left gripper for the {object_name} cleanup rehearsal.",
        )
        lift = move_tcp_relative(
            recorder,
            robot,
            ik,
            left_delta=(0.0, 0.0, 0.04),
            right_delta=(0.0, 0.0, 0.0),
            steps=ARGS.motion_steps,
            settle_steps=ARGS.settle_steps,
            stage_id=stage_id,
            phase=f"stage4_{object_name}_lift",
            intent=f"Lift the left TCP for the {object_name} cleanup rehearsal.",
        )
        lower = move_tcp_relative(
            recorder,
            robot,
            ik,
            left_delta=(0.0, 0.0, -0.04),
            right_delta=(0.0, 0.0, 0.0),
            steps=ARGS.motion_steps,
            settle_steps=ARGS.settle_steps,
            stage_id=stage_id,
            phase=f"stage4_{object_name}_lower",
            intent=f"Lower the left TCP at the sink for the {object_name} rehearsal.",
        )
        release = command_gripper(
            recorder,
            robot,
            gripper_drivers,
            sides=("left",),
            target=PROFILE.keyboard_positions[0],
            steps=ARGS.gripper_steps,
            stage_id=stage_id,
            phase=f"stage4_{object_name}_release",
            intent=f"Release the left gripper at the sink for the {object_name} rehearsal.",
        )
        cycle = {
            "object": object_name,
            "gates": [close, lift, lower, release],
            "motion_gate_passed": all(
                item["passed"] for item in (close, lift, lower, release)
            ),
            "object_grasp_observed": False,
            "object_in_sink_observed": False,
        }
        object_cycles.append(cycle)
        gates.extend((close, lift, lower, release))
    after = task_object_snapshot(stage, object_paths)
    stages.append(
        stage_record(
            stage_id,
            start_step=start_step,
            end_step=recorder.sim_step,
            gates=gates,
            before_objects=before,
            after_objects=after,
            additional_metrics={
                "official_sink_center": rounded(sink_center),
                "official_sink_reached": False,
                "object_cycles": object_cycles,
                "objects_in_sink_observed": None,
                "objects_required": 4,
            },
        )
    )
    return stages


def validate_arguments() -> None:
    if ARGS.trace_hz < 1:
        raise ValueError("--trace-hz must be positive")
    if ARGS.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    if ARGS.motion_steps < 1 or ARGS.gripper_steps < 1:
        raise ValueError("motion and gripper steps must be positive")
    if (
        not math.isfinite(ARGS.feed_hold_seconds)
        or ARGS.feed_hold_seconds < 3.0
    ):
        raise ValueError("--feed-hold-seconds must be at least 3.0")
    if ARGS.base_max_steps < 1:
        raise ValueError("--base-max-steps must be positive")
    if (
        not math.isfinite(ARGS.base_max_speed)
        or ARGS.base_max_speed <= 0.0
    ):
        raise ValueError("--base-max-speed must be finite and positive")


def main() -> None:
    validate_arguments()
    started_wall = time.time()
    output_dir = ARGS.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    room_path = Path(ARGS.room_usd).expanduser()
    robot_path = Path(ARGS.robot_usd).expanduser()
    if not room_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError(
            f"Missing room or robot asset: {room_path}, {robot_path}"
        )

    groups = core._load_joint_groups(
        Path(ARGS.franka_root).expanduser(),
        ARGS.embodiment,
        include_browser_commands=False,
    )
    ARGS.task = "task3"
    robot_position = room_scene.resolve_robot_position(ARGS)
    robot_yaw = room_scene.resolve_robot_yaw(ARGS)
    print("FOUR_STAGE_REHEARSAL_PHASE build_official_scene", flush=True)
    room_scene.build_stage(
        omni.kit.app.get_app(),
        room_path=room_path,
        robot_path=robot_path,
        task="task3",
        robot_position=robot_position,
        robot_rotation=room_scene.yaw_to_quat(robot_yaw),
        robot_yaw=robot_yaw,
        head_placement=ARGS.head_placement,
        dynamic_beans=True,
    )
    physics_scene_path = core._find_physics_scene_path() or "/physicsScene"
    world = World(
        physics_prim_path=physics_scene_path,
        stage_units_in_meters=1.0,
        physics_dt=1.0 / ARGS.physics_hz,
        rendering_dt=1.0 / ARGS.render_hz,
    )
    core.prepare_robot_prim(ROBOT_PRIM_PATH, ARGS)
    core._configure_drives(
        ROBOT_PRIM_PATH,
        lambda joint_name: get_profile_drive_gains(
            PROFILE.name, joint_name
        ),
    )
    root_path = core._find_articulation_root_path(ROBOT_PRIM_PATH)
    robot = SingleArticulation(
        prim_path=root_path,
        name="airsign_four_stage_rehearsal_robot",
    )
    world.scene.add(robot)
    world.reset()
    (
        _group_indices,
        _coupled_indices,
        steering_ids,
        drive_ids,
        _spine_keyboard_controller,
        _arm_keyboard_teleop,
    ) = core.setup_robot_control(robot, groups, ARGS)
    for _ in range(ARGS.settle_steps):
        world.step(render=ARGS.render)

    gripper_drivers = discover_gripper_drivers(robot)
    driver_names = {details[0] for details in gripper_drivers.values()}
    core._configure_drives(
        ROBOT_PRIM_PATH,
        lambda joint_name: (
            {
                "stiffness": 500.0,
                "damping": 50.0,
                "max_force": 200.0,
            }
            if joint_name in driver_names
            else None
        ),
    )
    ik = create_raw_dual_arm_lula(
        robot.dof_names,
        robot.get_joint_positions,
        project_root=TASK3_ROOT,
    )
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available")
    object_paths = {
        name: find_prim_path(stage, source_name)
        for name, source_name in TASK_OBJECT_NAMES.items()
    }
    head_path = find_prim_path(stage, "head")
    from grading import TASK3_BEAN_RECOVERY_REGION, TASK3_SINK_REGION

    recovery_center = np.asarray(
        (
            TASK3_BEAN_RECOVERY_REGION.center.x,
            TASK3_BEAN_RECOVERY_REGION.center.y,
            TASK3_BEAN_RECOVERY_REGION.center.z,
        ),
        dtype=np.float64,
    )
    sink_center = np.asarray(
        (
            0.5
            * (
                TASK3_SINK_REGION.bounds.x_min
                + TASK3_SINK_REGION.bounds.x_max
            ),
            0.5
            * (
                TASK3_SINK_REGION.bounds.y_min
                + TASK3_SINK_REGION.bounds.y_max
            ),
            TASK3_SINK_REGION.tabletop_z,
        ),
        dtype=np.float64,
    )
    recorder = TraceRecorder(
        world=world,
        robot=robot,
        ik=ik,
        stage=stage,
        object_paths=object_paths,
        gripper_drivers=gripper_drivers,
        physics_hz=float(ARGS.physics_hz),
        trace_hz=ARGS.trace_hz,
        render=ARGS.render,
    )
    recorder.set_phase(
        0,
        "initial_state",
        intent="Record the settled official Task 3 initial state.",
    )

    print("FOUR_STAGE_REHEARSAL_PHASE execute_stages_1_to_4", flush=True)
    stage_results = run_all_stages(
        recorder,
        robot,
        ik,
        steering_ids,
        drive_ids,
        gripper_drivers,
        stage,
        object_paths,
        head_path,
        recovery_center,
        sink_center,
    )
    recorder.set_phase(
        4,
        "final_state",
        intent="Record the final actuator and task-object state.",
    )

    controller_passed = all(
        result["motion_gate_passed"] for result in stage_results
    )
    script_path = Path(__file__).resolve()
    provenance = {
        "official_benchmark_commit": os.environ.get(
            "EBIM_BENCHMARK_COMMIT", git_revision(REPO_ROOT)
        ),
        "controller_sha256": sha256_file(script_path),
        "controller_path_in_runtime": str(script_path),
        "simulator": "Isaac Sim 5.1.0",
        "container_image": os.environ.get(
            "AIRSIGN_CAPTURE_CONTAINER_IMAGE",
            "isaac-sim-5.1.0:ebim2026",
        ),
        "container_image_id": os.environ.get(
            "AIRSIGN_CAPTURE_CONTAINER_IMAGE_ID", "unavailable"
        ),
        "python": platform.python_version(),
        "head_placement": ARGS.head_placement,
        "physics_hz": float(ARGS.physics_hz),
        "trace_hz": ARGS.trace_hz,
        "room_asset": str(room_path),
        "room_asset_sha256": sha256_file(room_path),
        "robot_asset": str(robot_path),
        "robot_asset_sha256": sha256_file(robot_path),
    }
    replay = {
        "schema_version": 1,
        "kind": "airsign_ebim_track3_four_stage_browser_replay",
        "classification": "measured_actuator_rehearsal",
        "coordinate_frame": "official_isaac_world_metres_z_up",
        "official_stage_completion_claimed": False,
        "official_score_claimed": False,
        "benchmark_score": None,
        "task_objects_teleported": False,
        "robot_links_teleported": False,
        "task_object_mutation_api_used": False,
        "dof_names": list(robot.dof_names),
        "task_object_prim_paths": object_paths,
        "official_regions": {
            "bean_recovery": {
                "center": rounded(recovery_center),
                "radius_metres": 0.2,
            },
            "sink": {
                "center": rounded(sink_center),
                "bounds": {
                    "x": [
                        TASK3_SINK_REGION.bounds.x_min,
                        TASK3_SINK_REGION.bounds.x_max,
                    ],
                    "y": [
                        TASK3_SINK_REGION.bounds.y_min,
                        TASK3_SINK_REGION.bounds.y_max,
                    ],
                    "tabletop_z": TASK3_SINK_REGION.tabletop_z,
                },
            },
        },
        "provenance": provenance,
        "events": recorder.events,
        "frames": recorder.frames,
    }
    metrics = {
        "schema_version": 1,
        "kind": "airsign_ebim_track3_four_stage_rehearsal_metrics",
        "classification": "participant_controller_rehearsal",
        "passed": controller_passed,
        "pass_definition": (
            "all actuator navigation, IK, gripper, and hold gates passed"
        ),
        "official_stage_completion_claimed": False,
        "official_score_claimed": False,
        "benchmark_score": None,
        "official_benchmark_max_score": 16,
        "benchmark_score_reason": (
            "Object transport, spoon bean occupancy, recovered mass, and "
            "sink-placement outcomes were not established by this run."
        ),
        "stages": stage_results,
        "trace": {
            "frame_count": len(recorder.frames),
            "event_count": len(recorder.events),
            "duration_seconds": round(
                recorder.sim_step / float(ARGS.physics_hz), 6
            ),
            "trace_hz": ARGS.trace_hz,
        },
        "provenance": provenance,
        "wall_duration_seconds": round(time.time() - started_wall, 3),
    }
    replay_path = output_dir / "replay_trace.json"
    metrics_path = output_dir / "metrics.json"
    replay_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "files": {
            replay_path.name: {
                "sha256": sha256_file(replay_path),
                "bytes": replay_path.stat().st_size,
            },
            metrics_path.name: {
                "sha256": sha256_file(metrics_path),
                "bytes": metrics_path.stat().st_size,
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "passed": controller_passed,
        "classification": metrics["classification"],
        "stages_recorded": [1, 2, 3, 4],
        "frame_count": len(recorder.frames),
        "duration_seconds": metrics["trace"]["duration_seconds"],
        "official_stage_completion_claimed": False,
        "official_score_claimed": False,
        "benchmark_score": None,
        "output_dir": str(output_dir),
        "files": manifest["files"],
    }
    print(
        "FOUR_STAGE_REHEARSAL_RESULT "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    if not controller_passed:
        raise RuntimeError(
            "One or more four-stage actuator rehearsal gates failed"
        )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        exit_code = 1
    finally:
        SIMULATION_APP.close()
    raise SystemExit(exit_code)
