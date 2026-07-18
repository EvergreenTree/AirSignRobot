#!/usr/bin/env python3
"""Physical, articulation-only Stage 1 development gates for EBiM Task 3.

This controller builds the official public Task 3 scene and moves only robot
articulation degrees of freedom.  It can exercise a cup grasp/lift/release
gate and a bimanual tray lift/transport gate.  Task-object poses are measured,
never commanded.

The public benchmark snapshot does not expose the organizer's randomized
Stage 1 target provider or a live competition scorer.  Consequently, even a
passing physical gate is development evidence, not an official stage
completion or score.
"""

from __future__ import annotations

import argparse
import ast
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
        "--gate",
        choices=("inspect", "cup", "tray-lift", "tray-transport", "all"),
        default="cup",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/EBiM_Challenge/airsign-stage1"),
    )
    parser.add_argument(
        "--targets-json",
        type=Path,
        default=None,
        help=(
            "Optional external target-provider record. It is provenance only "
            "until an organizer-authenticated provider contract is public."
        ),
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
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--motion-steps", type=int, default=180)
    parser.add_argument("--motion-settle-steps", type=int, default=60)
    parser.add_argument("--gripper-steps", type=int, default=180)
    parser.add_argument("--base-max-steps", type=int, default=7000)
    parser.add_argument("--base-min-speed", type=float, default=0.08)
    parser.add_argument("--base-max-speed", type=float, default=0.28)
    parser.add_argument("--base-max-accel", type=float, default=0.35)
    parser.add_argument("--transport-max-speed", type=float, default=0.14)
    parser.add_argument("--transport-max-accel", type=float, default=0.20)
    parser.add_argument("--gripper-max-force", type=float, default=30.0)
    parser.add_argument("--gripper-stiffness", type=float, default=350.0)
    parser.add_argument("--gripper-damping", type=float, default=35.0)
    parser.add_argument("--gripper-effort-abort", type=float, default=40.0)
    parser.add_argument("--arm-effort-abort", type=float, default=180.0)
    parser.add_argument("--base-stall-seconds", type=float, default=1.0)
    parser.add_argument("--base-stall-grace-seconds", type=float, default=1.0)
    parser.add_argument("--base-stall-distance", type=float, default=0.015)
    parser.add_argument("--cup-grasp-yaw-deg", type=float, default=90.0)
    parser.add_argument("--cup-pregrasp-clearance", type=float, default=0.12)
    parser.add_argument("--cup-grasp-z-offset", type=float, default=0.025)
    parser.add_argument("--cup-lift-height", type=float, default=0.16)
    parser.add_argument("--tray-pregrasp-clearance", type=float, default=0.12)
    parser.add_argument("--tray-grasp-z-offset", type=float, default=0.018)
    parser.add_argument("--tray-edge-overhang", type=float, default=0.015)
    parser.add_argument("--tray-lift-height", type=float, default=0.14)
    parser.add_argument("--dining-base-x", type=float, default=-5.05)
    parser.add_argument("--dining-base-y", type=float, default=1.25)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render simulator steps. Evidence is captured either way.",
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
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ROBOT_PRIM_PATH = "/World/Robot"
TASK_OBJECT_NAMES = {
    "tray": "simple_tray",
    "plate": "plate2",
    "cup": "cup",
    "bowl": "bowl2",
    "spoon": "spoon2",
}
RULEBOOK_STAGE1_OBJECTS = ("plate", "cup", "bowl", "spoon")
DINING_TABLE_AABB_Y = (1.540908, 2.340908)
NORTH_TRANSIT_Y = 2.85
NORTH_TRANSIT_CENTERLINE_CLEARANCE = (
    NORTH_TRANSIT_Y - DINING_TABLE_AABB_Y[1]
)
PLANAR_NORM_EPSILON = 1e-9

# These APIs can directly mutate object or link state.  The guard walks the
# controller's AST before scene construction and fails closed if any call is
# introduced.  The official scene builder remains the sole initialization
# boundary.
FORBIDDEN_ATTRIBUTE_CALLS = frozenset(
    {
        "set_world_pose",
        "set_local_pose",
        "set_default_state",
        "set_linear_velocity",
        "set_angular_velocity",
        "set_rigid_bodies_enabled_under",
        "set_rigid_bodies_kinematic_under",
        "translate_prim_preserving_rotation",
        "AddTranslateOp",
        "AddOrientOp",
        "AddTransformOp",
        "ClearXformOpOrder",
        "Set",
    }
)
FORBIDDEN_DIRECT_CALLS = frozenset(
    {"eval", "exec", "setattr", "delattr", "__import__"}
)


def rounded(values: Iterable[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def enforce_mutation_guard(script_path: Path) -> dict[str, Any]:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in FORBIDDEN_ATTRIBUTE_CALLS
        ):
            violations.append(
                {
                    "line": int(node.lineno),
                    "call": node.func.attr,
                    "kind": "forbidden_attribute_call",
                }
            )
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_DIRECT_CALLS
        ):
            violations.append(
                {
                    "line": int(node.lineno),
                    "call": node.func.id,
                    "kind": "forbidden_dynamic_call",
                }
            )
    result = {
        "passed": not violations,
        "controller_sha256": sha256_file(script_path),
        "ast_call_count": sum(
            isinstance(node, ast.Call) for node in ast.walk(tree)
        ),
        "forbidden_attribute_calls": sorted(FORBIDDEN_ATTRIBUTE_CALLS),
        "forbidden_direct_calls": sorted(FORBIDDEN_DIRECT_CALLS),
        "violations": violations,
        "scope": "participant/stage1_table_setup.py",
        "scene_initialization_boundary": (
            "scene_robot_room_keyboard.build_stage before controller motion"
        ),
    }
    if violations:
        raise RuntimeError(
            "Controller mutation guard rejected source: "
            + json.dumps(violations, sort_keys=True)
        )
    return result


def load_target_provider(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "missing",
            "path": None,
            "sha256": None,
            "schema_valid": False,
            "organizer_authenticated": False,
            "used_for_official_scoring": False,
            "reason": (
                "The pinned public benchmark does not expose the randomized "
                "Stage 1 target-provider contract."
            ),
            "assignments": None,
        }
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--targets-json root must be an object")
    assignments = payload.get("assignments")
    valid = payload.get("stage") == 1 and isinstance(assignments, dict)
    normalized: dict[str, list[float]] = {}
    if valid:
        for name, value in assignments.items():
            if name not in RULEBOOK_STAGE1_OBJECTS:
                valid = False
                break
            if (
                not isinstance(value, list)
                or len(value) != 3
                or not all(
                    isinstance(item, (int, float))
                    and math.isfinite(float(item))
                    for item in value
                )
            ):
                valid = False
                break
            normalized[name] = [float(item) for item in value]
    return {
        "status": "external_unverified",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema_valid": bool(valid),
        "organizer_authenticated": False,
        "used_for_official_scoring": False,
        "reason": (
            "External targets are retained as provenance but cannot be "
            "authenticated against a public organizer provider contract."
        ),
        "assignments": normalized if valid else None,
    }


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


def quaternion_normalized(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("Quaternion must be finite and non-zero")
    return result / norm


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = quaternion_normalized(left)
    rw, rx, ry, rz = quaternion_normalized(right)
    return quaternion_normalized(
        np.asarray(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dtype=np.float64,
        )
    )


def top_down_orientation(yaw_degrees: float) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    yaw_rotation = np.asarray(
        (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        dtype=np.float64,
    )
    tcp_z_down = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float64)
    return quaternion_multiply(yaw_rotation, tcp_z_down)


def quaternion_slerp(
    start: np.ndarray,
    target: np.ndarray,
    fraction: float,
) -> np.ndarray:
    first = quaternion_normalized(start)
    second = quaternion_normalized(target)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return quaternion_normalized(first + fraction * (second - first))
    theta = math.acos(dot)
    sine = math.sin(theta)
    return quaternion_normalized(
        math.sin((1.0 - fraction) * theta) / sine * first
        + math.sin(fraction * theta) / sine * second
    )


def quaternion_error_degrees(
    measured: np.ndarray,
    target: np.ndarray,
) -> float:
    dot = abs(
        float(
            np.dot(
                quaternion_normalized(measured),
                quaternion_normalized(target),
            )
        )
    )
    return math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


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


def hold_joint_positions(
    robot: SingleArticulation,
    indices: np.ndarray,
) -> None:
    """Fail-closed articulation stop at the currently measured positions."""
    measured = np.asarray(
        robot.get_joint_positions(), dtype=np.float64
    )[indices]
    robot.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=np.asarray(measured, dtype=np.float32),
            joint_indices=np.asarray(indices, dtype=np.int64),
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


def prim_record(stage: Any, path: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {path}")
    purposes = (
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
    )
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    transform = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return {
        "prim_path": path,
        "pose": {
            "position": rounded(translation),
            "orientation_wxyz": rounded(
                (
                    rotation.GetReal(),
                    imaginary[0],
                    imaginary[1],
                    imaginary[2],
                )
            ),
        },
        "aabb": {
            "min": rounded(bounds.GetMin()),
            "max": rounded(bounds.GetMax()),
        },
    }


def task_object_snapshot(
    stage: Any,
    object_paths: dict[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        name: prim_record(stage, path)
        for name, path in object_paths.items()
    }


def rigid_body_record(stage: Any, path: str) -> dict[str, Any]:
    root = stage.GetPrimAtPath(path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Invalid prim path: {path}")
    records: list[dict[str, Any]] = []
    candidates = [root]
    candidates.extend(Usd.PrimRange(root))
    seen: set[str] = set()
    for prim in candidates:
        prim_path = str(prim.GetPath())
        if prim_path in seen:
            continue
        seen.add(prim_path)
        api = UsdPhysics.RigidBodyAPI(prim)
        if not api:
            continue
        enabled_attr = api.GetRigidBodyEnabledAttr()
        kinematic_attr = api.GetKinematicEnabledAttr()
        enabled = (
            bool(enabled_attr.Get())
            if enabled_attr and enabled_attr.HasAuthoredValueOpinion()
            else True
        )
        kinematic = (
            bool(kinematic_attr.Get())
            if kinematic_attr and kinematic_attr.HasAuthoredValueOpinion()
            else False
        )
        records.append(
            {
                "prim_path": prim_path,
                "rigid_body_enabled": enabled,
                "kinematic_enabled": kinematic,
            }
        )
    return {
        "root_path": path,
        "rigid_body_count": len(records),
        "all_enabled": bool(records)
        and all(record["rigid_body_enabled"] for record in records),
        "all_dynamic": bool(records)
        and all(not record["kinematic_enabled"] for record in records),
        "bodies": records,
    }


def environment_top_level_inventory(stage: Any) -> list[dict[str, Any]]:
    """Return read-only top-level room bounds for waypoint diagnosis."""
    root = stage.GetPrimAtPath("/World/Environment/RobotRoom/Asset")
    if not root or not root.IsValid():
        return []
    records: list[dict[str, Any]] = []
    candidates = list(root.GetChildren())
    nested_root = stage.GetPrimAtPath(
        "/World/Environment/RobotRoom/Asset/root"
    )
    if nested_root and nested_root.IsValid():
        candidates.extend(nested_root.GetChildren())
    for prim in candidates:
        try:
            record = prim_record(stage, str(prim.GetPath()))
        except Exception:  # noqa: BLE001
            continue
        minimum = np.asarray(record["aabb"]["min"], dtype=np.float64)
        maximum = np.asarray(record["aabb"]["max"], dtype=np.float64)
        if not np.all(np.isfinite(minimum)) or not np.all(
            np.isfinite(maximum)
        ):
            continue
        records.append(
            {
                "name": prim.GetName(),
                "prim_path": record["prim_path"],
                "position": record["pose"]["position"],
                "aabb": record["aabb"],
            }
        )
    return records


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


def measured_efforts(robot: SingleArticulation) -> np.ndarray | None:
    try:
        values = robot.get_measured_joint_efforts()
    except Exception:  # noqa: BLE001
        return None
    if values is None:
        return None
    result = np.asarray(values, dtype=np.float64)
    return result if result.ndim == 1 else None


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
        self.current_phase = "initializing"

    def set_phase(self, phase: str, *, intent: str) -> None:
        self.current_phase = phase
        print(
            "STAGE1_TABLE_SETUP_PHASE "
            f"phase={phase} step={self.sim_step}",
            flush=True,
        )
        self.events.append(
            {
                "simulation_step": self.sim_step,
                "time_seconds": round(
                    self.sim_step / self.physics_hz, 6
                ),
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
        efforts = measured_efforts(self.robot)
        self.frames.append(
            {
                "simulation_step": self.sim_step,
                "time_seconds": round(
                    self.sim_step / self.physics_hz, 6
                ),
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
                "gripper_driver_positions": {
                    side: round(float(joint_positions[index]), 6)
                    for side, (_name, index) in self.gripper_drivers.items()
                },
                "gripper_driver_efforts": {
                    side: (
                        None
                        if (
                            efforts is None
                            or not math.isfinite(float(efforts[index]))
                        )
                        else round(float(efforts[index]), 6)
                    )
                    for side, (_name, index) in self.gripper_drivers.items()
                },
                "task_objects": {
                    name: record["pose"]
                    for name, record in task_object_snapshot(
                        self.stage, self.object_paths
                    ).items()
                },
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
    phase: str,
    intent: str,
) -> dict[str, Any]:
    recorder.set_phase(phase, intent=intent)
    indices = np.asarray(
        [drivers[side][1] for side in sides], dtype=np.int64
    )
    initial = np.asarray(robot.get_joint_positions())[indices]
    controller = robot.get_articulation_controller()
    peak_effort = {side: 0.0 for side in sides}
    samples = 0
    aborted = False
    abort_reason: str | None = None
    commanded_steps = 0
    telemetry_initially_available = False
    telemetry_nonfinite = False
    emergency_open_attempted = False
    emergency_open_steps = 0
    start_step = recorder.sim_step
    initial_efforts = measured_efforts(robot)
    if initial_efforts is not None:
        selected = initial_efforts[indices]
        telemetry_initially_available = bool(np.all(np.isfinite(selected)))
        telemetry_nonfinite = not telemetry_initially_available
    if not telemetry_initially_available:
        aborted = True
        abort_reason = (
            "finite gripper effort telemetry unavailable before command"
        )
    else:
        for step in range(1, steps + 1):
            commanded_steps = step
            fraction = step / steps
            targets = initial + fraction * (target - initial)
            controller.apply_action(
                ArticulationAction(
                    joint_positions=np.asarray(
                        targets, dtype=np.float32
                    ),
                    joint_indices=indices,
                )
            )
            recorder.step()
            efforts = measured_efforts(robot)
            if efforts is None:
                aborted = True
                abort_reason = (
                    "gripper effort telemetry disappeared during command"
                )
                break
            selected = efforts[indices]
            if not np.all(np.isfinite(selected)):
                telemetry_nonfinite = True
                aborted = True
                abort_reason = (
                    "non-finite gripper effort telemetry during command"
                )
                break
            samples += 1
            for side, value in zip(sides, selected, strict=True):
                peak_effort[side] = max(
                    peak_effort[side], abs(float(value))
                )
            if max(peak_effort.values()) > ARGS.gripper_effort_abort:
                aborted = True
                abort_reason = (
                    "measured gripper-driver effort exceeded "
                    "--gripper-effort-abort"
                )
                break
    if not aborted and samples < 1:
        aborted = True
        abort_reason = "no finite gripper effort sample was recorded"
    if aborted:
        closing_command = bool(
            target > float(np.mean(initial)) + 1e-6
        )
        if closing_command and commanded_steps > 0:
            # Close commands occur only while the object is supported on its
            # table/tray.  Back away from a high-force partial close before
            # terminating; lift/lower aborts use a different hold policy.
            emergency_open_attempted = True
            current = np.asarray(robot.get_joint_positions())[indices]
            emergency_open_steps = min(120, steps)
            for recovery_step in range(1, emergency_open_steps + 1):
                fraction = recovery_step / emergency_open_steps
                recovery_target = current + fraction * (
                    PROFILE.keyboard_positions[0] - current
                )
                controller.apply_action(
                    ArticulationAction(
                        joint_positions=np.asarray(
                            recovery_target, dtype=np.float32
                        ),
                        joint_indices=indices,
                    )
                )
                recorder.step()
        else:
            hold_joint_positions(robot, indices)
    measured = np.asarray(robot.get_joint_positions())[indices]
    finite = bool(np.all(np.isfinite(measured)))
    return {
        "phase": phase,
        "start_step": start_step,
        "end_step": recorder.sim_step,
        "commanded_steps": commanded_steps,
        "requested_steps": steps,
        "sides": list(sides),
        "target_position": round(float(target), 6),
        "initial_positions": {
            side: round(float(value), 6)
            for side, value in zip(sides, initial, strict=True)
        },
        "measured_positions": {
            side: round(float(value), 6)
            for side, value in zip(sides, measured, strict=True)
        },
        "peak_abs_measured_effort": {
            side: (
                round(value, 6) if samples else None
            )
            for side, value in peak_effort.items()
        },
        "effort_sample_count": samples,
        "effort_telemetry_initially_available": (
            telemetry_initially_available
        ),
        "effort_telemetry_nonfinite": telemetry_nonfinite,
        "configured_drive_max_force": ARGS.gripper_max_force,
        "effort_abort_threshold": ARGS.gripper_effort_abort,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "emergency_open_attempted": emergency_open_attempted,
        "emergency_open_steps": emergency_open_steps,
        "abort_recovery_policy": (
            "ramp_open_while_object_supported"
            if emergency_open_attempted
            else "hold_current_articulation"
        ),
        "passed": bool(finite and samples > 0 and not aborted),
        "pass_definition": (
            "finite measured gripper state after ramped articulation command "
            "without exceeding the participant safety-effort abort"
        ),
        "safety_limitation": (
            "Joint effort is not equivalent to task-object contact force; "
            "the organizer contact-force contract is unavailable."
        ),
    }


def move_tcp_pose(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
    *,
    left_target: np.ndarray | None,
    right_target: np.ndarray | None,
    left_orientation: np.ndarray | None,
    right_orientation: np.ndarray | None,
    steps: int,
    settle_steps: int,
    phase: str,
    intent: str,
    position_tolerance: float = 0.035,
    orientation_tolerance_degrees: float = 8.0,
) -> dict[str, Any]:
    recorder.set_phase(phase, intent=intent)
    base_position, base_orientation = world_pose(robot)
    initial_left, initial_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    target_left_position = np.asarray(
        initial_left[0] if left_target is None else left_target,
        dtype=np.float64,
    )
    target_right_position = np.asarray(
        initial_right[0] if right_target is None else right_target,
        dtype=np.float64,
    )
    target_left_orientation = quaternion_normalized(
        initial_left[1]
        if left_orientation is None
        else left_orientation
    )
    target_right_orientation = quaternion_normalized(
        initial_right[1]
        if right_orientation is None
        else right_orientation
    )
    start_step = recorder.sim_step
    left_successes = 0
    right_successes = 0
    attempted_motion_steps = 0
    attempted_settle_steps = 0
    aborted = False
    abort_reason: str | None = None
    names = list(robot.dof_names)
    arm_indices = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if (
                name.startswith("left_fr3v2_joint")
                or name.startswith("right_fr3v2_joint")
                or name == "franka_spine_vertical_joint"
            )
        ],
        dtype=np.int64,
    )
    peak_arm_effort = 0.0
    arm_effort_samples = 0
    effort_telemetry_initially_available = False
    effort_telemetry_nonfinite = False
    initial_efforts = measured_efforts(robot)
    if arm_indices.size and initial_efforts is not None:
        selected = initial_efforts[arm_indices]
        effort_telemetry_initially_available = bool(
            np.all(np.isfinite(selected))
        )
        effort_telemetry_nonfinite = (
            not effort_telemetry_initially_available
        )
    if not effort_telemetry_initially_available:
        aborted = True
        abort_reason = "finite arm effort telemetry unavailable before motion"
    for step in range(1, steps + 1):
        if aborted:
            break
        attempted_motion_steps = step
        fraction = step / steps
        desired_left_position = (
            initial_left[0]
            + fraction * (target_left_position - initial_left[0])
        )
        desired_right_position = (
            initial_right[0]
            + fraction * (target_right_position - initial_right[0])
        )
        desired_left_orientation = quaternion_slerp(
            initial_left[1], target_left_orientation, fraction
        )
        desired_right_orientation = quaternion_slerp(
            initial_right[1], target_right_orientation, fraction
        )
        base_position, base_orientation = world_pose(robot)
        result = ik.solve(
            desired_left_position,
            desired_right_position,
            desired_left_orientation,
            desired_right_orientation,
            spine_position=spine_position(robot),
            base_position=base_position,
            base_orientation_wxyz=base_orientation,
        )
        if not (result.left_succeeded and result.right_succeeded):
            aborted = True
            abort_reason = "raw Lula IK failed; no failed solution was applied"
            break
        left_successes += int(result.left_succeeded)
        right_successes += int(result.right_succeeded)
        apply_targets(robot, result.combined)
        recorder.step()
        efforts = measured_efforts(robot)
        if efforts is None:
            aborted = True
            abort_reason = "arm effort telemetry disappeared during motion"
            break
        selected = efforts[arm_indices]
        if not np.all(np.isfinite(selected)):
            effort_telemetry_nonfinite = True
            aborted = True
            abort_reason = "non-finite arm effort telemetry during motion"
            break
        arm_effort_samples += 1
        peak_arm_effort = max(
            peak_arm_effort,
            float(np.max(np.abs(selected))),
        )
        if peak_arm_effort > ARGS.arm_effort_abort:
            aborted = True
            abort_reason = (
                "measured arm-joint effort exceeded --arm-effort-abort"
            )
            break
    for settle_step in range(1, settle_steps + 1):
        if aborted:
            break
        attempted_settle_steps = settle_step
        base_position, base_orientation = world_pose(robot)
        result = ik.solve(
            target_left_position,
            target_right_position,
            target_left_orientation,
            target_right_orientation,
            spine_position=spine_position(robot),
            base_position=base_position,
            base_orientation_wxyz=base_orientation,
        )
        if not (result.left_succeeded and result.right_succeeded):
            aborted = True
            abort_reason = (
                "raw Lula IK failed during settle; no failed solution was "
                "applied"
            )
            break
        apply_targets(robot, result.combined)
        recorder.step()
        efforts = measured_efforts(robot)
        if efforts is None:
            aborted = True
            abort_reason = "arm effort telemetry disappeared during settle"
            break
        selected = efforts[arm_indices]
        if not np.all(np.isfinite(selected)):
            effort_telemetry_nonfinite = True
            aborted = True
            abort_reason = "non-finite arm effort telemetry during settle"
            break
        arm_effort_samples += 1
        peak_arm_effort = max(
            peak_arm_effort,
            float(np.max(np.abs(selected))),
        )
        if peak_arm_effort > ARGS.arm_effort_abort:
            aborted = True
            abort_reason = (
                "measured arm-joint effort exceeded --arm-effort-abort"
            )
            break
    if not aborted and arm_effort_samples < 1:
        aborted = True
        abort_reason = "no finite arm effort sample was recorded"
    if aborted and arm_indices.size:
        hold_joint_positions(robot, arm_indices)
    base_position, base_orientation = world_pose(robot)
    final_left, final_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    left_position_error = float(
        np.linalg.norm(final_left[0] - target_left_position)
    )
    right_position_error = float(
        np.linalg.norm(final_right[0] - target_right_position)
    )
    left_orientation_error = quaternion_error_degrees(
        final_left[1], target_left_orientation
    )
    right_orientation_error = quaternion_error_degrees(
        final_right[1], target_right_orientation
    )
    passed = bool(
        not aborted
        and left_successes == steps
        and right_successes == steps
        and left_position_error <= position_tolerance
        and right_position_error <= position_tolerance
        and left_orientation_error <= orientation_tolerance_degrees
        and right_orientation_error <= orientation_tolerance_degrees
    )
    return {
        "phase": phase,
        "start_step": start_step,
        "end_step": recorder.sim_step,
        "motion_steps": steps,
        "attempted_motion_steps": attempted_motion_steps,
        "requested_settle_steps": settle_steps,
        "attempted_settle_steps": attempted_settle_steps,
        "left_solve_successes": left_successes,
        "right_solve_successes": right_successes,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "peak_abs_measured_arm_effort": round(peak_arm_effort, 6),
        "arm_effort_sample_count": arm_effort_samples,
        "effort_telemetry_initially_available": (
            effort_telemetry_initially_available
        ),
        "effort_telemetry_nonfinite": effort_telemetry_nonfinite,
        "arm_effort_abort_threshold": ARGS.arm_effort_abort,
        "abort_recovery_policy": (
            "hold_current_arm_articulation_no_automatic_release"
        ),
        "target": {
            "left": {
                "position": rounded(target_left_position),
                "orientation_wxyz": rounded(target_left_orientation),
            },
            "right": {
                "position": rounded(target_right_position),
                "orientation_wxyz": rounded(target_right_orientation),
            },
        },
        "final": {
            "left": {
                "position": rounded(final_left[0]),
                "orientation_wxyz": rounded(final_left[1]),
            },
            "right": {
                "position": rounded(final_right[0]),
                "orientation_wxyz": rounded(final_right[1]),
            },
        },
        "position_errors_metres": {
            "left": round(left_position_error, 6),
            "right": round(right_position_error, 6),
        },
        "orientation_errors_degrees": {
            "left": round(left_orientation_error, 6),
            "right": round(right_orientation_error, 6),
        },
        "position_tolerance_metres": position_tolerance,
        "orientation_tolerance_degrees": orientation_tolerance_degrees,
        "passed": passed,
        "pass_definition": (
            "all raw-Lula solves succeeded and measured TCP position and "
            "orientation errors were within tolerance without a safety abort"
        ),
        "safety_limitation": (
            "Joint effort is a participant-side abort signal, not an "
            "organizer task-object contact-force measurement."
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
    max_accel: float,
    phase: str,
    intent: str,
) -> dict[str, Any]:
    recorder.set_phase(phase, intent=intent)
    controller = robot.get_articulation_controller()
    position_tolerance = 0.04
    yaw_tolerance = math.radians(3.0)
    dt = 1.0 / recorder.physics_hz
    command = np.zeros(3, dtype=np.float64)
    settled_steps = 0
    start_step = recorder.sim_step
    path_length = 0.0
    previous_position, _ = world_pose(robot)
    steps = 0
    aborted = False
    abort_reason: str | None = None
    drive_indices = np.asarray(drive_ids, dtype=np.int64)
    stall_window_steps = max(
        2, int(round(ARGS.base_stall_seconds * recorder.physics_hz))
    )
    stall_grace_steps = int(
        round(ARGS.base_stall_grace_seconds * recorder.physics_hz)
    )
    position_history: list[np.ndarray] = []
    minimum_speed_floor_engagement_steps = 0
    minimum_speed_floor_first_step: int | None = None
    minimum_speed_floor_last_step: int | None = None
    try:
        for steps in range(1, max_steps + 1):
            position, orientation = world_pose(robot)
            path_length += float(
                np.linalg.norm(position[:2] - previous_position[:2])
            )
            previous_position = position
            position_history.append(position[:2].copy())
            if len(position_history) > stall_window_steps:
                position_history.pop(0)
            yaw = yaw_from_wxyz(orientation)
            world_error = (
                np.asarray(target_xy, dtype=np.float64) - position[:2]
            )
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            body_error = np.asarray(
                (
                    cos_yaw * world_error[0]
                    + sin_yaw * world_error[1],
                    -sin_yaw * world_error[0]
                    + cos_yaw * world_error[1],
                ),
                dtype=np.float64,
            )
            yaw_error = wrap_to_pi(target_yaw - yaw)
            distance = float(np.linalg.norm(world_error))
            if (
                distance <= position_tolerance
                and abs(yaw_error) <= yaw_tolerance
            ):
                desired = np.zeros(3, dtype=np.float64)
                if float(np.linalg.norm(command)) <= 0.015:
                    settled_steps += 1
            else:
                settled_steps = 0
                planar = np.zeros(2, dtype=np.float64)
                if distance > position_tolerance:
                    planar = 1.2 * body_error
                    planar_norm = float(np.linalg.norm(planar))
                    if not math.isfinite(planar_norm):
                        aborted = True
                        abort_reason = (
                            "non-finite planar base command; safety abort"
                        )
                        break
                    if planar_norm <= PLANAR_NORM_EPSILON:
                        planar.fill(0.0)
                    elif planar_norm > max_speed:
                        planar *= max_speed / planar_norm
                    elif (
                        PLANAR_NORM_EPSILON
                        < planar_norm
                        < ARGS.base_min_speed
                    ):
                        planar *= ARGS.base_min_speed / planar_norm
                        minimum_speed_floor_engagement_steps += 1
                        minimum_speed_floor_last_step = steps
                        if minimum_speed_floor_first_step is None:
                            minimum_speed_floor_first_step = steps
                desired = np.asarray(
                    (
                        planar[0],
                        planar[1],
                        float(
                            np.clip(1.6 * yaw_error, -0.35, 0.35)
                        ),
                    ),
                    dtype=np.float64,
                )
            delta = np.clip(
                desired - command,
                -max_accel * dt,
                max_accel * dt,
            )
            command += delta
            steering_targets, drive_targets = core._compute_drive_targets(
                robot.get_joint_positions(),
                steering_ids,
                float(command[0]),
                float(command[1]),
                float(command[2]),
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
                    joint_indices=drive_indices,
                )
            )
            recorder.step()
            if (
                steps >= stall_grace_steps + stall_window_steps
                and len(position_history) == stall_window_steps
                and float(np.linalg.norm(command[:2]))
                >= ARGS.base_min_speed
                and float(
                    np.linalg.norm(
                        position_history[-1] - position_history[0]
                    )
                )
                < ARGS.base_stall_distance
            ):
                aborted = True
                abort_reason = (
                    "base motion stalled under command; treated as a "
                    "collision/obstruction safety abort"
                )
                break
            if settled_steps >= 30:
                break
    finally:
        controller.apply_action(
            ArticulationAction(
                joint_velocities=np.zeros(
                    len(drive_ids), dtype=np.float32
                ),
                joint_indices=drive_indices,
            )
        )
    for _ in range(30):
        recorder.step()
    position, orientation = world_pose(robot)
    final_yaw = yaw_from_wxyz(orientation)
    position_error = float(np.linalg.norm(target_xy - position[:2]))
    yaw_error = abs(wrap_to_pi(target_yaw - final_yaw))
    return {
        "phase": phase,
        "start_step": start_step,
        "end_step": recorder.sim_step,
        "steps": steps,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "target_xy": rounded(target_xy),
        "final_xy": rounded(position[:2]),
        "position_error_metres": round(position_error, 6),
        "yaw_error_degrees": round(math.degrees(yaw_error), 6),
        "path_length_metres": round(path_length, 6),
        "max_speed_metres_per_second": max_speed,
        "max_command_accel": max_accel,
        "minimum_planar_command_metres_per_second": (
            ARGS.base_min_speed
        ),
        "minimum_planar_command_floor_engaged": bool(
            minimum_speed_floor_engagement_steps
        ),
        "minimum_planar_command_floor_engagement_steps": (
            minimum_speed_floor_engagement_steps
        ),
        "minimum_planar_command_floor_first_step": (
            minimum_speed_floor_first_step
        ),
        "minimum_planar_command_floor_last_step": (
            minimum_speed_floor_last_step
        ),
        "minimum_planar_command_floor_scope": (
            "planar position error outside the 0.04m final tolerance only"
        ),
        "stall_window_seconds": ARGS.base_stall_seconds,
        "stall_grace_seconds": ARGS.base_stall_grace_seconds,
        "stall_distance_threshold_metres": ARGS.base_stall_distance,
        "position_tolerance_metres": position_tolerance,
        "yaw_tolerance_degrees": math.degrees(yaw_tolerance),
        "passed": bool(
            not aborted
            and position_error <= position_tolerance
            and yaw_error <= yaw_tolerance
        ),
        "pass_definition": (
            "acceleration-limited closed-loop base pose within tolerance "
            "without stall/collision safety abort"
        ),
    }


def settle(
    recorder: TraceRecorder,
    steps: int,
    *,
    phase: str,
    intent: str,
) -> None:
    recorder.set_phase(phase, intent=intent)
    for _ in range(steps):
        recorder.step()


def object_position(
    snapshot: dict[str, dict[str, Any]],
    name: str,
) -> np.ndarray:
    return np.asarray(snapshot[name]["pose"]["position"], dtype=np.float64)


def object_aabb(
    snapshot: dict[str, dict[str, Any]],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(snapshot[name]["aabb"]["min"], dtype=np.float64),
        np.asarray(snapshot[name]["aabb"]["max"], dtype=np.float64),
    )


def fail_closed_result(
    *,
    gate_name: str,
    gates: list[dict[str, Any]],
    before: dict[str, dict[str, Any]],
    recorder: TraceRecorder,
    reason: str,
) -> dict[str, Any]:
    """Finish a development gate immediately after a failed prerequisite."""
    after = task_object_snapshot(recorder.stage, recorder.object_paths)
    return {
        "gate": gate_name,
        "classification": "physical_development_gate",
        "passed": False,
        "fail_closed": True,
        "failure_reason": reason,
        "recovery_policy": (
            "base drive commanded zero; arm motion aborts hold current "
            "articulation with no automatic release of a possibly suspended "
            "object; any supported-object close-effort recovery is recorded "
            "on its gripper gate"
        ),
        "gates": gates,
        "task_objects_before": before,
        "task_objects_after": after,
        "official_stage_complete": False,
        "official_score": None,
    }


def run_cup_gate(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
    steering_ids: list[int],
    drive_ids: list[int],
    drivers: dict[str, tuple[str, int]],
) -> dict[str, Any]:
    before = task_object_snapshot(recorder.stage, recorder.object_paths)
    cup_start = object_position(before, "cup")
    _cup_min, cup_max = object_aabb(before, "cup")
    base_start, base_orientation = world_pose(robot)
    left_start, _right_start = ik.current_end_effector_poses(
        base_start,
        base_orientation,
        spine_position(robot),
    )
    tcp_offset = np.asarray(left_start[0][:2] - base_start[:2])
    staging_xy = cup_start[:2] - tcp_offset
    nominal_yaw = yaw_from_wxyz(base_orientation)
    grasp_orientation = top_down_orientation(ARGS.cup_grasp_yaw_deg)
    pregrasp = np.asarray(
        (
            cup_start[0],
            cup_start[1],
            cup_max[2] + ARGS.cup_pregrasp_clearance,
        ),
        dtype=np.float64,
    )
    grasp = np.asarray(
        (
            cup_start[0],
            cup_start[1],
            cup_start[2] + ARGS.cup_grasp_z_offset,
        ),
        dtype=np.float64,
    )
    gates: list[dict[str, Any]] = []
    route_waypoints = (
        (
            "cup_route_north_clearance",
            np.asarray((base_start[0], NORTH_TRANSIT_Y), dtype=np.float64),
            "Move north into the measured table-to-wall clearance strip.",
        ),
        (
            "cup_route_clear_table_north",
            np.asarray((-0.60, NORTH_TRANSIT_Y), dtype=np.float64),
            "Traverse north of the dining table toward its east corridor.",
        ),
        (
            "cup_route_descend_east_corridor",
            np.asarray((-0.60, 0.72), dtype=np.float64),
            "Move south through the measured table-to-wall corridor.",
        ),
        (
            "cup_route_enter_doorway",
            np.asarray((-4.15, 0.72), dtype=np.float64),
            "Align with the measured 1.2m doorway opening.",
        ),
        (
            "cup_route_cross_doorway",
            np.asarray((-4.15, -0.72), dtype=np.float64),
            "Cross the doorway before shifting toward the cup.",
        ),
        (
            "cup_base_staging",
            staging_xy,
            "Drive to the cup-aligned kitchen stance.",
        ),
    )
    for route_phase, route_target, route_intent in route_waypoints:
        gate = drive_base_to(
            recorder,
            robot,
            steering_ids,
            drive_ids,
            route_target,
            nominal_yaw,
            max_steps=ARGS.base_max_steps,
            max_speed=ARGS.base_max_speed,
            max_accel=ARGS.base_max_accel,
            phase=route_phase,
            intent=route_intent,
        )
        gate["route_source"] = (
            "read-only bounds from the pinned official room asset"
        )
        gate["route_geometry"] = {
            "dining_table_aabb_y_metres": rounded(
                DINING_TABLE_AABB_Y
            ),
            "north_transit_centerline_y_metres": NORTH_TRANSIT_Y,
            "north_transit_centerline_clearance_from_table_metres": round(
                NORTH_TRANSIT_CENTERLINE_CLEARANCE, 6
            ),
        }
        gates.append(gate)
        if not gate["passed"]:
            return fail_closed_result(
                gate_name="cup_grasp_lift_release",
                gates=gates,
                before=before,
                recorder=recorder,
                reason=f"Navigation prerequisite failed at {route_phase}.",
            )
    gates.append(
        command_gripper(
            recorder,
            robot,
            drivers,
            sides=("left",),
            target=PROFILE.keyboard_positions[0],
            steps=ARGS.gripper_steps,
            phase="cup_open",
            intent="Open the left Robotiq before entering the grasp corridor.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Left gripper open safety gate failed.",
        )
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=pregrasp,
            right_target=None,
            left_orientation=grasp_orientation,
            right_orientation=None,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="cup_pregrasp",
            intent=(
                "Move above the cup with TCP Z down and finger opening "
                "axis spanning the cup diameter."
            ),
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Cup pregrasp IK or effort safety gate failed.",
        )
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=grasp,
            right_target=None,
            left_orientation=grasp_orientation,
            right_orientation=None,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="cup_vertical_descent",
            intent=(
                "Descend vertically from the clearance waypoint to the "
                "measured cup centreline."
            ),
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Cup vertical-descent IK or effort safety gate failed.",
        )
    gates.append(
        command_gripper(
            recorder,
            robot,
            drivers,
            sides=("left",),
            target=PROFILE.keyboard_positions[1],
            steps=ARGS.gripper_steps,
            phase="cup_close",
            intent="Close the left Robotiq around the dynamic cup.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Cup close effort safety gate failed.",
        )
    lift_target = grasp + np.asarray((0.0, 0.0, ARGS.cup_lift_height))
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=lift_target,
            right_target=None,
            left_orientation=grasp_orientation,
            right_orientation=None,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="cup_lift",
            intent="Lift through articulation motion only.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Cup lift IK or effort safety gate failed.",
        )
    after_lift = task_object_snapshot(
        recorder.stage, recorder.object_paths
    )
    cup_lifted = object_position(after_lift, "cup")
    lift_delta = cup_lifted - cup_start
    non_cup_displacements = {
        name: float(
            np.linalg.norm(
                object_position(after_lift, name)
                - object_position(before, name)
            )
        )
        for name in before
        if name != "cup"
    }
    lift_evidence = {
        "phase": "cup_physical_lift_outcome",
        "cup_start_position": rounded(cup_start),
        "cup_after_lift_position": rounded(cup_lifted),
        "cup_displacement": rounded(lift_delta),
        "cup_vertical_lift_metres": round(float(lift_delta[2]), 6),
        "cup_horizontal_drift_metres": round(
            float(np.linalg.norm(lift_delta[:2])), 6
        ),
        "maximum_non_cup_displacement_metres": round(
            max(non_cup_displacements.values()), 6
        ),
        "non_cup_displacements_metres": {
            name: round(value, 6)
            for name, value in non_cup_displacements.items()
        },
        "passed": bool(
            lift_delta[2] >= 0.08
            and np.linalg.norm(lift_delta[:2]) <= 0.12
            and max(non_cup_displacements.values()) <= 0.08
        ),
        "pass_definition": (
            "dynamic cup rose at least 8cm without excessive horizontal "
            "or bystander-object displacement"
        ),
    }
    gates.append(lift_evidence)
    if not lift_evidence["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Dynamic cup lift outcome predicate failed.",
        )
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=grasp,
            right_target=None,
            left_orientation=grasp_orientation,
            right_orientation=None,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="cup_lower",
            intent="Lower the grasped cup to its original table region.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Cup lower IK or effort safety gate failed.",
        )
    gates.append(
        command_gripper(
            recorder,
            robot,
            drivers,
            sides=("left",),
            target=PROFILE.keyboard_positions[0],
            steps=ARGS.gripper_steps,
            phase="cup_release",
            intent="Open the left Robotiq to physically release the cup.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Cup release gripper safety gate failed.",
        )
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=pregrasp,
            right_target=None,
            left_orientation=grasp_orientation,
            right_orientation=None,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="cup_retreat",
            intent="Retreat vertically after opening the gripper.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name="cup_grasp_lift_release",
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Cup retreat IK or effort safety gate failed.",
        )
    settle(
        recorder,
        ARGS.settle_steps,
        phase="cup_post_release_settle",
        intent="Allow the released dynamic cup to settle.",
    )
    after_release = task_object_snapshot(
        recorder.stage, recorder.object_paths
    )
    cup_final = object_position(after_release, "cup")
    release_delta = cup_final - cup_start
    release_evidence = {
        "phase": "cup_physical_release_outcome",
        "cup_final_position": rounded(cup_final),
        "cup_displacement_from_start": rounded(release_delta),
        "horizontal_error_metres": round(
            float(np.linalg.norm(release_delta[:2])), 6
        ),
        "vertical_error_metres": round(abs(float(release_delta[2])), 6),
        "passed": bool(
            np.linalg.norm(release_delta[:2]) <= 0.10
            and abs(float(release_delta[2])) <= 0.06
        ),
        "pass_definition": (
            "released cup settled within 10cm horizontally and 6cm "
            "vertically of its initial table pose"
        ),
    }
    gates.append(release_evidence)
    return {
        "gate": "cup_grasp_lift_release",
        "classification": "physical_development_gate",
        "passed": all(bool(gate["passed"]) for gate in gates),
        "gates": gates,
        "task_objects_before": before,
        "task_objects_after": after_release,
        "official_stage_complete": False,
        "official_score": None,
    }


def tray_payload_relative_positions(
    snapshot: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    tray = object_position(snapshot, "tray")
    return {
        name: object_position(snapshot, name) - tray
        for name in RULEBOOK_STAGE1_OBJECTS
    }


def run_tray_gate(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
    steering_ids: list[int],
    drive_ids: list[int],
    drivers: dict[str, tuple[str, int]],
    *,
    transport: bool,
) -> dict[str, Any]:
    before = task_object_snapshot(recorder.stage, recorder.object_paths)
    tray_start = object_position(before, "tray")
    tray_min, tray_max = object_aabb(before, "tray")
    payload_before = tray_payload_relative_positions(before)
    base_start, base_orientation = world_pose(robot)
    left_start, right_start = ik.current_end_effector_poses(
        base_start,
        base_orientation,
        spine_position(robot),
    )
    mean_tcp_offset = (
        0.5 * (left_start[0][:2] + right_start[0][:2])
        - base_start[:2]
    )
    staging_xy = tray_start[:2] - mean_tcp_offset
    nominal_yaw = yaw_from_wxyz(base_orientation)
    grasp_orientation = top_down_orientation(90.0)
    grasp_z = tray_max[2] + ARGS.tray_grasp_z_offset
    left_grasp = np.asarray(
        (
            tray_max[0] + ARGS.tray_edge_overhang,
            tray_start[1],
            grasp_z,
        ),
        dtype=np.float64,
    )
    right_grasp = np.asarray(
        (
            tray_min[0] - ARGS.tray_edge_overhang,
            tray_start[1],
            grasp_z,
        ),
        dtype=np.float64,
    )
    pregrasp_z = tray_max[2] + ARGS.tray_pregrasp_clearance
    left_pregrasp = left_grasp.copy()
    right_pregrasp = right_grasp.copy()
    left_pregrasp[2] = pregrasp_z
    right_pregrasp[2] = pregrasp_z
    gates: list[dict[str, Any]] = []
    route_waypoints = (
        (
            "tray_route_north_clearance",
            np.asarray((base_start[0], NORTH_TRANSIT_Y), dtype=np.float64),
        ),
        (
            "tray_route_clear_table_north",
            np.asarray((-0.60, NORTH_TRANSIT_Y), dtype=np.float64),
        ),
        (
            "tray_route_descend_east_corridor",
            np.asarray((-0.60, 0.72), dtype=np.float64),
        ),
        (
            "tray_route_enter_doorway",
            np.asarray((-4.15, 0.72), dtype=np.float64),
        ),
        (
            "tray_route_cross_doorway",
            np.asarray((-4.15, -0.72), dtype=np.float64),
        ),
        ("tray_base_staging", staging_xy),
    )
    for route_phase, route_target in route_waypoints:
        gate = drive_base_to(
            recorder,
            robot,
            steering_ids,
            drive_ids,
            route_target,
            nominal_yaw,
            max_steps=ARGS.base_max_steps,
            max_speed=ARGS.base_max_speed,
            max_accel=ARGS.base_max_accel,
            phase=route_phase,
            intent=(
                "Follow the read-only official-room clearance route toward "
                "the symmetric tray stance."
            ),
        )
        gate["route_source"] = (
            "read-only bounds from the pinned official room asset"
        )
        gate["route_geometry"] = {
            "dining_table_aabb_y_metres": rounded(
                DINING_TABLE_AABB_Y
            ),
            "north_transit_centerline_y_metres": NORTH_TRANSIT_Y,
            "north_transit_centerline_clearance_from_table_metres": round(
                NORTH_TRANSIT_CENTERLINE_CLEARANCE, 6
            ),
        }
        gates.append(gate)
        if not gate["passed"]:
            return fail_closed_result(
                gate_name=(
                    "tray_bimanual_lift_and_transport"
                    if transport
                    else "tray_bimanual_lift"
                ),
                gates=gates,
                before=before,
                recorder=recorder,
                reason=f"Navigation prerequisite failed at {route_phase}.",
            )
    gates.append(
        command_gripper(
            recorder,
            robot,
            drivers,
            sides=("left", "right"),
            target=PROFILE.keyboard_positions[0],
            steps=ARGS.gripper_steps,
            phase="tray_open",
            intent="Open both Robotiq grippers above the tray.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Bimanual gripper-open safety gate failed.",
        )
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=left_pregrasp,
            right_target=right_pregrasp,
            left_orientation=grasp_orientation,
            right_orientation=grasp_orientation,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="tray_bimanual_pregrasp",
            intent=(
                "Move both TCPs above opposing tray edges with mirrored "
                "clearance and common top-down orientation."
            ),
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Tray pregrasp IK or effort safety gate failed.",
        )
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=left_grasp,
            right_target=right_grasp,
            left_orientation=grasp_orientation,
            right_orientation=grasp_orientation,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="tray_bimanual_descent",
            intent="Descend both TCPs together onto opposing tray edges.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Tray descent IK or effort safety gate failed.",
        )
    gates.append(
        command_gripper(
            recorder,
            robot,
            drivers,
            sides=("left", "right"),
            target=PROFILE.keyboard_positions[1],
            steps=ARGS.gripper_steps,
            phase="tray_bimanual_close",
            intent="Close both grippers on the dynamic tray.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Tray close effort safety gate failed.",
        )
    lift_delta = np.asarray((0.0, 0.0, ARGS.tray_lift_height))
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=left_grasp + lift_delta,
            right_target=right_grasp + lift_delta,
            left_orientation=grasp_orientation,
            right_orientation=grasp_orientation,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="tray_bimanual_lift",
            intent="Lift the tray with synchronized arm articulation.",
        )
    )
    if not gates[-1]["passed"]:
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Tray lift IK or effort safety gate failed.",
        )
    after_lift = task_object_snapshot(
        recorder.stage, recorder.object_paths
    )
    tray_lifted = object_position(after_lift, "tray")
    payload_after_lift = tray_payload_relative_positions(after_lift)
    tray_delta = tray_lifted - tray_start
    payload_drifts = {
        name: float(
            np.linalg.norm(payload_after_lift[name] - payload_before[name])
        )
        for name in RULEBOOK_STAGE1_OBJECTS
    }
    lift_outcome = {
        "phase": "tray_physical_lift_outcome",
        "tray_start_position": rounded(tray_start),
        "tray_after_lift_position": rounded(tray_lifted),
        "tray_displacement": rounded(tray_delta),
        "tray_vertical_lift_metres": round(float(tray_delta[2]), 6),
        "payload_relative_drifts_metres": {
            name: round(value, 6)
            for name, value in payload_drifts.items()
        },
        "maximum_payload_relative_drift_metres": round(
            max(payload_drifts.values()), 6
        ),
        "passed": bool(
            tray_delta[2] >= 0.07
            and max(payload_drifts.values()) <= 0.10
        ),
        "pass_definition": (
            "dynamic tray rose at least 7cm while all four task objects "
            "remained within 10cm of their tray-relative starting poses"
        ),
    }
    gates.append(lift_outcome)
    if not lift_outcome["passed"]:
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason="Dynamic tray lift outcome predicate failed.",
        )
    transport_outcome: dict[str, Any] | None = None
    if transport:
        transport_waypoints = (
            (
                "tray_transport_door_south",
                np.asarray((-4.15, -0.72), dtype=np.float64),
            ),
            (
                "tray_transport_door_north",
                np.asarray((-4.15, 0.72), dtype=np.float64),
            ),
            (
                "tray_transport_enter_east_corridor",
                np.asarray((-0.60, 0.72), dtype=np.float64),
            ),
            (
                "tray_transport_ascend_east_corridor",
                np.asarray((-0.60, NORTH_TRANSIT_Y), dtype=np.float64),
            ),
            (
                "tray_transport_clear_table_north",
                np.asarray(
                    (ARGS.dining_base_x, NORTH_TRANSIT_Y),
                    dtype=np.float64,
                ),
            ),
            (
                "tray_slow_transport",
                np.asarray(
                    (ARGS.dining_base_x, ARGS.dining_base_y),
                    dtype=np.float64,
                ),
            ),
        )
        for route_phase, route_target in transport_waypoints:
            gate = drive_base_to(
                recorder,
                robot,
                steering_ids,
                drive_ids,
                route_target,
                nominal_yaw,
                max_steps=ARGS.base_max_steps,
                max_speed=ARGS.transport_max_speed,
                max_accel=ARGS.transport_max_accel,
                phase=route_phase,
                intent=(
                    "Slowly transport the physically grasped dynamic tray "
                    "through the measured doorway clearance toward dining."
                ),
            )
            gate["route_source"] = (
                "read-only bounds from the pinned official room asset"
            )
            gate["route_geometry"] = {
                "dining_table_aabb_y_metres": rounded(
                    DINING_TABLE_AABB_Y
                ),
                "north_transit_centerline_y_metres": NORTH_TRANSIT_Y,
                "north_transit_centerline_clearance_from_table_metres": round(
                    NORTH_TRANSIT_CENTERLINE_CLEARANCE, 6
                ),
                "transport_route": (
                    "reverse the outbound east-corridor/north-table route"
                ),
            }
            gates.append(gate)
            if not gate["passed"]:
                return fail_closed_result(
                    gate_name="tray_bimanual_lift_and_transport",
                    gates=gates,
                    before=before,
                    recorder=recorder,
                    reason=(
                        "Tray transport navigation failed closed at "
                        f"{route_phase}."
                    ),
                )
        after_transport = task_object_snapshot(
            recorder.stage, recorder.object_paths
        )
        tray_final = object_position(after_transport, "tray")
        payload_after = tray_payload_relative_positions(after_transport)
        transport_payload_drifts = {
            name: float(
                np.linalg.norm(payload_after[name] - payload_before[name])
            )
            for name in RULEBOOK_STAGE1_OBJECTS
        }
        in_public_dining_proxy = bool(
            -5.8 <= tray_final[0] <= 0.1
            and 0.25 < tray_final[1] <= 3.6
        )
        transport_outcome = {
            "phase": "tray_physical_transport_outcome",
            "tray_final_position": rounded(tray_final),
            "tray_xy_displacement_metres": round(
                float(np.linalg.norm(tray_final[:2] - tray_start[:2])), 6
            ),
            "public_development_dining_region_proxy": in_public_dining_proxy,
            "proxy_source": (
                "scripts/evaluation/task3/grading.py public development region"
            ),
            "proxy_is_assigned_target": False,
            "payload_relative_drifts_metres": {
                name: round(value, 6)
                for name, value in transport_payload_drifts.items()
            },
            "maximum_payload_relative_drift_metres": round(
                max(transport_payload_drifts.values()), 6
            ),
            "passed": bool(
                in_public_dining_proxy
                and max(transport_payload_drifts.values()) <= 0.15
            ),
            "pass_definition": (
                "tray entered the public development dining-region proxy "
                "with payload relative drift no greater than 15cm"
            ),
        }
        gates.append(transport_outcome)
    after = task_object_snapshot(recorder.stage, recorder.object_paths)
    return {
        "gate": (
            "tray_bimanual_lift_and_transport"
            if transport
            else "tray_bimanual_lift"
        ),
        "classification": "physical_development_gate",
        "passed": all(bool(gate["passed"]) for gate in gates),
        "gates": gates,
        "task_objects_before": before,
        "task_objects_after": after,
        "transport_outcome": transport_outcome,
        "official_stage_complete": False,
        "official_score": None,
    }


def validate_arguments() -> None:
    integer_bounds = {
        "--trace-hz": (ARGS.trace_hz, 1, 120),
        "--settle-steps": (ARGS.settle_steps, 1, 2400),
        "--motion-steps": (ARGS.motion_steps, 1, 2400),
        "--motion-settle-steps": (
            ARGS.motion_settle_steps,
            1,
            1200,
        ),
        "--gripper-steps": (ARGS.gripper_steps, 1, 2400),
        "--base-max-steps": (ARGS.base_max_steps, 1, 12000),
    }
    for name, (value, minimum, maximum) in integer_bounds.items():
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be an integer in [{minimum}, {maximum}]"
            )
    bounded_fields = {
        "--physics-hz": (float(ARGS.physics_hz), 30.0, 500.0),
        "--render-hz": (float(ARGS.render_hz), 1.0, 240.0),
        "--base-min-speed": (ARGS.base_min_speed, 0.03, 0.08),
        "--base-max-speed": (ARGS.base_max_speed, 0.03, 0.40),
        "--base-max-accel": (ARGS.base_max_accel, 0.03, 0.80),
        "--transport-max-speed": (
            ARGS.transport_max_speed,
            0.03,
            0.25,
        ),
        "--transport-max-accel": (
            ARGS.transport_max_accel,
            0.03,
            0.50,
        ),
        "--gripper-max-force": (ARGS.gripper_max_force, 1.0, 50.0),
        "--gripper-stiffness": (
            ARGS.gripper_stiffness,
            50.0,
            800.0,
        ),
        "--gripper-damping": (ARGS.gripper_damping, 5.0, 100.0),
        "--gripper-effort-abort": (
            ARGS.gripper_effort_abort,
            1.0,
            100.0,
        ),
        "--arm-effort-abort": (ARGS.arm_effort_abort, 20.0, 250.0),
        "--base-stall-seconds": (ARGS.base_stall_seconds, 0.25, 3.0),
        "--base-stall-grace-seconds": (
            ARGS.base_stall_grace_seconds,
            0.25,
            3.0,
        ),
        "--base-stall-distance": (
            ARGS.base_stall_distance,
            0.002,
            0.05,
        ),
        "--cup-pregrasp-clearance": (
            ARGS.cup_pregrasp_clearance,
            0.06,
            0.25,
        ),
        "--cup-lift-height": (ARGS.cup_lift_height, 0.08, 0.30),
        "--tray-pregrasp-clearance": (
            ARGS.tray_pregrasp_clearance,
            0.06,
            0.25,
        ),
        "--tray-lift-height": (ARGS.tray_lift_height, 0.07, 0.30),
    }
    for name, (value, minimum, maximum) in bounded_fields.items():
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be finite and in [{minimum}, {maximum}]"
            )
    if ARGS.base_min_speed > min(
        ARGS.base_max_speed, ARGS.transport_max_speed
    ):
        raise ValueError(
            "--base-min-speed must not exceed either base speed cap"
        )
    geometry_bounds = {
        "--cup-grasp-z-offset": (
            ARGS.cup_grasp_z_offset,
            -0.03,
            0.08,
        ),
        "--tray-grasp-z-offset": (
            ARGS.tray_grasp_z_offset,
            -0.03,
            0.08,
        ),
        "--tray-edge-overhang": (
            ARGS.tray_edge_overhang,
            0.0,
            0.05,
        ),
        "--cup-grasp-yaw-deg": (
            ARGS.cup_grasp_yaw_deg,
            -180.0,
            180.0,
        ),
        "--dining-base-x": (ARGS.dining_base_x, -5.30, -4.70),
        "--dining-base-y": (ARGS.dining_base_y, 0.90, 1.60),
    }
    for name, (value, minimum, maximum) in geometry_bounds.items():
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be finite and in [{minimum}, {maximum}]"
            )
    optional_robot_pose_bounds = {
        "--robot-x": (ARGS.robot_x, -4.65, -4.55),
        "--robot-y": (ARGS.robot_y, 2.65, 2.75),
        "--robot-z": (ARGS.robot_z, -0.02, 0.05),
        "--robot-yaw": (ARGS.robot_yaw, -95.0, -85.0),
    }
    for name, (value, minimum, maximum) in (
        optional_robot_pose_bounds.items()
    ):
        if value is None:
            continue
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} override must be finite and in "
                f"[{minimum}, {maximum}] for the validated room route"
            )
    ratio = float(ARGS.physics_hz) / ARGS.trace_hz
    if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise ValueError(
            "--trace-hz must divide --physics-hz exactly"
        )


def build_provenance(
    script_path: Path,
    room_path: Path,
    robot_path: Path,
    configured_robot_position: Iterable[float],
    configured_robot_yaw: float,
) -> dict[str, Any]:
    return {
        "official_benchmark_commit": os.environ.get(
            "EBIM_BENCHMARK_COMMIT",
            os.environ.get("EBIM_COMMIT", git_revision(REPO_ROOT)),
        ),
        "controller_sha256": sha256_file(script_path),
        "controller_path_in_runtime": str(script_path),
        "simulator": "Isaac Sim 5.1.0",
        "container_image": os.environ.get(
            "AIRSIGN_CAPTURE_CONTAINER_IMAGE",
            "unavailable",
        ),
        "container_image_id": os.environ.get(
            "AIRSIGN_CAPTURE_CONTAINER_IMAGE_ID", "unavailable"
        ),
        "python": platform.python_version(),
        "head_placement": ARGS.head_placement,
        "requested_robot_pose_overrides": {
            "x": ARGS.robot_x,
            "y": ARGS.robot_y,
            "z": ARGS.robot_z,
            "yaw_degrees": ARGS.robot_yaw,
        },
        "configured_robot_start": {
            "position": rounded(configured_robot_position),
            "yaw_degrees": round(float(configured_robot_yaw), 6),
        },
        "physics_hz": float(ARGS.physics_hz),
        "trace_hz": ARGS.trace_hz,
        "room_asset": str(room_path),
        "room_asset_sha256": sha256_file(room_path),
        "robot_asset": str(robot_path),
        "robot_asset_sha256": sha256_file(robot_path),
    }


def write_evidence(
    output_dir: Path,
    *,
    mutation_guard: dict[str, Any],
    target_provider: dict[str, Any],
    provenance: dict[str, Any],
    recorder: TraceRecorder,
    physical_results: list[dict[str, Any]],
    rigid_bodies: dict[str, Any],
    environment_inventory: list[dict[str, Any]],
    started_wall: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_passed = all(
        bool(result["passed"]) for result in physical_results
    )
    trace = {
        "schema_version": 1,
        "kind": "airsign_ebim_track3_stage1_physical_gate_trace",
        "classification": "measured_physical_development_evidence",
        "coordinate_frame": "official_isaac_world_metres_z_up",
        "task_objects_teleported": False,
        "robot_links_teleported": False,
        "task_object_mutation_api_used": False,
        "official_stage_completion_claimed": False,
        "official_score_claimed": False,
        "benchmark_score": None,
        "events": recorder.events,
        "frames": recorder.frames,
        "provenance": provenance,
    }
    metrics = {
        "schema_version": 1,
        "kind": "airsign_ebim_track3_stage1_physical_gate_metrics",
        "classification": "participant_physical_development_gate",
        "gate_requested": ARGS.gate,
        "passed": physical_passed,
        "pass_definition": (
            "all requested articulation-only physical development gates passed"
        ),
        "official_stage_completion_claimed": False,
        "official_stage_complete": False,
        "official_score_claimed": False,
        "official_stage_score": None,
        "official_stage_max_score": 4,
        "official_claim_blockers": [
            "organizer randomized assigned-target provider is unavailable",
            "live organizer scorer contract is unavailable",
            "public development grader does not verify grasp/contact/release",
            (
                "participant joint-effort aborts are not organizer-validated "
                "task-object contact-force compliance"
            ),
        ],
        "target_provider": target_provider,
        "mutation_guard": mutation_guard,
        "rigid_body_state_before_motion": rigid_bodies,
        "environment_top_level_inventory": environment_inventory,
        "all_task_objects_dynamic": all(
            record["all_enabled"] and record["all_dynamic"]
            for record in rigid_bodies.values()
        ),
        "physical_results": physical_results,
        "trace": {
            "frame_count": len(recorder.frames),
            "event_count": len(recorder.events),
            "duration_seconds": round(
                recorder.sim_step / recorder.physics_hz, 6
            ),
            "trace_hz": recorder.trace_hz,
        },
        "provenance": provenance,
        "wall_duration_seconds": round(time.time() - started_wall, 3),
    }
    trace_path = output_dir / "trajectory.json"
    metrics_path = output_dir / "metrics.json"
    trace_path.write_text(
        json.dumps(
            trace, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            metrics, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "controller_sha256": provenance["controller_sha256"],
        "official_benchmark_commit": provenance[
            "official_benchmark_commit"
        ],
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (trace_path, metrics_path)
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "passed": physical_passed,
        "classification": metrics["classification"],
        "gate_requested": ARGS.gate,
        "official_stage_completion_claimed": False,
        "official_score_claimed": False,
        "benchmark_score": None,
        "target_provider_status": target_provider["status"],
        "all_task_objects_dynamic": metrics["all_task_objects_dynamic"],
        "output_dir": str(output_dir),
        "files": manifest["files"],
    }


def main() -> bool:
    validate_arguments()
    started_wall = time.time()
    script_path = Path(__file__).resolve()
    mutation_guard = enforce_mutation_guard(script_path)
    target_provider = load_target_provider(ARGS.targets_json)
    room_path = Path(ARGS.room_usd).expanduser()
    robot_path = Path(ARGS.robot_usd).expanduser()
    if not room_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError(
            f"Missing room or robot asset: {room_path}, {robot_path}"
        )
    output_dir = ARGS.output_dir.expanduser().resolve()
    groups = core._load_joint_groups(
        Path(ARGS.franka_root).expanduser(),
        ARGS.embodiment,
        include_browser_commands=False,
    )
    ARGS.task = "task3"
    robot_position = room_scene.resolve_robot_position(ARGS)
    robot_yaw = room_scene.resolve_robot_yaw(ARGS)
    print("STAGE1_TABLE_SETUP_PHASE build_official_scene", flush=True)
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
        name="airsign_stage1_table_setup_robot",
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
    drivers = discover_gripper_drivers(robot)
    driver_names = {details[0] for details in drivers.values()}
    core._configure_drives(
        ROBOT_PRIM_PATH,
        lambda joint_name: (
            {
                "stiffness": ARGS.gripper_stiffness,
                "damping": ARGS.gripper_damping,
                "max_force": ARGS.gripper_max_force,
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
    rigid_bodies = {
        name: rigid_body_record(stage, path)
        for name, path in object_paths.items()
    }
    environment_inventory = environment_top_level_inventory(stage)
    recorder = TraceRecorder(
        world=world,
        robot=robot,
        ik=ik,
        stage=stage,
        object_paths=object_paths,
        gripper_drivers=drivers,
        physics_hz=float(ARGS.physics_hz),
        trace_hz=ARGS.trace_hz,
        render=ARGS.render,
    )
    recorder.set_phase(
        "initial_state",
        intent="Record settled official public Task 3 scene state.",
    )
    results: list[dict[str, Any]] = []
    if ARGS.gate == "inspect":
        results.append(
            {
                "gate": "read_only_scene_and_controller_inspection",
                "classification": "read_only_development_gate",
                "passed": bool(
                    mutation_guard["passed"]
                    and all(
                        record["all_enabled"] and record["all_dynamic"]
                        for record in rigid_bodies.values()
                    )
                ),
                "task_objects": task_object_snapshot(stage, object_paths),
                "official_stage_complete": False,
                "official_score": None,
            }
        )
    cup_result: dict[str, Any] | None = None
    if ARGS.gate in ("cup", "all"):
        cup_result = run_cup_gate(
            recorder,
            robot,
            ik,
            steering_ids,
            drive_ids,
            drivers,
        )
        results.append(cup_result)
    if ARGS.gate in ("tray-lift", "tray-transport", "all"):
        if ARGS.gate == "all" and not bool(cup_result["passed"]):
            results.append(
                {
                    "gate": "tray_bimanual_lift_and_transport",
                    "classification": "fail_closed_skip",
                    "passed": False,
                    "skipped": True,
                    "reason": (
                        "The preceding cup physical gate failed; no further "
                        "task motion was authorized."
                    ),
                    "official_stage_complete": False,
                    "official_score": None,
                }
            )
        else:
            results.append(
                run_tray_gate(
                    recorder,
                    robot,
                    ik,
                    steering_ids,
                    drive_ids,
                    drivers,
                    transport=ARGS.gate in ("tray-transport", "all"),
                )
            )
    recorder.set_phase(
        "final_state",
        intent="Record final measured articulation and task-object state.",
    )
    provenance = build_provenance(
        script_path,
        room_path,
        robot_path,
        robot_position,
        robot_yaw,
    )
    result = write_evidence(
        output_dir,
        mutation_guard=mutation_guard,
        target_provider=target_provider,
        provenance=provenance,
        recorder=recorder,
        physical_results=results,
        rigid_bodies=rigid_bodies,
        environment_inventory=environment_inventory,
        started_wall=started_wall,
    )
    print(
        "STAGE1_TABLE_SETUP_RESULT "
        + json.dumps(result, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return bool(result["passed"])


if __name__ == "__main__":
    exit_code = 0
    try:
        if not main():
            exit_code = 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        exit_code = 1
    finally:
        SIMULATION_APP.close()
    raise SystemExit(exit_code)
