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
import hashlib
import importlib.util
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
from base_motion_monitor import (  # noqa: E402
    BaseMotionMonitor,
    BaseMotionObservation,
)
from executable_provenance import (  # noqa: E402
    executable_source_guard,
    sha256_file,
)
from isaac_collision_geometry import (  # noqa: E402
    CollisionProxySet,
    extract_collision_proxy_set,
    live_collision_geometry_containment_record,
    resolve_enabled_dynamic_rigid_body_descendant,
)
from joint_command_guard import (  # noqa: E402
    arm_and_spine_effort_record,
    joint_command_slew_record,
    joint_tracking_error_dwell_record,
)
from se2_route_validator import (  # noqa: E402
    Pose2,
    RouteValidationConfig,
    base_nonplanar_deviation_record,
    proxy_geometry_sha256,
    route_certificate_is_valid,
    route_execution_tube_record,
    route_sha256,
    validate_route,
)
from isaacsim_fr3duo_teleop_bridge_args import (  # noqa: E402
    add_common_bridge_args,
)

# Nonzero base motion stays fail-closed until an Isaac run establishes a
# conservative swept stopping envelope and that bound is carried in, and
# replay-verified from, each route certificate.  Fixed geometric gaps alone are
# not a braking proof.
BASE_MOTION_STOPPING_ENVELOPE_CERTIFIED = False
BASE_MOTION_STOPPING_ENVELOPE_LIMITATION = (
    "nonzero wheel commands are disabled because no measured, conservative "
    "swept stopping envelope is bound to the route certificate"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        choices=(
            "inspect",
            "cup-preflight",
            "cup",
            "tray-lift",
            "tray-transport",
            "all",
        ),
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
    parser.add_argument(
        "--arm-max-command-slew-rad",
        type=float,
        default=0.025,
        help=(
            "Maximum absolute change between adjacent applied arm commands. "
            "A violation aborts before application; targets are not clamped."
        ),
    )
    parser.add_argument(
        "--arm-tracking-error-threshold-rad",
        type=float,
        default=0.12,
        help=(
            "Independent maximum target-to-measured arm error before the "
            "tracking dwell begins."
        ),
    )
    parser.add_argument(
        "--arm-tracking-error-dwell-seconds",
        type=float,
        default=0.25,
        help=(
            "Continuous time above the tracking-error threshold required "
            "for a fail-closed abort."
        ),
    )
    parser.add_argument("--base-stall-seconds", type=float, default=1.0)
    parser.add_argument("--base-stall-grace-seconds", type=float, default=1.0)
    parser.add_argument("--base-stall-distance", type=float, default=0.015)
    parser.add_argument(
        "--base-steering-timeout-seconds", type=float, default=4.0
    )
    parser.add_argument(
        "--base-drive-response-rad-s", type=float, default=0.10
    )
    parser.add_argument(
        "--route-clearance-margin", type=float, default=0.08
    )
    parser.add_argument(
        "--route-max-translation-step", type=float, default=0.025
    )
    parser.add_argument(
        "--route-max-yaw-step-deg", type=float, default=2.0
    )
    parser.add_argument(
        "--route-cross-track-tolerance", type=float, default=0.04
    )
    parser.add_argument(
        "--route-operational-cross-track-tolerance",
        type=float,
        default=0.02,
        help=(
            "Smaller runtime cross-track limit. The gap to the certificate "
            "limit is diagnostic containment margin, not a braking proof."
        ),
    )
    parser.add_argument(
        "--route-yaw-tracking-tolerance-deg", type=float, default=2.0
    )
    parser.add_argument(
        "--route-operational-yaw-tolerance-deg",
        type=float,
        default=1.0,
        help=(
            "Smaller runtime yaw limit. The gap to the certificate limit is "
            "diagnostic containment margin, not a braking proof."
        ),
    )
    parser.add_argument(
        "--dynamic-geometry-certificate-allowance-metres",
        type=float,
        default=0.08,
        help=(
            "Clearance reserved in every route certificate for live "
            "articulation/collider motion relative to the captured base."
        ),
    )
    parser.add_argument(
        "--dynamic-geometry-operational-limit-metres",
        type=float,
        default=0.04,
        help=(
            "Smaller live collider-containment limit that triggers a stop "
            "before the certificate allowance is exhausted."
        ),
    )
    parser.add_argument(
        "--navigation-stow-height", type=float, default=1.15
    )
    parser.add_argument(
        "--navigation-stow-forward", type=float, default=0.48
    )
    parser.add_argument(
        "--navigation-stow-lateral", type=float, default=0.28
    )
    parser.add_argument("--cup-grasp-yaw-deg", type=float, default=90.0)
    parser.add_argument("--cup-pregrasp-clearance", type=float, default=0.12)
    parser.add_argument("--cup-grasp-z-offset", type=float, default=0.025)
    parser.add_argument("--cup-lift-height", type=float, default=0.16)
    parser.add_argument("--tray-pregrasp-clearance", type=float, default=0.12)
    parser.add_argument("--tray-grasp-z-offset", type=float, default=0.018)
    parser.add_argument("--tray-edge-overhang", type=float, default=0.015)
    parser.add_argument("--tray-lift-height", type=float, default=0.14)
    parser.add_argument(
        "--tray-payload-envelope-metres",
        type=float,
        default=0.15,
        help=(
            "Additional loaded-geometry clearance reserved for measured "
            "payload-to-base drift after the post-lift collision capture."
        ),
    )
    parser.add_argument(
        "--tray-payload-operational-limit-metres",
        type=float,
        default=0.10,
        help=(
            "Live loaded-compound containment limit; the gap to the tray "
            "payload envelope is diagnostic margin, not a braking proof."
        ),
    )
    parser.add_argument("--dining-base-x", type=float, default=-5.05)
    parser.add_argument("--dining-base-y", type=float, default=1.25)
    parser.add_argument(
        "--render",
        action="store_true",
        help=(
            "Reserved for future substep-safe rendering; currently rejected "
            "because one rendered World.step may batch physics substeps."
        ),
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
AIRSIGN_EXECUTED_SOURCE_NAMES = (
    "stage1_table_setup.py",
    "base_motion_monitor.py",
    "isaac_collision_geometry.py",
    "joint_command_guard.py",
    "se2_route_validator.py",
    "executable_provenance.py",
)
AIRSIGN_RUNTIME_DATA_NAMES = ("upstream_source_manifest.json",)
DIRECT_UPSTREAM_HELPER_MODULES = (
    "scene_robot_room_keyboard",
    "gripper_profiles",
    "isaacsim_fr3duo_teleop_bridge_args",
    "isaacsim_fr3duo_teleop_bridge_core",
    "dual_arm_lula",
)

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
        "set_joint_positions",
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


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def direct_upstream_helper_records() -> dict[str, dict[str, Any]]:
    manifest_path = (
        Path(__file__).resolve().parent / "upstream_source_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("revision"), str)
        or not isinstance(manifest.get("helpers"), dict)
        or set(manifest["helpers"]) != set(DIRECT_UPSTREAM_HELPER_MODULES)
    ):
        raise RuntimeError("upstream helper manifest is invalid")
    runtime_revision = os.environ.get(
        "EBIM_BENCHMARK_COMMIT",
        os.environ.get("EBIM_COMMIT", git_revision(REPO_ROOT)),
    )
    if runtime_revision != manifest["revision"]:
        raise RuntimeError(
            "runtime benchmark revision differs from the authenticated "
            "upstream helper manifest"
        )
    records: dict[str, dict[str, Any]] = {}
    for module_name in DIRECT_UPSTREAM_HELPER_MODULES:
        expected = manifest["helpers"][module_name]
        if not (
            isinstance(expected, dict)
            and isinstance(expected.get("repository_path"), str)
            and isinstance(expected.get("bytes"), int)
            and not isinstance(expected.get("bytes"), bool)
            and isinstance(expected.get("sha256"), str)
        ):
            raise RuntimeError(
                f"invalid upstream helper manifest entry: {module_name}"
            )
        module = sys.modules.get(module_name)
        supplied_path = getattr(module, "__file__", None)
        if not supplied_path:
            raise RuntimeError(
                f"direct upstream helper has no source path: {module_name}"
            )
        path = Path(supplied_path).resolve()
        if path.suffix == ".pyc":
            try:
                source_candidate = Path(
                    importlib.util.source_from_cache(str(path))
                )
            except (NotImplementedError, ValueError):
                source_candidate = path.with_suffix(".py")
            if source_candidate.is_file():
                path = source_candidate.resolve()
        if not path.is_file():
            raise RuntimeError(
                f"direct upstream helper source is missing: {path}"
            )
        try:
            repository_path = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            repository_path = None
        observed_sha256 = sha256_file(path)
        if (
            repository_path != expected["repository_path"]
            or path.stat().st_size != expected["bytes"]
            or observed_sha256 != expected["sha256"]
        ):
            raise RuntimeError(
                "runtime upstream helper differs from the pinned manifest: "
                f"{module_name}"
            )
        records[module_name] = {
            "runtime_path": str(path),
            "official_repository_path": repository_path,
            "bytes": path.stat().st_size,
            "sha256": observed_sha256,
            "pinned_manifest_revision": manifest["revision"],
            "pinned_manifest_path": (
                "participant/upstream_source_manifest.json"
            ),
            "matches_pinned_manifest": True,
        }
    return records


def resolved_airsign_module_records(
    source_root: Path,
) -> dict[str, dict[str, Any]]:
    """Bind imported AirSign modules to the exact guarded source files."""

    records: dict[str, dict[str, Any]] = {}
    for source_name in AIRSIGN_EXECUTED_SOURCE_NAMES:
        if source_name == "stage1_table_setup.py":
            continue
        module_name = Path(source_name).stem
        module = sys.modules.get(module_name)
        if module is None:
            raise RuntimeError(
                f"guarded AirSign module is not imported: {module_name}"
            )
        try:
            supplied_path = module.__file__
        except AttributeError as error:
            raise RuntimeError(
                f"guarded AirSign module has no source path: {module_name}"
            ) from error
        if not supplied_path:
            raise RuntimeError(
                f"guarded AirSign module has no source path: {module_name}"
            )
        runtime_path = Path(supplied_path).resolve()
        expected_path = (source_root / source_name).resolve()
        if runtime_path.suffix == ".pyc":
            try:
                runtime_path = Path(
                    importlib.util.source_from_cache(str(runtime_path))
                ).resolve()
            except (NotImplementedError, ValueError):
                runtime_path = runtime_path.with_suffix(".py").resolve()
        if runtime_path != expected_path or not expected_path.is_file():
            raise RuntimeError(
                "guarded AirSign module resolved outside the expected "
                f"source set: {module_name} -> {runtime_path}"
            )
        records[module_name] = {
            "runtime_path": str(runtime_path),
            "expected_path": str(expected_path),
            "repository_path": f"participant/{source_name}",
            "bytes": expected_path.stat().st_size,
            "sha256": sha256_file(expected_path),
            "path_matches_expected": True,
        }
    return records


def enforce_mutation_guard(script_path: Path) -> dict[str, Any]:
    source_root = script_path.parent
    result = executable_source_guard(
        (
            source_root / name
            for name in AIRSIGN_EXECUTED_SOURCE_NAMES
        ),
        root=source_root,
        forbidden_attribute_calls=FORBIDDEN_ATTRIBUTE_CALLS,
        forbidden_direct_calls=FORBIDDEN_DIRECT_CALLS,
    )
    result.update(
        {
            "controller_sha256": sha256_file(script_path),
            "scope": [
                f"participant/{name}"
                for name in AIRSIGN_EXECUTED_SOURCE_NAMES
            ],
            "source_root_in_runtime": str(source_root),
            "resolved_airsign_modules": (
                resolved_airsign_module_records(source_root)
            ),
            "scene_initialization_boundary": (
                "scene_robot_room_keyboard.build_stage before controller motion"
            ),
            "limitation": (
                "This is a provenance-bound syntactic policy check, not a "
                "Python sandbox or a complete proof against reflective or "
                "native-code mutation."
            ),
        }
    )
    if result["violations"]:
        raise RuntimeError(
            "Controller mutation guard rejected source: "
            + json.dumps(result["violations"], sort_keys=True)
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


def collision_prim_inventory(stage: Any) -> dict[str, Any]:
    """Inventory enabled collision prims without treating bounds as clearance."""

    purposes = [
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
    ]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    records: list[dict[str, Any]] = []
    unresolved: list[str] = []
    try:
        prims = Usd.PrimRange.Stage(
            stage, Usd.TraverseInstanceProxies()
        )
    except Exception:  # noqa: BLE001
        prims = stage.Traverse()
    seen_paths: set[str] = set()
    for prim in prims:
        path = str(prim.GetPath())
        if path in seen_paths:
            continue
        seen_paths.add(path)
        collision_attr = prim.GetAttribute("physics:collisionEnabled")
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (
            has_collision_api
            or (collision_attr and collision_attr.IsValid())
        ):
            continue
        enabled_value = (
            collision_attr.Get()
            if collision_attr and collision_attr.IsValid()
            else True
        )
        if enabled_value is False:
            continue
        record: dict[str, Any] = {
            "prim_path": path,
            "prim_type": str(prim.GetTypeName()),
            "has_collision_api": bool(has_collision_api),
            "instance": bool(prim.IsInstance()),
            "instanceable": bool(prim.IsInstanceable()),
            "under_robot": path == ROBOT_PRIM_PATH
            or path.startswith(f"{ROBOT_PRIM_PATH}/"),
        }
        try:
            bounds = bbox_cache.ComputeWorldBound(
                prim
            ).ComputeAlignedRange()
            minimum = np.asarray(bounds.GetMin(), dtype=np.float64)
            maximum = np.asarray(bounds.GetMax(), dtype=np.float64)
            finite = bool(
                np.all(np.isfinite(minimum))
                and np.all(np.isfinite(maximum))
                and np.all(minimum <= maximum)
                and np.max(np.abs(np.concatenate((minimum, maximum))))
                < 1e20
            )
            record["world_aabb"] = {
                "min": rounded(minimum) if finite else None,
                "max": rounded(maximum) if finite else None,
            }
            record["finite_world_bound"] = finite
            if not finite:
                unresolved.append(path)
        except Exception as error:  # noqa: BLE001
            record["world_aabb"] = {"min": None, "max": None}
            record["finite_world_bound"] = False
            record["bound_error"] = (
                f"{type(error).__name__}: {error}"
            )
            unresolved.append(path)
        records.append(record)
    records.sort(key=lambda item: item["prim_path"])
    robot_records = [record for record in records if record["under_robot"]]
    environment_records = [
        record for record in records if not record["under_robot"]
    ]
    return {
        "classification": "read_only_collision_prim_inventory",
        "clearance_certificate": False,
        "limitation": (
            "World-aligned bounds are diagnostic inventory only; they are "
            "not a full-robot swept-volume clearance proof."
        ),
        "enabled_collision_prim_count": len(records),
        "robot_collision_prim_count": len(robot_records),
        "environment_collision_prim_count": len(environment_records),
        "finite_world_bound_count": sum(
            bool(record["finite_world_bound"]) for record in records
        ),
        "unresolved_prim_paths": unresolved,
        "complete_finite_world_bound_coverage": not unresolved
        and bool(robot_records)
        and bool(environment_records),
        "records": records,
    }


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
        self.base_control: dict[str, Any] | None = None
        self.arm_last_command_by_name: dict[str, float] | None = None
        self.arm_tracking_error_consecutive_exceeded_steps = 0
        self.arm_tracking_last_observation_sim_step: int | None = None

    def set_base_control(
        self, diagnostics: dict[str, Any] | None
    ) -> None:
        self.base_control = diagnostics

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
                "base_control": self.base_control,
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
    abort_recovery_applied = False
    abort_recovery_reason = "not_required"
    names = list(robot.dof_names)
    arm_joint_names = tuple(
        name
        for name in names
        if (
            name.startswith("left_fr3v2_joint")
            or name.startswith("right_fr3v2_joint")
        )
    )
    spine_joint_name = "franka_spine_vertical_joint"
    arm_indices = np.asarray(
        [names.index(name) for name in arm_joint_names],
        dtype=np.int64,
    )
    initial_joint_positions = np.asarray(
        robot.get_joint_positions(), dtype=np.float64
    )
    if recorder.arm_last_command_by_name is None:
        initial_command_baseline_targets = {
            name: (
                float(initial_joint_positions[names.index(name)])
                if (
                    initial_joint_positions.ndim == 1
                    and len(initial_joint_positions) == len(names)
                )
                else math.nan
            )
            for name in arm_joint_names
        }
        initial_command_baseline_source = (
            "measured_articulation_before_first_arm_command"
        )
    else:
        initial_command_baseline_targets = dict(
            recorder.arm_last_command_by_name
        )
        initial_command_baseline_source = (
            "previous_applied_arm_command_from_trace_recorder"
        )
    previous_command_targets = dict(initial_command_baseline_targets)
    command_slew_check_count = 0
    first_command_slew_check: dict[str, Any] | None = None
    worst_command_slew_check: dict[str, Any] | None = None
    command_slew_violation: dict[str, Any] | None = None
    tracking_error_check_count = 0
    first_tracking_error_check: dict[str, Any] | None = None
    worst_tracking_error_check: dict[str, Any] | None = None
    tracking_error_dwell_violation: dict[str, Any] | None = None
    tracking_error_threshold_exceedance_count = 0
    tracking_error_observation_gap_reset = bool(
        recorder.arm_tracking_last_observation_sim_step is not None
        and recorder.arm_tracking_last_observation_sim_step
        != recorder.sim_step
    )
    if tracking_error_observation_gap_reset:
        recorder.arm_tracking_error_consecutive_exceeded_steps = 0
    tracking_error_consecutive_exceeded_steps = (
        recorder.arm_tracking_error_consecutive_exceeded_steps
    )
    initial_tracking_error_consecutive_exceeded_steps = (
        tracking_error_consecutive_exceeded_steps
    )
    max_tracking_error_consecutive_exceeded_steps = (
        tracking_error_consecutive_exceeded_steps
    )
    tracking_error_dwell_steps = max(
        1,
        int(
            math.ceil(
                ARGS.arm_tracking_error_dwell_seconds
                * float(ARGS.physics_hz)
            )
        ),
    )
    peak_arm_effort = 0.0
    peak_arm_effort_joint: str | None = None
    peak_arm_effort_by_joint = {
        name: 0.0 for name in arm_joint_names
    }
    peak_abs_spine_force = 0.0
    arm_effort_samples = 0
    spine_force_samples = 0
    effort_telemetry_initially_available = False
    effort_telemetry_nonfinite = False
    initial_effort_record: dict[str, Any] | None = None

    def assess_command_slew(
        targets: dict[str, float],
    ) -> dict[str, Any]:
        nonlocal command_slew_check_count
        nonlocal first_command_slew_check
        nonlocal worst_command_slew_check
        nonlocal command_slew_violation
        record = joint_command_slew_record(
            previous_targets=previous_command_targets,
            targets=targets,
            expected_names=arm_joint_names,
            max_abs_delta_rad=ARGS.arm_max_command_slew_rad,
        )
        command_slew_check_count += 1
        if first_command_slew_check is None:
            first_command_slew_check = record
        candidate = record.get("max_abs_delta_rad")
        current_worst = (
            None
            if worst_command_slew_check is None
            else worst_command_slew_check.get("max_abs_delta_rad")
        )
        if (
            worst_command_slew_check is None
            or (
                candidate is not None
                and (
                    current_worst is None
                    or float(candidate) > float(current_worst)
                )
            )
        ):
            worst_command_slew_check = record
        if not record["passed"]:
            command_slew_violation = record
        return record

    def assess_tracking_error(
        targets: dict[str, float],
    ) -> dict[str, Any]:
        nonlocal tracking_error_check_count
        nonlocal first_tracking_error_check
        nonlocal worst_tracking_error_check
        nonlocal tracking_error_dwell_violation
        nonlocal tracking_error_threshold_exceedance_count
        nonlocal tracking_error_consecutive_exceeded_steps
        nonlocal max_tracking_error_consecutive_exceeded_steps
        record = joint_tracking_error_dwell_record(
            dof_names=names,
            measured_positions=robot.get_joint_positions(),
            targets=targets,
            expected_names=arm_joint_names,
            max_abs_error_rad=ARGS.arm_tracking_error_threshold_rad,
            dwell_steps=tracking_error_dwell_steps,
            prior_consecutive_exceeded_steps=(
                tracking_error_consecutive_exceeded_steps
            ),
        )
        tracking_error_check_count += 1
        if first_tracking_error_check is None:
            first_tracking_error_check = record
        candidate = record.get("max_abs_error_rad")
        current_worst = (
            None
            if worst_tracking_error_check is None
            else worst_tracking_error_check.get("max_abs_error_rad")
        )
        if (
            worst_tracking_error_check is None
            or (
                candidate is not None
                and (
                    current_worst is None
                    or float(candidate) > float(current_worst)
                )
            )
        ):
            worst_tracking_error_check = record
        if record.get("threshold_exceeded"):
            tracking_error_threshold_exceedance_count += 1
        consecutive = record.get("consecutive_exceeded_steps")
        if consecutive is not None:
            tracking_error_consecutive_exceeded_steps = int(consecutive)
            recorder.arm_tracking_error_consecutive_exceeded_steps = (
                tracking_error_consecutive_exceeded_steps
            )
            recorder.arm_tracking_last_observation_sim_step = (
                recorder.sim_step
            )
            max_tracking_error_consecutive_exceeded_steps = max(
                max_tracking_error_consecutive_exceeded_steps,
                tracking_error_consecutive_exceeded_steps,
            )
        if not record["passed"]:
            tracking_error_dwell_violation = record
        return record

    def accumulate_effort(record: dict[str, Any]) -> None:
        nonlocal peak_arm_effort
        nonlocal peak_arm_effort_joint
        nonlocal peak_abs_spine_force
        for name, value in record.get(
            "arm_effort_by_name", {}
        ).items():
            magnitude = abs(float(value))
            peak_arm_effort_by_joint[name] = max(
                peak_arm_effort_by_joint.get(name, 0.0),
                magnitude,
            )
            if magnitude > peak_arm_effort:
                peak_arm_effort = magnitude
                peak_arm_effort_joint = name
        spine_force = record.get("spine_force_newtons")
        if spine_force is not None:
            peak_abs_spine_force = max(
                peak_abs_spine_force, abs(float(spine_force))
            )

    def describe_effort_failure(
        record: dict[str, Any],
        *,
        context: str,
    ) -> str:
        if record.get("arm_threshold_exceeded"):
            return (
                "measured revolute-arm effort "
                f"{record.get('peak_abs_arm_effort')} at "
                f"{record.get('peak_abs_arm_effort_joint')} exceeded "
                f"--arm-effort-abort during {context}"
            )
        return f"{record.get('reason')} during {context}"

    initial_efforts = measured_efforts(robot)
    if initial_efforts is not None:
        initial_effort_record = arm_and_spine_effort_record(
            dof_names=names,
            measured_efforts=initial_efforts,
            arm_joint_names=arm_joint_names,
            spine_joint_name=spine_joint_name,
            arm_effort_abort_threshold=ARGS.arm_effort_abort,
        )
        effort_telemetry_initially_available = bool(
            initial_effort_record.get("peak_abs_arm_effort") is not None
            and initial_effort_record.get("spine_force_newtons") is not None
        )
        accumulate_effort(initial_effort_record)
        if not initial_effort_record["passed"]:
            aborted = True
            abort_reason = describe_effort_failure(
                initial_effort_record, context="initial telemetry"
            )
            effort_telemetry_nonfinite = "non-finite" in abort_reason
    if not effort_telemetry_initially_available:
        aborted = True
        abort_reason = (
            abort_reason
            or "finite arm and spine effort telemetry unavailable before motion"
        )
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
        slew = assess_command_slew(result.combined)
        if not slew["passed"]:
            aborted = True
            abort_reason = (
                "IK adjacent-command slew gate failed before command: "
                f"{slew['reason']}"
            )
            break
        apply_targets(robot, result.combined)
        previous_command_targets = {
            name: float(result.combined[name])
            for name in arm_joint_names
        }
        recorder.arm_last_command_by_name = dict(
            previous_command_targets
        )
        recorder.step()
        tracking = assess_tracking_error(previous_command_targets)
        if not tracking["passed"]:
            aborted = True
            abort_reason = (
                "arm target-to-measured tracking-error dwell gate failed "
                f"during motion: {tracking['reason']}"
            )
            break
        efforts = measured_efforts(robot)
        if efforts is None:
            aborted = True
            abort_reason = "arm effort telemetry disappeared during motion"
            break
        effort_record = arm_and_spine_effort_record(
            dof_names=names,
            measured_efforts=efforts,
            arm_joint_names=arm_joint_names,
            spine_joint_name=spine_joint_name,
            arm_effort_abort_threshold=ARGS.arm_effort_abort,
        )
        accumulate_effort(effort_record)
        if effort_record.get("spine_force_newtons") is not None:
            spine_force_samples += 1
        if effort_record.get("peak_abs_arm_effort") is not None:
            arm_effort_samples += 1
        if not effort_record["passed"]:
            effort_telemetry_nonfinite = (
                "non-finite" in str(effort_record.get("reason"))
            )
            aborted = True
            abort_reason = describe_effort_failure(
                effort_record, context="motion"
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
        slew = assess_command_slew(result.combined)
        if not slew["passed"]:
            aborted = True
            abort_reason = (
                "IK adjacent-command slew gate failed before settle "
                f"command: {slew['reason']}"
            )
            break
        apply_targets(robot, result.combined)
        previous_command_targets = {
            name: float(result.combined[name])
            for name in arm_joint_names
        }
        recorder.arm_last_command_by_name = dict(
            previous_command_targets
        )
        recorder.step()
        tracking = assess_tracking_error(previous_command_targets)
        if not tracking["passed"]:
            aborted = True
            abort_reason = (
                "arm target-to-measured tracking-error dwell gate failed "
                f"during settle: {tracking['reason']}"
            )
            break
        efforts = measured_efforts(robot)
        if efforts is None:
            aborted = True
            abort_reason = "arm effort telemetry disappeared during settle"
            break
        effort_record = arm_and_spine_effort_record(
            dof_names=names,
            measured_efforts=efforts,
            arm_joint_names=arm_joint_names,
            spine_joint_name=spine_joint_name,
            arm_effort_abort_threshold=ARGS.arm_effort_abort,
        )
        accumulate_effort(effort_record)
        if effort_record.get("spine_force_newtons") is not None:
            spine_force_samples += 1
        if effort_record.get("peak_abs_arm_effort") is not None:
            arm_effort_samples += 1
        if not effort_record["passed"]:
            effort_telemetry_nonfinite = (
                "non-finite" in str(effort_record.get("reason"))
            )
            aborted = True
            abort_reason = describe_effort_failure(
                effort_record, context="settle"
            )
            break
    if not aborted and arm_effort_samples < 1:
        aborted = True
        abort_reason = "no finite arm effort sample was recorded"
    if aborted:
        measured_for_hold = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        if not arm_indices.size:
            abort_recovery_reason = "no_arm_joint_indices_available"
        elif (
            measured_for_hold.ndim != 1
            or len(measured_for_hold) != len(names)
            or not np.all(np.isfinite(measured_for_hold[arm_indices]))
        ):
            abort_recovery_reason = (
                "measured_arm_positions_invalid_no_new_command_applied"
            )
        else:
            hold_joint_positions(robot, arm_indices)
            held_targets = {
                name: float(measured_for_hold[names.index(name)])
                for name in arm_joint_names
            }
            recorder.arm_last_command_by_name = held_targets
            recorder.arm_tracking_error_consecutive_exceeded_steps = 0
            recorder.arm_tracking_last_observation_sim_step = (
                recorder.sim_step
            )
            abort_recovery_applied = True
            abort_recovery_reason = (
                "held_finite_measured_arm_articulation"
            )
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
        "peak_abs_measured_arm_effort_joint": peak_arm_effort_joint,
        "peak_abs_measured_arm_effort_by_joint": {
            name: round(value, 6)
            for name, value in peak_arm_effort_by_joint.items()
        },
        "initial_measured_arm_effort_by_joint": (
            None
            if initial_effort_record is None
            else initial_effort_record.get("arm_effort_by_name")
        ),
        "initial_measured_spine_force_newtons": (
            None
            if initial_effort_record is None
            else initial_effort_record.get("spine_force_newtons")
        ),
        "peak_abs_measured_spine_force_newtons": round(
            peak_abs_spine_force, 6
        ),
        "arm_effort_sample_count": arm_effort_samples,
        "spine_force_sample_count": spine_force_samples,
        "effort_telemetry_initially_available": (
            effort_telemetry_initially_available
        ),
        "effort_telemetry_nonfinite": effort_telemetry_nonfinite,
        "arm_effort_abort_threshold": ARGS.arm_effort_abort,
        "spine_force_abort_threshold_newtons": None,
        "spine_force_policy": (
            "finite telemetry required; measured prismatic force is recorded "
            "separately and is not compared with the revolute-arm threshold"
        ),
        "initial_command_baseline_by_joint": {
            name: (
                round(value, 6) if math.isfinite(value) else None
            )
            for name, value in initial_command_baseline_targets.items()
        },
        "initial_command_baseline_source": initial_command_baseline_source,
        "command_slew_check_count": command_slew_check_count,
        "first_command_slew_check": first_command_slew_check,
        "worst_command_slew_check": worst_command_slew_check,
        "command_slew_violation": command_slew_violation,
        "arm_max_command_slew_rad": ARGS.arm_max_command_slew_rad,
        "final_active_arm_command_by_joint": (
            None
            if recorder.arm_last_command_by_name is None
            else {
                name: round(float(value), 6)
                for name, value in (
                    recorder.arm_last_command_by_name.items()
                )
            }
        ),
        "tracking_error_check_count": tracking_error_check_count,
        "first_tracking_error_check": first_tracking_error_check,
        "worst_tracking_error_check": worst_tracking_error_check,
        "tracking_error_dwell_violation": (
            tracking_error_dwell_violation
        ),
        "tracking_error_threshold_exceedance_count": (
            tracking_error_threshold_exceedance_count
        ),
        "tracking_error_initial_consecutive_exceeded_steps": (
            initial_tracking_error_consecutive_exceeded_steps
        ),
        "tracking_error_observation_gap_reset": (
            tracking_error_observation_gap_reset
        ),
        "tracking_error_final_consecutive_exceeded_steps": (
            tracking_error_consecutive_exceeded_steps
        ),
        "tracking_error_max_consecutive_exceeded_steps": (
            max_tracking_error_consecutive_exceeded_steps
        ),
        "arm_tracking_error_threshold_rad": (
            ARGS.arm_tracking_error_threshold_rad
        ),
        "arm_tracking_error_dwell_seconds_configured": (
            ARGS.arm_tracking_error_dwell_seconds
        ),
        "arm_tracking_error_dwell_steps": tracking_error_dwell_steps,
        "arm_tracking_error_dwell_seconds_effective": round(
            tracking_error_dwell_steps / float(ARGS.physics_hz), 6
        ),
        "arm_tracking_error_calibration": {
            "scope": "participant_side_safety_gate",
            "threshold_basis": (
                "independent target-to-measured error limit, not the "
                "adjacent-command slew limit"
            ),
            "dwell_basis": (
                "configured seconds converted upward to whole physics steps; "
                "an observation gap resets the consecutive counter"
            ),
            "official_benchmark_threshold": False,
        },
        "joint_command_safety_policy": (
            "proposed targets are compared with the previous applied command "
            "for slew; the active command is compared with measured joints "
            "for sustained tracking error; commands are never clamped"
        ),
        "command_clamping_used": False,
        "abort_recovery_applied": abort_recovery_applied,
        "abort_recovery_reason": abort_recovery_reason,
        "abort_recovery_policy": (
            "hold finite measured arm articulation; if measurement is "
            "invalid, do not issue a new command; never release automatically"
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


def prepare_navigation_posture(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
) -> tuple[list[dict[str, Any]], CollisionProxySet]:
    """Lift, retract, settle, then capture posture-specific collision proxies."""

    gates: list[dict[str, Any]] = []
    base_position, base_orientation = world_pose(robot)
    current_left, current_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine_position(robot),
    )
    lift_height = max(
        ARGS.navigation_stow_height,
        float(current_left[0][2]),
        float(current_right[0][2]),
    )
    left_lift = np.asarray(
        (
            current_left[0][0],
            current_left[0][1],
            lift_height,
        ),
        dtype=np.float64,
    )
    right_lift = np.asarray(
        (
            current_right[0][0],
            current_right[0][1],
            lift_height,
        ),
        dtype=np.float64,
    )
    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=left_lift,
            right_target=right_lift,
            left_orientation=None,
            right_orientation=None,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="navigation_stow_lift",
            intent=(
                "Raise both empty grippers vertically before retracting them "
                "into the mobile-base navigation envelope."
            ),
        )
    )
    if not gates[-1]["passed"]:
        return gates, CollisionProxySet(
            robot=(),
            environment=(),
            support_surface_paths=(),
            candidate_robot_paths=(),
            candidate_environment_paths=(),
            unresolved_paths=("navigation stow lift failed",),
            traversal_backend="not_attempted",
        )

    base_position, base_orientation = world_pose(robot)
    yaw = yaw_from_wxyz(base_orientation)

    def target_from_body(forward: float, lateral: float) -> np.ndarray:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return np.asarray(
            (
                base_position[0] + cosine * forward - sine * lateral,
                base_position[1] + sine * forward + cosine * lateral,
                lift_height,
            ),
            dtype=np.float64,
        )

    gates.append(
        move_tcp_pose(
            recorder,
            robot,
            ik,
            left_target=target_from_body(
                ARGS.navigation_stow_forward,
                ARGS.navigation_stow_lateral,
            ),
            right_target=target_from_body(
                ARGS.navigation_stow_forward,
                -ARGS.navigation_stow_lateral,
            ),
            left_orientation=None,
            right_orientation=None,
            steps=ARGS.motion_steps,
            settle_steps=ARGS.motion_settle_steps,
            phase="navigation_stow_retract",
            intent=(
                "Retract both raised empty grippers to a symmetric compact "
                "navigation posture."
            ),
        )
    )
    if not gates[-1]["passed"]:
        return gates, CollisionProxySet(
            robot=(),
            environment=(),
            support_surface_paths=(),
            candidate_robot_paths=(),
            candidate_environment_paths=(),
            unresolved_paths=("navigation stow retraction failed",),
            traversal_backend="not_attempted",
        )

    base_position, _ = world_pose(robot)
    root_path = core._find_articulation_root_path(ROBOT_PRIM_PATH)
    try:
        collision_proxies = extract_collision_proxy_set(
            recorder.stage,
            robot_path=ROBOT_PRIM_PATH,
            base_frame_path=root_path,
            base_world_z=float(base_position[2]),
        )
    except Exception as error:  # noqa: BLE001
        collision_proxies = CollisionProxySet(
            robot=(),
            environment=(),
            support_surface_paths=(),
            candidate_robot_paths=(),
            candidate_environment_paths=(),
            unresolved_paths=(
                f"extractor [{type(error).__name__}: {error}]",
            ),
            traversal_backend="failed",
        )
    gates.append(
        {
            "phase": "navigation_collision_proxy_coverage",
            "classification": (
                "posture_specific_enabled_collider_coverage_gate"
            ),
            "passed": collision_proxies.complete,
            "post_stow_proxy_coverage": collision_proxy_coverage_summary(
                collision_proxies
            ),
            "collision_proxy_inventory": collision_proxies.record(),
            "pass_definition": (
                "every enabled robot and environment collision prim has a "
                "finite posture-specific proxy and one support surface is "
                "explicitly identified"
            ),
        }
    )
    return gates, collision_proxies


def capture_current_navigation_posture(
    recorder: TraceRecorder,
) -> tuple[list[dict[str, Any]], CollisionProxySet]:
    """Capture current collision geometry without an articulation command."""

    base_position, _ = world_pose(recorder.robot)
    root_path = core._find_articulation_root_path(ROBOT_PRIM_PATH)
    try:
        collision_proxies = extract_collision_proxy_set(
            recorder.stage,
            robot_path=ROBOT_PRIM_PATH,
            base_frame_path=root_path,
            base_world_z=float(base_position[2]),
        )
    except Exception as error:  # noqa: BLE001
        collision_proxies = CollisionProxySet(
            robot=(),
            environment=(),
            support_surface_paths=(),
            candidate_robot_paths=(),
            candidate_environment_paths=(),
            unresolved_paths=(
                f"extractor [{type(error).__name__}: {error}]",
            ),
            traversal_backend="failed",
        )
    gate = {
        "phase": "current_asset_posture_collision_proxy_coverage",
        "classification": "read_only_enabled_collider_coverage_gate",
        "passed": collision_proxies.complete,
        "current_posture_proxy_coverage": collision_proxy_coverage_summary(
            collision_proxies,
            capture_label="current_asset_posture_no_command",
        ),
        "collision_proxy_inventory": collision_proxies.record(),
        "motion_scope": (
            "no base, arm, gripper, or task-object command was issued"
        ),
        "pass_definition": (
            "every enabled robot and environment collision prim in the "
            "unmodified current asset posture has a finite proxy and one "
            "support surface is explicitly identified"
        ),
    }
    return [gate], collision_proxies


def collision_proxy_coverage_summary(
    collision_proxies: CollisionProxySet,
    *,
    capture_label: str = "compact_navigation_stow",
) -> dict[str, Any]:
    """Return concise coverage evidence for a posture-specific proxy set."""

    return {
        "capture_posture": capture_label,
        "complete": collision_proxies.complete,
        "traversal_backend": collision_proxies.traversal_backend,
        "robot_proxy_count": len(collision_proxies.robot),
        "candidate_robot_collision_prim_count": len(
            collision_proxies.candidate_robot_paths
        ),
        "environment_proxy_count": len(collision_proxies.environment),
        "candidate_environment_collision_prim_count": len(
            collision_proxies.candidate_environment_paths
        ),
        "robot_geometry_witness_count": len(
            collision_proxies.robot_geometry_witnesses
        ),
        "environment_geometry_witness_count": len(
            collision_proxies.environment_geometry_witnesses
        ),
        "support_surface_count": len(
            collision_proxies.support_surface_paths
        ),
        "support_surface_paths": list(
            collision_proxies.support_surface_paths
        ),
        "unresolved_path_count": len(collision_proxies.unresolved_paths),
        "attached_payload_root_paths": list(
            collision_proxies.attached_payload_root_paths
        ),
        "attached_payload_proxy_count": len(
            collision_proxies.attached_payload_proxy_paths
        ),
        "missing_attached_payload_root_paths": list(
            collision_proxies.missing_attached_payload_root_paths
        ),
    }


def payload_positions_in_base_frame(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    object_names: Iterable[str],
) -> dict[str, tuple[float, float, float]]:
    """Measure selected task-object origins in the current base-yaw frame."""

    base_position, base_orientation = world_pose(robot)
    yaw = yaw_from_wxyz(base_orientation)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    result: dict[str, tuple[float, float, float]] = {}
    for name in object_names:
        path = recorder.object_paths[name]
        prim = recorder.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"invalid monitored payload prim: {path}")
        translation = xform_cache.GetLocalToWorldTransform(
            prim
        ).ExtractTranslation()
        dx = float(translation[0]) - float(base_position[0])
        dy = float(translation[1]) - float(base_position[1])
        dz = float(translation[2]) - float(base_position[2])
        relative = (
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            dz,
        )
        if not all(math.isfinite(value) for value in relative):
            raise RuntimeError(
                f"non-finite monitored payload pose: {path}"
            )
        result[name] = relative
    return result


def drive_base_to(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    steering_ids: list[int],
    drive_ids: list[int],
    *,
    certified_waypoints: tuple[Pose2, ...],
    route_certificate: dict[str, Any],
    collision_proxies: CollisionProxySet,
    dynamic_geometry_certificate_allowance_metres: float,
    dynamic_geometry_operational_limit_metres: float,
    segment_index: int,
    max_steps: int,
    max_speed: float,
    max_accel: float,
    phase: str,
    intent: str,
    payload_reference_by_name: (
        dict[str, tuple[float, float, float]] | None
    ) = None,
    payload_drift_tolerance_metres: float | None = None,
) -> dict[str, Any]:
    recorder.set_phase(phase, intent=intent)
    controller = robot.get_articulation_controller()
    position_tolerance = (
        ARGS.route_operational_cross_track_tolerance
    )
    yaw_tolerance = math.radians(
        ARGS.route_operational_yaw_tolerance_deg
    )
    certificate_yaw_tolerance = math.radians(
        ARGS.route_yaw_tracking_tolerance_deg
    )
    target_xy = np.zeros(2, dtype=np.float64)
    target_yaw = 0.0
    segment_route = (
        Pose2(0.0, 0.0, 0.0),
        Pose2(0.0, 0.0, 0.0),
    )
    runtime_route_sha256: str | None = None
    certificate_route_sha256 = route_certificate.get(
        "validated_inputs", {}
    ).get("route_sha256")
    certificate_sha256 = route_certificate.get("certificate_sha256")
    certificate_binding_error: str | None = None
    initial_failure_classification = "route_certificate_binding_violation"
    maximum_robot_planar_radius = 0.0
    maximum_robot_spatial_radius = 0.0
    certificate_base_z = 0.0
    dynamic_certificate_allowance = 0.0
    dynamic_operational_limit = 0.0
    collision_base_frame_path: str | None = None
    payload_monitor_enabled = (
        payload_reference_by_name is not None
        or payload_drift_tolerance_metres is not None
    )
    normalized_payload_reference: dict[
        str, tuple[float, float, float]
    ] = {}
    normalized_payload_tolerance: float | None = None
    try:
        if not bool(route_certificate.get("passed", False)):
            raise ValueError("route certificate is not passing")
        if not route_certificate_is_valid(route_certificate):
            raise ValueError(
                "route certificate digest or deterministic replay is invalid"
            )
        if not collision_proxies.complete:
            raise ValueError(
                "live collision geometry capture is incomplete"
            )
        collision_base_frame_path = core._find_articulation_root_path(
            ROBOT_PRIM_PATH
        )
        dynamic_certificate_allowance = float(
            dynamic_geometry_certificate_allowance_metres
        )
        dynamic_operational_limit = float(
            dynamic_geometry_operational_limit_metres
        )
        if (
            not math.isfinite(dynamic_certificate_allowance)
            or not math.isfinite(dynamic_operational_limit)
            or dynamic_operational_limit < 0.0
            or dynamic_certificate_allowance
            - dynamic_operational_limit
            < 0.02
        ):
            raise ValueError(
                "dynamic geometry limits require a finite non-negative "
                "operational limit and at least 0.02m reaction reserve"
            )
        if (
            not isinstance(segment_index, int)
            or isinstance(segment_index, bool)
            or segment_index < 0
            or segment_index + 1 >= len(certified_waypoints)
        ):
            raise ValueError("certified segment index is out of range")
        runtime_route_sha256 = route_sha256(certified_waypoints)
        if runtime_route_sha256 != certificate_route_sha256:
            raise ValueError(
                "runtime route does not match the certificate route digest"
            )
        validated_inputs = route_certificate.get("validated_inputs", {})
        if (
            validated_inputs.get("robot_geometry_sha256")
            != proxy_geometry_sha256(collision_proxies.robot)
            or validated_inputs.get("obstacle_geometry_sha256")
            != proxy_geometry_sha256(collision_proxies.environment)
        ):
            raise ValueError(
                "runtime collision proxy geometry does not match the "
                "certificate inputs"
            )
        certified_clearance = float(
            route_certificate.get("config", {}).get(
                "clearance_margin"
            )
        )
        certificate_base_z = float(
            route_certificate.get("config", {}).get("base_z")
        )
        if not math.isfinite(certificate_base_z):
            raise ValueError("certificate base z is non-finite")
        if not math.isclose(
            certified_clearance,
            ARGS.route_clearance_margin
            + dynamic_certificate_allowance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "route certificate does not bind the requested dynamic "
                "geometry allowance"
            )
        envelope = route_certificate.get("execution_envelope", {})
        if not math.isclose(
            float(envelope.get("translation_tolerance_metres")),
            ARGS.route_cross_track_tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "certificate translation reserve differs from runtime limit"
            )
        if not math.isclose(
            float(envelope.get("yaw_tolerance_rad")),
            certificate_yaw_tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "certificate yaw reserve differs from runtime limit"
            )
        maximum_robot_planar_radius = float(
            envelope.get("maximum_robot_planar_radius_metres")
        )
        maximum_robot_spatial_radius = float(
            envelope.get("maximum_robot_spatial_radius_metres")
        )
        if (
            not math.isfinite(maximum_robot_planar_radius)
            or maximum_robot_planar_radius < 0.0
            or not math.isfinite(maximum_robot_spatial_radius)
            or maximum_robot_spatial_radius
            < maximum_robot_planar_radius
        ):
            raise ValueError(
                "certificate robot radii are invalid"
            )
        segment_route = (
            certified_waypoints[segment_index],
            certified_waypoints[segment_index + 1],
        )
        target_xy = np.asarray(
            (segment_route[1].x, segment_route[1].y),
            dtype=np.float64,
        )
        target_yaw = float(segment_route[1].yaw)
        if payload_monitor_enabled:
            if (
                payload_reference_by_name is None
                or payload_drift_tolerance_metres is None
                or not payload_reference_by_name
            ):
                raise ValueError(
                    "payload monitoring requires a non-empty reference and "
                    "a drift tolerance"
                )
            normalized_payload_tolerance = float(
                payload_drift_tolerance_metres
            )
            if (
                not math.isfinite(normalized_payload_tolerance)
                or normalized_payload_tolerance <= 0.0
            ):
                raise ValueError(
                    "payload drift tolerance must be finite and positive"
                )
            normalized_payload_reference = {
                str(name): tuple(float(value) for value in values)
                for name, values in payload_reference_by_name.items()
            }
            if any(
                len(values) != 3
                or not all(math.isfinite(value) for value in values)
                for values in normalized_payload_reference.values()
            ):
                raise ValueError(
                    "payload reference positions must be finite 3-D values"
                )
            required_clearance = (
                ARGS.route_clearance_margin
                + dynamic_certificate_allowance
            )
            if not math.isclose(
                certified_clearance,
                required_clearance,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "route certificate does not bind the loaded geometry "
                    "allowance"
                )
        if not BASE_MOTION_STOPPING_ENVELOPE_CERTIFIED:
            initial_failure_classification = (
                "base_motion_stopping_envelope_uncertified"
            )
            raise RuntimeError(BASE_MOTION_STOPPING_ENVELOPE_LIMITATION)
    except Exception as error:  # noqa: BLE001 - fail before wheel motion
        certificate_binding_error = (
            f"{type(error).__name__}: {error}"
        )
    dt = 1.0 / recorder.physics_hz
    command = np.zeros(3, dtype=np.float64)
    settled_steps = 0
    start_step = recorder.sim_step
    path_length = 0.0
    previous_position, _ = world_pose(robot)
    steps = 0
    aborted = certificate_binding_error is not None
    abort_reason: str | None = certificate_binding_error
    abort_classification: str | None = (
        initial_failure_classification
        if certificate_binding_error is not None
        else None
    )
    drive_indices = np.asarray(drive_ids, dtype=np.int64)
    steering_indices = np.asarray(steering_ids, dtype=np.int64)
    try:
        steering_alignment_error = float(
            core.STEERING_FULL_SPEED_ERROR_RAD
        )
    except AttributeError:
        steering_alignment_error = math.radians(8.0)
    motion_monitor = BaseMotionMonitor(
        physics_hz=recorder.physics_hz,
        minimum_planar_command_speed=ARGS.base_min_speed,
        steering_alignment_error_rad=steering_alignment_error,
        steering_timeout_seconds=ARGS.base_steering_timeout_seconds,
        stall_grace_seconds=ARGS.base_stall_grace_seconds,
        stall_window_seconds=ARGS.base_stall_seconds,
        stall_distance_metres=ARGS.base_stall_distance,
        measured_drive_response_rad_s=ARGS.base_drive_response_rad_s,
    )
    minimum_speed_floor_engagement_steps = 0
    minimum_speed_floor_first_step: int | None = None
    minimum_speed_floor_last_step: int | None = None
    steering_alignment_wait_steps = 0
    steering_alignment_first_step: int | None = None
    stall_monitor_armed_first_step: int | None = None
    max_abs_steering_error_degrees = 0.0
    max_abs_drive_target_rad_s = 0.0
    max_abs_measured_drive_velocity_rad_s = 0.0
    peak_abs_measured_drive_effort = 0.0
    drive_effort_sample_count = 0
    first_nonzero_wheel_command_sim_step: int | None = None
    base_telemetry_nonfinite = False
    final_window_displacement: float | None = None
    maximum_cross_track_error = 0.0
    maximum_route_yaw_error = 0.0
    maximum_runtime_yaw_arc = 0.0
    maximum_payload_to_base_drift = 0.0
    live_geometry_check_count = 0
    live_geometry_post_stop_check_count = 0
    maximum_live_geometry_planar_escape = 0.0
    maximum_live_geometry_z_escape = 0.0
    maximum_nonplanar_base_escape = 0.0
    maximum_combined_world_geometry_escape = 0.0
    first_live_geometry_failure: dict[str, Any] | None = None
    final_live_geometry_record: dict[str, Any] | None = None

    def compact_live_geometry_record(
        record: dict[str, Any],
        *,
        timing: str,
    ) -> dict[str, Any]:
        return {
            "timing": timing,
            "simulation_step": recorder.sim_step,
            "passed": bool(record.get("passed", False)),
            "failure": record.get("failure"),
            "reason": record.get("reason"),
            "dynamic_allowance_metres": record.get(
                "dynamic_allowance_metres"
            ),
            "inventory_exact": record.get("inventory_exact"),
            "maximum_planar_escape_metres": record.get(
                "maximum_planar_escape_metres"
            ),
            "maximum_z_escape_metres": record.get(
                "maximum_z_escape_metres"
            ),
            "nonplanar_base": record.get("nonplanar_base"),
            "combined_world_escape_metres": record.get(
                "combined_world_escape_metres"
            ),
            "missing_proxy_paths": record.get("missing_proxy_paths", []),
            "unexpected_proxy_paths": record.get(
                "unexpected_proxy_paths", []
            ),
            "missing_witness_keys": record.get(
                "missing_witness_keys", []
            ),
            "unexpected_witness_keys": record.get(
                "unexpected_witness_keys", []
            ),
            "unresolved_live_geometry": record.get(
                "unresolved_live_geometry", []
            ),
        }

    def measure_live_collision_geometry(
        allowance: float,
        *,
        timing: str,
        post_stop: bool = False,
    ) -> dict[str, Any]:
        nonlocal live_geometry_check_count
        nonlocal live_geometry_post_stop_check_count
        nonlocal maximum_live_geometry_planar_escape
        nonlocal maximum_live_geometry_z_escape
        nonlocal maximum_nonplanar_base_escape
        nonlocal maximum_combined_world_geometry_escape
        nonlocal first_live_geometry_failure
        nonlocal final_live_geometry_record
        if collision_base_frame_path is None:
            record = {
                "passed": False,
                "failure": "route_certificate_binding_violation",
                "reason": "collision base frame was not authenticated",
                "dynamic_allowance_metres": allowance,
            }
        else:
            record = live_collision_geometry_containment_record(
                recorder.stage,
                collision_proxies,
                robot_path=ROBOT_PRIM_PATH,
                base_frame_path=collision_base_frame_path,
                dynamic_allowance=allowance,
            )
            base_position, base_orientation = world_pose(robot)
            nonplanar = base_nonplanar_deviation_record(
                position_z=float(base_position[2]),
                orientation_wxyz=tuple(
                    float(value) for value in base_orientation
                ),
                reference_z=certificate_base_z,
                maximum_robot_radius=maximum_robot_spatial_radius,
                tolerance=allowance,
            )
            live_planar_escape = record.get(
                "maximum_planar_escape_metres"
            )
            live_z_escape = record.get("maximum_z_escape_metres")
            base_escape = nonplanar.get("combined_escape_metres")
            components_valid = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (
                    live_planar_escape,
                    live_z_escape,
                    base_escape,
                )
            )
            combined_world_escape = (
                max(
                    float(live_planar_escape),
                    float(live_z_escape),
                )
                + float(base_escape)
                if components_valid
                else None
            )
            combined_within_allowance = bool(
                combined_world_escape is not None
                and combined_world_escape <= allowance + 1e-12
            )
            record = {
                **record,
                "nonplanar_base": nonplanar,
                "combined_world_escape_metres": combined_world_escape,
                "passed": bool(
                    record.get("passed", False)
                    and nonplanar.get("passed", False)
                    and combined_within_allowance
                ),
            }
            if not bool(nonplanar.get("passed", False)):
                record["failure"] = nonplanar.get("failure")
                record["reason"] = nonplanar.get("reason")
            elif (
                bool(record.get("robot_geometry"))
                and bool(record.get("environment_geometry"))
                and not combined_within_allowance
            ):
                record["failure"] = (
                    "combined_world_geometry_envelope_violation"
                )
                record["reason"] = (
                    "summed live relative-geometry and non-planar base "
                    "escape exceeded the single dynamic allowance"
                )
        live_geometry_check_count += 1
        if post_stop:
            live_geometry_post_stop_check_count += 1
        planar_escape = record.get("maximum_planar_escape_metres")
        z_escape = record.get("maximum_z_escape_metres")
        if isinstance(planar_escape, (int, float)) and math.isfinite(
            float(planar_escape)
        ):
            maximum_live_geometry_planar_escape = max(
                maximum_live_geometry_planar_escape,
                float(planar_escape),
            )
        if isinstance(z_escape, (int, float)) and math.isfinite(
            float(z_escape)
        ):
            maximum_live_geometry_z_escape = max(
                maximum_live_geometry_z_escape,
                float(z_escape),
            )
        nonplanar_record = record.get("nonplanar_base")
        if isinstance(nonplanar_record, dict):
            combined_escape = nonplanar_record.get(
                "combined_escape_metres"
            )
            if isinstance(combined_escape, (int, float)) and math.isfinite(
                float(combined_escape)
            ):
                maximum_nonplanar_base_escape = max(
                    maximum_nonplanar_base_escape,
                    float(combined_escape),
                )
        combined_world_escape = record.get(
            "combined_world_escape_metres"
        )
        if isinstance(
            combined_world_escape, (int, float)
        ) and math.isfinite(float(combined_world_escape)):
            maximum_combined_world_geometry_escape = max(
                maximum_combined_world_geometry_escape,
                float(combined_world_escape),
            )
        compact = compact_live_geometry_record(record, timing=timing)
        final_live_geometry_record = compact
        if not compact["passed"] and first_live_geometry_failure is None:
            first_live_geometry_failure = compact
        return record

    def measure_payload_to_base_drift() -> float:
        nonlocal maximum_payload_to_base_drift
        if not payload_monitor_enabled:
            return 0.0
        measured = payload_positions_in_base_frame(
            recorder,
            robot,
            normalized_payload_reference,
        )
        if set(measured) != set(normalized_payload_reference):
            raise RuntimeError(
                "monitored payload name set changed during transport"
            )
        maximum = max(
            math.dist(measured[name], normalized_payload_reference[name])
            for name in measured
        )
        if not math.isfinite(maximum):
            raise RuntimeError("payload drift measurement is non-finite")
        maximum_payload_to_base_drift = max(
            maximum_payload_to_base_drift, maximum
        )
        return maximum

    try:
        for steps in range(1, max_steps + 1):
            if aborted:
                break
            position, orientation = world_pose(robot)
            path_length += float(
                np.linalg.norm(position[:2] - previous_position[:2])
            )
            previous_position = position
            yaw = yaw_from_wxyz(orientation)
            if route_sha256(certified_waypoints) != certificate_route_sha256:
                aborted = True
                abort_classification = "route_certificate_binding_violation"
                abort_reason = (
                    "certified route digest changed before a wheel command"
                )
                break
            tube_record = route_execution_tube_record(
                Pose2(
                    float(position[0]),
                    float(position[1]),
                    yaw,
                ),
                segment_route,
                translation_tolerance=(
                    ARGS.route_operational_cross_track_tolerance
                ),
                yaw_tolerance_rad=yaw_tolerance,
            )
            deviation = tube_record["deviation"]
            if deviation is None:
                aborted = True
                abort_classification = str(tube_record["failure"])
                abort_reason = str(tube_record["reason"])
                break
            cross_track_error = float(
                deviation["cross_track_metres"]
            )
            route_yaw_error = float(deviation["yaw_error_rad"])
            route_yaw_arc = (
                2.0
                * maximum_robot_planar_radius
                * math.sin(0.5 * route_yaw_error)
            )
            maximum_cross_track_error = max(
                maximum_cross_track_error, cross_track_error
            )
            maximum_route_yaw_error = max(
                maximum_route_yaw_error, route_yaw_error
            )
            maximum_runtime_yaw_arc = max(
                maximum_runtime_yaw_arc, route_yaw_arc
            )
            if not tube_record["passed"]:
                aborted = True
                abort_classification = str(tube_record["failure"])
                abort_reason = (
                    "measured base pose left the certified SE(2) tube before "
                    "a wheel command"
                )
                break
            try:
                payload_drift = measure_payload_to_base_drift()
            except Exception as error:  # noqa: BLE001
                aborted = True
                abort_classification = "payload_telemetry_unavailable"
                abort_reason = (
                    "payload-to-base monitoring failed before a wheel command: "
                    f"{type(error).__name__}: {error}"
                )
                break
            if (
                normalized_payload_tolerance is not None
                and payload_drift > normalized_payload_tolerance
            ):
                aborted = True
                abort_classification = (
                    "payload_geometry_envelope_violation"
                )
                abort_reason = (
                    "measured payload-to-base drift exceeded the clearance "
                    "reserved by the loaded route certificate before a wheel "
                    "command"
                )
                break
            live_geometry_before = measure_live_collision_geometry(
                dynamic_operational_limit,
                timing="immediately_before_physics_step",
            )
            if not bool(live_geometry_before.get("passed", False)):
                aborted = True
                abort_classification = str(
                    live_geometry_before.get("failure")
                    or "dynamic_geometry_envelope_violation"
                )
                abort_reason = (
                    "live articulated robot/payload collision geometry left "
                    "the operational containment envelope before a wheel "
                    "command: "
                    + str(live_geometry_before.get("reason"))
                )
                break
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
            joint_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            joint_velocities = robot.get_joint_velocities()
            efforts = measured_efforts(robot)
            if joint_velocities is None:
                aborted = True
                abort_classification = "base_telemetry_unavailable"
                abort_reason = (
                    "measured base joint velocity telemetry unavailable"
                )
                break
            joint_velocities = np.asarray(
                joint_velocities, dtype=np.float64
            )
            if (
                joint_velocities.ndim != 1
                or not np.all(
                    np.isfinite(
                        joint_velocities[
                            np.concatenate(
                                (steering_indices, drive_indices)
                            )
                        ]
                    )
                )
                or efforts is None
                or not np.all(
                    np.isfinite(
                        efforts[
                            np.concatenate(
                                (steering_indices, drive_indices)
                            )
                        ]
                    )
                )
            ):
                base_telemetry_nonfinite = True
                aborted = True
                abort_classification = "base_telemetry_unavailable"
                abort_reason = (
                    "finite steering, wheel, and effort telemetry is required"
                )
                break
            steering_targets, drive_targets = core._compute_drive_targets(
                joint_positions,
                steering_ids,
                float(command[0]),
                float(command[1]),
                float(command[2]),
            )
            measured_steering = joint_positions[steering_indices]
            steering_errors = np.asarray(
                [
                    wrap_to_pi(float(target) - float(measured))
                    for target, measured in zip(
                        steering_targets, measured_steering
                    )
                ],
                dtype=np.float64,
            )
            measured_drive_velocities = joint_velocities[drive_indices]
            measured_drive_efforts = efforts[drive_indices]
            max_abs_steering_error_degrees = max(
                max_abs_steering_error_degrees,
                math.degrees(
                    float(np.max(np.abs(steering_errors)))
                ),
            )
            max_abs_drive_target_rad_s = max(
                max_abs_drive_target_rad_s,
                float(np.max(np.abs(drive_targets))),
            )
            if (
                first_nonzero_wheel_command_sim_step is None
                and float(np.max(np.abs(drive_targets))) > 1e-9
            ):
                first_nonzero_wheel_command_sim_step = recorder.sim_step
            max_abs_measured_drive_velocity_rad_s = max(
                max_abs_measured_drive_velocity_rad_s,
                float(np.max(np.abs(measured_drive_velocities))),
            )
            peak_abs_measured_drive_effort = max(
                peak_abs_measured_drive_effort,
                float(np.max(np.abs(measured_drive_efforts))),
            )
            drive_effort_sample_count += 1
            recorder.set_base_control(
                {
                    "measurement_timing": "immediately_before_physics_step",
                    "target_xy": rounded(target_xy),
                    "world_position_error": rounded(world_error),
                    "body_position_error": rounded(body_error),
                    "distance_error_metres": round(distance, 6),
                    "yaw_error_degrees": round(
                        math.degrees(yaw_error), 6
                    ),
                    "desired_body_twist": rounded(desired),
                    "limited_body_twist": rounded(command),
                    "steering_measured_rad": rounded(
                        measured_steering
                    ),
                    "steering_target_rad": rounded(
                        steering_targets
                    ),
                    "steering_error_degrees": rounded(
                        np.degrees(steering_errors)
                    ),
                    "drive_target_rad_s": rounded(drive_targets),
                    "drive_measured_velocity_rad_s": rounded(
                        measured_drive_velocities
                    ),
                    "drive_measured_effort": rounded(
                        measured_drive_efforts
                    ),
                    "external_contact_telemetry": "unavailable",
                }
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
            live_geometry_after = measure_live_collision_geometry(
                dynamic_operational_limit,
                timing="immediately_after_physics_step",
            )
            if not bool(live_geometry_after.get("passed", False)):
                aborted = True
                abort_classification = str(
                    live_geometry_after.get("failure")
                    or "dynamic_geometry_envelope_violation"
                )
                abort_reason = (
                    "live articulated robot/payload collision geometry left "
                    "the operational containment envelope after a physics "
                    "step: "
                    + str(live_geometry_after.get("reason"))
                )
                break
            position_after, orientation_after = world_pose(robot)
            tube_record_after = route_execution_tube_record(
                Pose2(
                    float(position_after[0]),
                    float(position_after[1]),
                    yaw_from_wxyz(orientation_after),
                ),
                segment_route,
                translation_tolerance=(
                    ARGS.route_operational_cross_track_tolerance
                ),
                yaw_tolerance_rad=yaw_tolerance,
            )
            deviation_after = tube_record_after["deviation"]
            if deviation_after is None:
                aborted = True
                abort_classification = str(
                    tube_record_after["failure"]
                )
                abort_reason = str(tube_record_after["reason"])
                break
            cross_track_error = float(
                deviation_after["cross_track_metres"]
            )
            route_yaw_error = float(
                deviation_after["yaw_error_rad"]
            )
            route_yaw_arc = (
                2.0
                * maximum_robot_planar_radius
                * math.sin(0.5 * route_yaw_error)
            )
            maximum_cross_track_error = max(
                maximum_cross_track_error, cross_track_error
            )
            maximum_route_yaw_error = max(
                maximum_route_yaw_error, route_yaw_error
            )
            maximum_runtime_yaw_arc = max(
                maximum_runtime_yaw_arc, route_yaw_arc
            )
            if not tube_record_after["passed"]:
                aborted = True
                abort_classification = str(tube_record_after["failure"])
                abort_reason = (
                    "measured base pose left the certified SE(2) tube"
                )
                break
            try:
                payload_drift = measure_payload_to_base_drift()
            except Exception as error:  # noqa: BLE001
                aborted = True
                abort_classification = "payload_telemetry_unavailable"
                abort_reason = (
                    "payload-to-base monitoring failed after a wheel command: "
                    f"{type(error).__name__}: {error}"
                )
                break
            if (
                normalized_payload_tolerance is not None
                and payload_drift > normalized_payload_tolerance
            ):
                aborted = True
                abort_classification = (
                    "payload_geometry_envelope_violation"
                )
                abort_reason = (
                    "measured payload-to-base drift exceeded the clearance "
                    "reserved by the loaded route certificate"
                )
                break
            joint_positions_after = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            joint_velocities_after = robot.get_joint_velocities()
            efforts_after = measured_efforts(robot)
            if joint_velocities_after is None or efforts_after is None:
                aborted = True
                abort_classification = "base_telemetry_unavailable"
                abort_reason = (
                    "base telemetry disappeared after the physics step"
                )
                break
            joint_velocities_after = np.asarray(
                joint_velocities_after, dtype=np.float64
            )
            measured_steering_after = joint_positions_after[
                steering_indices
            ]
            steering_errors_after = tuple(
                wrap_to_pi(float(target) - float(measured))
                for target, measured in zip(
                    steering_targets, measured_steering_after
                )
            )
            decision = motion_monitor.update(
                BaseMotionObservation(
                    position_xy=(
                        float(position_after[0]),
                        float(position_after[1]),
                    ),
                    planar_command_speed=float(
                        np.linalg.norm(command[:2])
                    ),
                    steering_errors_rad=steering_errors_after,
                    drive_targets_rad_s=tuple(
                        float(value) for value in drive_targets
                    ),
                    measured_drive_velocities_rad_s=tuple(
                        float(value)
                        for value in joint_velocities_after[drive_indices]
                    ),
                    measured_drive_efforts=tuple(
                        float(value)
                        for value in efforts_after[drive_indices]
                    ),
                    external_contact_observed=None,
                )
            )
            steering_alignment_wait_steps = max(
                steering_alignment_wait_steps,
                decision.steering_wait_steps,
            )
            if (
                decision.steering_aligned
                and decision.drive_authorized
                and steering_alignment_first_step is None
            ):
                steering_alignment_first_step = steps
            if (
                decision.stall_monitor_armed
                and stall_monitor_armed_first_step is None
            ):
                stall_monitor_armed_first_step = steps
            final_window_displacement = (
                decision.window_displacement_metres
            )
            if decision.abort_classification is not None:
                aborted = True
                abort_classification = decision.abort_classification
                abort_reason = decision.abort_reason
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
        recorder.set_base_control(
            {
                "measurement_timing": "post_command_stop",
                "limited_body_twist": [0.0, 0.0, 0.0],
                "drive_target_rad_s": [
                    0.0 for _ in drive_ids
                ],
                "external_contact_telemetry": "unavailable",
            }
        )
    post_stop_route_check_count = 0
    post_stop_route_failure: dict[str, Any] | None = None
    for settle_index in range(30):
        for timing in (
            "before_post_stop_physics_step",
            "after_post_stop_physics_step",
        ):
            if timing.startswith("after"):
                recorder.step()
            geometry_record = measure_live_collision_geometry(
                dynamic_certificate_allowance,
                timing=f"{timing}_{settle_index + 1}",
                post_stop=True,
            )
            if not bool(geometry_record.get("passed", False)):
                aborted = True
                if abort_classification is None:
                    abort_classification = str(
                        geometry_record.get("failure")
                        or "post_stop_dynamic_geometry_violation"
                    )
                    abort_reason = (
                        "live collision geometry left the full certified "
                        "allowance during zero-command settling: "
                        + str(geometry_record.get("reason"))
                    )
            if certificate_binding_error is None:
                post_position, post_orientation = world_pose(robot)
                post_route = route_execution_tube_record(
                    Pose2(
                        float(post_position[0]),
                        float(post_position[1]),
                        yaw_from_wxyz(post_orientation),
                    ),
                    segment_route,
                    translation_tolerance=(
                        ARGS.route_cross_track_tolerance
                    ),
                    yaw_tolerance_rad=certificate_yaw_tolerance,
                )
                post_stop_route_check_count += 1
                deviation = post_route.get("deviation")
                if isinstance(deviation, dict):
                    post_cross_track = float(
                        deviation["cross_track_metres"]
                    )
                    post_yaw_error = float(
                        deviation["yaw_error_rad"]
                    )
                    maximum_cross_track_error = max(
                        maximum_cross_track_error,
                        post_cross_track,
                    )
                    maximum_route_yaw_error = max(
                        maximum_route_yaw_error,
                        post_yaw_error,
                    )
                    maximum_runtime_yaw_arc = max(
                        maximum_runtime_yaw_arc,
                        2.0
                        * maximum_robot_planar_radius
                        * math.sin(0.5 * post_yaw_error),
                    )
                if not bool(post_route.get("passed", False)):
                    abbreviated = {
                        "timing": f"{timing}_{settle_index + 1}",
                        "simulation_step": recorder.sim_step,
                        "failure": post_route.get("failure"),
                        "reason": post_route.get("reason"),
                        "deviation": deviation,
                    }
                    if post_stop_route_failure is None:
                        post_stop_route_failure = abbreviated
                    aborted = True
                    if abort_classification is None:
                        abort_classification = str(
                            post_route.get("failure")
                            or "post_stop_route_tube_violation"
                        )
                        abort_reason = (
                            "base pose left the full certified route tube "
                            "during zero-command settling"
                        )
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
        "abort_classification": abort_classification,
        "abort_reason": abort_reason,
        "certificate_sha256": certificate_sha256,
        "route_sha256": runtime_route_sha256,
        "certificate_route_sha256": certificate_route_sha256,
        "certified_segment_index": segment_index,
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
            "planar position error outside the configured "
            f"{position_tolerance:.6g}m operational endpoint tolerance only"
        ),
        "stall_window_seconds": ARGS.base_stall_seconds,
        "stall_grace_seconds": ARGS.base_stall_grace_seconds,
        "stall_distance_threshold_metres": ARGS.base_stall_distance,
        "stall_monitor_basis": (
            "consecutive steering-aligned nonzero wheel-target motion only"
        ),
        "stall_monitor_armed_first_step": (
            stall_monitor_armed_first_step
        ),
        "stall_window_final_displacement_metres": (
            None
            if final_window_displacement is None
            else round(final_window_displacement, 6)
        ),
        "steering_alignment_error_degrees": round(
            math.degrees(steering_alignment_error), 6
        ),
        "steering_alignment_timeout_seconds": (
            ARGS.base_steering_timeout_seconds
        ),
        "steering_alignment_max_wait_steps": (
            steering_alignment_wait_steps
        ),
        "steering_alignment_first_step": steering_alignment_first_step,
        "max_abs_steering_error_degrees": round(
            max_abs_steering_error_degrees, 6
        ),
        "max_abs_drive_target_rad_s": round(
            max_abs_drive_target_rad_s, 6
        ),
        "max_abs_measured_drive_velocity_rad_s": round(
            max_abs_measured_drive_velocity_rad_s, 6
        ),
        "peak_abs_measured_drive_effort": round(
            peak_abs_measured_drive_effort, 6
        ),
        "drive_effort_sample_count": drive_effort_sample_count,
        "first_nonzero_wheel_command_sim_step": (
            first_nonzero_wheel_command_sim_step
        ),
        "base_motion_stopping_envelope_certified": (
            BASE_MOTION_STOPPING_ENVELOPE_CERTIFIED
        ),
        "base_motion_nonzero_commands_authorized": bool(
            BASE_MOTION_STOPPING_ENVELOPE_CERTIFIED
            and certificate_binding_error is None
        ),
        "base_motion_stopping_envelope_limitation": (
            None
            if BASE_MOTION_STOPPING_ENVELOPE_CERTIFIED
            else BASE_MOTION_STOPPING_ENVELOPE_LIMITATION
        ),
        "base_telemetry_nonfinite": base_telemetry_nonfinite,
        "external_contact_telemetry": "unavailable",
        "classification_limitation": (
            "contact-backed obstruction cannot be claimed while external "
            "contact telemetry is unavailable"
        ),
        "maximum_cross_track_error_metres": round(
            maximum_cross_track_error, 6
        ),
        "maximum_route_yaw_error_degrees": round(
            math.degrees(maximum_route_yaw_error), 6
        ),
        "maximum_runtime_yaw_arc_metres": round(
            maximum_runtime_yaw_arc, 6
        ),
        "certificate_cross_track_tolerance_metres": (
            ARGS.route_cross_track_tolerance
        ),
        "operational_cross_track_tolerance_metres": (
            ARGS.route_operational_cross_track_tolerance
        ),
        "execution_translation_reserve_metres": (
            ARGS.route_cross_track_tolerance
        ),
        "certificate_yaw_tolerance_degrees": (
            ARGS.route_yaw_tracking_tolerance_deg
        ),
        "operational_yaw_tolerance_degrees": (
            ARGS.route_operational_yaw_tolerance_deg
        ),
        "dynamic_geometry_certificate_allowance_metres": (
            dynamic_certificate_allowance
        ),
        "dynamic_geometry_operational_limit_metres": (
            dynamic_operational_limit
        ),
        "dynamic_geometry_certificate_operational_gap_metres": (
            dynamic_certificate_allowance - dynamic_operational_limit
        ),
        "live_geometry_check_count": live_geometry_check_count,
        "live_geometry_post_stop_check_count": (
            live_geometry_post_stop_check_count
        ),
        "maximum_live_geometry_planar_escape_metres": round(
            maximum_live_geometry_planar_escape, 6
        ),
        "maximum_live_geometry_z_escape_metres": round(
            maximum_live_geometry_z_escape, 6
        ),
        "maximum_nonplanar_base_escape_metres": round(
            maximum_nonplanar_base_escape, 6
        ),
        "maximum_combined_world_geometry_escape_metres": round(
            maximum_combined_world_geometry_escape, 6
        ),
        "first_live_geometry_failure": first_live_geometry_failure,
        "final_live_geometry_record": final_live_geometry_record,
        "live_geometry_containment_passed": (
            first_live_geometry_failure is None
        ),
        "post_stop_route_check_count": post_stop_route_check_count,
        "post_stop_route_failure": post_stop_route_failure,
        "payload_monitor_enabled": payload_monitor_enabled,
        "payload_reference_by_name": {
            name: rounded(values)
            for name, values in sorted(
                normalized_payload_reference.items()
            )
        },
        "payload_drift_tolerance_metres": (
            normalized_payload_tolerance
        ),
        "maximum_payload_to_base_drift_metres": round(
            maximum_payload_to_base_drift, 6
        ),
        "payload_geometry_envelope_passed": bool(
            first_live_geometry_failure is None
            and abort_classification
            not in {
                "payload_telemetry_unavailable",
                "payload_geometry_envelope_violation",
                "collision_geometry_inventory_mismatch",
                "dynamic_geometry_envelope_violation",
                "invalid_live_geometry_capture",
                "live_geometry_unresolved",
                "invalid_nonplanar_base_pose",
                "nonplanar_base_envelope_violation",
                "combined_dynamic_geometry_violation",
                "combined_world_geometry_envelope_violation",
            }
        ),
        "execution_tube_passed": bool(
            certificate_binding_error is None
            and first_live_geometry_failure is None
            and post_stop_route_failure is None
            and abort_classification
            not in {
                "route_certificate_binding_violation",
                "cross_track_violation",
                "route_yaw_violation",
                "payload_telemetry_unavailable",
                "payload_geometry_envelope_violation",
                "collision_geometry_inventory_mismatch",
                "dynamic_geometry_envelope_violation",
                "invalid_live_geometry_capture",
                "live_geometry_unresolved",
                "invalid_nonplanar_base_pose",
                "nonplanar_base_envelope_violation",
                "combined_dynamic_geometry_violation",
                "combined_world_geometry_envelope_violation",
            }
        ),
        "position_tolerance_metres": position_tolerance,
        "yaw_tolerance_degrees": math.degrees(yaw_tolerance),
        "passed": bool(
            not aborted
            and position_error <= position_tolerance
            and yaw_error <= yaw_tolerance
        ),
        "pass_definition": (
            "acceleration-limited closed-loop base pose within tolerance "
            "inside smaller operational SE(2)/live-geometry envelopes, with "
            "every pre-step, post-step, and zero-command settling sample "
            "remaining inside the full certificate-bound envelopes and "
            "without a steering, telemetry, or post-alignment motion abort; "
            "currently unreachable while nonzero base motion remains "
            "fail-closed pending a certificate-bound stopping envelope"
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


def make_navigation_waypoints(
    robot: SingleArticulation,
    route_targets: Iterable[np.ndarray],
    *,
    target_yaw: float,
) -> tuple[Pose2, ...]:
    base_position, base_orientation = world_pose(robot)
    current_yaw = yaw_from_wxyz(base_orientation)
    waypoints = (
        Pose2(
            float(base_position[0]),
            float(base_position[1]),
            current_yaw,
        ),
        *(
        Pose2(float(target[0]), float(target[1]), target_yaw)
        for target in route_targets
        ),
    )
    return tuple(waypoints)


def validate_navigation_route(
    robot: SingleArticulation,
    collision_proxies: CollisionProxySet,
    waypoints: tuple[Pose2, ...],
    *,
    phase: str,
    capture_label: str = "compact_navigation_stow",
    geometry_deformation_allowance_metres: float | None = None,
) -> dict[str, Any]:
    dynamic_allowance = (
        ARGS.dynamic_geometry_certificate_allowance_metres
        if geometry_deformation_allowance_metres is None
        else float(geometry_deformation_allowance_metres)
    )
    base_position, _base_orientation = world_pose(robot)
    certificate = validate_route(
        collision_proxies.robot,
        collision_proxies.environment,
        waypoints,
        RouteValidationConfig(
            max_translation_step=ARGS.route_max_translation_step,
            max_yaw_step_rad=math.radians(
                ARGS.route_max_yaw_step_deg
            ),
            clearance_margin=(
                ARGS.route_clearance_margin
                + dynamic_allowance
            ),
            execution_translation_tolerance=(
                ARGS.route_cross_track_tolerance
            ),
            execution_yaw_tolerance_rad=math.radians(
                ARGS.route_yaw_tracking_tolerance_deg
            ),
            base_z=float(base_position[2]),
            collision_coverage_complete=collision_proxies.complete,
        ),
    )
    return {
        "phase": phase,
        "classification": (
            "participant_conservative_full_robot_route_preflight"
        ),
        "passed": bool(certificate["passed"]),
        "certificate": certificate,
        "post_stow_proxy_coverage": collision_proxy_coverage_summary(
            collision_proxies,
            capture_label=capture_label,
        ),
        "collision_proxy_inventory_complete": (
            collision_proxies.complete
        ),
        "robot_proxy_count": len(collision_proxies.robot),
        "environment_proxy_count": len(
            collision_proxies.environment
        ),
        "support_surface_paths": list(
            collision_proxies.support_surface_paths
        ),
        "route_waypoints": [
            {
                "x": float(waypoint.x),
                "y": float(waypoint.y),
                "yaw_rad": float(waypoint.yaw),
            }
            for waypoint in waypoints
        ],
        "route_sha256": certificate.get(
            "validated_inputs", {}
        ).get("route_sha256"),
        "execution_translation_reserve_metres": (
            ARGS.route_cross_track_tolerance
        ),
        "execution_yaw_reserve_degrees": (
            ARGS.route_yaw_tracking_tolerance_deg
        ),
        "geometry_deformation_allowance_metres": (
            dynamic_allowance
        ),
        "pass_definition": (
            "complete enabled-collider coverage and no inflated 2.5-D "
            "collision-proxy overlap over the entire sampled SE(2) route"
        ),
        "limitation": (
            "This conservative participant preflight is not an organizer "
            "collision or scoring contract."
        ),
    }


def live_geometry_authorization_gate(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    collision_proxies: CollisionProxySet,
    *,
    route_certificate: dict[str, Any],
    phase: str,
    operational_limit_metres: float,
    certificate_allowance_metres: float,
) -> dict[str, Any]:
    """Require a live exact-witness recapture before any base command."""

    base_frame_path = core._find_articulation_root_path(ROBOT_PRIM_PATH)
    operational = live_collision_geometry_containment_record(
        recorder.stage,
        collision_proxies,
        robot_path=ROBOT_PRIM_PATH,
        base_frame_path=base_frame_path,
        dynamic_allowance=operational_limit_metres,
    )
    certificate = live_collision_geometry_containment_record(
        recorder.stage,
        collision_proxies,
        robot_path=ROBOT_PRIM_PATH,
        base_frame_path=base_frame_path,
        dynamic_allowance=certificate_allowance_metres,
    )
    base_position, base_orientation = world_pose(robot)
    reference_z = float(
        route_certificate.get("config", {}).get("base_z")
    )
    maximum_radius = float(
        route_certificate.get("execution_envelope", {}).get(
            "maximum_robot_spatial_radius_metres"
        )
    )
    operational_nonplanar = base_nonplanar_deviation_record(
        position_z=float(base_position[2]),
        orientation_wxyz=tuple(float(value) for value in base_orientation),
        reference_z=reference_z,
        maximum_robot_radius=maximum_radius,
        tolerance=operational_limit_metres,
    )
    certificate_nonplanar = base_nonplanar_deviation_record(
        position_z=float(base_position[2]),
        orientation_wxyz=tuple(float(value) for value in base_orientation),
        reference_z=reference_z,
        maximum_robot_radius=maximum_radius,
        tolerance=certificate_allowance_metres,
    )

    def combined_passed(
        containment: dict[str, Any],
        nonplanar: dict[str, Any],
        allowance: float,
    ) -> tuple[bool, float | None]:
        components = (
            containment.get("maximum_planar_escape_metres"),
            containment.get("maximum_z_escape_metres"),
            nonplanar.get("combined_escape_metres"),
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in components
        ):
            return False, None
        combined = max(
            float(components[0]), float(components[1])
        ) + float(components[2])
        return bool(
            containment.get("passed", False)
            and nonplanar.get("passed", False)
            and combined <= allowance + 1e-12
        ), combined

    operational_passed, operational_combined = combined_passed(
        operational,
        operational_nonplanar,
        operational_limit_metres,
    )
    certificate_passed, certificate_combined = combined_passed(
        certificate,
        certificate_nonplanar,
        certificate_allowance_metres,
    )
    return {
        "phase": phase,
        "classification": (
            "pre_command_live_collision_geometry_authorization_gate"
        ),
        "passed": bool(
            operational_passed
            and certificate_passed
            and certificate_allowance_metres - operational_limit_metres
            >= 0.02
        ),
        "operational_limit_metres": operational_limit_metres,
        "certificate_allowance_metres": certificate_allowance_metres,
        "reaction_reserve_metres": (
            certificate_allowance_metres - operational_limit_metres
        ),
        "operational_containment": operational,
        "certificate_containment": certificate,
        "operational_nonplanar_base": operational_nonplanar,
        "certificate_nonplanar_base": certificate_nonplanar,
        "operational_combined_world_escape_metres": (
            operational_combined
        ),
        "certificate_combined_world_escape_metres": (
            certificate_combined
        ),
        "pass_definition": (
            "the exact captured collider/Boundable witness inventory "
            "recaptures from current full USD transforms inside both the "
            "smaller operational envelope and the full certificate allowance"
        ),
    }


def run_cup_gate(
    recorder: TraceRecorder,
    robot: SingleArticulation,
    ik: Any,
    steering_ids: list[int],
    drive_ids: list[int],
    drivers: dict[str, tuple[str, int]],
    *,
    execute: bool = True,
) -> dict[str, Any]:
    before = task_object_snapshot(recorder.stage, recorder.object_paths)
    requested_gate_name = (
        "cup_grasp_lift_release"
        if execute
        else "cup_navigation_posture_and_route_preflight"
    )
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
    capture_label = "compact_navigation_stow"
    if execute:
        posture_gates, collision_proxies = prepare_navigation_posture(
            recorder,
            robot,
            ik,
        )
    else:
        capture_label = "current_asset_posture_no_command"
        posture_gates, collision_proxies = capture_current_navigation_posture(
            recorder
        )
    gates.extend(posture_gates)
    if not posture_gates or not all(
        bool(gate.get("passed", False)) for gate in posture_gates
    ):
        return fail_closed_result(
            gate_name=requested_gate_name,
            gates=gates,
            before=before,
            recorder=recorder,
            reason=(
                "Compact navigation posture or posture-specific collider "
                "coverage failed before route validation or any base command."
            ),
        )
    certified_route = make_navigation_waypoints(
        robot,
        (route_target for _phase, route_target, _intent in route_waypoints),
        target_yaw=nominal_yaw,
    )
    preflight = validate_navigation_route(
        robot,
        collision_proxies,
        certified_route,
        phase="cup_route_full_robot_preflight",
        capture_label=capture_label,
    )
    gates.append(preflight)
    if not preflight["passed"]:
        return fail_closed_result(
            gate_name=requested_gate_name,
            gates=gates,
            before=before,
            recorder=recorder,
            reason=(
                "Full-robot route preflight failed before any base command."
            ),
        )
    live_authorization = live_geometry_authorization_gate(
        recorder,
        robot,
        collision_proxies,
        route_certificate=preflight["certificate"],
        phase="cup_live_geometry_authorization",
        operational_limit_metres=(
            ARGS.dynamic_geometry_operational_limit_metres
        ),
        certificate_allowance_metres=(
            ARGS.dynamic_geometry_certificate_allowance_metres
        ),
    )
    gates.append(live_authorization)
    if not live_authorization["passed"]:
        return fail_closed_result(
            gate_name=requested_gate_name,
            gates=gates,
            before=before,
            recorder=recorder,
            reason=(
                "Live robot collider recapture failed before any base "
                "command."
            ),
        )
    if not execute:
        after = task_object_snapshot(
            recorder.stage, recorder.object_paths
        )
        return {
            "gate": "cup_navigation_posture_and_route_preflight",
            "classification": (
                "participant_navigation_posture_and_route_preflight"
            ),
            "passed": True,
            "gates": gates,
            "task_objects_before": before,
            "task_objects_after": after,
            "motion_scope": (
                "read-only current-posture geometry and route validation; no "
                "base, arm, gripper, or task-object command was issued"
            ),
            "official_stage_complete": False,
            "official_score": None,
        }
    for segment_index, (
        route_phase,
        _route_target,
        route_intent,
    ) in enumerate(route_waypoints):
        gate = drive_base_to(
            recorder,
            robot,
            steering_ids,
            drive_ids,
            certified_waypoints=certified_route,
            route_certificate=preflight["certificate"],
            collision_proxies=collision_proxies,
            dynamic_geometry_certificate_allowance_metres=(
                ARGS.dynamic_geometry_certificate_allowance_metres
            ),
            dynamic_geometry_operational_limit_metres=(
                ARGS.dynamic_geometry_operational_limit_metres
            ),
            segment_index=segment_index,
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
    posture_gates, collision_proxies = prepare_navigation_posture(
        recorder,
        robot,
        ik,
    )
    gates.extend(posture_gates)
    if not posture_gates or not all(
        bool(gate.get("passed", False)) for gate in posture_gates
    ):
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason=(
                "Compact navigation posture or posture-specific collider "
                "coverage failed before route validation or any base command."
            ),
        )
    certified_route = make_navigation_waypoints(
        robot,
        (route_target for _phase, route_target in route_waypoints),
        target_yaw=nominal_yaw,
    )
    preflight = validate_navigation_route(
        robot,
        collision_proxies,
        certified_route,
        phase="tray_route_full_robot_preflight",
    )
    gates.append(preflight)
    if not preflight["passed"]:
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason=(
                "Full-robot tray approach preflight failed before any base "
                "command."
            ),
        )
    live_authorization = live_geometry_authorization_gate(
        recorder,
        robot,
        collision_proxies,
        route_certificate=preflight["certificate"],
        phase="tray_live_geometry_authorization",
        operational_limit_metres=(
            ARGS.dynamic_geometry_operational_limit_metres
        ),
        certificate_allowance_metres=(
            ARGS.dynamic_geometry_certificate_allowance_metres
        ),
    )
    gates.append(live_authorization)
    if not live_authorization["passed"]:
        return fail_closed_result(
            gate_name=(
                "tray_bimanual_lift_and_transport"
                if transport
                else "tray_bimanual_lift"
            ),
            gates=gates,
            before=before,
            recorder=recorder,
            reason=(
                "Live robot collider recapture failed before any base "
                "command."
            ),
        )
    for segment_index, (
        route_phase,
        _route_target,
    ) in enumerate(route_waypoints):
        gate = drive_base_to(
            recorder,
            robot,
            steering_ids,
            drive_ids,
            certified_waypoints=certified_route,
            route_certificate=preflight["certificate"],
            collision_proxies=collision_proxies,
            dynamic_geometry_certificate_allowance_metres=(
                ARGS.dynamic_geometry_certificate_allowance_metres
            ),
            dynamic_geometry_operational_limit_metres=(
                ARGS.dynamic_geometry_operational_limit_metres
            ),
            segment_index=segment_index,
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
        recorder.set_phase(
            "tray_loaded_collision_capture",
            intent=(
                "Recapture the extended robot, physically lifted tray, and "
                "four dynamic payload objects before any loaded wheel command."
            ),
        )
        loaded_capture_step = recorder.sim_step
        attached_payload_roots = tuple(
            recorder.object_paths[name]
            for name in ("tray", *RULEBOOK_STAGE1_OBJECTS)
        )
        base_position_for_loaded_proxies, _ = world_pose(robot)
        loaded_root_path = core._find_articulation_root_path(
            ROBOT_PRIM_PATH
        )
        try:
            loaded_collision_proxies = extract_collision_proxy_set(
                recorder.stage,
                robot_path=ROBOT_PRIM_PATH,
                base_frame_path=loaded_root_path,
                base_world_z=float(base_position_for_loaded_proxies[2]),
                attached_payload_root_paths=attached_payload_roots,
            )
        except Exception as error:  # noqa: BLE001
            loaded_collision_proxies = CollisionProxySet(
                robot=(),
                environment=(),
                support_surface_paths=(),
                candidate_robot_paths=(),
                candidate_environment_paths=(),
                unresolved_paths=(
                    f"loaded extractor [{type(error).__name__}: {error}]",
                ),
                traversal_backend="failed",
                attached_payload_root_paths=attached_payload_roots,
            )
        loaded_coverage_gate = {
            "phase": "tray_loaded_collision_proxy_coverage",
            "classification": (
                "post_lift_loaded_compound_enabled_collider_coverage_gate"
            ),
            "capture_sim_step": loaded_capture_step,
            "passed": loaded_collision_proxies.complete,
            "loaded_proxy_coverage": collision_proxy_coverage_summary(
                loaded_collision_proxies,
                capture_label=(
                    "post_lift_extended_arms_with_tray_and_four_objects"
                ),
            ),
            "collision_proxy_inventory": (
                loaded_collision_proxies.record()
            ),
            "pass_definition": (
                "every enabled robot, tray, payload, and environment collider "
                "is represented, with every requested attached payload root "
                "present in the robot-relative compound geometry"
            ),
        }
        gates.append(loaded_coverage_gate)
        if not loaded_coverage_gate["passed"]:
            return fail_closed_result(
                gate_name="tray_bimanual_lift_and_transport",
                gates=gates,
                before=before,
                recorder=recorder,
                reason=(
                    "Loaded tray/payload collider coverage failed before any "
                    "loaded base command."
                ),
            )
        try:
            loaded_payload_reference = payload_positions_in_base_frame(
                recorder,
                robot,
                ("tray", *RULEBOOK_STAGE1_OBJECTS),
            )
        except Exception as error:  # noqa: BLE001
            gates.append(
                {
                    "phase": "tray_loaded_payload_reference",
                    "capture_sim_step": loaded_capture_step,
                    "classification": (
                        "loaded_payload_to_base_reference_gate"
                    ),
                    "passed": False,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            return fail_closed_result(
                gate_name="tray_bimanual_lift_and_transport",
                gates=gates,
                before=before,
                recorder=recorder,
                reason=(
                    "Loaded payload-to-base reference capture failed before "
                    "any loaded base command."
                ),
            )
        gates.append(
            {
                "phase": "tray_loaded_payload_reference",
                "capture_sim_step": loaded_capture_step,
                "classification": (
                    "loaded_payload_to_base_reference_gate"
                ),
                "passed": True,
                "payload_reference_by_name": {
                    name: rounded(values)
                    for name, values in sorted(
                        loaded_payload_reference.items()
                    )
                },
                "reserved_drift_metres": (
                    ARGS.tray_payload_envelope_metres
                ),
            }
        )
        loaded_certified_route = make_navigation_waypoints(
            robot,
            (
                route_target
                for _phase, route_target in transport_waypoints
            ),
            target_yaw=nominal_yaw,
        )
        loaded_preflight = validate_navigation_route(
            robot,
            loaded_collision_proxies,
            loaded_certified_route,
            phase="tray_loaded_return_route_preflight",
            capture_label=(
                "post_lift_extended_arms_with_tray_and_four_objects"
            ),
            geometry_deformation_allowance_metres=(
                ARGS.tray_payload_envelope_metres
            ),
        )
        loaded_preflight["capture_sim_step"] = loaded_capture_step
        loaded_preflight["authorization_sim_step"] = recorder.sim_step
        loaded_preflight["loaded_geometry_model"] = (
            "post-lift compound proxy plus reserved payload deformation "
            "allowance; runtime payload-to-base drift is checked separately"
        )
        gates.append(loaded_preflight)
        if not loaded_preflight["passed"]:
            return fail_closed_result(
                gate_name="tray_bimanual_lift_and_transport",
                gates=gates,
                before=before,
                recorder=recorder,
                reason=(
                    "Loaded tray return-route preflight failed before any "
                    "loaded base command."
                ),
            )
        loaded_live_authorization = live_geometry_authorization_gate(
            recorder,
            robot,
            loaded_collision_proxies,
            route_certificate=loaded_preflight["certificate"],
            phase="tray_loaded_live_geometry_authorization",
            operational_limit_metres=(
                ARGS.tray_payload_operational_limit_metres
            ),
            certificate_allowance_metres=(
                ARGS.tray_payload_envelope_metres
            ),
        )
        gates.append(loaded_live_authorization)
        if not loaded_live_authorization["passed"]:
            return fail_closed_result(
                gate_name="tray_bimanual_lift_and_transport",
                gates=gates,
                before=before,
                recorder=recorder,
                reason=(
                    "Live loaded robot/payload collider recapture failed "
                    "before any loaded base command."
                ),
            )
        for segment_index, (
            route_phase,
            _route_target,
        ) in enumerate(transport_waypoints):
            gate = drive_base_to(
                recorder,
                robot,
                steering_ids,
                drive_ids,
                certified_waypoints=loaded_certified_route,
                route_certificate=loaded_preflight["certificate"],
                collision_proxies=loaded_collision_proxies,
                dynamic_geometry_certificate_allowance_metres=(
                    ARGS.tray_payload_envelope_metres
                ),
                dynamic_geometry_operational_limit_metres=(
                    ARGS.tray_payload_operational_limit_metres
                ),
                segment_index=segment_index,
                max_steps=ARGS.base_max_steps,
                max_speed=ARGS.transport_max_speed,
                max_accel=ARGS.transport_max_accel,
                phase=route_phase,
                intent=(
                    "Slowly transport the physically grasped dynamic tray "
                    "through the measured doorway clearance toward dining."
                ),
                payload_reference_by_name=loaded_payload_reference,
                payload_drift_tolerance_metres=(
                    ARGS.tray_payload_envelope_metres
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
    if ARGS.render:
        raise ValueError(
            "--render is not supported by the fail-closed Stage 1 controller "
            "until every batched physics substep can be monitored"
        )
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
        "--arm-max-command-slew-rad": (
            ARGS.arm_max_command_slew_rad,
            0.001,
            0.20,
        ),
        "--arm-tracking-error-threshold-rad": (
            ARGS.arm_tracking_error_threshold_rad,
            0.02,
            0.50,
        ),
        "--arm-tracking-error-dwell-seconds": (
            ARGS.arm_tracking_error_dwell_seconds,
            0.05,
            2.0,
        ),
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
        "--base-steering-timeout-seconds": (
            ARGS.base_steering_timeout_seconds,
            0.5,
            10.0,
        ),
        "--base-drive-response-rad-s": (
            ARGS.base_drive_response_rad_s,
            0.01,
            5.0,
        ),
        "--route-clearance-margin": (
            ARGS.route_clearance_margin,
            0.01,
            0.20,
        ),
        "--route-max-translation-step": (
            ARGS.route_max_translation_step,
            0.005,
            0.05,
        ),
        "--route-max-yaw-step-deg": (
            ARGS.route_max_yaw_step_deg,
            0.5,
            5.0,
        ),
        "--route-cross-track-tolerance": (
            ARGS.route_cross_track_tolerance,
            0.01,
            0.10,
        ),
        "--route-operational-cross-track-tolerance": (
            ARGS.route_operational_cross_track_tolerance,
            0.0,
            0.08,
        ),
        "--route-yaw-tracking-tolerance-deg": (
            ARGS.route_yaw_tracking_tolerance_deg,
            0.25,
            5.0,
        ),
        "--route-operational-yaw-tolerance-deg": (
            ARGS.route_operational_yaw_tolerance_deg,
            0.0,
            4.0,
        ),
        "--dynamic-geometry-certificate-allowance-metres": (
            ARGS.dynamic_geometry_certificate_allowance_metres,
            0.04,
            0.20,
        ),
        "--dynamic-geometry-operational-limit-metres": (
            ARGS.dynamic_geometry_operational_limit_metres,
            0.0,
            0.15,
        ),
        "--navigation-stow-height": (
            ARGS.navigation_stow_height,
            1.0,
            1.4,
        ),
        "--navigation-stow-forward": (
            ARGS.navigation_stow_forward,
            0.35,
            0.65,
        ),
        "--navigation-stow-lateral": (
            ARGS.navigation_stow_lateral,
            0.20,
            0.38,
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
        "--tray-payload-envelope-metres": (
            ARGS.tray_payload_envelope_metres,
            0.05,
            0.30,
        ),
        "--tray-payload-operational-limit-metres": (
            ARGS.tray_payload_operational_limit_metres,
            0.0,
            0.25,
        ),
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
    if (
        ARGS.route_cross_track_tolerance
        - ARGS.route_operational_cross_track_tolerance
        < 0.01
    ):
        raise ValueError(
            "route certificate cross-track tolerance must exceed the "
            "operational limit by at least 0.01m"
        )
    if (
        ARGS.route_yaw_tracking_tolerance_deg
        - ARGS.route_operational_yaw_tolerance_deg
        < 0.5
    ):
        raise ValueError(
            "route certificate yaw tolerance must exceed the operational "
            "limit by at least 0.5 degrees"
        )
    dynamic_allowance_gap = (
        ARGS.dynamic_geometry_certificate_allowance_metres
        - ARGS.dynamic_geometry_operational_limit_metres
    )
    if dynamic_allowance_gap < 0.02:
        raise ValueError(
            "dynamic geometry certificate allowance must exceed the "
            "operational limit by at least 0.02m"
        )
    loaded_allowance_gap = (
        ARGS.tray_payload_envelope_metres
        - ARGS.tray_payload_operational_limit_metres
    )
    if loaded_allowance_gap < 0.02:
        raise ValueError(
            "tray payload envelope must exceed its operational limit by "
            "at least 0.02m"
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
    mutation_guard: dict[str, Any],
    authenticated_upstream_helpers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    executed_sources = [
        {
            **record,
            "repository_path": f"participant/{record['path']}",
            "runtime_path": str(script_path.parent / str(record["path"])),
        }
        for record in mutation_guard["sources"]
    ]
    entrypoint_path = Path(
        os.environ.get(
            "AIRSIGN_ENTRYPOINT_PATH",
            "/usr/local/bin/airsign",
        )
    ).resolve()
    if not entrypoint_path.is_file():
        raise RuntimeError(
            f"AirSign runtime entrypoint is missing: {entrypoint_path}"
        )
    executed_sources.append(
        {
            "path": "airsign",
            "repository_path": "scripts/airsign",
            "runtime_path": str(entrypoint_path),
            "bytes": entrypoint_path.stat().st_size,
            "sha256": sha256_file(entrypoint_path),
            "ast_call_count": None,
        }
    )
    for data_name in AIRSIGN_RUNTIME_DATA_NAMES:
        data_path = (script_path.parent / data_name).resolve()
        if not data_path.is_file():
            raise RuntimeError(
                f"AirSign runtime provenance data is missing: {data_path}"
            )
        executed_sources.append(
            {
                "path": data_name,
                "repository_path": f"participant/{data_name}",
                "runtime_path": str(data_path),
                "bytes": data_path.stat().st_size,
                "sha256": sha256_file(data_path),
                "ast_call_count": None,
            }
        )
    source_set_records = sorted(
        (
            {
                "repository_path": record["repository_path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for record in executed_sources
        ),
        key=lambda record: str(record["repository_path"]),
    )
    source_set_sha256 = hashlib.sha256(
        json.dumps(
            source_set_records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "airsign_repository": (
            "https://github.com/EvergreenTree/AirSignRobot"
        ),
        "airsign_source_revision": os.environ.get(
            "AIRSIGN_REVISION",
            "unavailable",
        ),
        "official_benchmark_commit": os.environ.get(
            "EBIM_BENCHMARK_COMMIT",
            os.environ.get("EBIM_COMMIT", git_revision(REPO_ROOT)),
        ),
        "controller_sha256": sha256_file(script_path),
        "controller_path_in_runtime": str(script_path),
        "source_set_sha256": source_set_sha256,
        "python_source_set_sha256": mutation_guard[
            "source_set_sha256"
        ],
        "executed_sources": executed_sources,
        "direct_upstream_helpers": authenticated_upstream_helpers,
        "simulator": "Isaac Sim 5.1.0",
        "container_image": os.environ.get(
            "AIRSIGN_CAPTURE_CONTAINER_IMAGE",
            "unavailable",
        ),
        "container_image_id": os.environ.get(
            "AIRSIGN_CAPTURE_CONTAINER_IMAGE_ID", "unavailable"
        ),
        "container_repo_digest": os.environ.get(
            "AIRSIGN_CAPTURE_CONTAINER_REPO_DIGEST"
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
    collision_inventory: dict[str, Any],
    collision_proxy_inventory: dict[str, Any],
    started_wall: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_passed = all(
        bool(result["passed"]) for result in physical_results
    )
    trace = {
        "schema_version": 2,
        "kind": "airsign_ebim_track3_stage1_physical_gate_trace",
        "classification": "measured_physical_development_evidence",
        "coordinate_frame": "official_isaac_world_metres_z_up",
        "task_objects_teleported": False,
        "robot_links_teleported": False,
        "robot_link_initialization_boundary": (
            "Isaac World.reset applied the asset default before evidence "
            "recording; the ready-pose setup helper and set_joint_positions "
            "were not used"
        ),
        "task_object_mutation_api_used": False,
        "official_stage_completion_claimed": False,
        "official_score_claimed": False,
        "benchmark_score": None,
        "events": recorder.events,
        "frames": recorder.frames,
        "provenance": provenance,
    }
    metrics = {
        "schema_version": 2,
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
        "collision_prim_inventory": collision_inventory,
        "collision_proxy_inventory": collision_proxy_inventory,
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
    canonical_provenance = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    manifest = {
        "schema_version": 2,
        "controller_sha256": provenance["controller_sha256"],
        "source_set_sha256": provenance["source_set_sha256"],
        "airsign_source_revision": provenance[
            "airsign_source_revision"
        ],
        "container_image_id": provenance["container_image_id"],
        "provenance_sha256": hashlib.sha256(
            canonical_provenance.encode("utf-8")
        ).hexdigest(),
        "provenance": provenance,
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
    authenticated_upstream_helpers = direct_upstream_helper_records()
    target_provider = load_target_provider(ARGS.targets_json)
    room_path = Path(ARGS.room_usd).expanduser()
    robot_path = Path(ARGS.robot_usd).expanduser()
    if not room_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError(
            f"Missing room or robot asset: {room_path}, {robot_path}"
        )
    output_dir = ARGS.output_dir.expanduser().resolve()
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
    # Do not call setup_robot_control(): the pinned official helper invokes
    # set_joint_positions() to teleport a ready pose.  Keep the initialized
    # asset state and use only measured, guarded articulation targets after the
    # evidence recorder starts.
    steering_ids, drive_ids = core._find_drive_joint_ids(
        list(robot.dof_names)
    )
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
    object_asset_root_paths = {
        name: find_prim_path(stage, source_name)
        for name, source_name in TASK_OBJECT_NAMES.items()
    }
    dynamic_body_resolution = {
        name: resolve_enabled_dynamic_rigid_body_descendant(stage, path)
        for name, path in object_asset_root_paths.items()
    }
    failed_body_resolution = {
        name: record
        for name, record in dynamic_body_resolution.items()
        if not bool(record.get("passed", False))
    }
    if failed_body_resolution:
        raise RuntimeError(
            "Task-object dynamic rigid-body resolution failed closed: "
            + json.dumps(failed_body_resolution, sort_keys=True)
        )
    object_paths = {
        name: str(record["dynamic_rigid_body_path"])
        for name, record in dynamic_body_resolution.items()
    }
    rigid_bodies = {
        name: {
            **rigid_body_record(stage, object_asset_root_paths[name]),
            "selected_dynamic_rigid_body_path": object_paths[name],
            "selection": dynamic_body_resolution[name],
        }
        for name in object_asset_root_paths
    }
    environment_inventory = environment_top_level_inventory(stage)
    collision_inventory = collision_prim_inventory(stage)
    base_pose_for_proxies, _ = world_pose(robot)
    try:
        collision_proxies = extract_collision_proxy_set(
            stage,
            robot_path=ROBOT_PRIM_PATH,
            base_frame_path=root_path,
            base_world_z=float(base_pose_for_proxies[2]),
        )
    except Exception as error:  # noqa: BLE001
        collision_proxies = CollisionProxySet(
            robot=(),
            environment=(),
            support_surface_paths=(),
            candidate_robot_paths=(),
            candidate_environment_paths=(),
            unresolved_paths=(
                f"extractor [{type(error).__name__}: {error}]",
            ),
            traversal_backend="failed",
        )
    collision_proxy_inventory = collision_proxies.record()
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
    if ARGS.gate in ("cup-preflight", "cup", "all"):
        cup_result = run_cup_gate(
            recorder,
            robot,
            ik,
            steering_ids,
            drive_ids,
            drivers,
            execute=ARGS.gate != "cup-preflight",
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
        mutation_guard,
        authenticated_upstream_helpers,
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
        collision_inventory=collision_inventory,
        collision_proxy_inventory=collision_proxy_inventory,
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
