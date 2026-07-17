#!/usr/bin/env python3
"""Read-only Task 3 scene inventory plus Robotiq driver-joint gate.

This isolated probe creates the official Task 3 scene, commands only the two
Robotiq outer-knuckle articulation DOFs through ``ArticulationAction`` in an
open -> closed -> open sequence, and prints a compact JSON result.  It never
sets transforms, velocities, or poses on task objects or robot links.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

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
from gripper_profiles import get_gripper_profile, get_profile_drive_gains  # noqa: E402
from isaacsim_fr3duo_teleop_bridge_args import add_common_bridge_args  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room-usd", type=Path, default=room_scene.asset_path("robot_room.usd")
    )
    parser.add_argument("--robot-usd", type=Path, default=None)
    parser.add_argument("--robot-x", type=float, default=None)
    parser.add_argument("--robot-y", type=float, default=None)
    parser.add_argument("--robot-z", type=float, default=None)
    parser.add_argument("--robot-yaw", type=float, default=None)
    parser.add_argument("--head-placement", type=room_scene.head_placement_arg, default="A")
    parser.add_argument("--settle-steps", type=int, default=150)
    parser.add_argument("--command-steps", type=int, default=150)
    parser.add_argument("--position-tolerance", type=float, default=0.06)
    add_common_bridge_args(parser)
    parser.set_defaults(headless=True)
    return parser


ARGS = build_arg_parser().parse_args()
PROFILE = get_gripper_profile("robotiq")
if ARGS.robot_usd is None:
    ARGS.robot_usd = PROFILE.robot_usd

from isaacsim import SimulationApp  # noqa: E402

SIMULATION_APP = SimulationApp({"headless": ARGS.headless, "width": 1280, "height": 720})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.robot_motion.motion_generation")
SIMULATION_APP.update()
enable_extension("isaacsim.ros2.bridge")
SIMULATION_APP.update()

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
import isaacsim_fr3duo_teleop_bridge_core as core  # noqa: E402
import omni.kit.app  # noqa: E402


ROBOT_PRIM_PATH = "/World/Robot"
TASK_OBJECT_NAMES = {
    "tray": "simple_tray",
    "plate": "plate2",
    "cup": "cup",
    "bowl": "bowl2",
    "spoon": "spoon2",
    "head": "head",
}


def rounded(values: Any) -> list[float]:
    return [round(float(value), 6) for value in values]


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
    raise RuntimeError(f"Could not find task prim named {name!r}")


def prim_record(stage: Any, path: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {path}")
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes).ComputeWorldBound(prim).ComputeAlignedRange()
    minimum, maximum = bbox.GetMin(), bbox.GetMax()
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return {
        "prim_path": path,
        "pose": {
            "position": rounded(translation),
            "orientation_wxyz": rounded((rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2])),
        },
        "aabb": {"min": rounded(minimum), "max": rounded(maximum)},
    }


def beans_record(stage: Any) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    paths = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetName().startswith("Bean_")
    )
    if not paths:
        raise RuntimeError("No coffee-bean prims found")
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    xform_cache = UsdGeom.XformCache()
    minima, maxima, positions = [], [], []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minima.append(np.asarray(bounds.GetMin(), dtype=np.float64))
        maxima.append(np.asarray(bounds.GetMax(), dtype=np.float64))
        positions.append(np.asarray(xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation(), dtype=np.float64))
    return {
        "scope_path": "/World/Scene/CoffeeBeans",
        "count": len(paths),
        "aggregate_aabb": {"min": rounded(np.min(minima, axis=0)), "max": rounded(np.max(maxima, axis=0))},
        "centroid": rounded(np.mean(positions, axis=0)),
    }


def logical_regions() -> dict[str, Any]:
    from grading import TASK3_BEAN_RECOVERY_REGION, TASK3_SINK_REGION

    recovery = TASK3_BEAN_RECOVERY_REGION
    sink = TASK3_SINK_REGION
    center = recovery.center
    radius = float(recovery.radius)
    sink_center = (0.5 * (sink.bounds.x_min + sink.bounds.x_max), 0.5 * (sink.bounds.y_min + sink.bounds.y_max), sink.tabletop_z)
    return {
        "recovery": {
            "kind": "official_grader_sphere_region",
            "source": "scripts/evaluation/task3/grading.py:TASK3_BEAN_RECOVERY_REGION",
            "pose": {"position": rounded((center.x, center.y, center.z)), "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "aabb": {"min": rounded((center.x - radius, center.y - radius, center.z - radius)), "max": rounded((center.x + radius, center.y + radius, center.z + radius))},
            "radius_metres": radius,
        },
        "sink": {
            "kind": "official_grader_2d_sink_region",
            "source": "scripts/evaluation/task3/grading.py:TASK3_SINK_REGION",
            "pose": {"position": rounded(sink_center), "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "aabb": {"min": rounded((sink.bounds.x_min, sink.bounds.y_min, sink.tabletop_z)), "max": rounded((sink.bounds.x_max, sink.bounds.y_max, sink.tabletop_z))},
            "tabletop_z_metres": round(float(sink.tabletop_z), 6),
        },
    }


def discover_drivers(robot: SingleArticulation) -> dict[str, tuple[str, int]]:
    names = list(robot.dof_names)
    result: dict[str, tuple[str, int]] = {}
    for side in ("left", "right"):
        candidates = [
            (index, name)
            for index, name in enumerate(names)
            if name.startswith(f"{side}_") and "outer_knuckle_joint" in name
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one {side} outer-knuckle driver, found {candidates}")
        index, name = candidates[0]
        result[side] = (name, index)
    return result


def command_phase(
    world: World,
    robot: SingleArticulation,
    drivers: dict[str, tuple[str, int]],
    target: float,
    steps: int,
    tolerance: float,
    label: str,
) -> dict[str, Any]:
    indices = np.asarray([drivers["left"][1], drivers["right"][1]], dtype=np.int64)
    targets = np.asarray([target, target], dtype=np.float32)
    controller = robot.get_articulation_controller()
    for _ in range(steps):
        controller.apply_action(ArticulationAction(joint_positions=targets, joint_indices=indices))
        world.step(render=False)
    measured = robot.get_joint_positions()[indices]
    errors = np.abs(measured - targets)
    return {
        "phase": label,
        "target_position": round(float(target), 6),
        "measured_positions": {"left": round(float(measured[0]), 6), "right": round(float(measured[1]), 6)},
        "absolute_errors": {"left": round(float(errors[0]), 6), "right": round(float(errors[1]), 6)},
        "passed": bool(np.all(errors <= tolerance)),
    }


def main() -> None:
    if ARGS.settle_steps < 0 or ARGS.command_steps < 1 or not math.isfinite(ARGS.position_tolerance) or ARGS.position_tolerance <= 0.0:
        raise ValueError("Invalid step count or position tolerance")
    room_path = Path(ARGS.room_usd).expanduser()
    robot_path = Path(ARGS.robot_usd).expanduser()
    if not room_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError(f"Missing room or robot asset: {room_path}, {robot_path}")

    ARGS.task = "task3"
    robot_position = room_scene.resolve_robot_position(ARGS)
    robot_yaw = room_scene.resolve_robot_yaw(ARGS)
    print("GRIPPER_SCENE_GATE2_PHASE build_stage", flush=True)
    room_scene.build_stage(
        omni.kit.app.get_app(), room_path=room_path, robot_path=robot_path, task="task3",
        robot_position=robot_position, robot_rotation=room_scene.yaw_to_quat(robot_yaw),
        robot_yaw=robot_yaw, head_placement=ARGS.head_placement, dynamic_beans=True,
    )
    physics_scene_path = core._find_physics_scene_path() or "/physicsScene"
    world = World(physics_prim_path=physics_scene_path, stage_units_in_meters=1.0, physics_dt=1.0 / ARGS.physics_hz, rendering_dt=1.0 / ARGS.render_hz)
    core.prepare_robot_prim(ROBOT_PRIM_PATH, ARGS)
    core._configure_drives(ROBOT_PRIM_PATH, lambda joint_name: get_profile_drive_gains(PROFILE.name, joint_name))
    root_path = core._find_articulation_root_path(ROBOT_PRIM_PATH)
    robot = SingleArticulation(prim_path=root_path, name="task3_gripper_scene_gate2_robot")
    world.scene.add(robot)
    world.reset()
    for _ in range(ARGS.settle_steps):
        world.step(render=False)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available after scene creation")
    drivers = discover_drivers(robot)
    discovered_driver_names = {details[0] for details in drivers.values()}
    # The Robotiq asset ships its outer-knuckle drives without usable position
    # gains in this headless scene.  Configure exactly the names discovered
    # from this articulation; all subsequent motion is ArticulationAction.
    core._configure_drives(
        ROBOT_PRIM_PATH,
        lambda joint_name: (
            {"stiffness": 500.0, "damping": 50.0, "max_force": 200.0}
            if joint_name in discovered_driver_names
            else None
        ),
    )
    print("GRIPPER_SCENE_GATE2_PHASE command_open_closed_open", flush=True)
    phases = [
        command_phase(world, robot, drivers, PROFILE.keyboard_positions[0], ARGS.command_steps, ARGS.position_tolerance, "open_initial"),
        command_phase(world, robot, drivers, PROFILE.keyboard_positions[1], ARGS.command_steps, ARGS.position_tolerance, "closed"),
        command_phase(world, robot, drivers, PROFILE.keyboard_positions[0], ARGS.command_steps, ARGS.position_tolerance, "open_final"),
    ]
    print("GRIPPER_SCENE_GATE2_PHASE read_scene", flush=True)
    task_prims = {label: prim_record(stage, find_prim_path(stage, name)) for label, name in TASK_OBJECT_NAMES.items()}
    scene = {**task_prims, "beans": beans_record(stage), **logical_regions()}
    payload = {
        "kind": "robotiq_gripper_open_closed_open_and_task3_scene_inventory",
        "passed": all(phase["passed"] for phase in phases),
        "head_placement": ARGS.head_placement,
        "driver_joint_names_discovered": {side: details[0] for side, details in drivers.items()},
        "driver_joint_indices_discovered": {side: details[1] for side, details in drivers.items()},
        "command_method": "ArticulationAction.joint_positions_only",
        "position_tolerance": ARGS.position_tolerance,
        "phases": phases,
        "scene": scene,
        "task_objects_teleported": False,
        "robot_links_teleported": False,
        "task_object_mutation_api_used": False,
    }
    print("GRIPPER_SCENE_GATE2_RESULT " + json.dumps(payload, sort_keys=True), flush=True)
    if not payload["passed"]:
        raise RuntimeError("Measured gripper driver positions did not reach targets")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:  # noqa: BLE001 - preserve failure details and nonzero exit
        traceback.print_exc()
        exit_code = 1
    finally:
        SIMULATION_APP.close()
    raise SystemExit(exit_code)
