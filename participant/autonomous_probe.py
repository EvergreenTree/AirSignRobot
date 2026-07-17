#!/usr/bin/env python3
"""Validate actuator-driven dual-arm IK in the official EBiM Task 3 scene.

This is the first gate for the autonomous participant controller. It builds the
official Robotiq scene, moves both end effectors upward through Lula IK and
articulation position targets, and emits a machine-readable result. It never
teleports task objects or robot links.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK3_ROOT = REPO_ROOT / "task3_isaacsim"
SHARED_SCENES_DIR = REPO_ROOT / "scripts" / "scenes"
TASK2_SCRIPTS_DIR = REPO_ROOT / "task2_isaacsim" / "scripts"
TASK3_SCRIPTS_DIR = TASK3_ROOT / "scripts"
TASK3_COMMON_DIR = TASK3_SCRIPTS_DIR / "common"
for module_dir in (
    SHARED_SCENES_DIR,
    TASK2_SCRIPTS_DIR,
    TASK3_SCRIPTS_DIR,
    TASK3_COMMON_DIR,
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
    parser.add_argument("--lift-metres", type=float, default=0.04)
    parser.add_argument("--motion-steps", type=int, default=120)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--base-distance-metres", type=float, default=0.40)
    parser.add_argument("--base-max-steps", type=int, default=2880)
    parser.add_argument("--base-max-speed", type=float, default=0.25)
    add_common_bridge_args(parser)
    parser.set_defaults(
        headless=True,
        arm_teleop_gripper_open=None,
        arm_teleop_gripper_closed=None,
    )
    return parser


args_cli = build_arg_parser().parse_args()
profile_cli = get_gripper_profile("robotiq")
if args_cli.robot_usd is None:
    args_cli.robot_usd = profile_cli.robot_usd
if args_cli.arm_teleop_gripper_open is None:
    args_cli.arm_teleop_gripper_open = profile_cli.keyboard_positions[0]
if args_cli.arm_teleop_gripper_closed is None:
    args_cli.arm_teleop_gripper_closed = profile_cli.keyboard_positions[1]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": args_cli.headless, "width": 1280, "height": 720}
)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.robot_motion.motion_generation")
simulation_app.update()
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import isaacsim_fr3duo_teleop_bridge_core as core  # noqa: E402
import omni.kit.app  # noqa: E402
from dual_arm_lula import create_raw_dual_arm_lula  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

ROBOT_PRIM_PATH = "/World/Robot"


def _spine_position(robot: SingleArticulation) -> float:
    names = list(robot.dof_names)
    try:
        index = names.index("franka_spine_vertical_joint")
    except ValueError:
        return 0.0
    positions = robot.get_joint_positions()
    return float(positions[index])


def _world_pose(robot: SingleArticulation) -> tuple[np.ndarray, np.ndarray]:
    position, orientation = robot.get_world_pose()
    return (
        np.asarray(position, dtype=np.float64),
        np.asarray(orientation, dtype=np.float64),
    )


def _apply_targets(
    robot: SingleArticulation,
    target_by_name: dict[str, float],
) -> None:
    names = list(robot.dof_names)
    ordered = [
        (names.index(name), float(value))
        for name, value in target_by_name.items()
    ]
    indices = np.asarray([index for index, _ in ordered], dtype=np.int64)
    positions = np.asarray([value for _, value in ordered], dtype=np.float32)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=positions,
            joint_indices=indices,
        )
    )


def _pose_record(
    pose: tuple[np.ndarray, np.ndarray],
) -> dict[str, list[float]]:
    position, orientation = pose
    return {
        "position": [round(float(value), 6) for value in position],
        "orientation_wxyz": [
            round(float(value), 6) for value in orientation
        ],
    }


def _yaw_from_wxyz(orientation: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in orientation)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _drive_base_to(
    world: World,
    robot: SingleArticulation,
    steering_ids: list[int],
    drive_ids: list[int],
    target_xy: np.ndarray,
    target_yaw: float,
    *,
    max_steps: int,
    max_speed: float,
) -> dict[str, float | int | bool | list[float]]:
    controller = robot.get_articulation_controller()
    position_tolerance = 0.03
    yaw_tolerance = math.radians(2.0)
    settled_steps = 0
    steps = 0

    for steps in range(1, max_steps + 1):
        position, orientation = _world_pose(robot)
        yaw = _yaw_from_wxyz(orientation)
        world_error = np.asarray(target_xy, dtype=np.float64) - position[:2]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        body_error = np.array(
            (
                cos_yaw * world_error[0] + sin_yaw * world_error[1],
                -sin_yaw * world_error[0] + cos_yaw * world_error[1],
            )
        )
        yaw_error = _wrap_to_pi(target_yaw - yaw)
        distance = float(np.linalg.norm(world_error))

        if distance <= position_tolerance and abs(yaw_error) <= yaw_tolerance:
            settled_steps += 1
            vx = 0.0
            vy = 0.0
            wz = 0.0
        else:
            settled_steps = 0
            body_command = 1.4 * body_error
            command_norm = float(np.linalg.norm(body_command))
            minimum_speed = min(0.16, max_speed)
            if 0.0 < command_norm < minimum_speed:
                body_command *= minimum_speed / command_norm
            command_norm = float(np.linalg.norm(body_command))
            if command_norm > max_speed:
                body_command *= max_speed / command_norm
            vx = float(body_command[0])
            vy = float(body_command[1])
            wz = float(np.clip(2.0 * yaw_error, -0.6, 0.6))

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
                joint_indices=np.asarray(steering_ids, dtype=np.int64),
            )
        )
        controller.apply_action(
            ArticulationAction(
                joint_velocities=drive_targets,
                joint_indices=np.asarray(drive_ids, dtype=np.int64),
            )
        )
        world.step(render=False)
        if settled_steps >= 30:
            break

    position, orientation = _world_pose(robot)
    final_yaw = _yaw_from_wxyz(orientation)
    position_error = float(np.linalg.norm(target_xy - position[:2]))
    yaw_error = abs(_wrap_to_pi(target_yaw - final_yaw))
    return {
        "passed": (
            position_error <= position_tolerance
            and yaw_error <= yaw_tolerance
        ),
        "steps": steps,
        "final_xy": [
            round(float(position[0]), 6),
            round(float(position[1]), 6),
        ],
        "target_xy": [
            round(float(target_xy[0]), 6),
            round(float(target_xy[1]), 6),
        ],
        "position_error_metres": round(position_error, 6),
        "yaw_error_degrees": round(math.degrees(yaw_error), 6),
        "position_tolerance_metres": position_tolerance,
        "yaw_tolerance_degrees": math.degrees(yaw_tolerance),
    }


def main() -> None:
    print("AUTONOMOUS_PROBE_PHASE validate_arguments", flush=True)
    if not math.isfinite(args_cli.lift_metres):
        raise ValueError("--lift-metres must be finite")
    if args_cli.motion_steps < 1 or args_cli.settle_steps < 0:
        raise ValueError("step counts must be non-negative")
    if (
        not math.isfinite(args_cli.base_distance_metres)
        or args_cli.base_distance_metres <= 0.0
        or args_cli.base_max_steps < 1
        or not math.isfinite(args_cli.base_max_speed)
        or args_cli.base_max_speed <= 0.0
    ):
        raise ValueError("base probe arguments must be finite and positive")

    room_path = Path(args_cli.room_usd).expanduser()
    robot_path = Path(args_cli.robot_usd).expanduser()
    franka_root = Path(args_cli.franka_root).expanduser()
    for required in (room_path, robot_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    groups = core._load_joint_groups(
        franka_root,
        args_cli.embodiment,
        include_browser_commands=False,
    )
    args_cli.task = "task3"
    robot_position = room_scene.resolve_robot_position(args_cli)
    robot_yaw = room_scene.resolve_robot_yaw(args_cli)

    print("AUTONOMOUS_PROBE_PHASE build_stage", flush=True)
    room_scene.build_stage(
        omni.kit.app.get_app(),
        room_path=room_path,
        robot_path=robot_path,
        task="task3",
        robot_position=robot_position,
        robot_rotation=room_scene.yaw_to_quat(robot_yaw),
        robot_yaw=robot_yaw,
        head_placement=args_cli.head_placement,
        dynamic_beans=True,
    )

    physics_scene_path = core._find_physics_scene_path() or "/physicsScene"
    world = World(
        physics_prim_path=physics_scene_path,
        stage_units_in_meters=1.0,
        physics_dt=1.0 / args_cli.physics_hz,
        rendering_dt=1.0 / args_cli.render_hz,
    )
    core.prepare_robot_prim(ROBOT_PRIM_PATH, args_cli)
    core._configure_drives(
        ROBOT_PRIM_PATH,
        lambda joint_name: get_profile_drive_gains(
            profile_cli.name, joint_name
        ),
    )
    articulation_root_path = core._find_articulation_root_path(ROBOT_PRIM_PATH)
    robot = SingleArticulation(
        prim_path=articulation_root_path,
        name="task3_probe_robot",
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
    ) = core.setup_robot_control(robot, groups, args_cli)
    for _ in range(args_cli.settle_steps):
        world.step(render=False)

    print("AUTONOMOUS_PROBE_PHASE initialize_raw_lula", flush=True)
    ik = create_raw_dual_arm_lula(
        robot.dof_names,
        robot.get_joint_positions,
        project_root=TASK3_ROOT,
    )
    print("AUTONOMOUS_PROBE_PHASE read_initial_tcp", flush=True)
    base_position, base_orientation = _world_pose(robot)
    spine = _spine_position(robot)
    initial_left, initial_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        spine,
    )
    target_left = initial_left[0] + np.array(
        (0.0, 0.0, args_cli.lift_metres)
    )
    target_right = initial_right[0] + np.array(
        (0.0, 0.0, args_cli.lift_metres)
    )

    print("AUTONOMOUS_PROBE_PHASE execute_motion", flush=True)
    solve_successes = 0
    for step in range(1, args_cli.motion_steps + 1):
        fraction = step / args_cli.motion_steps
        desired_left = (
            initial_left[0] + fraction * (target_left - initial_left[0])
        )
        desired_right = (
            initial_right[0] + fraction * (target_right - initial_right[0])
        )
        base_position, base_orientation = _world_pose(robot)
        result = ik.solve(
            desired_left,
            desired_right,
            initial_left[1],
            initial_right[1],
            spine_position=_spine_position(robot),
            base_position=base_position,
            base_orientation_wxyz=base_orientation,
        )
        if result.left_succeeded and result.right_succeeded:
            solve_successes += 1
        _apply_targets(robot, result.combined)
        world.step(render=False)

    for _ in range(args_cli.settle_steps):
        world.step(render=False)
    base_position, base_orientation = _world_pose(robot)
    final_left, final_right = ik.current_end_effector_poses(
        base_position,
        base_orientation,
        _spine_position(robot),
    )
    left_error = float(np.linalg.norm(final_left[0] - target_left))
    right_error = float(np.linalg.norm(final_right[0] - target_right))
    tolerance = 0.025
    arm_passed = (
        solve_successes == args_cli.motion_steps
        and left_error <= tolerance
        and right_error <= tolerance
    )

    print("AUTONOMOUS_PROBE_PHASE execute_base_outbound", flush=True)
    base_start, base_start_orientation = _world_pose(robot)
    base_start_yaw = _yaw_from_wxyz(base_start_orientation)
    forward_world = np.array(
        (math.cos(base_start_yaw), math.sin(base_start_yaw))
    )
    outbound_target = (
        base_start[:2] + args_cli.base_distance_metres * forward_world
    )
    outbound = _drive_base_to(
        world,
        robot,
        steering_ids,
        drive_ids,
        outbound_target,
        base_start_yaw,
        max_steps=args_cli.base_max_steps,
        max_speed=args_cli.base_max_speed,
    )
    print("AUTONOMOUS_PROBE_PHASE execute_base_return", flush=True)
    returned = _drive_base_to(
        world,
        robot,
        steering_ids,
        drive_ids,
        base_start[:2],
        base_start_yaw,
        max_steps=args_cli.base_max_steps,
        max_speed=args_cli.base_max_speed,
    )
    passed = arm_passed and bool(outbound["passed"]) and bool(
        returned["passed"]
    )
    payload = {
        "kind": "actuator_driven_dual_arm_and_base_probe",
        "passed": passed,
        "gripper_profile": "robotiq",
        "head_placement": args_cli.head_placement,
        "motion_steps": args_cli.motion_steps,
        "dual_solve_successes": solve_successes,
        "target_lift_metres": args_cli.lift_metres,
        "tolerance_metres": tolerance,
        "left_error_metres": round(left_error, 6),
        "right_error_metres": round(right_error, 6),
        "initial": {
            "left": _pose_record(initial_left),
            "right": _pose_record(initial_right),
        },
        "target": {
            "left_position": [
                round(float(value), 6) for value in target_left
            ],
            "right_position": [
                round(float(value), 6) for value in target_right
            ],
        },
        "final": {
            "left": _pose_record(final_left),
            "right": _pose_record(final_right),
        },
        "task_objects_teleported": False,
        "base": {
            "start_xy": [
                round(float(base_start[0]), 6),
                round(float(base_start[1]), 6),
            ],
            "commanded_distance_metres": args_cli.base_distance_metres,
            "outbound": outbound,
            "return": returned,
        },
    }
    print(
        "AUTONOMOUS_PROBE_RESULT " + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    if not passed:
        raise RuntimeError("Actuator-driven arm-and-base probe failed")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:  # noqa: BLE001 - preserve a nonzero process result
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
