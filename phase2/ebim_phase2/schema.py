"""Released schemas: preserve native training layouts, translate only at the boundary."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

CAMERAS = ("observation.images.head", "observation.images.wrist_left", "observation.images.wrist_right")
JOINT_ACTION_INDICES = np.array([*range(7), *range(8, 15)])
JOINT_STATE_INDICES = np.array([*range(7), *range(21, 28)])


@dataclass(frozen=True)
class Profile:
    task: int
    state_dim: int
    action_dim: int
    gripper_state_indices: tuple[int, int]
    force_slices: tuple[slice, slice]


PROFILES = {
    1: Profile(1, 62, 23, (20, 41), (slice(14, 17), slice(35, 38))),
    2: Profile(2, 42, 17, (7, 28), (slice(15, 18), slice(36, 39))),
}


def check_metadata(info: dict, task: int) -> None:
    p = PROFILES[task]
    if info["codebase_version"] != "v2.1" or info["fps"] != 20:
        raise ValueError("Expected the audited LeRobot v2.1 / 20 Hz release")
    for key, dim in (("observation.state", p.state_dim), ("action", p.action_dim)):
        feature = info["features"][key]
        if feature["shape"] != [dim] or len(feature["names"]) != dim:
            raise ValueError(f"Unexpected {key} schema")
    names = info["features"]["action"]["names"]
    if not ("left" in names[0] and "joint1" in names[0] and "percent" in names[7]
            and "right" in names[8] and "joint1" in names[8] and "percent" in names[15]
            and names[-1] == "spine_target_height_value"):
        raise ValueError("Action semantic order changed")
    states = info['features']['observation.state']['names']
    for arm, action_offset, state_offset in (('left', 0, 0), ('right', 8, 21)):
        for joint in range(7):
            for name in (names[action_offset+joint], states[state_offset+joint]):
                if arm not in name or not name.endswith(f'joint{joint+1}'):
                    raise ValueError('Joint semantic order changed')
    for index in p.gripper_state_indices:
        if not states[index].endswith('knuckle_joint'):
            raise ValueError('Measured gripper semantic order changed')
    for component in p.force_slices:
        if not all(name.endswith(f'force_{axis}') for name, axis in zip(states[component], 'xyz')):
            raise ValueError('Wrench semantic order changed')
    if task == 1:
        axes = ('linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z')
        if not all(name.endswith(axis) for name, axis in zip(names[16:22], axes)) or not states[61].endswith('spine_z'):
            raise ValueError('Base/spine semantic order changed')
    for camera in CAMERAS:
        if info["features"][camera]["dtype"] != "video":
            raise ValueError(f"Missing video camera: {camera}")


def action23(native: np.ndarray, task: int, *, spine_scale_m_per_native: float | None = None,
             spine_offset_m: float | None = None, joint_action_scale=None,
             joint_action_offset_rad=None) -> np.ndarray:
    native = np.asarray(native, dtype=np.float64)
    if native.shape != (PROFILES[task].action_dim,) or not np.isfinite(native).all():
        raise ValueError("Invalid native action")
    result = native.copy() if task == 1 else np.zeros(23)
    result[:16] = native[:16]
    # Task2 contains values around 434, while Task1 contains values around .3.
    # The published feature name does not establish mm, ticks or an origin.
    # Never silently label native Task2 spine values as metres.
    if task == 2 or spine_scale_m_per_native is not None or spine_offset_m is not None:
        if (spine_scale_m_per_native is None or spine_offset_m is None
                or not np.isfinite([spine_scale_m_per_native, spine_offset_m]).all()
                or spine_scale_m_per_native <= 0):
            raise ValueError("Task2 requires calibrated spine_scale_m_per_native and spine_offset_m")
        result[22] = native[-1] * spine_scale_m_per_native + spine_offset_m
    if joint_action_scale is not None or joint_action_offset_rad is not None:
        scale, offset = np.asarray(joint_action_scale), np.asarray(joint_action_offset_rad)
        if (scale.shape != (14,) or offset.shape != (14,) or not np.isfinite(scale).all()
                or not np.isfinite(offset).all() or np.any(scale == 0)):
            raise ValueError('Joint command conversion needs 14 finite nonzero scales and 14 finite offsets')
        result[JOINT_ACTION_INDICES] = result[JOINT_ACTION_INDICES]*scale+offset
    return result


def command_dict(action: np.ndarray) -> dict:
    if np.shape(action) != (23,) or not np.isfinite(action).all():
        raise ValueError("Expected finite action23")
    return {"left_joint_positions_rad": action[:7].tolist(), "left_gripper_width_percent": float(action[7]),
            "right_joint_positions_rad": action[8:15].tolist(), "right_gripper_width_percent": float(action[15]),
            "base_twist": action[16:22].tolist(), "spine_height_m": float(action[22])}
