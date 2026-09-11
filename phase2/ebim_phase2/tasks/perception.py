"""Calibrated RGB-D projection and task-goal construction.

Pixel keypoints must come from a detector/tracker or recorded annotations.
This module does not invent detections, target assignments, or material-side
labels. Construct fixed goals once per acquisition/placement attempt; do not
continually add a lift offset to the moving object's current location.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import Pose, quat_multiply, vector


@dataclass(frozen=True)
class RGBDCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float
    world_from_camera: Pose
    calibration_id: str

    def __post_init__(self):
        if not self.calibration_id or not all(math.isfinite(x) for x in (self.fx, self.fy, self.cx, self.cy, self.depth_scale)):
            raise ValueError("finite camera calibration and calibration_id required")
        if min(self.fx, self.fy, self.depth_scale) <= 0:
            raise ValueError("focal lengths/depth scale must be positive")


def project_keypoint(depth, uv, calibration: RGBDCalibration, *, radius: int = 1,
                     max_depth_spread_m: float = 0.02) -> tuple[float, float, float]:
    """Median depth in a small registered RGB-D patch; reject depth edges."""
    array = np.asarray(depth)
    if array.ndim != 2 or radius < 0 or radius > 5:
        raise ValueError("depth must be H×W; patch radius must be in [0,5]")
    u, v = vector(uv, 2, "pixel uv")
    height, width = array.shape
    if not 0 <= u < width or not 0 <= v < height:
        raise ValueError("keypoint outside calibrated image")
    x, y = int(round(u)), int(round(v))
    x, y = min(x, width-1), min(y, height-1)
    patch = array[max(0, y-radius):min(height, y+radius+1), max(0, x-radius):min(width, x+radius+1)].astype(float)*calibration.depth_scale
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < max(1, patch.size//2):
        raise ValueError("insufficient valid keypoint depth")
    if np.ptp(valid) > max_depth_spread_m:
        raise ValueError("depth discontinuity at keypoint; reacquire before contact")
    z = float(np.median(valid))
    camera_point = np.array([(u-calibration.cx)*z/calibration.fx, (v-calibration.cy)*z/calibration.fy, z])
    transform = calibration.world_from_camera
    q = np.asarray(transform.wxyz)
    rotated = quat_multiply(quat_multiply(q, np.r_[0., camera_point]), q*np.array([1., -1., -1., -1.]))[1:]
    return tuple(rotated + transform.xyz)


def pad_goals(source_grasps: dict[str, Pose], target_grasps: dict[str, Pose], *,
              clearance_m: float = 0.05, peel_offsets_world: dict[str, tuple] | None = None) -> dict[str, Pose]:
    """Build one attempt's selected-arm goals from calibrated TCP grasp poses.

    Source/target poses must already include the side-specific tool offset and
    intended orientation. Raising uses world +Z, independent of wrist pose.
    Peel displacement is explicitly supplied by the site material calibration.
    """
    if not source_grasps or set(source_grasps) != set(target_grasps) or set(source_grasps)-{'left','right'}:
        raise ValueError('Matching selected-arm grasp poses required at source and assigned target')
    if not math.isfinite(clearance_m) or not 0 < clearance_m <= 0.3:
        raise ValueError("clearance_m must be in (0,0.3]")
    result = {}
    for arm in source_grasps:
        source, target = source_grasps[arm], target_grasps[arm]
        raised_source = Pose(tuple(np.asarray(source.xyz)+(0, 0, clearance_m)), source.wxyz)
        raised_target = Pose(tuple(np.asarray(target.xyz)+(0, 0, clearance_m)), target.wxyz)
        result.update({f"pad_pregrasp_{arm}": raised_source, f"pad_grasp_{arm}": source,
                       f"pad_lift_{arm}": raised_source, f"pad_hover_{arm}": raised_target,
                       f"pad_place_{arm}": target})
        if peel_offsets_world is not None:
            offset = vector(peel_offsets_world[arm], 3, "calibrated peel offset")
            result[f"pad_peel_{arm}"] = Pose(tuple(np.asarray(source.xyz)+offset), source.wxyz)
    return result
