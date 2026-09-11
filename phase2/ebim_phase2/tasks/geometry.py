"""Small geometric tools using measured poses and a measured obstacle map."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np


def vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def quat_multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


@dataclass(frozen=True)
class Pose:
    xyz: tuple[float, float, float]
    wxyz: tuple[float, float, float, float]

    def __post_init__(self):
        vector(self.xyz, 3, "xyz")
        quat = vector(self.wxyz, 4, "wxyz")
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3):
            raise ValueError("pose quaternion must be normalized (wxyz)")

    def offset(self, xyz=(0., 0., 0.), wxyz=(1., 0., 0., 0.)) -> "Pose":
        """Compose a local offset; every pose is in the same calibrated frame."""
        q = np.asarray(self.wxyz)
        inverse = q * np.array([1., -1., -1., -1.])
        delta = quat_multiply(quat_multiply(q, np.r_[0., vector(xyz, 3, "offset")]), inverse)[1:]
        rotation = quat_multiply(q, vector(wxyz, 4, "offset quaternion"))
        return Pose(tuple(np.asarray(self.xyz) + delta), tuple(rotation))


def pose_error(current: Pose, target: Pose) -> np.ndarray:
    """World-frame translational error and shortest quaternion log error."""
    inverse = np.asarray(current.wxyz) * np.array([1., -1., -1., -1.])
    error = quat_multiply(target.wxyz, inverse)
    if error[0] < 0:
        error *= -1
    norm = np.linalg.norm(error[1:])
    rotation = np.zeros(3) if norm < 1e-10 else error[1:] * (2 * math.atan2(norm, error[0]) / norm)
    return np.r_[np.asarray(target.xyz) - current.xyz, rotation]


@dataclass(frozen=True)
class Rectangle:
    minimum: tuple[float, float]
    maximum: tuple[float, float]

    def __post_init__(self):
        lo, hi = vector(self.minimum, 2, "minimum"), vector(self.maximum, 2, "maximum")
        if np.any(lo > hi):
            raise ValueError("rectangle minimum exceeds maximum")

    def intersects(self, start, goal, margin: float) -> bool:
        """Exact slab intersection of segment with footprint-expanded AABB."""
        near, far = 0., 1.
        for axis in range(2):
            low, high = self.minimum[axis]-margin, self.maximum[axis]+margin
            delta = goal[axis] - start[axis]
            if abs(delta) < 1e-12:
                if start[axis] < low or start[axis] > high:
                    return False
            else:
                t0, t1 = (low-start[axis])/delta, (high-start[axis])/delta
                near, far = max(near, min(t0, t1)), min(far, max(t0, t1))
                if near > far:
                    return False
        return True


def route(start, goal, obstacles: tuple[Rectangle, ...], clearance: float) -> list[tuple[float, float]]:
    """Visibility graph around inflated measured rectangles; never cuts corners."""
    start, goal = tuple(vector(start, 2, "start")), tuple(vector(goal, 2, "goal"))
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("clearance must be positive")
    if len(obstacles) > 128:
        raise ValueError("obstacle map exceeds bounded planning budget")
    clear = lambda a, b: not any(rect.intersects(a, b, clearance) for rect in obstacles)
    if not clear(start, start) or not clear(goal, goal):
        raise ValueError("start or goal intersects the measured obstacle envelope")
    if clear(start, goal):
        return [start, goal]
    points = [start, goal]
    for rect in obstacles:
        lo, hi = np.asarray(rect.minimum)-clearance-0.005, np.asarray(rect.maximum)+clearance+0.005
        points.extend((float(x), float(y)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                      if clear((x, y), (x, y)))
    queue, costs, previous = [(0., 0)], {0: 0.}, {}
    while queue:
        cost, index = heapq.heappop(queue)
        if cost > costs[index]:
            continue
        if index == 1:
            indices = [1]
            while indices[-1] != 0:
                indices.append(previous[indices[-1]])
            return [points[i] for i in reversed(indices)]
        for nxt, point in enumerate(points):
            if nxt == index or not clear(points[index], point):
                continue
            candidate = cost + math.dist(points[index], point)
            if candidate < costs.get(nxt, math.inf):
                costs[nxt], previous[nxt] = candidate, index
                heapq.heappush(queue, (candidate, nxt))
    raise ValueError("no collision-cleared route through measured map")
