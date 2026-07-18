#!/usr/bin/env python3
"""Deterministic, import-safe SE(2) route validation for prism proxies.

Robot collision geometry is represented as a union of convex prisms in the
robot base frame. Environment collision geometry is represented by convex
prisms in the world frame. The validator samples translation and shortest-path
yaw interpolation, then performs conservative 2.5-D separating-axis tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


Point2 = tuple[float, float]


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


@dataclass(frozen=True, order=True)
class Pose2:
    x: float
    y: float
    yaw: float

    def validate(self) -> None:
        if not all(_finite(value) for value in (self.x, self.y, self.yaw)):
            raise ValueError("route poses must be finite")


@dataclass(frozen=True)
class PrismProxy:
    vertices_xy: tuple[Point2, ...]
    z_min: float
    z_max: float
    source_path: str

    def validate(self) -> None:
        if len(self.vertices_xy) < 3:
            raise ValueError(
                f"{self.source_path or '<unknown>'} has fewer than 3 vertices"
            )
        if not self.source_path:
            raise ValueError("proxy source_path must be non-empty")
        if not all(
            _finite(value)
            for point in self.vertices_xy
            for value in point
        ):
            raise ValueError(f"{self.source_path} has non-finite vertices")
        if not (
            _finite(self.z_min)
            and _finite(self.z_max)
            and self.z_min <= self.z_max
        ):
            raise ValueError(f"{self.source_path} has invalid z bounds")
        if abs(_polygon_area(self.vertices_xy)) <= 1e-12:
            raise ValueError(f"{self.source_path} has a degenerate polygon")


@dataclass(frozen=True)
class RouteValidationConfig:
    max_translation_step: float = 0.025
    max_yaw_step_rad: float = math.radians(2.0)
    clearance_margin: float = 0.08
    base_z: float = 0.0
    collision_coverage_complete: bool = True

    def validate(self) -> None:
        if not (
            _finite(self.max_translation_step)
            and self.max_translation_step > 0.0
            and _finite(self.max_yaw_step_rad)
            and self.max_yaw_step_rad > 0.0
            and _finite(self.clearance_margin)
            and self.clearance_margin >= 0.0
            and _finite(self.base_z)
        ):
            raise ValueError("route validation limits must be finite")
        if not self.collision_coverage_complete:
            raise ValueError("collision geometry coverage is incomplete")


def convex_hull(points: Iterable[Point2]) -> tuple[Point2, ...]:
    """Return a deterministic counter-clockwise convex hull."""

    unique = sorted(
        {
            (float(point[0]), float(point[1]))
            for point in points
            if len(point) == 2
        }
    )
    if len(unique) < 3:
        raise ValueError("at least three unique points are required")
    if not all(_finite(value) for point in unique for value in point):
        raise ValueError("convex hull points must be finite")

    def cross(origin: Point2, a: Point2, b: Point2) -> float:
        return (
            (a[0] - origin[0]) * (b[1] - origin[1])
            - (a[1] - origin[1]) * (b[0] - origin[0])
        )

    lower: list[Point2] = []
    for point in unique:
        while len(lower) >= 2 and cross(
            lower[-2], lower[-1], point
        ) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[Point2] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(
            upper[-2], upper[-1], point
        ) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3 or abs(_polygon_area(hull)) <= 1e-12:
        raise ValueError("points do not form a non-degenerate polygon")
    return hull


def transform_polygon(vertices: Sequence[Point2], pose: Pose2) -> tuple[Point2, ...]:
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    return tuple(
        (
            pose.x + cosine * point[0] - sine * point[1],
            pose.y + sine * point[0] + cosine * point[1],
        )
        for point in vertices
    )


def prisms_overlap(
    robot: PrismProxy,
    obstacle: PrismProxy,
    pose: Pose2,
    *,
    margin: float,
    base_z: float = 0.0,
) -> bool:
    """Return true when two prisms overlap or approach within ``margin``."""

    robot_z_min = base_z + robot.z_min
    robot_z_max = base_z + robot.z_max
    if (
        robot_z_max + margin < obstacle.z_min
        or obstacle.z_max + margin < robot_z_min
    ):
        return False
    robot_world = transform_polygon(robot.vertices_xy, pose)
    return _polygons_overlap_with_margin(
        robot_world, obstacle.vertices_xy, margin
    )


def sample_route(
    waypoints: Sequence[Pose2],
    config: RouteValidationConfig,
) -> list[dict[str, Any]]:
    config.validate()
    if not waypoints:
        raise ValueError("at least one route waypoint is required")
    for pose in waypoints:
        pose.validate()
    if len(waypoints) == 1:
        return [
            {
                "pose": waypoints[0],
                "segment_index": 0,
                "fraction": 0.0,
                "interpolation_allowance": 0.0,
            }
        ]

    samples: list[dict[str, Any]] = []
    for segment_index, (start, end) in enumerate(
        zip(waypoints, waypoints[1:])
    ):
        distance = math.hypot(end.x - start.x, end.y - start.y)
        yaw_delta = wrap_to_pi(end.yaw - start.yaw)
        count = max(
            1,
            int(math.ceil(distance / config.max_translation_step)),
            int(math.ceil(abs(yaw_delta) / config.max_yaw_step_rad)),
        )
        actual_ds = distance / count
        actual_dyaw = abs(yaw_delta) / count
        start_index = 0 if segment_index == 0 else 1
        for index in range(start_index, count + 1):
            fraction = index / count
            samples.append(
                {
                    "pose": Pose2(
                        start.x + fraction * (end.x - start.x),
                        start.y + fraction * (end.y - start.y),
                        wrap_to_pi(start.yaw + fraction * yaw_delta),
                    ),
                    "segment_index": segment_index,
                    "fraction": fraction,
                    "translation_step": actual_ds,
                    "yaw_step_rad": actual_dyaw,
                }
            )
    return samples


def validate_route(
    robot_proxies: Sequence[PrismProxy],
    obstacle_proxies: Sequence[PrismProxy],
    waypoints: Sequence[Pose2],
    config: RouteValidationConfig = RouteValidationConfig(),
) -> dict[str, Any]:
    """Validate a complete route and return a deterministic certificate."""

    try:
        config.validate()
        if not robot_proxies:
            raise ValueError("robot collision proxy set is empty")
        if not obstacle_proxies:
            raise ValueError("environment collision proxy set is empty")
        for proxy in (*robot_proxies, *obstacle_proxies):
            proxy.validate()
        samples = sample_route(waypoints, config)
    except Exception as error:  # noqa: BLE001 - fail-closed certificate
        return _certificate(
            {
                "passed": False,
                "failure": "invalid_or_incomplete_geometry",
                "reason": f"{type(error).__name__}: {error}",
                "checked_samples": 0,
                "first_unsafe_sample": None,
                "robot_proxy_count": len(robot_proxies),
                "obstacle_proxy_count": len(obstacle_proxies),
                "config": asdict(config),
            }
        )

    ordered_robot = sorted(robot_proxies, key=lambda item: item.source_path)
    ordered_obstacles = sorted(
        obstacle_proxies, key=lambda item: item.source_path
    )
    maximum_radius = max(
        math.hypot(point[0], point[1])
        for proxy in ordered_robot
        for point in proxy.vertices_xy
    )
    checked_pairs = 0
    for sample_index, sample in enumerate(samples):
        translation_allowance = 0.5 * float(
            sample.get("translation_step", 0.0)
        )
        yaw_allowance = 2.0 * maximum_radius * math.sin(
            0.25 * float(sample.get("yaw_step_rad", 0.0))
        )
        effective_margin = (
            config.clearance_margin
            + translation_allowance
            + yaw_allowance
        )
        pose = sample["pose"]
        for robot in ordered_robot:
            for obstacle in ordered_obstacles:
                checked_pairs += 1
                if prisms_overlap(
                    robot,
                    obstacle,
                    pose,
                    margin=effective_margin,
                    base_z=config.base_z,
                ):
                    return _certificate(
                        {
                            "passed": False,
                            "failure": "preflight_overlap",
                            "reason": (
                                "inflated robot collision proxy overlaps an "
                                "environment collision proxy"
                            ),
                            "checked_samples": sample_index + 1,
                            "checked_proxy_pairs": checked_pairs,
                            "total_samples": len(samples),
                            "first_unsafe_sample": {
                                "sample_index": sample_index,
                                "segment_index": sample["segment_index"],
                                "fraction": round(
                                    float(sample["fraction"]), 9
                                ),
                                "pose": _pose_record(pose),
                                "effective_margin_metres": round(
                                    effective_margin, 9
                                ),
                                "robot_source_path": robot.source_path,
                                "obstacle_source_path": (
                                    obstacle.source_path
                                ),
                            },
                            "robot_proxy_count": len(ordered_robot),
                            "obstacle_proxy_count": len(
                                ordered_obstacles
                            ),
                            "config": asdict(config),
                        }
                    )

    return _certificate(
        {
            "passed": True,
            "failure": None,
            "reason": None,
            "checked_samples": len(samples),
            "checked_proxy_pairs": checked_pairs,
            "total_samples": len(samples),
            "first_unsafe_sample": None,
            "robot_proxy_count": len(ordered_robot),
            "obstacle_proxy_count": len(ordered_obstacles),
            "robot_source_paths": [
                proxy.source_path for proxy in ordered_robot
            ],
            "obstacle_source_paths": [
                proxy.source_path for proxy in ordered_obstacles
            ],
            "config": asdict(config),
        }
    )


def distance_to_polyline(
    pose: Pose2, waypoints: Sequence[Pose2]
) -> float:
    """Return planar cross-track distance from ``pose`` to a route."""

    pose.validate()
    if not waypoints:
        raise ValueError("at least one waypoint is required")
    for waypoint in waypoints:
        waypoint.validate()
    if len(waypoints) == 1:
        return math.hypot(
            pose.x - waypoints[0].x, pose.y - waypoints[0].y
        )
    distances = []
    for start, end in zip(waypoints, waypoints[1:]):
        dx = end.x - start.x
        dy = end.y - start.y
        denominator = dx * dx + dy * dy
        if denominator <= 1e-18:
            distances.append(math.hypot(pose.x - start.x, pose.y - start.y))
            continue
        fraction = min(
            1.0,
            max(
                0.0,
                (
                    (pose.x - start.x) * dx
                    + (pose.y - start.y) * dy
                )
                / denominator,
            ),
        )
        nearest_x = start.x + fraction * dx
        nearest_y = start.y + fraction * dy
        distances.append(
            math.hypot(pose.x - nearest_x, pose.y - nearest_y)
        )
    return min(distances)


def _pose_record(pose: Pose2) -> dict[str, float]:
    return {
        "x": round(pose.x, 9),
        "y": round(pose.y, 9),
        "yaw_rad": round(pose.yaw, 9),
    }


def _polygon_area(vertices: Sequence[Point2]) -> float:
    return 0.5 * sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(vertices, (*vertices[1:], vertices[0]))
    )


def _axes(vertices: Sequence[Point2]) -> Iterable[Point2]:
    for start, end in zip(vertices, (*vertices[1:], vertices[0])):
        edge_x = end[0] - start[0]
        edge_y = end[1] - start[1]
        length = math.hypot(edge_x, edge_y)
        if length <= 1e-15:
            continue
        yield (-edge_y / length, edge_x / length)


def _project(vertices: Sequence[Point2], axis: Point2) -> tuple[float, float]:
    values = [
        point[0] * axis[0] + point[1] * axis[1]
        for point in vertices
    ]
    return min(values), max(values)


def _polygons_overlap_with_margin(
    left: Sequence[Point2],
    right: Sequence[Point2],
    margin: float,
) -> bool:
    if not _finite(margin) or margin < 0.0:
        raise ValueError("polygon margin must be finite and non-negative")
    for axis in (*tuple(_axes(left)), *tuple(_axes(right))):
        left_min, left_max = _project(left, axis)
        right_min, right_max = _project(right, axis)
        gap = max(right_min - left_max, left_min - right_max, 0.0)
        if gap > margin:
            return False
    return True


def _certificate(record: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {
        **record,
        "certificate_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }
