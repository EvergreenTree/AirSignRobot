#!/usr/bin/env python3
"""Deterministic, import-safe SE(2) route validation for prism proxies.

Robot collision geometry is represented as a union of convex prisms in the
robot base frame. Environment collision geometry is represented by convex
prisms in the world frame. The validator samples translation and shortest-path
yaw interpolation, then performs conservative 2.5-D separating-axis tests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, fields
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
    execution_translation_tolerance: float = 0.04
    execution_yaw_tolerance_rad: float = math.radians(2.0)
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
            and _finite(self.execution_translation_tolerance)
            and self.execution_translation_tolerance >= 0.0
            and _finite(self.execution_yaw_tolerance_rad)
            and self.execution_yaw_tolerance_rad >= 0.0
            and self.execution_yaw_tolerance_rad <= math.pi
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

    segment_specs: list[dict[str, Any]] = []
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
        segment_specs.append(
            {
                "segment_index": segment_index,
                "start": start,
                "end": end,
                "yaw_delta": yaw_delta,
                "count": count,
                "translation_step": actual_ds,
                "yaw_step_rad": actual_dyaw,
            }
        )

    samples: list[dict[str, Any]] = []
    for spec in segment_specs:
        segment_index = int(spec["segment_index"])
        start = spec["start"]
        end = spec["end"]
        yaw_delta = float(spec["yaw_delta"])
        count = int(spec["count"])
        actual_ds = float(spec["translation_step"])
        actual_dyaw = float(spec["yaw_step_rad"])
        start_index = 0 if segment_index == 0 else 1
        for index in range(start_index, count + 1):
            fraction = index / count
            sample_ds = actual_ds
            sample_dyaw = actual_dyaw
            if (
                index == count
                and segment_index + 1 < len(segment_specs)
            ):
                next_spec = segment_specs[segment_index + 1]
                sample_ds = max(
                    sample_ds,
                    float(next_spec["translation_step"]),
                )
                sample_dyaw = max(
                    sample_dyaw,
                    float(next_spec["yaw_step_rad"]),
                )
            samples.append(
                {
                    "pose": Pose2(
                        start.x + fraction * (end.x - start.x),
                        start.y + fraction * (end.y - start.y),
                        wrap_to_pi(start.yaw + fraction * yaw_delta),
                    ),
                    "segment_index": segment_index,
                    "fraction": fraction,
                    "translation_step": sample_ds,
                    "yaw_step_rad": sample_dyaw,
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
        _require_unique_source_paths(robot_proxies, label="robot")
        _require_unique_source_paths(obstacle_proxies, label="environment")
        samples = sample_route(waypoints, config)
    except Exception as error:  # noqa: BLE001 - fail-closed certificate
        return _certificate(
            {
                "certificate_schema_version": 2,
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

    ordered_robot = sorted(
        robot_proxies, key=lambda item: _canonical_json(_proxy_record(item))
    )
    ordered_obstacles = sorted(
        obstacle_proxies,
        key=lambda item: _canonical_json(_proxy_record(item)),
    )
    validated_inputs = _validated_input_record(
        ordered_robot,
        ordered_obstacles,
        waypoints,
    )
    maximum_radius = max(
        math.hypot(point[0], point[1])
        for proxy in ordered_robot
        for point in proxy.vertices_xy
    )
    maximum_spatial_radius = max(
        math.sqrt(
            point[0] * point[0]
            + point[1] * point[1]
            + max(abs(proxy.z_min), abs(proxy.z_max)) ** 2
        )
        for proxy in ordered_robot
        for point in proxy.vertices_xy
    )
    execution_yaw_allowance = (
        2.0
        * maximum_radius
        * math.sin(0.5 * config.execution_yaw_tolerance_rad)
    )
    execution_envelope = {
        "translation_tolerance_metres": (
            config.execution_translation_tolerance
        ),
        "yaw_tolerance_rad": config.execution_yaw_tolerance_rad,
        "maximum_robot_planar_radius_metres": maximum_radius,
        "maximum_robot_spatial_radius_metres": maximum_spatial_radius,
        "yaw_arc_allowance_metres": execution_yaw_allowance,
        "total_planar_allowance_metres": (
            config.execution_translation_tolerance
            + execution_yaw_allowance
        ),
    }
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
            + config.execution_translation_tolerance
            + execution_yaw_allowance
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
                            "certificate_schema_version": 2,
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
                            "execution_envelope": execution_envelope,
                            "validated_inputs": validated_inputs,
                            "config": asdict(config),
                        }
                    )

    return _certificate(
        {
            "certificate_schema_version": 2,
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
            "execution_envelope": execution_envelope,
            "validated_inputs": validated_inputs,
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


def route_pose_deviation(
    pose: Pose2,
    waypoints: Sequence[Pose2],
) -> dict[str, float | int]:
    """Measure translation and yaw deviation from the nearest route pose."""

    pose.validate()
    if not waypoints:
        raise ValueError("at least one waypoint is required")
    for waypoint in waypoints:
        waypoint.validate()
    if len(waypoints) == 1:
        return {
            "segment_index": 0,
            "fraction": 0.0,
            "cross_track_metres": math.hypot(
                pose.x - waypoints[0].x,
                pose.y - waypoints[0].y,
            ),
            "expected_yaw_rad": waypoints[0].yaw,
            "yaw_error_rad": abs(
                wrap_to_pi(pose.yaw - waypoints[0].yaw)
            ),
        }

    candidates: list[dict[str, float | int]] = []
    for segment_index, (start, end) in enumerate(
        zip(waypoints, waypoints[1:])
    ):
        dx = end.x - start.x
        dy = end.y - start.y
        denominator = dx * dx + dy * dy
        if denominator <= 1e-18:
            yaw_delta = wrap_to_pi(end.yaw - start.yaw)
            fraction = (
                0.0
                if abs(yaw_delta) <= 1e-18
                else min(
                    1.0,
                    max(
                        0.0,
                        wrap_to_pi(pose.yaw - start.yaw) / yaw_delta,
                    ),
                )
            )
        else:
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
        yaw_delta = wrap_to_pi(end.yaw - start.yaw)
        expected_yaw = wrap_to_pi(start.yaw + fraction * yaw_delta)
        candidates.append(
            {
                "segment_index": segment_index,
                "fraction": fraction,
                "cross_track_metres": math.hypot(
                    pose.x - nearest_x,
                    pose.y - nearest_y,
                ),
                "expected_yaw_rad": expected_yaw,
                "yaw_error_rad": abs(
                    wrap_to_pi(pose.yaw - expected_yaw)
                ),
            }
        )
    return min(
        candidates,
        key=lambda record: (
            float(record["cross_track_metres"]),
            float(record["yaw_error_rad"]),
            int(record["segment_index"]),
        ),
    )


def route_execution_tube_record(
    pose: Pose2,
    waypoints: Sequence[Pose2],
    *,
    translation_tolerance: float,
    yaw_tolerance_rad: float,
) -> dict[str, Any]:
    """Fail closed when a measured pose leaves its certified SE(2) tube."""

    try:
        translation_limit = float(translation_tolerance)
        yaw_limit = float(yaw_tolerance_rad)
        if (
            not _finite(translation_limit)
            or translation_limit < 0.0
            or not _finite(yaw_limit)
            or yaw_limit < 0.0
            or yaw_limit > math.pi
        ):
            raise ValueError(
                "execution-tube tolerances must be finite and non-negative"
            )
        deviation = route_pose_deviation(pose, waypoints)
        cross_track = float(deviation["cross_track_metres"])
        yaw_error = float(deviation["yaw_error_rad"])
    except Exception as error:  # noqa: BLE001 - pure fail-closed record
        return {
            "passed": False,
            "failure": "invalid_execution_tube",
            "reason": f"{type(error).__name__}: {error}",
            "translation_within_tolerance": False,
            "yaw_within_tolerance": False,
            "translation_tolerance_metres": translation_tolerance,
            "yaw_tolerance_rad": yaw_tolerance_rad,
            "deviation": None,
        }
    translation_passed = cross_track <= translation_limit
    yaw_passed = yaw_error <= yaw_limit
    failure = None
    if not translation_passed:
        failure = "cross_track_violation"
    elif not yaw_passed:
        failure = "route_yaw_violation"
    return {
        "passed": translation_passed and yaw_passed,
        "failure": failure,
        "reason": (
            None
            if failure is None
            else "measured pose left the certified SE(2) execution tube"
        ),
        "translation_within_tolerance": translation_passed,
        "yaw_within_tolerance": yaw_passed,
        "translation_tolerance_metres": translation_limit,
        "yaw_tolerance_rad": yaw_limit,
        "deviation": deviation,
    }


def base_nonplanar_deviation_record(
    *,
    position_z: float,
    orientation_wxyz: Sequence[float],
    reference_z: float,
    maximum_robot_radius: float,
    tolerance: float,
) -> dict[str, Any]:
    """Bound base z/tilt motion omitted by an SE(2) route model."""

    try:
        current_z = float(position_z)
        captured_z = float(reference_z)
        radius = float(maximum_robot_radius)
        limit = float(tolerance)
        quaternion = tuple(float(value) for value in orientation_wxyz)
        if len(quaternion) != 4:
            raise ValueError("base orientation must be a wxyz quaternion")
        if not all(
            _finite(value)
            for value in (
                current_z,
                captured_z,
                radius,
                limit,
                *quaternion,
            )
        ):
            raise ValueError("non-planar base inputs must be finite")
        if radius < 0.0 or limit < 0.0:
            raise ValueError(
                "base radius and non-planar tolerance must be non-negative"
            )
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-12:
            raise ValueError("base orientation quaternion is zero")
        _w, x, y, _z = (value / norm for value in quaternion)
        world_up_alignment = max(
            -1.0,
            min(1.0, 1.0 - 2.0 * (x * x + y * y)),
        )
        tilt = math.acos(world_up_alignment)
        z_error = abs(current_z - captured_z)
        tilt_arc = 2.0 * radius * math.sin(0.5 * tilt)
        combined_escape = z_error + tilt_arc
    except Exception as error:  # noqa: BLE001 - pure fail-closed record
        return {
            "passed": False,
            "failure": "invalid_nonplanar_base_pose",
            "reason": f"{type(error).__name__}: {error}",
            "z_error_metres": None,
            "tilt_rad": None,
            "tilt_arc_metres": None,
            "combined_escape_metres": None,
            "tolerance_metres": tolerance,
        }
    passed = combined_escape <= limit or math.isclose(
        combined_escape,
        limit,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    return {
        "passed": passed,
        "failure": None if passed else "nonplanar_base_envelope_violation",
        "reason": (
            None
            if passed
            else "base z/tilt motion exceeded the certified 2.5-D allowance"
        ),
        "z_error_metres": z_error,
        "tilt_rad": tilt,
        "tilt_arc_metres": tilt_arc,
        "combined_escape_metres": combined_escape,
        "tolerance_metres": limit,
    }


def route_sha256(waypoints: Sequence[Pose2]) -> str:
    """Hash the exact ordered finite route coordinates."""

    if not waypoints:
        raise ValueError("at least one waypoint is required")
    for waypoint in waypoints:
        waypoint.validate()
    return _sha256_json(
        [_input_pose_record(waypoint) for waypoint in waypoints]
    )


def proxy_geometry_sha256(proxies: Sequence[PrismProxy]) -> str:
    """Hash a finite proxy set independently of caller ordering."""

    if not proxies:
        raise ValueError("collision proxy set is empty")
    for proxy in proxies:
        proxy.validate()
    _require_unique_source_paths(proxies, label="collision")
    records = sorted(
        (_proxy_record(proxy) for proxy in proxies),
        key=_canonical_json,
    )
    return _sha256_json(records)


def route_certificate_is_valid(certificate: dict[str, Any]) -> bool:
    """Verify a passing certificate and independently replay its inputs.

    A digest alone only detects accidental mutation when an attacker cannot
    recompute it.  Passing schema-2 certificates therefore retain the exact
    route and proxy inputs and are reproduced here with the deterministic
    validator.  Runtime callers must still bind the replayed certificate to
    the route and live geometry they intend to execute.
    """

    if not isinstance(certificate, dict):
        return False
    supplied = certificate.get("certificate_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        return False
    record = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    try:
        expected = hashlib.sha256(
            _canonical_json(record).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        return False
    if not hmac.compare_digest(supplied, expected):
        return False
    if (
        certificate.get("certificate_schema_version") != 2
        or certificate.get("passed") is not True
    ):
        return False
    try:
        validated_inputs = certificate["validated_inputs"]
        if not isinstance(validated_inputs, dict):
            return False
        route_records = validated_inputs["route_waypoints"]
        robot_records = validated_inputs["robot_geometry"]
        obstacle_records = validated_inputs["obstacle_geometry"]
        if not (
            isinstance(route_records, list)
            and isinstance(robot_records, list)
            and isinstance(obstacle_records, list)
        ):
            return False
        route = tuple(_pose_from_record(value) for value in route_records)
        robot = tuple(
            _proxy_from_record(value) for value in robot_records
        )
        obstacles = tuple(
            _proxy_from_record(value) for value in obstacle_records
        )
        config_record = certificate["config"]
        if not isinstance(config_record, dict):
            return False
        expected_config_keys = {
            field.name for field in fields(RouteValidationConfig)
        }
        if set(config_record) != expected_config_keys:
            return False
        config = RouteValidationConfig(**config_record)
        replayed = validate_route(robot, obstacles, route, config)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    try:
        return hmac.compare_digest(
            _canonical_json(certificate),
            _canonical_json(replayed),
        )
    except (TypeError, ValueError):
        return False


def _pose_record(pose: Pose2) -> dict[str, float]:
    return {
        "x": round(pose.x, 9),
        "y": round(pose.y, 9),
        "yaw_rad": round(pose.yaw, 9),
    }


def _input_pose_record(pose: Pose2) -> dict[str, float]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "yaw_rad": float(pose.yaw),
    }


def _proxy_record(proxy: PrismProxy) -> dict[str, Any]:
    return {
        "source_path": proxy.source_path,
        "vertices_xy": [
            [float(point[0]), float(point[1])]
            for point in proxy.vertices_xy
        ],
        "z_min": float(proxy.z_min),
        "z_max": float(proxy.z_max),
    }


def _pose_from_record(value: Any) -> Pose2:
    if not isinstance(value, dict) or set(value) != {
        "x",
        "y",
        "yaw_rad",
    }:
        raise ValueError("route waypoint record has an invalid shape")
    pose = Pose2(
        float(value["x"]),
        float(value["y"]),
        float(value["yaw_rad"]),
    )
    pose.validate()
    return pose


def _proxy_from_record(value: Any) -> PrismProxy:
    if not isinstance(value, dict) or set(value) != {
        "source_path",
        "vertices_xy",
        "z_min",
        "z_max",
    }:
        raise ValueError("collision proxy record has an invalid shape")
    source_path = value["source_path"]
    vertices = value["vertices_xy"]
    if (
        not isinstance(source_path, str)
        or not isinstance(vertices, list)
        or any(
            not isinstance(point, list) or len(point) != 2
            for point in vertices
        )
    ):
        raise ValueError("collision proxy record contains invalid values")
    proxy = PrismProxy(
        vertices_xy=tuple(
            (float(point[0]), float(point[1])) for point in vertices
        ),
        z_min=float(value["z_min"]),
        z_max=float(value["z_max"]),
        source_path=source_path,
    )
    proxy.validate()
    return proxy


def _require_unique_source_paths(
    proxies: Sequence[PrismProxy],
    *,
    label: str,
) -> None:
    paths = [proxy.source_path for proxy in proxies]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} proxy source paths must be unique")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validated_input_record(
    robot_proxies: Sequence[PrismProxy],
    obstacle_proxies: Sequence[PrismProxy],
    waypoints: Sequence[Pose2],
) -> dict[str, Any]:
    route = [_input_pose_record(pose) for pose in waypoints]
    robot_geometry = [_proxy_record(proxy) for proxy in robot_proxies]
    obstacle_geometry = [_proxy_record(proxy) for proxy in obstacle_proxies]
    digests = {
        "route_sha256": route_sha256(waypoints),
        "robot_geometry_sha256": _sha256_json(robot_geometry),
        "obstacle_geometry_sha256": _sha256_json(obstacle_geometry),
    }
    return {
        **digests,
        "route_waypoints": route,
        "robot_geometry": robot_geometry,
        "obstacle_geometry": obstacle_geometry,
        "robot_proxy_count": len(robot_geometry),
        "obstacle_proxy_count": len(obstacle_geometry),
        "certificate_inputs_sha256": _sha256_json(
            {
                "route": route,
                "robot_geometry": robot_geometry,
                "obstacle_geometry": obstacle_geometry,
            }
        ),
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
    canonical = _canonical_json(record)
    return {
        **record,
        "certificate_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }
