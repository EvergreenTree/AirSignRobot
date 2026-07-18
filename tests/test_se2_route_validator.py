#!/usr/bin/env python3
"""CPU-only tests for conservative full-proxy route validation."""

from __future__ import annotations

import math
import unittest

from participant.se2_route_validator import (
    Pose2,
    PrismProxy,
    RouteValidationConfig,
    distance_to_polyline,
    sample_route,
    validate_route,
)


def box(
    source: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    z_min: float = 0.1,
    z_max: float = 1.0,
) -> PrismProxy:
    return PrismProxy(
        vertices_xy=(
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ),
        z_min=z_min,
        z_max=z_max,
        source_path=source,
    )


class Se2RouteValidatorTests(unittest.TestCase):
    def config(self, **overrides: object) -> RouteValidationConfig:
        values = {
            "max_translation_step": 0.025,
            "max_yaw_step_rad": math.radians(2.0),
            "clearance_margin": 0.0,
            "base_z": 0.0,
            "collision_coverage_complete": True,
        }
        values.update(overrides)
        return RouteValidationConfig(**values)

    def test_thin_midpoint_obstacle_is_not_skipped(self) -> None:
        robot = [box("/robot/base", -0.1, -0.1, 0.1, 0.1)]
        obstacle = [box("/room/thin", 0.49, -0.3, 0.51, 0.3)]
        result = validate_route(
            robot,
            obstacle,
            (Pose2(0.0, 0.0, 0.0), Pose2(1.0, 0.0, 0.0)),
            self.config(),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "preflight_overlap")
        self.assertGreater(result["checked_samples"], 1)

    def test_rotation_sweep_catches_long_footprint_corner(self) -> None:
        robot = [box("/robot/arm", -1.0, -0.05, 1.0, 0.05)]
        obstacle = [box("/room/post", 0.62, 0.62, 0.72, 0.72)]
        result = validate_route(
            robot,
            obstacle,
            (
                Pose2(0.0, 0.0, 0.0),
                Pose2(0.0, 0.0, math.radians(90.0)),
            ),
            self.config(),
        )
        self.assertFalse(result["passed"])
        self.assertNotIn(
            result["first_unsafe_sample"]["fraction"], (0.0, 1.0)
        )

    def test_yaw_wrap_uses_shortest_interpolation(self) -> None:
        samples = sample_route(
            (
                Pose2(0.0, 0.0, math.radians(179.0)),
                Pose2(0.0, 0.0, math.radians(-179.0)),
            ),
            self.config(),
        )
        self.assertLessEqual(len(samples), 3)
        total_delta = sum(
            abs(samples[index]["pose"].yaw - samples[index - 1]["pose"].yaw)
            if abs(
                samples[index]["pose"].yaw
                - samples[index - 1]["pose"].yaw
            )
            <= math.pi
            else 2.0 * math.pi
            - abs(
                samples[index]["pose"].yaw
                - samples[index - 1]["pose"].yaw
            )
            for index in range(1, len(samples))
        )
        self.assertAlmostEqual(total_delta, math.radians(2.0), places=6)

    def test_touching_inflated_boundary_is_unsafe(self) -> None:
        robot = [box("/robot/base", -0.1, -0.1, 0.1, 0.1)]
        obstacle = [box("/room/wall", 0.2, -0.1, 0.3, 0.1)]
        result = validate_route(
            robot,
            obstacle,
            (Pose2(0.0, 0.0, 0.0),),
            self.config(clearance_margin=0.1),
        )
        self.assertFalse(result["passed"])

    def test_extended_arm_rejects_base_only_route(self) -> None:
        base = [box("/robot/base", -0.2, -0.2, 0.2, 0.2)]
        extended = [
            *base,
            box("/robot/arm", 0.2, -0.05, 1.0, 0.05),
        ]
        obstacle = [box("/room/post", 0.75, -0.1, 0.85, 0.1)]
        route = (Pose2(0.0, 0.0, 0.0),)
        self.assertTrue(
            validate_route(base, obstacle, route, self.config())["passed"]
        )
        self.assertFalse(
            validate_route(
                extended, obstacle, route, self.config()
            )["passed"]
        )

    def test_payload_rejects_narrow_doorway(self) -> None:
        base = [box("/robot/base", -0.2, -0.2, 0.2, 0.2)]
        loaded = [
            *base,
            box("/payload/tray", -0.2, -0.55, 0.2, 0.55),
        ]
        doorway = [
            box("/room/left", -0.5, 0.45, 0.5, 0.60),
            box("/room/right", -0.5, -0.60, 0.5, -0.45),
        ]
        route = (
            Pose2(-1.0, 0.0, 0.0),
            Pose2(1.0, 0.0, 0.0),
        )
        self.assertTrue(
            validate_route(base, doorway, route, self.config())["passed"]
        )
        self.assertFalse(
            validate_route(loaded, doorway, route, self.config())["passed"]
        )

    def test_reverse_route_has_same_pass_result(self) -> None:
        robot = [box("/robot/base", -0.1, -0.1, 0.1, 0.1)]
        obstacle = [box("/room/far", 3.0, 3.0, 3.2, 3.2)]
        route = (
            Pose2(0.0, 0.0, 0.0),
            Pose2(1.0, 1.0, math.radians(30.0)),
        )
        forward = validate_route(
            robot, obstacle, route, self.config()
        )
        reverse = validate_route(
            robot, obstacle, tuple(reversed(route)), self.config()
        )
        self.assertEqual(forward["passed"], reverse["passed"])

    def test_unknown_and_invalid_geometry_fail_closed(self) -> None:
        robot = [box("/robot/base", -0.1, -0.1, 0.1, 0.1)]
        obstacle = [box("/room/far", 3.0, 3.0, 3.2, 3.2)]
        route = (Pose2(0.0, 0.0, 0.0),)
        self.assertFalse(
            validate_route(
                robot,
                obstacle,
                route,
                self.config(collision_coverage_complete=False),
            )["passed"]
        )
        self.assertFalse(
            validate_route([], obstacle, route, self.config())["passed"]
        )
        self.assertFalse(
            validate_route(
                robot,
                obstacle,
                (Pose2(float("nan"), 0.0, 0.0),),
                self.config(),
            )["passed"]
        )

    def test_certificate_is_deterministic_under_proxy_ordering(self) -> None:
        robot = [
            box("/robot/b", -0.1, -0.1, 0.0, 0.1),
            box("/robot/a", 0.0, -0.1, 0.1, 0.1),
        ]
        obstacle = [
            box("/room/b", 3.0, 3.0, 3.2, 3.2),
            box("/room/a", -3.2, -3.2, -3.0, -3.0),
        ]
        route = (
            Pose2(0.0, 0.0, 0.0),
            Pose2(0.5, 0.0, 0.0),
        )
        first = validate_route(
            robot, obstacle, route, self.config()
        )
        second = validate_route(
            tuple(reversed(robot)),
            tuple(reversed(obstacle)),
            route,
            self.config(),
        )
        self.assertEqual(
            first["certificate_sha256"], second["certificate_sha256"]
        )

    def test_cross_track_distance_detects_corridor_exit(self) -> None:
        route = (
            Pose2(0.0, 0.0, 0.0),
            Pose2(2.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(
            distance_to_polyline(Pose2(1.0, 0.04, 0.0), route), 0.04
        )
        self.assertGreater(
            distance_to_polyline(Pose2(1.0, 0.25, 0.0), route), 0.20
        )


if __name__ == "__main__":
    unittest.main()
