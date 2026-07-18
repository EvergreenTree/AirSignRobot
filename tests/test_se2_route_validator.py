#!/usr/bin/env python3
"""CPU-only tests for conservative full-proxy route validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest

from participant.se2_route_validator import (
    Pose2,
    PrismProxy,
    RouteValidationConfig,
    base_nonplanar_deviation_record,
    distance_to_polyline,
    proxy_geometry_sha256,
    route_execution_tube_record,
    route_certificate_is_valid,
    route_pose_deviation,
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

    def test_internal_waypoint_uses_larger_adjacent_step_allowance(
        self,
    ) -> None:
        samples = sample_route(
            (
                Pose2(0.0, 0.0, 0.0),
                Pose2(0.001, 0.0, 0.0),
                Pose2(0.026, 0.0, 0.0),
            ),
            self.config(
                max_translation_step=0.025,
                execution_translation_tolerance=0.0,
                execution_yaw_tolerance_rad=0.0,
            ),
        )
        shared = next(
            sample
            for sample in samples
            if sample["pose"].x == 0.001
        )
        self.assertAlmostEqual(shared["translation_step"], 0.025)

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
        self.assertEqual(
            proxy_geometry_sha256(robot),
            proxy_geometry_sha256(tuple(reversed(robot))),
        )
        self.assertNotEqual(
            proxy_geometry_sha256(robot),
            proxy_geometry_sha256(
                (
                    robot[0],
                    box("/robot/a", 0.0, -0.1, 0.100001, 0.1),
                )
            ),
        )

    def test_certificate_binds_route_and_proxy_geometry(self) -> None:
        robot = [box("/robot/base", -0.1, -0.1, 0.1, 0.1)]
        obstacle = [box("/room/far", 3.0, 3.0, 3.2, 3.2)]
        route = (
            Pose2(0.0, 0.0, 0.0),
            Pose2(0.5, 0.0, 0.0),
        )
        baseline = validate_route(
            robot, obstacle, route, self.config()
        )
        moved_route = validate_route(
            robot,
            obstacle,
            (route[0], Pose2(0.500001, 0.0, 0.0)),
            self.config(),
        )
        moved_obstacle = validate_route(
            robot,
            [box("/room/far", 3.000001, 3.0, 3.2, 3.2)],
            route,
            self.config(),
        )
        self.assertNotEqual(
            baseline["certificate_sha256"],
            moved_route["certificate_sha256"],
        )
        self.assertNotEqual(
            baseline["certificate_sha256"],
            moved_obstacle["certificate_sha256"],
        )
        self.assertNotEqual(
            baseline["validated_inputs"]["route_sha256"],
            moved_route["validated_inputs"]["route_sha256"],
        )
        self.assertNotEqual(
            baseline["validated_inputs"]["obstacle_geometry_sha256"],
            moved_obstacle["validated_inputs"][
                "obstacle_geometry_sha256"
            ],
        )
        self.assertEqual(baseline["certificate_schema_version"], 2)
        self.assertEqual(
            baseline["validated_inputs"]["route_waypoints"],
            [
                {"x": 0.0, "y": 0.0, "yaw_rad": 0.0},
                {"x": 0.5, "y": 0.0, "yaw_rad": 0.0},
            ],
        )
        self.assertTrue(route_certificate_is_valid(baseline))
        tampered = {
            **baseline,
            "checked_samples": baseline["checked_samples"] + 1,
        }
        self.assertFalse(route_certificate_is_valid(tampered))

    def test_certificate_replay_rejects_rehashed_internal_tampering(
        self,
    ) -> None:
        certificate = validate_route(
            [box("/robot/base", -0.1, -0.1, 0.1, 0.1)],
            [box("/room/far", 3.0, 3.0, 3.2, 3.2)],
            (
                Pose2(0.0, 0.0, 0.0),
                Pose2(0.5, 0.0, 0.0),
            ),
            self.config(),
        )
        self.assertTrue(route_certificate_is_valid(certificate))
        tampered = copy.deepcopy(certificate)
        tampered["validated_inputs"]["route_sha256"] = "0" * 64
        unsigned = {
            key: value
            for key, value in tampered.items()
            if key != "certificate_sha256"
        }
        tampered["certificate_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertFalse(route_certificate_is_valid(tampered))

    def test_yaw_tolerance_above_pi_fails_closed(self) -> None:
        result = validate_route(
            [box("/robot/base", -0.1, -0.1, 0.1, 0.1)],
            [box("/room/far", 3.0, 3.0, 3.2, 3.2)],
            (Pose2(0.0, 0.0, 0.0),),
            self.config(execution_yaw_tolerance_rad=math.pi + 0.01),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(route_certificate_is_valid(result))
        tube = route_execution_tube_record(
            Pose2(0.0, 0.0, 0.0),
            (Pose2(0.0, 0.0, 0.0),),
            translation_tolerance=0.04,
            yaw_tolerance_rad=math.pi + 0.01,
        )
        self.assertFalse(tube["passed"])
        self.assertEqual(tube["failure"], "invalid_execution_tube")

    def test_nonplanar_base_gate_catches_z_and_tilt(self) -> None:
        level = base_nonplanar_deviation_record(
            position_z=0.01,
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            reference_z=0.0,
            maximum_robot_radius=1.0,
            tolerance=0.04,
        )
        self.assertTrue(level["passed"])
        tilted = base_nonplanar_deviation_record(
            position_z=0.0,
            orientation_wxyz=(
                math.cos(math.radians(2.0)),
                math.sin(math.radians(2.0)),
                0.0,
                0.0,
            ),
            reference_z=0.0,
            maximum_robot_radius=1.0,
            tolerance=0.04,
        )
        self.assertFalse(tilted["passed"])
        self.assertEqual(
            tilted["failure"], "nonplanar_base_envelope_violation"
        )
        invalid = base_nonplanar_deviation_record(
            position_z=0.0,
            orientation_wxyz=(0.0, 0.0, 0.0, 0.0),
            reference_z=0.0,
            maximum_robot_radius=1.0,
            tolerance=0.04,
        )
        self.assertFalse(invalid["passed"])
        self.assertEqual(invalid["failure"], "invalid_nonplanar_base_pose")

    def test_duplicate_proxy_source_paths_fail_closed(self) -> None:
        duplicate_robot = [
            box("/robot/duplicate", -0.2, -0.1, 0.0, 0.1),
            box("/robot/duplicate", 0.0, -0.1, 0.2, 0.1),
        ]
        result = validate_route(
            duplicate_robot,
            [box("/room/far", 3.0, 3.0, 3.2, 3.2)],
            (Pose2(0.0, 0.0, 0.0),),
            self.config(),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["failure"], "invalid_or_incomplete_geometry"
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

    def test_route_pose_deviation_interpolates_expected_yaw(self) -> None:
        route = (
            Pose2(0.0, 0.0, 0.0),
            Pose2(2.0, 0.0, math.radians(90.0)),
        )
        deviation = route_pose_deviation(
            Pose2(1.0, 0.03, math.radians(50.0)),
            route,
        )
        self.assertEqual(deviation["segment_index"], 0)
        self.assertAlmostEqual(deviation["fraction"], 0.5)
        self.assertAlmostEqual(
            deviation["cross_track_metres"], 0.03
        )
        self.assertAlmostEqual(
            deviation["expected_yaw_rad"], math.radians(45.0)
        )
        self.assertAlmostEqual(
            deviation["yaw_error_rad"], math.radians(5.0)
        )

    def test_route_pose_deviation_handles_in_place_wrap_rotation(self) -> None:
        route = (
            Pose2(1.0, 2.0, math.radians(179.0)),
            Pose2(1.0, 2.0, math.radians(-179.0)),
        )
        deviation = route_pose_deviation(
            Pose2(1.0, 2.0, math.pi),
            route,
        )
        self.assertAlmostEqual(deviation["fraction"], 0.5, places=6)
        self.assertAlmostEqual(
            deviation["expected_yaw_rad"], math.pi, places=6
        )
        self.assertAlmostEqual(deviation["yaw_error_rad"], 0.0, places=6)

    def test_execution_tube_translation_and_yaw_fail_independently(self) -> None:
        route = (
            Pose2(0.0, 0.0, 0.0),
            Pose2(2.0, 0.0, 0.0),
        )
        translation_failure = route_execution_tube_record(
            Pose2(1.0, 0.05, 0.0),
            route,
            translation_tolerance=0.04,
            yaw_tolerance_rad=math.radians(2.0),
        )
        self.assertEqual(
            translation_failure["failure"], "cross_track_violation"
        )
        self.assertTrue(translation_failure["yaw_within_tolerance"])

        yaw_failure = route_execution_tube_record(
            Pose2(1.0, 0.0, math.radians(3.0)),
            route,
            translation_tolerance=0.04,
            yaw_tolerance_rad=math.radians(2.0),
        )
        self.assertEqual(yaw_failure["failure"], "route_yaw_violation")
        self.assertTrue(yaw_failure["translation_within_tolerance"])

    def test_certificate_reserves_execution_pose_envelope(self) -> None:
        robot = [box("/robot/base", -0.1, -0.1, 0.1, 0.1)]
        obstacle = [box("/room/wall", 0.25, -0.1, 0.35, 0.1)]
        route = (Pose2(0.0, 0.0, 0.0),)
        without_envelope = validate_route(
            robot,
            obstacle,
            route,
            self.config(
                execution_translation_tolerance=0.0,
                execution_yaw_tolerance_rad=0.0,
            ),
        )
        with_envelope = validate_route(
            robot,
            obstacle,
            route,
            self.config(
                execution_translation_tolerance=0.15,
                execution_yaw_tolerance_rad=0.0,
            ),
        )
        self.assertTrue(without_envelope["passed"])
        self.assertFalse(with_envelope["passed"])
        self.assertEqual(
            with_envelope["execution_envelope"][
                "translation_tolerance_metres"
            ],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
