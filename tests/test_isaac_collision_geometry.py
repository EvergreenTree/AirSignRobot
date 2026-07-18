#!/usr/bin/env python3
"""CPU-only tests for import-safe USD collision-bound bookkeeping."""

from __future__ import annotations

import math
import unittest

from participant.isaac_collision_geometry import (
    CollisionGeometryWitness,
    CollisionProxySet,
    _extent_corner_values,
    _path_is_within_any,
    _validated_extent,
    collision_geometry_containment_record,
    enabled_dynamic_rigid_body_record,
    live_collision_geometry_containment_record,
)
from participant.se2_route_validator import PrismProxy


def proxy(path: str) -> PrismProxy:
    return PrismProxy(
        vertices_xy=(
            (-0.1, -0.1),
            (0.1, -0.1),
            (0.1, 0.1),
            (-0.1, 0.1),
        ),
        z_min=0.0,
        z_max=0.2,
        source_path=path,
    )


def box(
    path: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    z_min: float = 0.0,
    z_max: float = 0.2,
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
        source_path=path,
    )


def witness(
    collider_path: str,
    *,
    boundable_name: str = "mesh",
    frame: str,
) -> CollisionGeometryWitness:
    corners = (
        (-0.1, -0.1, 0.0),
        (-0.1, -0.1, 0.2),
        (-0.1, 0.1, 0.0),
        (-0.1, 0.1, 0.2),
        (0.1, -0.1, 0.0),
        (0.1, -0.1, 0.2),
        (0.1, 0.1, 0.0),
        (0.1, 0.1, 0.2),
    )
    return CollisionGeometryWitness(
        collider_path=collider_path,
        boundable_path=f"{collider_path}/{boundable_name}",
        local_corners=corners,
        captured_corners=corners,
        captured_frame=frame,
    )


class ExtentValidationTests(unittest.TestCase):
    def test_normalizes_vector_like_extent_and_enumerates_corners(
        self,
    ) -> None:
        extent = ([-2, -1, 0], [2, 3, 4])
        self.assertEqual(
            _validated_extent(extent),
            ((-2.0, -1.0, 0.0), (2.0, 3.0, 4.0)),
        )
        corners = _extent_corner_values(extent)
        self.assertEqual(len(corners), 8)
        self.assertEqual(len(set(corners)), 8)
        self.assertIn((-2.0, 3.0, 4.0), corners)
        self.assertIn((2.0, -1.0, 0.0), corners)

    def test_flat_axis_is_valid_for_thin_collision_geometry(self) -> None:
        local_min, local_max = _validated_extent(
            ((-1.0, -1.0, 0.0), (1.0, 1.0, 0.0))
        )
        self.assertEqual(local_min[2], local_max[2])

    def test_missing_malformed_nonfinite_and_inverted_fail_closed(
        self,
    ) -> None:
        invalid = (
            None,
            (),
            ((0.0, 0.0, 0.0),),
            ((0.0, 0.0), (1.0, 1.0, 1.0)),
            ((0.0, 0.0, 0.0), (1.0, math.nan, 1.0)),
            ((0.0, 2.0, 0.0), (1.0, 1.0, 1.0)),
        )
        for extent in invalid:
            with self.subTest(extent=extent):
                with self.assertRaises(ValueError):
                    _validated_extent(extent)


class CollisionProxyCompletenessTests(unittest.TestCase):
    def complete_set(self) -> CollisionProxySet:
        return CollisionProxySet(
            robot=(proxy("/robot/a"), proxy("/robot/b")),
            environment=(proxy("/room/wall"),),
            support_surface_paths=("/room/floor",),
            candidate_robot_paths=("/robot/a", "/robot/b"),
            candidate_environment_paths=("/room/floor", "/room/wall"),
            unresolved_paths=(),
            traversal_backend="unit-test",
            robot_geometry_witnesses=(
                witness("/robot/a", frame="base"),
                witness("/robot/b", frame="base"),
            ),
            environment_geometry_witnesses=(
                witness("/room/floor", frame="world"),
                witness("/room/wall", frame="world"),
            ),
            support_surface_proxies=(proxy("/room/floor"),),
        )

    def test_exact_candidate_accounting_is_complete(self) -> None:
        self.assertTrue(self.complete_set().complete)

    def test_missing_proxy_or_unresolved_path_is_incomplete(self) -> None:
        complete = self.complete_set()
        missing_robot = CollisionProxySet(
            **{
                **complete.__dict__,
                "robot": complete.robot[:1],
            }
        )
        unresolved = CollisionProxySet(
            **{
                **complete.__dict__,
                "unresolved_paths": ("/robot/b [missing bound]",),
            }
        )
        self.assertFalse(missing_robot.complete)
        self.assertFalse(unresolved.complete)

    def test_equal_counts_with_wrong_source_paths_are_incomplete(self) -> None:
        complete = self.complete_set()
        wrong_robot = CollisionProxySet(
            **{
                **complete.__dict__,
                "robot": (
                    proxy("/robot/a"),
                    proxy("/robot/not-a-candidate"),
                ),
            }
        )
        wrong_environment = CollisionProxySet(
            **{
                **complete.__dict__,
                "environment": (proxy("/room/not-a-candidate"),),
            }
        )
        self.assertFalse(wrong_robot.complete)
        self.assertFalse(wrong_environment.complete)

    def test_support_surface_must_not_escape_environment_accounting(
        self,
    ) -> None:
        complete = self.complete_set()
        extra_support = CollisionProxySet(
            **{
                **complete.__dict__,
                "support_surface_paths": (
                    "/room/floor",
                    "/room/unknown-floor",
                ),
            }
        )
        self.assertFalse(extra_support.complete)

    def test_witness_inventory_must_exactly_cover_candidates(self) -> None:
        complete = self.complete_set()
        missing = CollisionProxySet(
            **{
                **complete.__dict__,
                "robot_geometry_witnesses": (
                    complete.robot_geometry_witnesses[:1]
                ),
            }
        )
        wrong_frame = CollisionProxySet(
            **{
                **complete.__dict__,
                "robot_geometry_witnesses": (
                    witness("/robot/a", frame="world"),
                    witness("/robot/b", frame="base"),
                ),
            }
        )
        duplicate = CollisionProxySet(
            **{
                **complete.__dict__,
                "robot_geometry_witnesses": (
                    *complete.robot_geometry_witnesses,
                    complete.robot_geometry_witnesses[0],
                ),
            }
        )
        self.assertFalse(missing.complete)
        self.assertFalse(wrong_frame.complete)
        self.assertFalse(duplicate.complete)

    def test_requested_payload_root_requires_a_represented_proxy(self) -> None:
        complete = self.complete_set()
        missing_payload = CollisionProxySet(
            **{
                **complete.__dict__,
                "attached_payload_root_paths": (
                    "/room/unrepresented-tray",
                ),
            }
        )
        self.assertEqual(
            missing_payload.missing_attached_payload_root_paths,
            ("/room/unrepresented-tray",),
        )
        self.assertFalse(missing_payload.complete)

        represented_payload = CollisionProxySet(
            robot=(
                *complete.robot,
                proxy("/room/tray/collisions/mesh"),
            ),
            environment=complete.environment,
            support_surface_paths=complete.support_surface_paths,
            candidate_robot_paths=(
                *complete.candidate_robot_paths,
                "/room/tray/collisions/mesh",
            ),
            candidate_environment_paths=(
                complete.candidate_environment_paths
            ),
            unresolved_paths=(),
            traversal_backend="unit-test",
            attached_payload_root_paths=("/room/tray",),
            robot_geometry_witnesses=(
                *complete.robot_geometry_witnesses,
                witness(
                    "/room/tray/collisions/mesh",
                    frame="base",
                ),
            ),
            environment_geometry_witnesses=(
                complete.environment_geometry_witnesses
            ),
            support_surface_proxies=complete.support_surface_proxies,
        )
        self.assertEqual(
            represented_payload.attached_payload_proxy_paths,
            ("/room/tray/collisions/mesh",),
        )
        self.assertEqual(
            represented_payload.missing_attached_payload_root_paths, ()
        )
        self.assertTrue(represented_payload.complete)


class LiveGeometryContainmentTests(unittest.TestCase):
    def record(
        self,
        captured: PrismProxy,
        current: PrismProxy,
        *,
        allowance: float,
        captured_keys: tuple[tuple[str, str], ...] | None = None,
        current_keys: tuple[tuple[str, str], ...] | None = None,
    ) -> dict[str, object]:
        default_keys = ((captured.source_path, f"{captured.source_path}/mesh"),)
        return collision_geometry_containment_record(
            (captured,),
            (current,),
            captured_witness_keys=(
                default_keys if captured_keys is None else captured_keys
            ),
            current_witness_keys=(
                default_keys if current_keys is None else current_keys
            ),
            dynamic_allowance=allowance,
        )

    def test_translation_within_inflated_proxy_passes(self) -> None:
        captured = box("/robot/link", -0.1, -0.1, 0.1, 0.1)
        current = box("/robot/link", -0.06, -0.1, 0.14, 0.1)
        record = self.record(captured, current, allowance=0.04)
        self.assertTrue(record["passed"])
        self.assertAlmostEqual(
            float(record["maximum_planar_escape_metres"]), 0.04
        )

    def test_descendant_payload_rotation_escapes_planar_envelope(self) -> None:
        captured = box(
            "/room/tray/body/collider",
            -1.0,
            -0.1,
            1.0,
            0.1,
        )
        rotated = box(
            "/room/tray/body/collider",
            -0.1,
            -1.0,
            0.1,
            1.0,
        )
        record = self.record(captured, rotated, allowance=0.15)
        self.assertFalse(record["passed"])
        self.assertEqual(
            record["failure"], "dynamic_geometry_envelope_violation"
        )
        self.assertAlmostEqual(
            float(record["maximum_planar_escape_metres"]), 0.9
        )

    def test_vertical_escape_is_checked_independently(self) -> None:
        captured = box(
            "/robot/link",
            -0.1,
            -0.1,
            0.1,
            0.1,
            z_min=0.0,
            z_max=0.2,
        )
        current = box(
            "/robot/link",
            -0.1,
            -0.1,
            0.1,
            0.1,
            z_min=0.0,
            z_max=0.26,
        )
        record = self.record(captured, current, allowance=0.05)
        self.assertFalse(record["passed"])
        self.assertAlmostEqual(
            float(record["maximum_z_escape_metres"]), 0.06
        )

    def test_proxy_and_witness_inventory_must_match_exactly(self) -> None:
        captured = proxy("/robot/link")
        current = proxy("/robot/link")
        missing_witness = self.record(
            captured,
            current,
            allowance=0.1,
            current_keys=(),
        )
        unexpected_proxy = collision_geometry_containment_record(
            (captured,),
            (current, proxy("/robot/extra")),
            captured_witness_keys=(("/robot/link", "/robot/link/mesh"),),
            current_witness_keys=(
                ("/robot/link", "/robot/link/mesh"),
                ("/robot/extra", "/robot/extra/mesh"),
            ),
            dynamic_allowance=0.1,
        )
        self.assertEqual(
            missing_witness["failure"],
            "collision_geometry_inventory_mismatch",
        )
        self.assertEqual(
            unexpected_proxy["failure"],
            "collision_geometry_inventory_mismatch",
        )
        self.assertFalse(missing_witness["inventory_exact"])
        self.assertFalse(unexpected_proxy["inventory_exact"])

    def test_invalid_allowance_fails_before_importing_pxr(self) -> None:
        incomplete = CollisionProxySet(
            robot=(),
            environment=(),
            support_surface_paths=(),
            candidate_robot_paths=(),
            candidate_environment_paths=(),
            unresolved_paths=(),
            traversal_backend="unit-test",
        )
        record = live_collision_geometry_containment_record(
            object(),
            incomplete,
            robot_path="/robot",
            base_frame_path="/robot/base",
            dynamic_allowance=math.nan,
        )
        self.assertFalse(record["passed"])
        self.assertEqual(
            record["failure"], "invalid_live_geometry_capture"
        )
        self.assertIn("finite", str(record["reason"]))


class DynamicRigidBodyResolverTests(unittest.TestCase):
    def body(
        self,
        path: str,
        *,
        enabled: bool = True,
        kinematic: bool = False,
    ) -> dict[str, object]:
        return {
            "prim_path": path,
            "rigid_body_enabled": enabled,
            "kinematic_enabled": kinematic,
        }

    def test_exactly_one_enabled_dynamic_descendant_resolves(self) -> None:
        record = enabled_dynamic_rigid_body_record(
            "/room/tray",
            (
                self.body("/room/tray/body"),
                self.body(
                    "/room/tray/disabled-helper",
                    enabled=False,
                ),
            ),
        )
        self.assertTrue(record["passed"])
        self.assertEqual(
            record["dynamic_rigid_body_path"], "/room/tray/body"
        )

    def test_zero_or_multiple_dynamic_descendants_fail_closed(self) -> None:
        none = enabled_dynamic_rigid_body_record(
            "/room/tray",
            (self.body("/room/tray/kinematic", kinematic=True),),
        )
        multiple = enabled_dynamic_rigid_body_record(
            "/room/tray",
            (
                self.body("/room/tray/body-a"),
                self.body("/room/tray/body-b"),
            ),
        )
        self.assertFalse(none["passed"])
        self.assertFalse(multiple["passed"])
        self.assertIn("found 0", str(none["reason"]))
        self.assertIn("found 2", str(multiple["reason"]))

    def test_malformed_or_outside_body_record_fails_closed(self) -> None:
        outside = enabled_dynamic_rigid_body_record(
            "/room/tray",
            (self.body("/room/other/body"),),
        )
        non_boolean = enabled_dynamic_rigid_body_record(
            "/room/tray",
            (
                {
                    "prim_path": "/room/tray/body",
                    "rigid_body_enabled": 1,
                    "kinematic_enabled": False,
                },
            ),
        )
        self.assertFalse(outside["passed"])
        self.assertFalse(non_boolean["passed"])


class AttachedPathClassificationTests(unittest.TestCase):
    def test_exact_root_and_descendants_are_attached(self) -> None:
        roots = (
            "/World/Environment/tray",
            "/World/Environment/cup",
        )
        self.assertTrue(
            _path_is_within_any("/World/Environment/tray", roots)
        )
        self.assertTrue(
            _path_is_within_any(
                "/World/Environment/tray/collisions/mesh", roots
            )
        )
        self.assertFalse(
            _path_is_within_any("/World/Environment/tray_extra", roots)
        )
        self.assertFalse(
            _path_is_within_any("/World/Environment/table", roots)
        )


if __name__ == "__main__":
    unittest.main()
