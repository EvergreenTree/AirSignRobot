#!/usr/bin/env python3
"""CPU-only tests for fail-closed named articulation command checks."""

from __future__ import annotations

import math
import unittest

from participant.joint_command_guard import (
    arm_and_spine_effort_record,
    joint_target_continuity_record,
)


class JointCommandGuardTests(unittest.TestCase):
    def continuity(
        self,
        *,
        current: tuple[float, ...] = (0.0, 0.0, 0.0),
        targets: dict[str, float] | None = None,
        threshold: float = 0.025,
    ) -> dict[str, object]:
        return joint_target_continuity_record(
            dof_names=("left_1", "wheel", "right_1"),
            current_positions=current,
            targets=(
                {"left_1": 0.01, "right_1": -0.01}
                if targets is None
                else targets
            ),
            expected_names=("left_1", "right_1"),
            max_abs_delta_rad=threshold,
        )

    def test_interleaved_name_mapping_and_threshold_equality(self) -> None:
        record = self.continuity(
            current=(1.0, 99.0, -1.0),
            targets={"left_1": 1.025, "right_1": -1.025},
        )
        self.assertTrue(record["passed"])
        self.assertAlmostEqual(record["max_abs_delta_rad"], 0.025)
        self.assertEqual(record["max_abs_delta_joint"], "left_1")

    def test_excess_step_fails_without_angle_wrapping(self) -> None:
        record = self.continuity(
            targets={"left_1": 0.026, "right_1": 0.0}
        )
        self.assertFalse(record["passed"])
        wrap_jump = self.continuity(
            current=(3.13, 0.0, 0.0),
            targets={"left_1": -3.13, "right_1": 0.0},
            threshold=0.20,
        )
        self.assertFalse(wrap_jump["passed"])
        self.assertGreater(wrap_jump["max_abs_delta_rad"], 6.0)

    def test_invalid_inputs_fail_closed(self) -> None:
        cases = (
            self.continuity(targets={"left_1": 0.0}),
            self.continuity(
                targets={
                    "left_1": 0.0,
                    "right_1": 0.0,
                    "extra": 0.0,
                }
            ),
            self.continuity(current=(0.0, math.nan, 0.0)),
            self.continuity(
                targets={"left_1": math.inf, "right_1": 0.0}
            ),
            joint_target_continuity_record(
                dof_names=("left_1", "left_1", "right_1"),
                current_positions=(0.0, 0.0, 0.0),
                targets={"left_1": 0.0, "right_1": 0.0},
                expected_names=("left_1", "right_1"),
                max_abs_delta_rad=0.025,
            ),
            self.continuity(threshold=0.0),
        )
        self.assertTrue(all(record["passed"] is False for record in cases))

    def test_spine_force_is_not_compared_with_arm_torque_limit(self) -> None:
        record = arm_and_spine_effort_record(
            dof_names=("left_1", "spine", "right_1"),
            measured_efforts=(12.0, 678.0, -10.0),
            arm_joint_names=("left_1", "right_1"),
            spine_joint_name="spine",
            arm_effort_abort_threshold=180.0,
        )
        self.assertTrue(record["passed"])
        self.assertEqual(record["spine_force_newtons"], 678.0)
        self.assertEqual(record["peak_abs_arm_effort"], 12.0)

    def test_real_arm_threshold_violation_names_the_joint(self) -> None:
        record = arm_and_spine_effort_record(
            dof_names=("left_1", "spine", "right_1"),
            measured_efforts=(181.0, 20.0, -10.0),
            arm_joint_names=("left_1", "right_1"),
            spine_joint_name="spine",
            arm_effort_abort_threshold=180.0,
        )
        self.assertFalse(record["passed"])
        self.assertTrue(record["arm_threshold_exceeded"])
        self.assertEqual(record["peak_abs_arm_effort_joint"], "left_1")


if __name__ == "__main__":
    unittest.main()
