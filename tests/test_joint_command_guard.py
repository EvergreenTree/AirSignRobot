#!/usr/bin/env python3
"""CPU-only tests for fail-closed named articulation command checks."""

from __future__ import annotations

import json
import math
import unittest

from participant.joint_command_guard import (
    arm_and_spine_effort_record,
    joint_command_slew_record,
    joint_tracking_error_dwell_record,
)


class JointCommandGuardTests(unittest.TestCase):
    def slew(
        self,
        *,
        previous: dict[str, float] | None = None,
        targets: dict[str, float] | None = None,
        threshold: float = 0.025,
    ) -> dict[str, object]:
        return joint_command_slew_record(
            previous_targets=(
                {"left_1": 0.0, "right_1": 0.0}
                if previous is None
                else previous
            ),
            targets=(
                {"left_1": 0.01, "right_1": -0.01}
                if targets is None
                else targets
            ),
            expected_names=("left_1", "right_1"),
            max_abs_delta_rad=threshold,
        )

    def tracking(
        self,
        *,
        measured: tuple[float, ...] = (0.0, 99.0, 0.0),
        targets: dict[str, float] | None = None,
        threshold: float = 0.12,
        dwell_steps: int = 3,
        prior: int = 0,
    ) -> dict[str, object]:
        return joint_tracking_error_dwell_record(
            dof_names=("left_1", "wheel", "right_1"),
            measured_positions=measured,
            targets=(
                {"left_1": 0.01, "right_1": -0.01}
                if targets is None
                else targets
            ),
            expected_names=("left_1", "right_1"),
            max_abs_error_rad=threshold,
            dwell_steps=dwell_steps,
            prior_consecutive_exceeded_steps=prior,
        )

    def test_adjacent_command_slew_uses_previous_target(self) -> None:
        record = self.slew(
            previous={"left_1": 1.0, "right_1": -1.0},
            targets={"left_1": 1.025, "right_1": -1.025},
        )
        self.assertTrue(record["passed"])
        self.assertAlmostEqual(record["max_abs_delta_rad"], 0.025)
        self.assertEqual(record["max_abs_delta_joint"], "left_1")
        self.assertEqual(
            record["signal"], "target_to_previous_command_slew"
        )
        self.assertFalse(record["command_clamped"])

    def test_excess_step_fails_without_angle_wrapping(self) -> None:
        record = self.slew(
            targets={"left_1": 0.026, "right_1": 0.0}
        )
        self.assertFalse(record["passed"])
        wrap_jump = self.slew(
            previous={"left_1": 3.13, "right_1": 0.0},
            targets={"left_1": -3.13, "right_1": 0.0},
            threshold=0.20,
        )
        self.assertFalse(wrap_jump["passed"])
        self.assertGreater(wrap_jump["max_abs_delta_rad"], 6.0)

    def test_invalid_inputs_fail_closed(self) -> None:
        cases = (
            self.slew(targets={"left_1": 0.0}),
            self.slew(
                targets={
                    "left_1": 0.0,
                    "right_1": 0.0,
                    "extra": 0.0,
                }
            ),
            self.slew(
                previous={"left_1": math.nan, "right_1": 0.0}
            ),
            self.slew(
                targets={"left_1": math.inf, "right_1": 0.0}
            ),
            joint_command_slew_record(
                previous_targets={"left_1": 0.0, "right_1": 0.0},
                targets={"left_1": 0.0, "right_1": 0.0},
                expected_names=("left_1", "left_1"),
                max_abs_delta_rad=0.025,
            ),
            joint_command_slew_record(
                previous_targets={"left_1": 0.0},
                targets={"left_1": 0.0},
                expected_names=None,  # type: ignore[arg-type]
                max_abs_delta_rad=0.025,
            ),
            self.slew(threshold=0.0),
        )
        self.assertTrue(all(record["passed"] is False for record in cases))

    def test_tracking_error_is_a_separate_dwell_signal(self) -> None:
        slew = self.slew(
            previous={"left_1": 0.99, "right_1": -0.99},
            targets={"left_1": 1.0, "right_1": -1.0},
        )
        tracking = self.tracking(
            measured=(0.0, 99.0, 0.0),
            targets={"left_1": 1.0, "right_1": -1.0},
            prior=0,
        )
        self.assertTrue(slew["passed"])
        self.assertTrue(tracking["passed"])
        self.assertTrue(tracking["threshold_exceeded"])
        self.assertFalse(tracking["abort_required"])
        self.assertEqual(tracking["consecutive_exceeded_steps"], 1)
        self.assertEqual(tracking["status"], "dwell_pending")
        self.assertEqual(
            tracking["signal"], "target_to_measured_tracking_error"
        )

    def test_tracking_dwell_aborts_only_when_sustained(self) -> None:
        first = self.tracking(
            measured=(0.0, 99.0, 0.0),
            targets={"left_1": 0.2, "right_1": 0.0},
            prior=0,
        )
        second = self.tracking(
            measured=(0.0, 99.0, 0.0),
            targets={"left_1": 0.2, "right_1": 0.0},
            prior=int(first["consecutive_exceeded_steps"]),
        )
        third = self.tracking(
            measured=(0.0, 99.0, 0.0),
            targets={"left_1": 0.2, "right_1": 0.0},
            prior=int(second["consecutive_exceeded_steps"]),
        )
        self.assertTrue(first["passed"])
        self.assertTrue(second["passed"])
        self.assertFalse(third["passed"])
        self.assertTrue(third["dwell_reached"])
        self.assertTrue(third["abort_required"])
        self.assertEqual(third["consecutive_exceeded_steps"], 3)
        self.assertEqual(third["status"], "dwell_reached")

    def test_tracking_recovery_resets_dwell(self) -> None:
        recovered = self.tracking(
            measured=(0.2, 99.0, 0.0),
            targets={"left_1": 0.2, "right_1": 0.0},
            prior=2,
        )
        self.assertTrue(recovered["passed"])
        self.assertFalse(recovered["threshold_exceeded"])
        self.assertEqual(recovered["consecutive_exceeded_steps"], 0)
        self.assertEqual(recovered["remaining_dwell_steps"], 3)
        self.assertEqual(recovered["status"], "within_threshold")

    def test_tracking_threshold_equality_does_not_start_dwell(self) -> None:
        record = self.tracking(
            measured=(0.0, 99.0, 0.0),
            targets={"left_1": 0.12, "right_1": 0.0},
            prior=2,
        )
        self.assertTrue(record["passed"])
        self.assertFalse(record["threshold_exceeded"])
        self.assertEqual(record["consecutive_exceeded_steps"], 0)

    def test_invalid_tracking_inputs_abort_immediately(self) -> None:
        cases = (
            self.tracking(measured=(math.nan, 99.0, 0.0)),
            self.tracking(targets={"left_1": 0.0}),
            self.tracking(threshold=0.0),
            self.tracking(dwell_steps=0),
            self.tracking(prior=3),
            joint_tracking_error_dwell_record(
                dof_names=("left_1", "left_1", "right_1"),
                measured_positions=(0.0, 0.0, 0.0),
                targets={"left_1": 0.0, "right_1": 0.0},
                expected_names=("left_1", "right_1"),
                max_abs_error_rad=0.12,
                dwell_steps=3,
                prior_consecutive_exceeded_steps=0,
            ),
            joint_tracking_error_dwell_record(
                dof_names=None,  # type: ignore[arg-type]
                measured_positions=(0.0,),
                targets={"left_1": 0.0},
                expected_names=("left_1",),
                max_abs_error_rad=0.12,
                dwell_steps=3,
                prior_consecutive_exceeded_steps=0,
            ),
        )
        self.assertTrue(all(record["passed"] is False for record in cases))
        self.assertTrue(all(record["abort_required"] for record in cases))
        for record in cases:
            json.dumps(record, allow_nan=False)

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
