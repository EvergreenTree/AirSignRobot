#!/usr/bin/env python3
"""CPU-only tests for Stage 1 base-motion failure classification."""

from __future__ import annotations

import math
import unittest

from participant.base_motion_monitor import (
    BaseMotionMonitor,
    BaseMotionObservation,
)


class BaseMotionMonitorTests(unittest.TestCase):
    def monitor(self) -> BaseMotionMonitor:
        return BaseMotionMonitor(
            physics_hz=10.0,
            minimum_planar_command_speed=0.08,
            steering_alignment_error_rad=math.radians(8.0),
            steering_timeout_seconds=1.0,
            stall_grace_seconds=0.2,
            stall_window_seconds=0.3,
            stall_distance_metres=0.015,
        )

    def observation(
        self,
        *,
        position: tuple[float, float] = (0.0, 0.0),
        steering_error_degrees: float = 0.0,
        drive_target: float = 2.0,
        drive_velocity: float = 2.0,
        contact: bool | None = None,
    ) -> BaseMotionObservation:
        return BaseMotionObservation(
            position_xy=position,
            planar_command_speed=0.10,
            steering_errors_rad=(
                math.radians(steering_error_degrees),
                math.radians(steering_error_degrees),
            ),
            drive_targets_rad_s=(drive_target, drive_target),
            measured_drive_velocities_rad_s=(
                drive_velocity,
                drive_velocity,
            ),
            measured_drive_efforts=(1.0, 1.0),
            external_contact_observed=contact,
        )

    def test_ninety_degree_alignment_does_not_arm_stall(self) -> None:
        monitor = self.monitor()
        for _ in range(9):
            decision = monitor.update(
                self.observation(
                    steering_error_degrees=90.0,
                    drive_target=0.0,
                    drive_velocity=0.0,
                )
            )
            self.assertFalse(decision.stall_monitor_armed)
            self.assertIsNone(decision.abort_classification)

        decision = monitor.update(
            self.observation(
                steering_error_degrees=4.0,
                drive_target=2.0,
                drive_velocity=2.0,
            )
        )
        self.assertTrue(decision.steering_aligned)
        self.assertTrue(decision.drive_authorized)
        self.assertFalse(decision.stall_monitor_armed)
        self.assertIsNone(decision.abort_classification)

    def test_frozen_steering_has_specific_timeout(self) -> None:
        monitor = self.monitor()
        decision = None
        for _ in range(10):
            decision = monitor.update(
                self.observation(
                    steering_error_degrees=90.0,
                    drive_target=0.0,
                    drive_velocity=0.0,
                )
            )
        self.assertIsNotNone(decision)
        self.assertEqual(
            decision.abort_classification, "steering_alignment_timeout"
        )
        self.assertFalse(decision.stall_monitor_armed)

    def test_contact_obstruction_is_evidence_backed(self) -> None:
        monitor = self.monitor()
        decision = None
        for _ in range(5):
            decision = monitor.update(self.observation(contact=True))
        self.assertEqual(
            decision.abort_classification,
            "environment_contact_obstruction",
        )

    def test_no_wheel_response_is_drive_failure_without_contact(self) -> None:
        monitor = self.monitor()
        decision = None
        for _ in range(5):
            decision = monitor.update(
                self.observation(drive_velocity=0.0, contact=False)
            )
        self.assertEqual(
            decision.abort_classification, "drive_actuation_failure"
        )

    def test_wheel_spin_without_contact_is_traction_or_dynamics(self) -> None:
        monitor = self.monitor()
        decision = None
        for _ in range(5):
            decision = monitor.update(
                self.observation(drive_velocity=2.0, contact=False)
            )
        self.assertEqual(
            decision.abort_classification, "traction_or_dynamics_stall"
        )

    def test_unknown_contact_never_claims_collision(self) -> None:
        monitor = self.monitor()
        decision = None
        for _ in range(5):
            decision = monitor.update(
                self.observation(drive_velocity=0.0, contact=None)
            )
        self.assertEqual(
            decision.abort_classification,
            "unresolved_contact_or_drive_response_failure",
        )
        self.assertNotIn("collision", decision.abort_reason.lower())


if __name__ == "__main__":
    unittest.main()
