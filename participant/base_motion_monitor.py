#!/usr/bin/env python3
"""Import-safe base-motion diagnostics for the Stage 1 controller.

The official mobile-base helper suppresses wheel velocity while a swerve
module is turning toward a new direction.  A translation-stall detector must
therefore observe steering alignment and the resulting wheel targets, rather
than only the requested body velocity.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable


def _finite_tuple(values: Iterable[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite values")
    return result


@dataclass(frozen=True)
class BaseMotionObservation:
    """One physics-step observation of the mobile base."""

    position_xy: tuple[float, float]
    planar_command_speed: float
    steering_errors_rad: tuple[float, ...]
    drive_targets_rad_s: tuple[float, ...]
    measured_drive_velocities_rad_s: tuple[float, ...]
    measured_drive_efforts: tuple[float, ...] | None = None
    external_contact_observed: bool | None = None

    def __post_init__(self) -> None:
        position = _finite_tuple(self.position_xy, "position_xy")
        steering = _finite_tuple(
            self.steering_errors_rad, "steering_errors_rad"
        )
        targets = _finite_tuple(
            self.drive_targets_rad_s, "drive_targets_rad_s"
        )
        velocities = _finite_tuple(
            self.measured_drive_velocities_rad_s,
            "measured_drive_velocities_rad_s",
        )
        if len(position) != 2:
            raise ValueError("position_xy must contain exactly two values")
        if not (
            len(steering) == len(targets) == len(velocities)
        ):
            raise ValueError(
                "steering, target, and measured wheel arrays must match"
            )
        if not math.isfinite(float(self.planar_command_speed)):
            raise ValueError("planar_command_speed must be finite")
        if self.measured_drive_efforts is not None:
            efforts = _finite_tuple(
                self.measured_drive_efforts, "measured_drive_efforts"
            )
            if len(efforts) != len(targets):
                raise ValueError(
                    "measured drive efforts must match the wheel arrays"
                )


@dataclass(frozen=True)
class BaseMotionDecision:
    """State and any fail-closed decision produced for one observation."""

    translation_requested: bool
    steering_aligned: bool
    drive_authorized: bool
    stall_monitor_armed: bool
    steering_wait_steps: int
    aligned_motion_steps: int
    window_displacement_metres: float | None
    abort_classification: str | None
    abort_reason: str | None


class BaseMotionMonitor:
    """Distinguish steering alignment from true post-alignment immobility."""

    def __init__(
        self,
        *,
        physics_hz: float,
        minimum_planar_command_speed: float,
        steering_alignment_error_rad: float,
        steering_timeout_seconds: float,
        stall_grace_seconds: float,
        stall_window_seconds: float,
        stall_distance_metres: float,
        drive_target_epsilon_rad_s: float = 0.05,
        measured_drive_response_rad_s: float = 0.10,
    ) -> None:
        values = {
            "physics_hz": physics_hz,
            "minimum_planar_command_speed": minimum_planar_command_speed,
            "steering_alignment_error_rad": steering_alignment_error_rad,
            "steering_timeout_seconds": steering_timeout_seconds,
            "stall_grace_seconds": stall_grace_seconds,
            "stall_window_seconds": stall_window_seconds,
            "stall_distance_metres": stall_distance_metres,
            "drive_target_epsilon_rad_s": drive_target_epsilon_rad_s,
            "measured_drive_response_rad_s": (
                measured_drive_response_rad_s
            ),
        }
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in values.values()
        ):
            raise ValueError("all base-motion monitor limits must be positive")
        self.minimum_planar_command_speed = float(
            minimum_planar_command_speed
        )
        self.steering_alignment_error_rad = float(
            steering_alignment_error_rad
        )
        self.stall_distance_metres = float(stall_distance_metres)
        self.drive_target_epsilon_rad_s = float(
            drive_target_epsilon_rad_s
        )
        self.measured_drive_response_rad_s = float(
            measured_drive_response_rad_s
        )
        self.steering_timeout_steps = max(
            1, int(round(steering_timeout_seconds * physics_hz))
        )
        self.stall_grace_steps = max(
            0, int(round(stall_grace_seconds * physics_hz))
        )
        self.stall_window_steps = max(
            2, int(round(stall_window_seconds * physics_hz))
        )
        self._positions: deque[tuple[float, float]] = deque(
            maxlen=self.stall_window_steps
        )
        self.steering_wait_steps = 0
        self.aligned_motion_steps = 0

    def reset(self) -> None:
        self._positions.clear()
        self.steering_wait_steps = 0
        self.aligned_motion_steps = 0

    def update(
        self, observation: BaseMotionObservation
    ) -> BaseMotionDecision:
        translation_requested = (
            observation.planar_command_speed
            >= self.minimum_planar_command_speed
        )
        steering_aligned = (
            max(abs(value) for value in observation.steering_errors_rad)
            <= self.steering_alignment_error_rad
        )
        drive_authorized = (
            max(abs(value) for value in observation.drive_targets_rad_s)
            > self.drive_target_epsilon_rad_s
        )

        if not translation_requested:
            self.reset()
            return self._decision(
                translation_requested=False,
                steering_aligned=steering_aligned,
                drive_authorized=drive_authorized,
            )

        if not (steering_aligned and drive_authorized):
            self._positions.clear()
            self.aligned_motion_steps = 0
            self.steering_wait_steps += 1
            if self.steering_wait_steps >= self.steering_timeout_steps:
                return self._decision(
                    translation_requested=True,
                    steering_aligned=steering_aligned,
                    drive_authorized=drive_authorized,
                    abort_classification="steering_alignment_timeout",
                    abort_reason=(
                        "swerve modules did not align and authorize wheel "
                        "motion before the steering timeout"
                    ),
                )
            return self._decision(
                translation_requested=True,
                steering_aligned=steering_aligned,
                drive_authorized=drive_authorized,
            )

        self.steering_wait_steps = 0
        self.aligned_motion_steps += 1
        self._positions.append(tuple(observation.position_xy))
        monitor_armed = (
            self.aligned_motion_steps
            >= self.stall_grace_steps + self.stall_window_steps
            and len(self._positions) == self.stall_window_steps
        )
        displacement = self._window_displacement()
        if monitor_armed and displacement < self.stall_distance_metres:
            classification, reason = self._classify_stall(observation)
            return self._decision(
                translation_requested=True,
                steering_aligned=True,
                drive_authorized=True,
                monitor_armed=True,
                displacement=displacement,
                abort_classification=classification,
                abort_reason=reason,
            )

        return self._decision(
            translation_requested=True,
            steering_aligned=True,
            drive_authorized=True,
            monitor_armed=monitor_armed,
            displacement=displacement,
        )

    def _window_displacement(self) -> float | None:
        if len(self._positions) < 2:
            return None
        start = self._positions[0]
        end = self._positions[-1]
        return math.hypot(end[0] - start[0], end[1] - start[1])

    def _classify_stall(
        self, observation: BaseMotionObservation
    ) -> tuple[str, str]:
        wheel_response = max(
            abs(value)
            for value in observation.measured_drive_velocities_rad_s
        )
        contact = observation.external_contact_observed
        if contact is True:
            return (
                "environment_contact_obstruction",
                "aligned wheel motion produced insufficient base displacement "
                "while an external environment contact was measured",
            )
        if contact is False and (
            wheel_response < self.measured_drive_response_rad_s
        ):
            return (
                "drive_actuation_failure",
                "aligned nonzero wheel targets produced neither measured "
                "wheel response nor sufficient base displacement",
            )
        if contact is False:
            return (
                "traction_or_dynamics_stall",
                "measured wheel motion produced insufficient base "
                "displacement without an external contact",
            )
        if wheel_response < self.measured_drive_response_rad_s:
            return (
                "unresolved_contact_or_drive_response_failure",
                "aligned wheel targets produced neither measured wheel "
                "response nor sufficient base displacement; external "
                "contact telemetry was unavailable",
            )
        return (
            "unresolved_contact_or_traction_stall",
            "measured wheel motion produced insufficient base displacement; "
            "external contact telemetry was unavailable",
        )

    def _decision(
        self,
        *,
        translation_requested: bool,
        steering_aligned: bool,
        drive_authorized: bool,
        monitor_armed: bool = False,
        displacement: float | None = None,
        abort_classification: str | None = None,
        abort_reason: str | None = None,
    ) -> BaseMotionDecision:
        return BaseMotionDecision(
            translation_requested=translation_requested,
            steering_aligned=steering_aligned,
            drive_authorized=drive_authorized,
            stall_monitor_armed=monitor_armed,
            steering_wait_steps=self.steering_wait_steps,
            aligned_motion_steps=self.aligned_motion_steps,
            window_displacement_metres=displacement,
            abort_classification=abort_classification,
            abort_reason=abort_reason,
        )
