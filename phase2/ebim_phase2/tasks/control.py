"""Bounded Cartesian visual servoing and observable task transitions.

The runtime consumes sensor estimates, not USD prims, object setters, or
simulator-only annotations. Jacobians must map joint velocity to world-frame
TCP [vx, vy, vz, wx, wy, wz], matching the calibrated poses supplied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Mapping

import numpy as np

from .geometry import Pose, Rectangle, pose_error, quat_multiply, route, vector

ARMS = ("left", "right")
Q_SLICES = {"left": slice(0, 7), "right": slice(21, 28)}
A_SLICES = {"left": slice(0, 7), "right": slice(8, 15)}
GRIP_STATE = {"left": 20, "right": 41}
GRIP_ACTION = {"left": 7, "right": 15}
SENSOR_SOURCES = frozenset(("rgbd", "vision", "robot_state", "tactile", "force", "scale", "calibration", "localization", "lidar"))


@dataclass(frozen=True)
class Estimate:
    value: object
    timestamp: float
    confidence: float
    source: str

    def __post_init__(self):
        if self.source not in SENSOR_SOURCES:
            raise ValueError(f"unsupported observation source {self.source!r}; simulator truth is not a policy input")
        if not math.isfinite(self.timestamp) or not 0 <= self.confidence <= 1:
            raise ValueError("estimate timestamp/confidence is invalid")
        if isinstance(self.value, (float, int)) and not math.isfinite(self.value):
            raise ValueError("estimate value must be finite")


@dataclass(frozen=True)
class Observation:
    timestamp: float
    state: np.ndarray
    poses: Mapping[str, Estimate] = field(default_factory=dict)
    signals: Mapping[str, Estimate] = field(default_factory=dict)
    jacobians: Mapping[str, np.ndarray] = field(default_factory=dict)
    obstacles: tuple[Rectangle, ...] = ()
    obstacle_timestamp: float | None = None
    obstacle_stage: str | None = None
    base_xyyaw: tuple[float, float, float] | None = None
    state_profile: int = 1
    spine_height_m: float | None = None

    def __post_init__(self):
        if self.state_profile not in (1, 2):
            raise ValueError("state_profile must be 1 (62D) or 2 (42D)")
        vector(self.state, 62 if self.state_profile == 1 else 42, "organizer state")
        if self.state_profile == 2 and (self.spine_height_m is None or not math.isfinite(self.spine_height_m)):
            raise ValueError("42D Task2 state requires separately measured spine_height_m")
        if not math.isfinite(self.timestamp):
            raise ValueError("observation timestamp must be finite")
        for name, estimate in self.poses.items():
            if not isinstance(estimate.value, Pose):
                raise ValueError(f"pose estimate {name} must contain a Pose")
        for name, jacobian in self.jacobians.items():
            if np.shape(jacobian) != (6, 7) or not np.isfinite(jacobian).all():
                raise ValueError(f"{name} Jacobian must be finite [6,7]")
        if self.base_xyyaw is not None:
            vector(self.base_xyyaw, 3, "base_xyyaw")

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        poses = {key: Estimate(Pose(tuple(item["xyz"]), tuple(item["wxyz"])),
                               float(item["timestamp"]), float(item["confidence"]), item["source"])
                 for key, item in data.get("poses", {}).items()}
        signals = {key: Estimate(item["value"], float(item["timestamp"]),
                                 float(item["confidence"]), item["source"])
                   for key, item in data.get("signals", {}).items()}
        profile = int(data.get("state_profile", 1))
        return cls(float(data["timestamp"]), vector(data["state"], 42 if profile == 2 else 62, "state"), poses, signals,
                   {k: np.asarray(v, dtype=float) for k, v in data.get("jacobians", {}).items()},
                   tuple(Rectangle(tuple(x["minimum"]), tuple(x["maximum"])) for x in data.get("obstacles", [])),
                   data.get("obstacle_timestamp"), data.get("obstacle_stage"),
                   tuple(data["base_xyyaw"]) if "base_xyyaw" in data else None,
                   profile, data.get("spine_height_m"))

    def gripper_radians(self, arm: str) -> float:
        index = GRIP_STATE[arm] if self.state_profile == 1 else {"left": 7, "right": 28}[arm]
        return float(self.state[index])


@dataclass(frozen=True)
class ServoCalibration:
    """Site-measured limits and gripper endpoints; never guess radians→command.

    Real release fields named ``percent`` store native values in [0,1].
    Commands preserve that scale; calibration defines the open/closed direction.
    """

    calibration_id: str
    joint_min: tuple[float, ...]
    joint_max: tuple[float, ...]
    gripper_open_rad: tuple[float, float]
    gripper_closed_rad: tuple[float, float]
    gripper_open_command: tuple[float, float]
    gripper_closed_command: tuple[float, float]
    max_joint_speed: float = 0.4
    max_tcp_speed: float = 0.06
    max_angular_speed: float = 0.3
    max_base_speed: float = 0.15
    max_base_yaw_speed: float = 0.25
    base_radius: float = 0.51
    payload_radius: float = 0.15
    max_state_age: float = 0.25
    min_confidence: float = 0.8
    position_tolerance: float = 0.008
    angle_tolerance: float = 0.06
    head_force_limit_n: float | None = None

    def __post_init__(self):
        if not self.calibration_id:
            raise ValueError("calibration_id is required")
        lo, hi = vector(self.joint_min, 14, "joint_min"), vector(self.joint_max, 14, "joint_max")
        if np.any(lo >= hi):
            raise ValueError("joint_min must be below joint_max")
        opened = vector(self.gripper_open_rad, 2, "gripper_open_rad")
        closed = vector(self.gripper_closed_rad, 2, "gripper_closed_rad")
        if np.any(np.abs(opened - closed) < 1e-6):
            raise ValueError("gripper endpoints must be distinct")
        command_open = vector(self.gripper_open_command, 2, "gripper_open_command")
        command_closed = vector(self.gripper_closed_command, 2, "gripper_closed_command")
        if np.any(command_open == command_closed) or np.any(np.r_[command_open, command_closed] < 0) or np.any(np.r_[command_open, command_closed] > 1):
            raise ValueError("gripper command endpoints must be distinct and in [0,1]")
        for name in ("max_joint_speed", "max_tcp_speed", "max_angular_speed", "max_base_speed",
                     "max_base_yaw_speed", "base_radius", "payload_radius", "max_state_age",
                     "position_tolerance", "angle_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.min_confidence <= 1:
            raise ValueError("min_confidence must be in (0,1]")
        if self.head_force_limit_n is not None and (not math.isfinite(self.head_force_limit_n) or self.head_force_limit_n <= 0):
            raise ValueError("head_force_limit_n must be finite and positive")

    def gripper_command(self, arm: str, radians: float) -> float:
        i = ARMS.index(arm)
        fraction = np.clip((radians-self.gripper_closed_rad[i]) /
                           (self.gripper_open_rad[i]-self.gripper_closed_rad[i]), 0, 1)
        return float(self.gripper_closed_command[i] + fraction *
                     (self.gripper_open_command[i]-self.gripper_closed_command[i]))


@dataclass(frozen=True)
class Target:
    key: str
    offset_xyz: tuple[float, float, float] = (0., 0., 0.)
    offset_wxyz: tuple[float, float, float, float] = (1., 0., 0., 0.)


@dataclass(frozen=True)
class Waypoint:
    name: str
    stage: str
    targets: Mapping[str, Target] = field(default_factory=dict)
    grippers: Mapping[str, str] = field(default_factory=dict)
    require: Mapping[str, bool] = field(default_factory=dict)
    require_while: Mapping[str, bool] = field(default_factory=dict)
    navigate: str | None = None
    hold_seconds: float = 0.15
    timeout_seconds: float = 30.
    max_retries: int = 1
    retry_index: int | None = None
    near_head: bool = False
    gripper_feedback: bool = True

    def __post_init__(self):
        if set(self.targets) - set(ARMS) or set(self.grippers) - set(ARMS):
            raise ValueError("waypoint arm names must be left/right")
        if any(x not in ("open", "closed") for x in self.grippers.values()):
            raise ValueError("gripper command must be open/closed; native command comes from site calibration")
        if self.hold_seconds < 0 or self.timeout_seconds <= self.hold_seconds:
            raise ValueError("timeout must exceed nonnegative hold duration")


@dataclass(frozen=True)
class Decision:
    action: np.ndarray | None
    status: str
    stage: str
    waypoint: str
    reason: str
    retries: int

    def to_dict(self):
        return {"action": self.action.tolist() if self.action is not None else None, "status": self.status, "stage": self.stage,
                "waypoint": self.waypoint, "reason": self.reason, "retries": self.retries}


class MissingObservation(ValueError):
    pass


class Controller:
    """Runs calibrated task waypoints at 20 Hz, gated by fresh measurements.

    Optional IK callback: (arm, target_pose, measured_q) -> seven joint targets
    or None if unreachable. Without it, the measured Jacobian drives a damped
    least-squares servo. Neither backend supplies task success predicates.
    """

    def __init__(self, plan: tuple[Waypoint, ...], calibration: ServoCalibration,
                 ik: Callable | None = None, *, episode_timeout: float = 1800.):
        if not plan or not math.isfinite(episode_timeout) or episode_timeout <= 0:
            raise ValueError("nonempty plan and positive episode_timeout required")
        self.plan, self.calibration, self.ik = plan, calibration, ik
        self.episode_timeout = episode_timeout
        self.reset()

    def reset(self):
        self.index = 0
        self.started = self.waypoint_started = self.last_timestamp = None
        self.last_command_time = None
        self.stable_seconds = 0.
        self.stable_since = None
        self.commanded_grips = None
        self._path = None
        self._route_signature = None
        self.retries: dict[int, int] = {}
        self.status = "ACTIVE"

    def hold(self, observation: Observation) -> np.ndarray:
        state = np.asarray(observation.state)
        action = np.zeros(23, dtype=float)
        for arm in ARMS:
            action[A_SLICES[arm]] = state[Q_SLICES[arm]]
            i = ARMS.index(arm)
            action[GRIP_ACTION[arm]] = (self.calibration.gripper_command(arm, observation.gripper_radians(arm))
                                        if self.commanded_grips is None else self.commanded_grips[i])
        action[22] = state[61] if observation.state_profile == 1 else observation.spine_height_m
        return action

    def _read(self, estimates, key: str, now: float, *, static: bool = False):
        item = estimates.get(key)
        if item is None:
            raise MissingObservation(f"missing sensor estimate: {key}")
        exempt = static and item.source == "calibration"
        if item.confidence < self.calibration.min_confidence or (not exempt and not 0 <= now-item.timestamp <= self.calibration.max_state_age):
            raise MissingObservation(f"stale or uncertain sensor estimate: {key}")
        return item.value

    def _conditions(self, observation: Observation, conditions: Mapping[str, bool]):
        for key, expected in conditions.items():
            value = self._read(observation.signals, key, observation.timestamp)
            if not isinstance(value, (bool, np.bool_)):
                raise MissingObservation(f"predicate {key} must be boolean")
            if value != expected:
                return False
        return True

    def _servo(self, observation: Observation, waypoint: Waypoint, dt: float):
        action, reached = self.hold(observation), True
        for arm, target_spec in waypoint.targets.items():
            current = self._read(observation.poses, f"{arm}_tcp", observation.timestamp)
            target = self._read(observation.poses, target_spec.key, observation.timestamp, static=True)
            target = target.offset(target_spec.offset_xyz, target_spec.offset_wxyz)
            error = pose_error(current, target)
            reached &= np.linalg.norm(error[:3]) <= self.calibration.position_tolerance and np.linalg.norm(error[3:]) <= self.calibration.angle_tolerance
            q = np.asarray(observation.state)[Q_SLICES[arm]]
            # Limit pose increment before IK as well as limiting joint targets.
            for component, bound in ((slice(0, 3), self.calibration.max_tcp_speed), (slice(3, 6), self.calibration.max_angular_speed)):
                norm = np.linalg.norm(error[component])
                if norm > bound*dt:
                    error[component] *= bound*dt/norm
            if self.ik is None:
                if arm not in observation.jacobians:
                    raise MissingObservation(f"missing measured {arm} Jacobian or IK backend")
                jacobian = np.asarray(observation.jacobians[arm])
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.02**2*np.eye(6), error)
                desired = q + delta
            else:
                angle = np.linalg.norm(error[3:])
                increment = np.r_[math.cos(angle/2), error[3:] * (math.sin(angle/2)/angle)] if angle > 1e-10 else np.array([1., 0., 0., 0.])
                bounded_target = Pose(tuple(np.asarray(current.xyz)+error[:3]),
                                      tuple(quat_multiply(increment, current.wxyz)))
                desired = self.ik(arm, bounded_target, q.copy())
                if desired is None:
                    raise MissingObservation(f"{arm} target is unreachable")
                desired = vector(desired, 7, "IK result")
            part = slice(ARMS.index(arm)*7, (ARMS.index(arm)+1)*7)
            lo, hi = np.asarray(self.calibration.joint_min)[part], np.asarray(self.calibration.joint_max)[part]
            if np.any(q < lo-1e-3) or np.any(q > hi+1e-3):
                raise MissingObservation(f"measured {arm} joints outside calibrated limits")
            desired = np.clip(desired, lo, hi)
            action[A_SLICES[arm]] = q + np.clip(desired-q, -self.calibration.max_joint_speed*dt, self.calibration.max_joint_speed*dt)
        for arm, goal in waypoint.grippers.items():
            command = (self.calibration.gripper_open_command if goal == "open" else self.calibration.gripper_closed_command)[ARMS.index(arm)]
            measured = self.calibration.gripper_command(arm, observation.gripper_radians(arm))
            previous = action[GRIP_ACTION[arm]]
            # Full native [0,1] travel over at least one second.
            action[GRIP_ACTION[arm]] = previous + np.clip(command-previous, -dt, dt)
            if waypoint.gripper_feedback:
                reached &= abs(measured-command) <= 0.05
        if waypoint.navigate:
            reached &= self._navigate(observation, waypoint, action)
        return action, bool(reached)

    def _navigate(self, observation: Observation, waypoint: Waypoint, action: np.ndarray):
        timestamp = observation.obstacle_timestamp
        if observation.base_xyyaw is None or timestamp is None or not math.isfinite(timestamp) or not 0 <= observation.timestamp-timestamp <= self.calibration.max_state_age:
            raise MissingObservation("navigation needs fresh measured base pose and obstacle map")
        if observation.obstacle_stage != waypoint.stage:
            raise MissingObservation("obstacle map must be refreshed for this stage")
        pose = self._read(observation.poses, waypoint.navigate, observation.timestamp, static=True)
        base = np.asarray(observation.base_xyyaw)
        clearance = self.calibration.base_radius+self.calibration.payload_radius
        signature = (waypoint.stage, pose, observation.obstacles, clearance)
        if self._path is not None:
            while len(self._path) > 1 and math.dist(base[:2], self._path[0]) < 0.025:
                self._path.pop(0)
        if (self._path is None or signature != self._route_signature or
                any(rect.intersects(base[:2], self._path[0], clearance) for rect in observation.obstacles)):
            self._path = route(base[:2], pose.xyz[:2], observation.obstacles, clearance)[1:]
            self._route_signature = signature
        delta = np.asarray(self._path[0])-base[:2]
        distance = np.linalg.norm(delta)
        w, x, y, z = pose.wxyz
        target_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        yaw_error = math.atan2(math.sin(target_yaw-base[2]), math.cos(target_yaw-base[2]))
        if math.dist(base[:2], pose.xyz[:2]) <= 0.03 and abs(yaw_error) <= 0.04:
            return True
        velocity = delta * min(1., self.calibration.max_base_speed/max(distance, 1e-9))
        # This adapter's base Twist is body-frame linear xyz, angular xyz.
        # The site transport must convert its actual controller convention.
        c, s = math.cos(base[2]), math.sin(base[2])
        action[16:18] = [c*velocity[0]+s*velocity[1], -s*velocity[0]+c*velocity[1]]
        if distance < 0.15:
            action[21] = np.clip(yaw_error, -self.calibration.max_base_yaw_speed, self.calibration.max_base_yaw_speed)
        return False

    def step(self, observation: Observation, *, now: float | None = None) -> Decision:
        waypoint = self.plan[min(self.index, len(self.plan)-1)]
        held = self.hold(observation)
        def result(action=held, status=None, reason=""):
            effective = status or self.status
            if effective in ("WAITING", "RETRY", "FAILED"):
                self.stable_since = None
            if effective in ("ACTIVE", "COMPLETE"):
                self.commanded_grips = (float(action[7]), float(action[15]))
            return Decision(action, effective, waypoint.stage, waypoint.name,
                            reason, self.retries.get(self.index, 0))
        now = time.time() if now is None else now
        if not math.isfinite(now) or not 0 <= now-observation.timestamp <= self.calibration.max_state_age:
            self.stable_seconds = 0.
            return result(action=None, status="WAITING", reason="stale robot observation")
        if self.last_timestamp is not None and observation.timestamp <= self.last_timestamp:
            self.stable_seconds = 0.
            return result(action=None, status="WAITING", reason="duplicate or backwards robot timestamp")
        if self.last_command_time is not None and now <= self.last_command_time:
            return result(action=None, status="WAITING", reason="duplicate or backwards command clock")
        dt = 0. if self.last_timestamp is None else observation.timestamp-self.last_timestamp
        command_dt = dt if self.last_command_time is None else min(dt, now-self.last_command_time)
        self.last_timestamp = observation.timestamp
        self.last_command_time = now
        if self.status in ("COMPLETE", "FAILED"):
            return result(reason="episode ended; reset before reuse")
        if self.started is None:
            self.started = self.waypoint_started = observation.timestamp
        if observation.timestamp-self.started >= self.episode_timeout:
            self.status = "FAILED"
            return result(reason="episode time limit")
        if observation.timestamp-self.waypoint_started >= waypoint.timeout_seconds:
            self.stable_seconds = 0.
            count = self.retries.get(self.index, 0)
            if count >= waypoint.max_retries:
                self.status = "FAILED"
                return result(reason="waypoint timeout; retry budget exhausted")
            self.retries[self.index] = count+1
            self.index = waypoint.retry_index if waypoint.retry_index is not None else self.index
            self.waypoint_started = observation.timestamp
            return result(status="RETRY", reason="waypoint timed out; reacquire measured scene")
        try:
            if not self._conditions(observation, {"collision_clear": True}):
                self.stable_seconds = 0.
                return result(status="WAITING", reason="collision clearance not confirmed")
            for key in ("pad_torn", "emergency_stop"):
                if key in observation.signals and self._read(observation.signals, key, observation.timestamp) is True:
                    self.status = "FAILED"
                    return result(reason=key)
            if waypoint.near_head:
                force_limit = self.calibration.head_force_limit_n
                if force_limit is None:
                    raise MissingObservation("near-head control requires site force-limit calibration")
                force = self._read(observation.signals, "head_force_n", observation.timestamp)
                if not isinstance(force, (int, float)) or isinstance(force, bool) or force < 0:
                    raise MissingObservation("head_force_n must be a nonnegative measurement")
                if force >= force_limit:
                    self.status = "FAILED"
                    return result(reason="site head-force limit exceeded")
            if not self._conditions(observation, waypoint.require_while):
                self.stable_seconds = 0.
                return result(status="WAITING", reason="motion precondition not met")
            action, reached = self._servo(observation, waypoint, min(max(command_dt, 0.), 0.05))
            conditions = self._conditions(observation, waypoint.require)
            if reached and conditions:
                if self.stable_since is None or dt > self.calibration.max_state_age:
                    self.stable_since = observation.timestamp
                self.stable_seconds = observation.timestamp-self.stable_since
                if self.stable_seconds >= waypoint.hold_seconds:
                    self.index += 1
                    self.waypoint_started = observation.timestamp
                    self.stable_seconds = 0.
                    self.stable_since = None
                    if self.index == len(self.plan):
                        self.status = "COMPLETE"
                    return result(action, reason="fresh measured completion criteria satisfied")
            else:
                self.stable_seconds = 0.
                self.stable_since = None
            return result(action, reason="tracking measured targets" if not reached else "waiting for measured task outcome")
        except (MissingObservation, ValueError, np.linalg.LinAlgError) as error:
            self.stable_seconds = 0.
            return result(status="WAITING", reason=str(error))
