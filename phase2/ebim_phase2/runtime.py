"""ACT inference and calibrated command limiting, over JSONL or a local HTTP service."""
from __future__ import annotations
import argparse
import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image
import torch

from .data import letterbox
from .model import ACT, ACTConfig
from .schema import CAMERAS, JOINT_ACTION_INDICES, JOINT_STATE_INDICES, PROFILES, action23, command_dict


def base_command_in_body_frame(action, state, native_frame):
    """Convert the recorded planar velocity convention into this API's body frame."""
    if native_frame not in ('body','world'):
        raise ValueError('Calibrate native_base_velocity_frame as body or world')
    result=np.asarray(action,dtype=float).copy()
    if native_frame=='world':
        quat=np.asarray(state,dtype=float)[45:49]  # Native base pose: xyz, quaternion xyzw.
        norm=np.linalg.norm(quat)
        if quat.shape!=(4,) or not np.isfinite(quat).all() or not .99<=norm<=1.01:
            raise ValueError('World-frame base commands require a valid measured base quaternion')
        x,y,z,w=quat/norm
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        c,s=math.cos(yaw),math.sin(yaw)
        vx,vy=result[16:18]
        result[16:18]=[c*vx+s*vy,-s*vx+c*vy]
    return result


class TemporalEnsemble:
    def __init__(self, horizon=32, decay=0.25):
        self.horizon, self.decay = horizon, decay
        self.chunks = []

    def reset(self):
        self.chunks.clear()

    def step(self, timestamp: float, chunk: np.ndarray) -> np.ndarray:
        if self.chunks and timestamp <= self.chunks[-1][0]:
            raise ValueError("Policy observations must advance in time")
        self.chunks = [(t, c) for t,c in self.chunks if round((timestamp-t)*20) < len(c)]
        self.chunks.append((timestamp, chunk.copy()))
        values, weights = [], []
        for t, c in self.chunks:
            age = max(0, round((timestamp-t)*20))
            if age < len(c):
                values.append(c[age])
                weights.append(math.exp(-self.decay*age))
        return np.average(values, weights=weights, axis=0)


class CommandLimiter:
    """Site limits must be explicit; these are not claimed ISO/manufacturer safety certification."""
    def __init__(self, config: dict):
        required = ["calibration_id", "joint_lower", "joint_upper", "joint_speed_rad_s", "gripper_open_knuckle",
                    "gripper_closed_knuckle", "gripper_open_command", "gripper_closed_command", "spine_lower_m",
                    "spine_upper_m", "spine_speed_m_s", "base_linear_speed_m_s", "base_yaw_speed_rad_s",
                    "max_force_n", "max_observation_age_s", "max_camera_skew_s"]
        missing = set(required)-set(config)
        if missing:
            raise ValueError(f"Site calibration missing: {sorted(missing)}")
        self.config = config
        for key in ("joint_lower", "joint_upper", "joint_speed_rad_s"):
            if np.shape(config[key]) != (14,) or not np.isfinite(config[key]).all():
                raise ValueError(f"{key} must contain 14 finite values")
        if np.any(np.array(config["joint_lower"]) >= config["joint_upper"]):
            raise ValueError("Joint lower bounds must be below upper bounds")
        if np.any(np.array(config["joint_speed_rad_s"]) <= 0):
            raise ValueError("Joint speed limits must be positive")
        for key in ("gripper_open_knuckle", "gripper_closed_knuckle", "gripper_open_command", "gripper_closed_command"):
            if np.shape(config[key]) != (2,) or not np.isfinite(config[key]).all():
                raise ValueError(f"Invalid {key}")
        if np.any(np.array(config["gripper_open_knuckle"]) == config["gripper_closed_knuckle"]):
            raise ValueError("Gripper calibration endpoints must differ")
        opened = np.array(config["gripper_open_command"])
        closed = np.array(config["gripper_closed_command"])
        if np.any(opened == closed) or np.any(np.r_[opened, closed] < 0) or np.any(np.r_[opened, closed] > 1):
            raise ValueError("Gripper command endpoints must be distinct and in native [0,1]")
        for key in ("spine_speed_m_s", "base_linear_speed_m_s", "base_yaw_speed_rad_s", "max_force_n",
                    "max_observation_age_s", "max_camera_skew_s"):
            if not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not all(math.isfinite(config[key]) for key in ("spine_lower_m", "spine_upper_m")) or not config["spine_lower_m"] < config["spine_upper_m"]:
            raise ValueError("Invalid spine bounds")
        self.reset()

    def reset(self):
        self.last_observation_timestamp = None
        self.last_command_time = None

    def gripper_commands(self, state, task):
        c = self.config
        measured = state[list(PROFILES[task].gripper_state_indices)]
        closed = np.array(c["gripper_closed_knuckle"])
        fraction = np.clip((measured-closed)/(np.array(c["gripper_open_knuckle"])-closed), 0, 1)
        return np.array(c["gripper_closed_command"]) + fraction*(np.array(c["gripper_open_command"])-c["gripper_closed_command"])

    def apply(self, proposal, state, task, record, now):
        c = self.config
        state = np.asarray(state, dtype=float)
        if state.shape != (PROFILES[task].state_dim,) or not np.isfinite(state).all():
            raise ValueError("Cannot command from invalid measured state")
        timestamp = float(record["timestamp_s"])
        spine = float(state[61] if task == 1 else record["spine_height_m"])
        if not math.isfinite(timestamp) or not math.isfinite(now) or not math.isfinite(spine):
            raise ValueError("Invalid timing/spine observation")
        reason = None
        if not 0 <= now-timestamp <= c["max_observation_age_s"]:
            reason = "stale_or_future_state"
        elif self.last_observation_timestamp is not None and timestamp <= self.last_observation_timestamp:
            reason = "nonmonotonic_observation"
        elif self.last_command_time is not None and now <= self.last_command_time:
            reason = "nonmonotonic_command_time"
        elif record.get("collision_imminent") is not False:
            reason = "collision_or_missing_collision_observation"
        elif any(np.linalg.norm(state[s]) > c["max_force_n"] for s in PROFILES[task].force_slices):
            reason = "force_limit"
        elif np.shape(proposal) != (23,) or not np.isfinite(proposal).all():
            reason = "nonfinite_prediction"
        if reason:
            # A stale pose is not a hold target: sending it can pull the robot
            # back toward an obsolete position. Stop updates; transport must
            # retain its deadman watchdog and local low-level hold behavior.
            return None, reason
        dt = 0.05
        if self.last_observation_timestamp is not None:
            dt = min(dt, timestamp-self.last_observation_timestamp, now-self.last_command_time)
        result = np.asarray(proposal, dtype=float).copy()
        q = state[JOINT_STATE_INDICES]
        delta = np.array(c["joint_speed_rad_s"])*dt
        lower = np.maximum(c["joint_lower"], q-delta)
        upper = np.minimum(c["joint_upper"], q+delta)
        if np.any(lower > upper):
            return None, "measured_joints_outside_limits"
        result[JOINT_ACTION_INDICES] = np.clip(result[JOINT_ACTION_INDICES], lower, upper)
        g0, g1 = np.array(c["gripper_closed_command"]), np.array(c["gripper_open_command"])
        result[[7, 15]] = np.clip(result[[7, 15]], np.minimum(g0, g1), np.maximum(g0, g1))
        result[18:21] = 0  # Planar base: prohibit z, roll and pitch commands.
        speed = np.linalg.norm(result[16:18])
        if speed > c["base_linear_speed_m_s"]:
            result[16:18] *= c["base_linear_speed_m_s"]/speed
        result[21] = np.clip(result[21], -c["base_yaw_speed_rad_s"], c["base_yaw_speed_rad_s"])
        lo, hi = max(c["spine_lower_m"], spine-c["spine_speed_m_s"]*dt), min(c["spine_upper_m"], spine+c["spine_speed_m_s"]*dt)
        if lo > hi:
            return None, "measured_spine_outside_limits"
        result[22] = np.clip(result[22], lo, hi)
        if task == 2:
            result[16:22] = 0  # No mobile commands were demonstrated in the Munich release.
        self.last_observation_timestamp, self.last_command_time = timestamp, now
        return result, "bounded_proposal"


class Policy:
    def __init__(self, checkpoint: Path, device="cuda", calibration: dict | None = None):
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("format") != "airsign_act_v1" or saved.get("task") not in PROFILES:
            raise ValueError("Unsupported checkpoint; legacy Task3/command checkpoints cannot be deployed")
        self.task = saved["task"]
        self.native_base_frame = None
        self.conversion = None
        if calibration:
            if self.task==1:
                self.native_base_frame=calibration.get('native_base_velocity_frame')
                if self.native_base_frame not in ('body','world'):
                    raise ValueError('Task1 calibration requires native_base_velocity_frame: body or world')
            required = ('joint_action_scale', 'joint_action_offset_rad',
                        'spine_scale_m_per_native', 'spine_offset_m')
            if any(key not in calibration for key in required):
                raise ValueError('Learned-policy calibration needs explicit native joint and spine conversions')
            self.conversion = {key: calibration[key] for key in required}
            # GELLO recorded targets are not certified as the physical servo's
            # coordinates by their field name. Even an identity map is explicit.
            action23(np.zeros(PROFILES[self.task].action_dim), self.task, **self.conversion)
        self.config = ACTConfig(**saved["model_config"])
        self.model = ACT(self.config, pretrained=False).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.stats = {k: np.array(v, dtype=np.float32) for k,v in saved["stats"].items() if k != "episodes"}
        self.schema = saved["schema"]
        self.ensemble = TemporalEnsemble(self.config.chunk_size)
        self.limiter = CommandLimiter(calibration) if calibration else None
        self.episode_id = None
        self.lock = threading.Lock()

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.episode_started_at = None
        self.ensemble.reset()
        if self.limiter:
            self.limiter.reset()

    @torch.inference_mode()
    def infer(self, record: dict, *, now: float | None = None):
        with self.lock:
            if not isinstance(record, dict):
                raise ValueError('Observation must be a JSON object')
            if record.get("episode_id") is None:
                raise ValueError("episode_id is required to reset temporal state")
            if record["episode_id"] != self.episode_id:
                self.reset(record["episode_id"])
            if int(record["task"]) != self.task:
                raise ValueError("Checkpoint/task mismatch")
            if record.get("state_names") != self.schema["observation.state"]["names"]:
                raise ValueError("State names/order must exactly match checkpoint schema")
            state = np.asarray(record["state"], dtype=np.float32)
            if state.shape != (self.config.state_dim,) or not np.isfinite(state).all():
                raise ValueError("Invalid state")
            timestamp = float(record["timestamp_s"])
            if not math.isfinite(timestamp):
                raise ValueError("Invalid observation timestamp")
            clock = time.time() if now is None else float(now)
            if not math.isfinite(clock):
                raise ValueError('Invalid receiver clock')
            if self.episode_started_at is None:
                self.episode_started_at = clock
            if self.task == 1 and clock-self.episode_started_at >= 1800:
                self.ensemble.reset()
                return self._result(record, timestamp, None, 'no_command', 'task1_time_limit')
            if self.limiter:
                # Reject stale/collision/force observations before image decoding
                # or model inference; check freshness again after inference.
                prior_timing = (self.limiter.last_observation_timestamp, self.limiter.last_command_time)
                _, preflight_reason = self.limiter.apply(np.zeros(23), state, self.task, record,
                                                        time.time() if now is None else now)
                if preflight_reason != "bounded_proposal":
                    self.ensemble.reset()
                    return self._result(record, timestamp, None, "no_command", preflight_reason)
                # Preflight is validation, not an emitted command. Timing state
                # must change only for the final command below.
                self.limiter.last_observation_timestamp, self.limiter.last_command_time = prior_timing
            skew_limit = self.limiter.config["max_camera_skew_s"] if self.limiter else 0.075
            images = []
            for key in CAMERAS:
                camera = record["images"][key]
                camera_time = float(camera["timestamp_s"])
                if not math.isfinite(camera_time) or abs(timestamp-camera_time) > skew_limit:
                    raise ValueError(f"Stale or unsynchronized camera: {key}")
                if "base64" in camera:
                    im = np.asarray(Image.open(BytesIO(base64.b64decode(camera["base64"], validate=True))).convert("RGB"))
                else:
                    im = np.asarray(Image.open(camera["path"]).convert("RGB"))
                images.append(letterbox(im, self.config.image_size).transpose(2,0,1))
            image_tensor = torch.from_numpy(np.stack(images)[None]).to(self.device)
            norm_state = (state-self.stats["state_mean"])/self.stats["state_std"]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                chunk, _, _ = self.model(image_tensor, torch.from_numpy(norm_state[None]).to(self.device))
            chunk = chunk[0].float().cpu().numpy()*self.stats["action_std"] + self.stats["action_mean"]
            native = self.ensemble.step(timestamp, chunk)
            if self.limiter is None:
                result = self._result(record, timestamp, None, "shadow_native_proposal",
                                      "native_command_conversion_not_calibrated")
                result["native_action"] = native.tolist()
                return result
            proposal = action23(native, self.task, **self.conversion)
            if self.task==1:
                proposal=base_command_in_body_frame(proposal,state,self.native_base_frame)
            action, reason = self.limiter.apply(proposal, state, self.task, record, time.time() if now is None else now)
            status = "bounded_proposal" if action is not None else "no_command"
            if action is None:
                self.ensemble.reset()
            result = self._result(record, timestamp, action, status, reason)
            result["native_action"] = native.tolist()
            return result

    def _result(self, record, timestamp, action, status, reason):
        return {"team": "AirSign", "task": self.task, "episode_id": self.episode_id,
                "observation_timestamp_s": timestamp, "status": status, "reason": reason,
                "action": action.tolist() if action is not None else None,
                "commands": command_dict(action) if action is not None else None,
                "native_action_dim": self.config.action_dim}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument('--route', type=Path, help='Task1 current assignment JSON; require fresh route_progress observations')
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy = Policy(args.checkpoint, args.device, json.loads(args.calibration.read_text()) if args.calibration else None)
    if args.route:
        from .cable import Route, SupervisedRoutingPolicy
        assignment=json.loads(args.route.read_text())
        policy=SupervisedRoutingPolicy(policy,Route(assignment['route_id'],tuple(assignment['fixtures']),tuple(assignment['directions'])))
    if args.port:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = json.dumps({"team": "AirSign", "task": policy.task, "ready": True,
                                      "mode": "bounded" if policy.limiter else "shadow"}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(payload)
            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 32*1024*1024:
                        raise ValueError("Invalid request size")
                    result = policy.infer(json.loads(self.rfile.read(length)))
                    code = 200
                except (ValueError, TypeError, KeyError, OSError) as e:
                    policy.ensemble.reset()
                    result, code = {"error": str(e), "status": "no_command", "action": None, "commands": None}, 400
                payload = json.dumps(result, allow_nan=False).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(payload)
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    else:
        for line in sys.stdin:
            if not line.strip():continue
            try:
                record = json.loads(line)
                result = policy.infer(record)
            except (ValueError, TypeError, KeyError, OSError) as error:
                policy.ensemble.reset()
                result={'status':'no_command','action':None,'commands':None,'error':str(error)}
            print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
