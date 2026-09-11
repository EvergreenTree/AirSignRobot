"""Robot kinematics and actuator diagnostics; scene geometry is not a Phase II problem definition."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from .download import sha256
from .schema import JOINT_ACTION_INDICES


def required_id(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"Model lacks required named object: {name}")
    return value


class RobotKinematics:
    """Read measured joint coordinates into a separate kinematic model, return TCP Jacobians.

    The model's robot/base frames must match the supplied site calibration. No task object
    pose is observed or changed. The data object is private, never the physics rollout.
    """
    def __init__(self, model, mapping):
        self.model, self.data, self.mapping = model, mujoco.MjData(model), mapping
        self.arm_joint_ids = {}
        self.tcp_ids = {}
        for arm in ("left", "right"):
            names = mapping[arm]["joints"]
            if len(names) != 7 or len(set(names)) != 7:
                raise ValueError("Expected seven distinct arm joints")
            self.arm_joint_ids[arm] = np.array([required_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
            self.tcp_ids[arm] = required_id(model, mujoco.mjtObj.mjOBJ_BODY, mapping[arm]["tcp_body"])

    def observe(self, left_q, right_q, extra_joints=None):
        if set(extra_joints or {}) != set(self.mapping.get('extra_joints', [])):
            raise ValueError('Supply every calibrated base/spine coordinate on each kinematic observation')
        for arm, q in (("left", left_q), ("right", right_q)):
            q = np.asarray(q)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError("Expected seven finite measured joint positions")
            self.data.qpos[self.model.jnt_qposadr[self.arm_joint_ids[arm]]] = q
        for name, value in (extra_joints or {}).items():
            if name not in self.mapping.get("extra_joints", []):
                raise ValueError(f"Extra joint is not calibrated: {name}")
            if not np.isfinite(value):
                raise ValueError("Nonfinite joint")
            jid = required_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(self.model, self.data)
        result = {}
        for arm in ("left", "right"):
            body = self.tcp_ids[arm]
            jp, jr = np.zeros((3,self.model.nv)), np.zeros((3,self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jp, jr, body)
            dofs = self.model.jnt_dofadr[self.arm_joint_ids[arm]]
            result[arm] = {"position": self.data.xpos[body].tolist(),
                           "quaternion_wxyz": self.data.xquat[body].tolist(),
                           "jacobian": np.concatenate([jp[:,dofs],jr[:,dofs]],0).tolist()}
        return result


class ActuatorDiagnostic:
    """Track measured/native joint targets using only MuJoCo actuators during a rollout."""
    def __init__(self, scene: Path, mapping: dict, initial_positions=None):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.mapping = mapping
        self.joints = []
        self.actuators = []
        for arm in ("left", "right"):
            self.joints.extend(required_id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in mapping[arm]["joints"])
            self.actuators.extend(required_id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in mapping[arm]["actuators"])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.dofadr = self.model.jnt_dofadr[self.joints]
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        if initial_positions is not None:
            initial_positions = np.asarray(initial_positions, dtype=float)
            if initial_positions.shape != (14,) or not np.isfinite(initial_positions).all():
                raise ValueError('Initial arm positions need 14 finite joint angles')
            # Initial conditions only. No rollout step writes to qpos/qvel.
            self.data.qpos[self.qadr] = initial_positions
            self.data.qvel[self.dofadr] = 0
        self.limits = self.model.jnt_range[self.joints]
        q = self.data.qpos[self.qadr]
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError('Scene keyframe has arm joints outside model limits; supply valid --initial-joints')
        mujoco.mj_forward(self.model,self.data)

    def step(self, command, duration=.05):
        command=np.asarray(command)
        if command.shape != (23,) or not np.isfinite(command).all():
            raise ValueError("Expected finite action23")
        target=command[JOINT_ACTION_INDICES]
        if np.any(target < self.limits[:, 0]) or np.any(target > self.limits[:, 1]):
            raise ValueError('Joint target exceeds the model limits')
        mode=self.mapping["arm_actuator_mode"]
        if mode not in ("position", "velocity"):
            raise ValueError("Calibrate actuator mode explicitly")
        steps=max(1,round(duration/self.model.opt.timestep))
        for _ in range(steps):
            controls=target if mode=="position" else np.clip(8*(target-self.data.qpos[self.qadr]),-.4,.4)
            for aid,value in zip(self.actuators,controls):
                self.data.ctrl[aid]=np.clip(value,*self.model.actuator_ctrlrange[aid])
            for arm,index in (("left",7),("right",15)):
                spec=self.mapping[arm]
                aid=required_id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,spec["gripper_actuator"])
                # Simulation calibration, never reused as physical gripper calibration.
                fraction=np.clip(command[index],0,1)
                self.data.ctrl[aid]=spec["gripper_closed_ctrl"]+fraction*(spec["gripper_open_ctrl"]-spec["gripper_closed_ctrl"])
            if "spine_actuator" in self.mapping:
                aid=required_id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,self.mapping["spine_actuator"])
                self.data.ctrl[aid]=np.clip(command[22],*self.model.actuator_ctrlrange[aid])
            # This component probe keeps the mobile base stationary; no invented base calibration.
            for name in self.mapping.get("base_velocity_actuators",[]):
                self.data.ctrl[required_id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)]=0
            mujoco.mj_step(self.model,self.data)
        if not np.isfinite(self.data.qpos).all():
            raise FloatingPointError("Simulation diverged")
        return {"time_s":self.data.time,"joint_positions":self.data.qpos[self.qadr].tolist(),
                "tracking_error_rad":(target-self.data.qpos[self.qadr]).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--mapping",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument('--initial-joints', type=Path, help='Explicit simulation-only initial joint condition JSON')
    args=p.parse_args()
    mapping=json.loads(args.mapping.read_text())
    initial = json.loads(args.initial_joints.read_text()) if args.initial_joints else None
    if initial and initial['joint_names'] != mapping['left']['joints']+mapping['right']['joints']:
        raise ValueError('Initial joint names disagree with actuator mapping')
    probe=ActuatorDiagnostic(args.scene,mapping,initial['joint_positions_rad'] if initial else None)
    q0=probe.data.qpos[probe.qadr].copy()
    command=np.zeros(23);command[JOINT_ACTION_INDICES]=q0;command[[7,15]]=1
    if "spine_actuator" in mapping:
        aid=required_id(probe.model,mujoco.mjtObj.mjOBJ_ACTUATOR,mapping["spine_actuator"])
        jid=probe.model.actuator_trnid[aid,0]
        command[22]=probe.data.qpos[probe.model.jnt_qposadr[jid]]
    command[0]+=0.03;command[8]-=0.03
    records=[probe.step(command) for _ in range(args.steps)]
    movement=probe.data.qpos[probe.qadr]-q0
    result={"team":"AirSign","scope":"actuator diagnostic only; not a Phase II task rollout",
            "scene_sha256":sha256(args.scene),"mapping_sha256":sha256(args.mapping),
            'initial_joints_sha256': sha256(args.initial_joints) if args.initial_joints else None,
            "mujoco":mujoco.__version__,"steps":args.steps,"joint_movement_rad":movement.tolist(),
            "max_tracking_error_rad":float(np.max(np.abs(records[-1]["tracking_error_rad"]))),
            "finite":True,"benchmark_score":None,"records":records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}))


if __name__=="__main__":
    main()
