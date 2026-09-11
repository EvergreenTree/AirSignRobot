---
name: ebim-task3-phase2
description: Resume AirSign EBiM Task 3 work on the two-host Mobile FR3 Duo, inspect wrist RGB-D and measured robot state, run individually reviewed finite motions, or prepare an evidence-backed Phase II submission. Use for this hardware integration, not generic robotics or simulated task completion.
---

# EBiM Task 3 on-site workflow

The repository root is two directories above this file. Start with
`phase2/task3_submission/README.md` for the last measured result and evidence.
The last session ended on 12 September 2026; historical readiness is not live readiness.

## Establish the current state

1. Read [hardware and recovery](references/hardware.md) and the site's current
   `TMR_Two_Host_Runtime_Guide.md` (troubleshooting sections 7-8) and `驱动启动.pdf`.
   Those private guides supply credentials; never copy credentials into source or reports.
2. Determine whether the operator is present and the session is still authorized.
   After departure, restrict work to offline packaging and read-only checks; do not resume motion.
3. Reach the hosts by SSH and inspect their actual kernel, arm status, FCI state,
   control ownership and running controllers. An SSH connection establishes Linux
   reachability, not readiness of the drive/spine/arm interfaces.
4. Acquire fresh camera images, depth, intrinsics and measured arm poses.
   `phase2/scripts/probe_hardware.py` is a finite subscription-only snapshot;
   its first received frame is not a continuous video feed.

## Interpret vision before movement

Read [frames and camera pitfalls](references/vision.md). Resolve left/right from
current launch serials plus geometry and images. Do not reuse an identity TF
between the two wrists. Do not index raw depth with RGB pixels without registration.
Use nominal URDF geometry only as a hypothesis until checked against measured poses
and image features. Session plate coordinates are historical evidence, never new-run targets.

## Use the tested low-level boundary

The preserved sources and exact on-site paths are documented in
`phase2/hardware/task3_onsite/README.md`. They are supervised primitives, not a
complete task policy. The arm helper enforces real-time scheduling; the state-only
probe does not need a control loop. Never change motion to `kIgnore` to overcome
a scheduling refusal. Do not use automatic homing or fault recovery as a probe.

Take normal control only when available, activate FCI, read the current pose,
then execute one finite, pose-anchored step after checking its swept space.
Read state and new images again before the next step. Stop on error, contact,
stale observations or changed ownership. The preserved helper's thresholds are
session engineering limits, not certified collision protection or grasp forces.
Avoid launching the site's dual-arm controller blindly: its documented startup
can move the right arm backwards.

The normal session supports `READ`, `STEP dx dy dz seconds`, and `RELEASE`.
Directions are in the arm base frame. Maximum displacement is 50 mm per call,
duration 3-8 seconds. It does not plan around obstacles or close the gripper.
The operator's physical emergency stop remains the immediate stop; network
release and process termination do not replace it.

## End and preserve evidence

Release owned control, verify FCI inactive and both arms idle, and use the normal
brake-lock procedure when available. Record only states actually observed;
unreachable hardware is not proof of locked brakes. Never send another movement
after the operator leaves.

Keep raw telemetry separate from inferred geometry and task outcomes. A successful
trajectory is not evidence of plate contact, grasp, lift, feeding or a benchmark
score. Record human assistance and perception/transport dependencies. Use the
current official issue form and the team's emailed deadline. A skill alone does
not satisfy the runnable-policy submission requirements.
