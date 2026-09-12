# AirSign: a skill-based visual action loop for Task 3

**EBiM Phase II Technical Report - Assisted Living & Feeding**

**Author:** Changqing Fu, AirSign

**Assigned site:** Shanghai

**Hardware session:** September 11-12, 2026

## Abstract

We present an installable Codex skill that connects a visual agent to a bounded,
measured-state robot action loop. The skill packages operational knowledge and
three runtime sources so a new agent can establish the two-host interface,
interpret camera/arm frames and issue short gamepad-style actions without writing
or starting a new C++ program for each decision. The target agent is Codex with
GPT-6 Astra; the model is supplied by the evaluator's host, not embedded or trained
in this package. The primary submission is this technical report.

During supervised work on Mobile FR3 Duo, earlier finite controllers produced
17.64 cm net right-arm displacement toward a plate and an estimated 57.96 degrees
of cumulative base turning. No plate contact, grasp, lift or complete task stage
was verified. The subsequent persistent interface passed offline guard and
latency checks, but has not been physically validated. We report these scopes
separately and make no official benchmark-score claim.

## 1. System and contribution

The skill is a portable folder containing SKILL.md, three runtime sources,
CMake input, hardware/frame/loop references, UI invocation metadata, a labelled
image fixture and its license. It does not rely on files outside its installed
directory. The three roles are:

| Component | Responsibility |
| --- | --- |
| Codex + Astra + installed skill | Read the image, interpret the reviewed goal, choose one action and assess the next observation |
| Camera relay on sensor host | Maintain a depth-one RGB subscription and return a newly captured, timestamp-checked frame |
| Native worker on real-time host | Retain the robot connection and execute one finite pose-anchored Cartesian pulse |

The agent's model/vision tools already exist in its Codex host. The skill provides
instructions and local tools; it does not start another model process or include
weights, an API client, account credentials or Docker. The operator supplies
normal robot initialization and the site's device/driver configuration.

The interface separates adaptation from execution. At session setup, the agent
checks host roles, ownership/FCI state, camera identity and arm-base axes. During
the fast loop it reuses the existing subscriber, SSH connection and native process.
This removes per-action compilation/process orchestration from the intended path.

## 2. Action and feedback contract

A view contains an image path, unique frame ID, measured pose, expiry and allowed
actions. The agent emits one line such as `<frame> R:Z-/fine`. Default pulses are
2 mm over 0.8 s; `/fine` selects 1 mm. Directions are expressed in the selected
arm's base coordinates. The R/L label does not select a physical IP or convert
camera-relative directions. The agent must establish that mapping before movement.

A frame permits one action and expires after five seconds. The bridge maps this
deadline conservatively to the worker's monotonic clock using a fresh round trip.
The native worker checks expiry again at execution. A successful pulse returns
a measured result and new camera frame; the next decision must use that new view.
Invalid or repeated frames cannot authorize another action, and busy requests
are rejected rather than queued.

The native process persists, but each pulse uses a finite libfranka control call.
It enforces real-time scheduling, initial idle/error state, agreement with the
reviewed measured pose, and guards on force changes, collision flags, joint
speed/change, orientation, displacement and timing. These are engineering guards,
not a certified collision-avoidance or contact controller. STOP, EOF and signals
request stopping; a stopped/faulted pulse requires inspection rather than automatic
recovery. The operator retains the physical emergency stop and normal ownership
cleanup responsibility.

## 3. Calibration and human assistance

The on-site analysis used 640x480 wrist RGB-D, measured arm transforms and nominal
vendor geometry. Current camera-side labels were checked against kinematics and
image features. An identity TF between the wrists was unsuitable as physical
hand-eye calibration; raw RGB and depth had different intrinsics and required
registration. The measured flange-to-configured-EE offset was 174 mm. Finger
contact geometry and full hand-eye calibration remained incomplete.

The operator initialized hardware, supervised the emergency stop, supplied
high-level directions and moved the desk in front of the robot. A remote visual
agent selected image/depth regions and reviewed finite movements. No left-arm
trajectory was executed during the plate approach. This was not an uninterrupted
autonomous task trial. The persistent gamepad prototype was developed afterward.

For pose dependence, our declaration is **Partly: agent-selected visual targets,
nominal calibration and externally supplied robot observations**. No simulator
ground-truth poses drove the real-hardware steps. No Task 3 trajectory dataset
was used for training, and no Task 3 learned weights are claimed.

## 4. Evidence and results

| Measurement | Result | Evidence scope |
| --- | --- | --- |
| Right-arm net endpoint displacement | 0.176448 m | Real measured initial/final O_T_EE; not path length or plate movement |
| Cumulative clockwise base turn | 57.9578 degrees | Sum of settled scan-registration increments, turns 2-21; not external metrology |
| Guard checks | 15 passed | Mock: expiry, reuse, bounds, STOP, EOF, busy rejection and camera loss |
| Ten local timing cycles | Median 924.14 ms; maximum 927.08 ms | Static fixture, synthetic 100 ms decision delay and 800 ms mock pulse |

![Right wrist during the plate approach](skills/ebim-task3-phase2/assets/approach.png)

The image is from after four arm translations, before the fifth. No post-fifth-step
image is archived. The final endpoint sample reports Idle with no current errors.
The final trajectory's stdout was not retained, so we do not infer its per-step
return or guard flag. Initial/final transforms, settled yaw increments and the
original benchmark statistics are retained in `evidence/results.json`.

An independent isolated skill test completed one specified 1 mm +X mock pulse
and exited cleanly. Separate image/tool calls initially exhausted the five-second
lease, and preliminary sessions reached the 30-second idle timeout. The successful
trial batched tool operations; it did not demonstrate visual decision-making
within the deadline. A misleading restart hint after worker exit was corrected
and covered by an additional test.

The timing result establishes local protocol overhead only. It does not measure
Astra reasoning, remote tool latency, SSH, a ROS camera stream or physical motion.
A roughly three-second decision budget is a design target. The five-second lease
rejects a late decision; it cannot force a hosted model to finish in time.

## 5. Reproduction and portability

Install the complete `ebim-task3-phase2` folder in the evaluator's supported local
Codex skills directory, select GPT-6 Astra, and invoke the skill. The ZIP provides
its instructions, source and fixture together. The README supplies installation
and offline build/run commands. Mock review needs Python 3.9+, CMake and C++17;
no external Python package or model/network call is used for that demo.

Physical review additionally needs the site's RT Linux/libfranka 0.20.4 environment,
ROS Jazzy camera host with NumPy/OpenCV, normally initialized FCI ownership,
persistent SSH, fresh calibration and an attended cell. A new agent must confirm
host-specific paths, addresses, clocks, executable architecture and coordinate
mapping. Credentials and proprietary guides remain operator-provided. The skill
records the verified interface facts and stopping boundaries rather than assuming
all future testbeds have the same configuration.

A clean relocated package build/demo and the offline guard suite validate the
software boundary. They do not establish complete robot-task performance. Earlier
full hardware records remain in commit `4256c63098453d76f177b2a6274a578752ca8048`;
the first persistent prototype is in `1d8f2c4f0b3421ed16e87efc4ab2a8d99b5f7981`.

## 6. Limitations and submission route

The packaged fast path provides RGB and finite translation of one selected arm.
Calibrated contact, gripper closing, wrist rotation, base navigation, simultaneous
bimanual control and the complete feeding sequence remain separate integration
work. Live execution of the persistent worker and an Astra-driven physical loop
have not been demonstrated. The skill makes that knowledge reusable; it does not
turn partial movement into a grasp or a completed stage.

We choose the **Technical Report** route advertised in the EBiM submissions
README. It states that a report requires no Dockerfile and is assessed on technical
maturity/readiness with 0.65 weighting. The report and installable skill are the
submission materials; no runnable-policy build/launch declaration is asserted.
At preparation, the issue chooser exposed only a policy template, so the filing
mechanism must be reconciled with the advertised report option. The organizer
retains acceptance and scoring authority.

## References

1. [EBiM submissions and technical-report option](https://github.com/EBiM-Benchmark/submissions)
2. [Current Phase II policy form](https://github.com/EBiM-Benchmark/submissions/blob/main/.github/ISSUE_TEMPLATE/phase2-submission.yml)
3. [Codex skill format and local installation](https://learn.chatgpt.com/docs/build-skills)
4. [GPT-6 Astra model reference](https://developers.openai.com/api/docs/models/gpt-6-astra)
5. [Franka Control Interface and libfranka](https://frankaemika.github.io/docs/libfranka.html)
