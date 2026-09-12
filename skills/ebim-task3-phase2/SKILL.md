---
name: ebim-task3-phase2
description: Adapt Codex to AirSign's supervised Mobile FR3 Duo visual action loop for EBiM Task 3. Inspect fresh wrist images and measured poses, resolve arm-base directions, and choose short frame-bound motion pulses. Includes an offline demo and two-host startup knowledge; not a completed autonomous feeding policy.
---

# EBiM Task 3: observe, decide, pulse

Use this installed folder as `SKILL_DIR`; every packaged path is relative to it.
Target agent: **Codex with GPT-6 Astra**, selected in the host application. The
skill supplies instructions and tools, not the model or an API credential.
The host needs local shell execution, a persistent process session and a local
image-viewing tool. Its account must provide the selected model and network
access. Record actual model/settings in new experiments; Astra's hardware-loop
latency has not been measured by this package.

## Select the mode

For offline review, follow [action-loop.md](references/action-loop.md).
`simulated: true` is a static fixture, not live perception. For a new attended
robot session, read [hardware.md](references/hardware.md) and
[vision.md](references/vision.md), then the live setup in action-loop.md.

The previous on-site session ended on September 12, 2026. Installation and old
prompts do not authorize new motion. Re-establish current operator presence,
authorized work, initialization and readiness. SSH access alone is not actuator
readiness. Credentials and proprietary manuals come from the site operator.

## Adapt before the fast loop

Identify the sensor host, RT host, target arm IP and camera side from live data.
Check measured pose, FCI/ownership/error state and the scene view. Commands use
**arm-base XYZ axes**, not image up/down, robot-body directions or world axes.
Verify their relation to the image using current pose and calibration; do not
guess from the R/L letter. Plan a free-space approach and establish clearance.
The RGB-only fast path cannot certify metric depth or finger-contact geometry.
Use STOP when a required transform, view or clearance cannot be established.

Keep the camera subscriber, SSH transport and native worker resident. Build C++
once per source change, not per action. Avoid rereading all references, launching
new probes or rewriting movement code between ordinary pulses.

## Fast decision contract

View the returned image using the host's image tool. The initial JSON response
contains `frame`, `image`, `pose`, `expires_in_ms` and the allowed actions. A
successful pulse returns the next observation under `next`.

```text
Input: fresh image, frame ID, measured pose and reviewed approach goal.
Output: <frame-id> R:X+ (or X-, Y+, Y-, Z+, Z- on the configured arm)
Default pulse: 2 mm / 0.8 s. Append /fine for 1 mm.
Use STOP for uncertainty, obstruction, contact or lost view.
Target decision time: about 3 s. Frame-to-dispatch deadline: 5 s.
One frame permits one pulse. Inspect next.image before another action.
```

LOOK refreshes expired/consumed views; STOP needs no frame; QUIT ends the session.
Never queue actions, retry an old frame ID or fabricate timestamps. Prefer one
host-tool invocation that sends a command, receives its result and displays
`next.image`. Keep planning/calibration outside the fast loop.

The native worker enforces RT scheduling, pose agreement and finite-motion
guards. A stopped/faulted pulse is not a completed movement: inspect state and
resolve the cause before restarting. Never weaken limits, substitute `kIgnore`
for motion, recover automatically or extend expiry to hide slow reasoning.
The operator's physical emergency stop remains the immediate stop; a software
STOP acknowledgement does not establish physical stopping.

## End and report

End the worker, release normally held control and confirm brake/FCI state through
the site's procedure. Do not claim cleanup succeeded after a lost connection.
After operator departure, continue offline work only.

Record image/frame IDs, actions, measured changes, latency, faults and assistance.
The on-site record showed 17.64 cm net right-arm approach; no plate contact,
grasp, lift or Task 3 stage was verified. The later persistent worker is mock-
tested and physically unverified. Gripper closing, rotation, base motion and
coordinated bimanual control are separate development, not available actions.
