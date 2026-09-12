# Fast supervised manipulation loop (experimental)

This prototype turns each visual decision into one small gamepad-style pulse.
It removes per-action native-process startup, separate state-probe subprocesses
and finite camera-probe sessions. C++ remains the local servo implementation.

```text
latest wrist frame + measured pose + frame ID
                  |
           f42 R:Z-/fine
                  |
      one native 1 mm / 0.8 s pulse
                  |
       new frame + pose + latency
```

The existing submission candidate `4256c63098453d76f177b2a6274a578752ca8048`
remains the record of on-site work. This later prototype is **offline tested,
not physically tested**, and is not evidence of a successful plate grasp.

## What changed in the loop

| Earlier session | This prototype |
| --- | --- |
| Start native translation process for each motion | One native process and one robot connection per session |
| Start a separate state probe | `READ` on the existing native process |
| Launch a finite ROS camera capture | Continuous subscription, depth-one latest-frame cache |
| Long command vectors and manual file handling | `<frame> R:X+` or `R:Z-/fine`; action returns the next image |
| Multi-second, up to 50 mm steps | Default 2 mm / 0.8 s; fine 1 mm / 0.8 s |
| Review age up to 90 s | Frame-bound dispatch deadline of 5 s, checked again by native worker |

This is a persistent **process and connection**, with a finite `robot.control`
call per pulse. Between pulses it is idle; it does not continuously stream an
unbounded velocity command while a model thinks. The native callback handles
the real-time trajectory, following the [libfranka callback model](https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Robot.html).
The model chooses direction and pulse size, not individual servo samples.

## Timing target

Budget approximately 0.2 s for a fresh image/state, 3 s for the decision, 0.8 s
for the pulse and 0.2 s for the next frame, leaving margin under 5 s. These are
targets, not measured hardware performance. The five-second bound enforced in
code is **capture-to-dispatch freshness**; it cannot make model reasoning finish
on time. A late decision is discarded and requires a new view.

`latency.jsonl` separates decision time, frame-to-dispatch age, command duration,
new-frame time and complete cycle time. The benchmark injects a fixed synthetic
decision delay. It measures neither actual model reasoning nor SSH/ROS/robot latency.

## Build and test without hardware

From the repository root, with CMake, a C++17 compiler, Python 3.9+ and pytest:

```bash
cmake -S phase2/interactive -B /tmp/airsign-gamepad-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/airsign-gamepad-build -j2
AIRSIGN_GAMEPAD_MOCK=/tmp/airsign-gamepad-build/gamepad_mock \
  python3 -m pytest -q phase2/interactive/test_gamepad.py
python3 phase2/interactive/benchmark.py \
  --worker /tmp/airsign-gamepad-build/gamepad_mock \
  --output /tmp/airsign-gamepad-benchmark --steps 10
```

Only the mock binary is built by default. It has no libfranka dependency or robot
connection. The tests exercise expired/reused views, native bounds, STOP, input
EOF, rejection while busy, image loss and refusing fixture images for live control.
Loopback HTTP is required for the tests. See `validation/` for measured mock results.

To interact with the mock, start a fixture camera server in one terminal:

```bash
python3 phase2/interactive/camera_server.py \
  --fixture phase2/task3_submission/evidence/images/before-final-step.png
```

Then keep this bridge running in another terminal:

```bash
python3 phase2/interactive/gamepad.py \
  --worker '["/tmp/airsign-gamepad-build/gamepad_mock","--mock"]' \
  --output /tmp/airsign-gamepad-frames
```

It prints a frame ID and image path. Send `FRAME_ID R:X+`, using that actual ID.
The result includes `next.frame` and `next.image`. `LOOK` refreshes a view; `STOP`
requires no frame; `QUIT` closes the worker. Normal motion is 2 mm and `/fine`
is 1 mm. This is a six-direction arm-base control, not camera-relative arrows.
The fixture is explicitly labelled simulated and is refused in a live session.

## Site integration for the next attended session

No on-site setup or motion was attempted while developing this prototype.
First use the preserved [hardware skill](../../skills/ebim-task3-phase2/SKILL.md)
and [on-site notes](../hardware/task3_onsite/README.md). Normal control ownership,
brake state and FCI activation are prerequisites; this worker does not obtain
ownership, unlock brakes, recover faults or change collision settings.

1. On the sensor host, source its ROS environment and run the read-only camera
   service with `--topic /wrist_camera_right/color/image_raw`. It requires ROS
   `rclpy`, `sensor_msgs`, NumPy and OpenCV. The topic must use source timestamps
   in that host's wall-clock domain. Future/old stamps and duplicate frames are
   rejected. A request waits for a newly captured frame, rather than serving an
   old image indefinitely. This fast path supplies RGB, not calibrated depth;
   contact planning still needs the existing RGB-D/calibration pipeline.
2. Tunnel its loopback port to the control workstation using one persistent SSH
   tunnel. The camera endpoint is bound to 127.0.0.1 and has no authentication of
   its own; do not expose it directly on the LAN. Requests expire after 1.2 s.
3. On the real-time host, opt in to building the live worker with
   `-DAIRSIGN_BUILD_LIVE_GAMEPAD=ON`, using the site's libfranka 0.20.4 CMake
   prefix. The executable requires `--execute --ip ARM_IP`. It enforces RT
   scheduling; the non-RT sensor host is not an alternative motion executor.
   Start it only under a valid, normally acquired FCI control session. Use a
   single persistent SSH process for stdin/stdout, with server-alive checking.
4. Point the bridge's `--worker` JSON argv at that SSH command and add `--live`.
   `--arm R` or `--arm L` names the configured arm. Confirm the IP/physical side
   and axis mapping before any pulse; the letter is not automatic identification.
   Test only an individually reviewed free-space pulse, then inspect real timing
   and tracking. The live build and ROS service have not been verified on the robot.

The bridge estimates a conservative offset between its clock and the worker's
monotonic clock using a fresh round trip for every view. The native deadline
therefore includes network transit; no synchronized workstation/RT-host clocks
are assumed. A 20 ms margin and a 250 ms round-trip gate are applied. This is an
engineering bound, not a guarantee against arbitrary clock faults or suspend.

## Failure behavior and remaining limits

There is one outstanding action and no motion queue. A new view invalidates the
previous one; invalid actions also consume their view. Native STOP invalidates
the previous pose anchor, interrupts an active pulse and latches the worker closed
after an interrupted motion. Signals and stdin EOF request stopping. Every pulse
is finite even if a network cable is pulled before SSH notices. The process exits
after 30 s without a queued READ/MOVE. Native failures terminate the session;
there is no automatic retry or recovery.

STOP acknowledgement means the request was received, not that physical stopping
has been verified. The operator's emergency stop remains necessary. Abrupt guard
termination can trigger the robot's own stop/reflex handling. Tool-force limits,
scene clearance and contact geometry still require site validation. Releasing
FCI ownership and locking brakes remain the responsibility of the session holder
when the worker exits; a process exit alone does not certify either state.

The prototype handles one selected arm at a time. Gripper closing, wrist rotation,
base movement, simultaneous two-arm control and automatic contact detection are
not included. The compact [agent prompt](AGENT_PROMPT.md) constrains the fast loop
to an already planned approach; it does not replace scene understanding.
