# Run and adapt the packaged action loop

`SKILL_DIR` is the folder containing this skill's SKILL.md. All runtime sources,
build input and the demo image are inside that folder. No Docker is required.

## Offline review

Requires Python 3.9+, CMake and a C++17 compiler; Python uses only its standard
library in mock mode. Keep the bridge's process session open:

```bash
cmake -S "$SKILL_DIR" -B "$SKILL_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SKILL_DIR/build" -j2
python3 "$SKILL_DIR/scripts/gamepad.py" --demo --output /tmp/airsign-frames
```

View the image named in the first JSON response, then send its actual frame ID
and an action. Example shape: `<frame> R:X+/fine`. Read `result.status`,
`result.pose`, `latency` and `next`; view `next.image` before using `next.frame`.
The fixture changes a mock pose, not the photographed scene, and is labelled
simulated. QUIT closes the session; retain its exit status after testing.

| Input | Meaning |
| --- | --- |
| LOOK | New image and native pose anchor; invalidates the previous view |
| `<frame> R:X+` | 2 mm in selected right-arm base +X over 0.8 s |
| `<frame> R:Z-/fine` | 1 mm in selected right-arm base -Z over 0.8 s |
| STOP | Invalidate the old anchor and request stopping; no frame needed |
| QUIT | Close bridge and worker |

Other directions are X-, Y+, Y-, Z+ and Z-. A left-arm session uses L: with
`--arm L`. That flag changes vocabulary, not the worker IP. Only one selected
arm is controlled at a time. No grasp command is implemented.

## Two-host live setup (experimental)

Read hardware.md and vision.md first. Copy this skill to the site hosts through
the approved SSH connection. Establish normal Desk ownership, brakes and FCI
readiness using the site procedure. A session holder must retain the normal
control token and release it afterward. The worker does not initialize hardware,
acquire ownership or recover faults.

On the sensor host:

```bash
source "$HOME/tmr_env.sh"
python3 "$SKILL_DIR/scripts/camera_server.py" \
  --topic /wrist_camera_right/color/image_raw --port 8765
```

This subscription-only mode needs the site's ROS Jazzy environment, rclpy,
sensor_msgs, NumPy and OpenCV. Source stamps must use that host's wall clock.
Requests wait for new captures and reject stale/future stamps. The service binds
to loopback; keep a persistent tunnel from the agent workstation, for example:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=1 \
  -o ServerAliveCountMax=2 -L 8765:127.0.0.1:8765 aup@172.16.0.100
```

Verify these historical addresses. The HTTP service has no separate auth: do
not expose it to the LAN. If changing ports, update the tunnel and --camera too.

On the real-time host, build against its installed libfranka:

```bash
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
cmake -S "$SKILL_DIR" -B "$SKILL_DIR/build" -DCMAKE_BUILD_TYPE=Release \
  -DAIRSIGN_BUILD_LIVE_GAMEPAD=ON \
  -DCMAKE_PREFIX_PATH="$HOME/ros2_ws/install/libfranka"
cmake --build "$SKILL_DIR/build" -j2
```

From the agent workstation, start the bridge with one persistent worker. Replace
its remote absolute path with the actual skill installation on the RT host:

```bash
python3 "$SKILL_DIR/scripts/gamepad.py" --live --arm R \
  --camera http://127.0.0.1:8765/frame --output /tmp/airsign-frames \
  --worker '["ssh","-T","-o","ServerAliveInterval=1","-o","ServerAliveCountMax=2","tmr-user@172.16.0.50","/absolute/skill/build/gamepad_franka --execute --ip 172.16.16.11"]'
```

Establish SSH authentication beforehand with an approved key/agent or an
already-authenticated multiplexed connection. Worker stdin carries the command
protocol, not passwords. Never put credentials in the argv array. On a changed
host, check executable architecture, libraries, camera identity, clocks and
arm-base axes before considering a reviewed free-space pulse.

## Timing and failure semantics

C++ keeps its connection resident; each movement is a finite libfranka control
call. Parsing, JSON and camera work stay outside the RT callback. Codex itself
chooses actions through the persistent process; no nested model/API is needed.

A view has one five-second dispatch lease. A fresh worker round trip establishes
a conservative monotonic-clock offset, with a 20 ms margin; RTT above 250 ms is
rejected. This handles ordinary offsets, not arbitrary clock faults. Model
latency is not guaranteed. Refresh an expired frame rather than extending it.

The native parser bounds requests to 5 mm and 0.6-1.0 s; the agent interface
exposes only 1 or 2 mm at 0.8 s. The worker checks measured-pose agreement with
READ, idle/error state, force changes, collision flags, joint speed/change,
orientation, travel and timing. These are engineering guards, not certified
collision avoidance or calibrated contact control. RGB does not certify a full
swept volume.

Busy requests are rejected, not queued. Invalid actions consume their view.
STOP invalidates the native anchor; an interrupted pulse closes the worker.
Signals and stdin EOF request stopping. Each pulse remains finite if SSH has
not yet detected a dropped connection. The worker exits after 30 seconds without
READ/MOVE. Camera loss after motion leaves no usable frame for a blind next
command. If the worker exits, the bridge ends with a restart instruction and a
nonzero status; LOOK cannot revive a terminated process. Inspect the cause and
re-establish the session. There is no automatic restart or recovery.

Keep latency.jsonl from the output folder. It separates decision delay, capture-
to-dispatch age, command-to-result duration, native pulse and next-frame time.
The prior 924 ms median used a static fixture and synthetic 100 ms decision delay;
it is not measured Astra/SSH/ROS/robot performance.
