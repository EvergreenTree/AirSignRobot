# Preserved on-site hardware primitives

These are the final source files used during AirSign's September 11-12 session.
`provenance.json` hashes each byte-identical copy. They are individually supervised
diagnostics and motion primitives. They do not implement autonomous plate grasping
or connect the Task 3 JSONL policy to the robot.

## Components and observed execution

| Source | Role | Observed result |
| --- | --- | --- |
| `tested/native_arm_translation.cpp` | Current-pose-anchored Cartesian translation | Right arm moved on the real-time host; 50 mm maximum request, 3-8 s |
| `tested/right_arm_session.py` | Normal authenticated control, read/step/release | Used interactively with credentials entered at a prompt |
| `../../scripts/probe_franka_state.cpp` | FCI read-only measured state | Used before/after arm movement; no motion/configuration calls |
| `../../scripts/probe_hardware.py` | Finite ROS camera/topic snapshot | Wrist RGB-D and base sensor snapshots |
| `tested/live_lidar.py` | Subscription-only local scan relay | Front/rear scans relayed at 20 Hz |
| `tested/lidar_motion_watch.py` | Local scan registration | Independent finite-turn displacement feedback |
| `tested/native_scene_turn.cpp` | Bounded clockwise chassis turn | Repeated small turns, guarded by fresh scan registration |
| `tested/run_scene_turn.sh` | One supervised turn and evidence capture | Requires the live relay and source paths below |
| `tested/read_base_feedback.cpp` | Read-only mobile state | Requires the site's extended libfranka fields |

The source has deliberate site-specific addresses and `/tmp` paths. Do not run
the files as generic startup scripts. Preserve the site's libfranka installation;
the mobile code requires its TMR extensions and is not stock arm-only libfranka.

## Build on the existing RT host (no actuation)

The observed host had kernel `5.15.148-rt-tegra` and libfranka 0.20.4 at
`~/ros2_ws/install/libfranka`. These packaging commands have not been rerun on
the robot after departure; the source files themselves were compiled and used
during the session.

```bash
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
cmake -S phase2/hardware/task3_onsite -B /tmp/airsign-task3-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/ros2_ws/install/libfranka"
cmake --build /tmp/airsign-task3-build -j2
```

Only arm and read-only probe targets build by default. Opting into
`-DAIRSIGN_BUILD_TMR=ON` additionally requires the site's extended libfranka and
`nlohmann_json` CMake package. No source here changes collision parameters,
homes an arm, or performs automatic error recovery.

## Interactive arm session

First follow the repository skill's live readiness, frame and ownership checks.
The session requires an enabled robot-right arm at `172.16.16.11`; it does not
unlock brakes. Copy the compiled files to the exact tested names:

```bash
cp /tmp/airsign-task3-build/native_arm_translation /tmp/airsign-native-arm-translation
cp /tmp/airsign-task3-build/probe_franka_state /tmp/airsign-probe-franka-state-v2
# The following takes control and activates FCI. Run only in an attended session.
python3 phase2/hardware/task3_onsite/tested/right_arm_session.py
```

Enter the site Desk password at the terminal prompt. Credentials/control tokens
are held in memory. `READ` refreshes telemetry; `STEP dx dy dz seconds` executes
one reviewed translation in the right arm base frame; `RELEASE` ends control.
There is no canned motion example because a valid displacement depends on the
current scene. Idle timeout is 180 seconds; a nonzero step result prevents further
steps. The session releases its owned token in `finally`; verify status afterward
because a network failure can also prevent cleanup. This is not an emergency-stop
replacement and does not guarantee brake locking.

The native helper enforces real-time scheduling, measured-pose agreement,
initial idle state and joint-speed limits. During motion it checks force/torque
changes, collision flags, joint changes, orientation change, travel and callback
timing. Bounds are engineering guards tested for this session, not calibrated
contact forces or a certified collision-avoidance system.

## Base dependencies

The preserved base launcher expects `/tmp/airsign-native-scene-turn`,
`/tmp/airsign-lidar-motion-watch.py`, `/tmp/airsign-probe-hardware.py` and an
already running `live_lidar.py` relay writing `/tmp/airsign-live-lidar.json`.
The relay subscribes to `/lidar_front/scan` and `/lidar_rear/scan`; the session
used `ROS_DOMAIN_ID=31 ROS_LOCALHOST_ONLY=1` for that relay. Observer dependencies
are NumPy and SciPy. Do not infer immediate readiness from the presence of a
stale `/tmp` file. The launcher requires a newly established scene reference.

The base can reposition steering modules at control startup even with a zero
body-velocity request. Review fresh wide views and all near returns before
control entry. The scan matcher excludes near self-returns and is not a safety
scanner. Historical turn logs are evidence, never instructions to replay turns.

## Reproduction boundary

On-site paths, build prerequisites and supervision are explicit. The review
Docker image neither builds these vendor-dependent binaries nor actuates them.
Packaging/build scripts were added after the site session and need host-side
verification before a future run. No agent runner, model credentials, site
perception bridge or complete physical-task launcher is supplied in this package.
