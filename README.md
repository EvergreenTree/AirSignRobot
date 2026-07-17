# AirSignRobot

**Team:** AirSign

**Competition track:** EBiM Task 3 — Assisted Living and Feeding

**Submission repository:** <https://github.com/EvergreenTree/AirSignRobot>

This repository is the Docker submission contract for AirSign's actuator-driven
Task 3 work. The image pins the exact official benchmark snapshot and Robotiq
robot asset used during validation, then exposes a GPU-free image smoke test and
two headless Isaac Sim participant workloads.

> **Evidence boundary:** the included workloads validate dual-arm IK,
> articulation-driven Robotiq motion, and closed-loop mobile-base control. They
> do not yet constitute a verified four-stage run, and this repository makes no
> official benchmark-score claim. The official rulebook score is 16 points; the
> repository's 18-point development grader is non-authoritative.

## Reproducibility lock

| Component | Pinned value |
|---|---|
| Simulator image | `nvcr.io/nvidia/isaac-sim:5.1.0@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54` |
| Official benchmark | `EBiM-Benchmark/benchmark@cb5184574f33611f943ff42aae461678ccb538e9` |
| Task 3 room asset SHA-256 | `696c71577f1874d815fe29c6a58c65f0f1a0a0fb15c0d8adbb5105209f5ff883` |
| Robot profile | Mobile FR3 Duo with Robotiq 2F-85 |
| Robotiq asset source | official commit `c2439d961b652b1eda6122bf530c58cb9559b219` |
| Robotiq asset SHA-256 | `aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c` |
| ROS middleware | ROS 2 Jazzy, Fast DDS over UDPv4 |

The Docker build pins the multi-platform Isaac Sim manifest, downloads the
official Task 3 room through Git LFS, and checks the benchmark revision, 56 MB
room asset, and 68 MB robot asset. A changed or unavailable pinned top-level
artifact makes the build fail instead of silently changing the simulation.

The pinned upstream `robot_room.usd` also contains dangling payload references
to `assets/ikea_knock_box.glb` and `assets/ikea_scale.glb`; neither file exists
in that official commit's Git tree. Isaac Sim reports them as non-fatal scene
warnings. The scored Task 3 objects and logical regions still load, and both
packaged component gates pass, but a full four-stage run has not been validated
against this incomplete upstream scene. This image does not substitute
unverified third-party geometry.

## Requirements

- Linux x86-64 host with a supported NVIDIA GPU
- NVIDIA driver compatible with Isaac Sim 5.1.0
- Docker Engine 24 or newer
- NVIDIA Container Toolkit configured for Docker
- Network access to GitHub and NVIDIA NGC while building
- Permission to use the NVIDIA Isaac Sim container and acceptance of its EULA

If NGC requests authentication, sign in interactively with `docker login
nvcr.io` before building. Do not put an NGC key in this repository or in a
Docker build argument.

## Build

From the repository root:

```bash
docker build --pull --tag airsignrobot:task3 .
```

The first build downloads the Isaac Sim base image, pinned benchmark and Task 3
room, and pinned Robotiq asset. It is intentionally large. No cloud
credentials, datasets, or model weights are required.

## Smoke test (no GPU required)

The default command validates the benchmark commit, robot-asset checksum,
controller syntax, and required image paths:

```bash
docker run --rm airsignrobot:task3
```

Expected final record:

```text
AIRSIGN_SMOKE_RESULT {"passed":true,...}
```

The process exits nonzero if any integrity check fails.

## Participant workload (GPU required)

Run the actuator-driven dual-arm and mobile-base workload headlessly:

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=8g \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  airsignrobot:task3 \
  participant \
  --head-placement A
```

This workload creates the official Task 3 room with dynamic beans and the
competition Robotiq robot. It lifts both TCPs 4 cm using Lula IK and articulation
position targets, then drives the mobile base 40 cm out and back using the
official wheel/steering target computation. It never teleports task objects or
robot links.

Success requires both a zero process status and a final line beginning with:

```text
AUTONOMOUS_PROBE_RESULT {"base":...,"passed": true,...}
```

The entrypoint checks that record itself, which avoids treating an Isaac
launcher status alone as proof of success. On a cold host, shader and extension
cache setup can make the first run substantially slower.

### Optional Robotiq and scene gate

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=8g \
  airsignrobot:task3 \
  gripper-gate \
  --head-placement A
```

This gate discovers the real left and right outer-knuckle driver joints,
commands an open → closed → open sequence only through articulation actions,
and reads the task-object and logical-region inventory without mutating it.
Success ends with `GRIPPER_SCENE_GATE2_RESULT` containing `"passed": true`.

## Container command contract

| Command | GPU | Meaning |
|---|---:|---|
| `smoke` | No | Integrity and syntax checks; this is the default |
| `participant [options]` | Yes | Dual-arm IK plus closed-loop base workload |
| `gripper-gate [options]` | Yes | Robotiq articulation and read-only scene gate |
| `shell [command ...]` | Depends | Debug shell or explicit command |
| `help` | No | Show the command summary |

Controller options are passed unchanged to the selected Python workload. For
example, a shorter deterministic controller run can be requested with:

```bash
docker run --rm --gpus all --shm-size=8g \
  airsignrobot:task3 participant \
  --head-placement A \
  --motion-steps 60 \
  --base-distance-metres 0.20
```

## Evidence and scoring strategy

Machine-readable records from the validated GCP run are in [`evidence/`](evidence/).
Each file explicitly distinguishes participant component validation from the
official benchmark score.

The implementation order follows the official lexicographic ranking:

1. maximize the highest completed stage;
2. maximize the official total out of 16;
3. reduce completion time only after reliability.

The official four-stage target is table setup, feeding with a three-second
hold, bean recovery, and cleanup. The supplied development integration check
is useful for geometry only: it directly moves scene prims, uses a different
18-point scale, and is not autonomous robot evidence.

## Security and privacy

The repository contains no Google Cloud credentials, NGC credentials, access
tokens, private URLs, or user-specific paths. `.dockerignore` is an allowlist,
so only the participant code, evidence, and entrypoint can enter the build
context.

## License

AirSign-authored files are licensed under Apache License 2.0. The Docker build
fetches the official EBiM benchmark and uses the NVIDIA Isaac Sim image; those
components retain their own licenses and terms. See [NOTICE](NOTICE).
