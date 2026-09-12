# AirSign — EBiM Phase II

The current real-robot codebase for Tasks 1, 2 and 3 is in **[phase2/](phase2/)**.
Start with the **[on-site migration guide](phase2/MIGRATION.md)**.

**Task 3, September 11-12 hardware session:** [submission package and technical
report](phase2/task3_submission/README.md). It preserves supervised base motion
and a partial right-arm plate approach. Plate grasp/lift and full task completion
were not verified. The [hardware skill](skills/ebim-task3-phase2/SKILL.md) captures
the startup, frame and control knowledge for another attended session.

The later [gamepad-style manipulation prototype](phase2/interactive/README.md)
adds a persistent native worker, latest-frame camera service and short action
prompts. Its timing and command guards are tested offline; live execution remains
unverified.

```bash
git clone https://github.com/EvergreenTree/AirSignRobot.git
cd AirSignRobot/phase2
python3 scripts/fetch_policies.py --output policies
```

The last command downloads and verifies the trained Task 1/2 weights from the
[Phase II release](https://github.com/EvergreenTree/AirSignRobot/releases/tag/phase2-v0.2.0).
Continue with Python installation and service startup in the migration guide.
The ACT migration commands run from `phase2/`; its Dockerfile and runtime are
there. The Task 3 evidence-review package uses repository-root commands as
documented in its own README.

[Implementation and research](phase2/README.md) ·
[Offline results](phase2/docs/results/RESULTS.md) ·
[Robot interface contract](phase2/docs/INTERFACE.md)

The trained policies and Task 3 feedback program have offline software validation.
Actual robot transport, calibration and task perception must be bound to the site.
The service starts in shadow mode until calibrated; see the integration contract.

<details>
<summary>Historical Task 3 reference and previous submission</summary>

# AirSignRobot

**Team:** AirSign

**Competition track:** EBiM Task 3 — Assisted Living and Feeding

**Submission repository:** <https://github.com/EvergreenTree/AirSignRobot>

**Interactive judge demo:** [Mobile FR3 Duo four-stage viewer](https://airsign-ebim-track3.chattytransformer.chatgpt.site/)

This repository is the Docker submission contract for AirSign's actuator-driven
Task 3 work. The image pins the exact official benchmark snapshot and Robotiq
robot asset used during validation, then exposes a GPU-free image smoke test and
four headless Isaac Sim participant workloads.

> **Evidence boundary:** the included workloads validate dual-arm IK,
> articulation-driven Robotiq motion, closed-loop mobile-base control, and a
> four-stage actuator rehearsal. The rehearsal does not verify rulebook object
> or bean outcomes, so it is not an official four-stage completion and this
> repository makes no benchmark-score claim. The official rulebook score is 16
> points; the repository's 18-point development grader is non-authoritative.

## Judge quickstart

The demo separates recorded, measured actuator clips and logs from a synthetic
interactive 3D design visualization. Neither is rulebook task-outcome evidence.
The canonical machine-readable actuator record is
[`evidence/four-stage-rehearsal/`](evidence/four-stage-rehearsal/).

| Review question | Current, verifiable answer |
|---|---|
| Robot and environment | Mobile FR3 Duo with Robotiq 2F-85 in the pinned official Task 3 room |
| Canonical measured result | Navigation, IK, Robotiq, and hold gates passed across four rehearsal intents |
| Rulebook object outcomes | Not established; maximum measured task-object displacement was `0.0 m` |
| Official stage completion claimed | No |
| Official benchmark score claimed | No — `benchmark_score` is `null` |
| Official maximum | 16 points |

Run the complete CPU-only repository check—no Docker, NGC login, GPU, or cloud
credential is needed:

```bash
python3 scripts/validate_submission.py
```

Expected final record:

```text
AIRSIGN_SUBMISSION_CHECK {"checks_passed": 9, "failures": [], "passed": true}
```

This verifies controller syntax, the canonical replay schema and hashes,
evidence claim boundaries, Stage 1 diagnostic manifests and detached exits,
controller provenance, Dockerfile packaging guards, the read-only CI contract,
and recognizable credential patterns. For a container-level review, continue
with the build and smoke test below.

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

The upstream
[`STATUS.md`](https://github.com/EBiM-Benchmark/benchmark/blob/cb5184574f33611f943ff42aae461678ccb538e9/STATUS.md)
at this exact revision also describes Task 3 as a runnable preview and leaves
force-limited grasping and a full four-stage run unverified. Accordingly, this
submission treats the included grading helpers as development aids and does not
infer an official score from them.

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

### Physical Stage 1 development gate

Run the fail-closed, articulation-only Table Setup development controller:

```bash
mkdir -p stage1-output
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=8g \
  -v "$PWD/stage1-output:/stage1-output" \
  airsignrobot:task3 \
  stage1-table-setup \
  --gate cup \
  --output-dir /stage1-output \
  --head-placement A
```

Available gates are `inspect`, `cup`, `tray-lift`, `tray-transport`, and `all`.
The controller constructs the pinned official scene, commands only robot
articulation degrees of freedom, records task-object poses read-only, and exits
nonzero when a navigation, IK, joint-effort, or physical-outcome gate fails. It
writes `trajectory.json`, `metrics.json`, and `manifest.json` with controller,
scene, and image provenance.

The tray gates are payload-stability experiments only. The tray is not one of
the four scored Stage 1 objects and a tray-gate pass is not a rulebook
placement.

This is explicitly a development workload. The public benchmark snapshot does
not expose the organizer's randomized Stage 1 target-provider or live scorer
contract, so even a passing grasp, lift, release, or transport gate keeps
`official_stage_complete: false` and `official_stage_score: null`.
The retained exact-entrypoint diagnostics are indexed under
[`evidence/stage1-physical-development/`](evidence/stage1-physical-development/);
both fail before manipulation and are labeled non-canonical.

### Four-stage browser replay

Record a deterministic actuator rehearsal covering the intent and motion phases
for Table Setup, Feeding, Bean Recovery, and Cleanup:

```bash
mkdir -p evidence-output
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=8g \
  -v "$PWD/evidence-output:/evidence-output" \
  airsignrobot:task3 \
  four-stage-rehearsal \
  --output-dir /evidence-output \
  --head-placement A
```

The workload writes:

- `replay_trace.json`: 10 Hz measured base, joint, gripper, TCP, and task-object
  states suitable for deterministic WebGL playback;
- `metrics.json`: independent actuator gates for all four stages, with every
  official completion and score field explicitly false or null;
- `manifest.json`: SHA-256 and byte count for the replay and metrics files.

Validate the bundle without running physics again:

```bash
docker run --rm \
  -v "$PWD/evidence-output:/evidence:ro" \
  airsignrobot:task3 \
  validate-replay /evidence
```

The validator checks frame ordering, all four stage IDs, joint-vector shape,
finite numeric values, file hashes, and the no-score/no-completion boundary.
An actuator-gate pass means only that the commanded navigation, IK, gripper, and
hold waypoints were reached. It does not mean tableware or beans moved.

## Container command contract

| Command | GPU | Meaning |
|---|---:|---|
| `smoke` | No | Integrity and syntax checks; this is the default |
| `participant [options]` | Yes | Dual-arm IK plus closed-loop base workload |
| `gripper-gate [options]` | Yes | Robotiq articulation and read-only scene gate |
| `stage1-table-setup [options]` | Yes | Physical Stage 1 development gates; no official score claim |
| `four-stage-rehearsal [options]` | Yes | Four-stage actuator trace; no outcome claim |
| `validate-replay DIRECTORY` | No | Validate replay schema, hashes, and claim boundary |
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

Machine-readable records from the validated GCP run are indexed in
[`evidence/`](evidence/). Judges should start with the canonical
[`four-stage-rehearsal`](evidence/four-stage-rehearsal/) bundle; supporting
component checks and clearly separated failed
[`Stage 1 physical-development diagnostics`](evidence/stage1-physical-development/)
are retained for traceability. Every score-bearing record distinguishes
participant component validation from the official benchmark score.

Research is designed backward from the Stage 4 end state, then executed and
verified forward in the rulebook order:

1. establish physical grasp, transport, release, and official predicates for
   Stage 1;
2. validate loaded-spoon dwell and return for Stage 2;
3. validate particle recovery and mass measurement for Stage 3;
4. validate contact-driven sink placements for Stage 4;
5. optimize completion time only after reliable continuous Stage 1 → 4 runs.

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

</details>
