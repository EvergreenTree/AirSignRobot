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

The submission-ready final record is:

```text
AIRSIGN_SUBMISSION_CHECK {"checks_passed": 10, "failures": [], "passed": true}
```

Until a fresh schema-2 GPU diagnostic has been captured and reviewed, the
validator intentionally exits nonzero with
`No passing schema-2 Stage 1 cup-preflight binds all executed sources and the
immutable container image`. Do not suppress that failure or relabel an older,
failed, or less-complete diagnostic.

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
git status --short  # keep the context clean if the label must be exact
docker build --pull \
  --build-arg AIRSIGN_REVISION="$(git rev-parse HEAD)" \
  --tag airsignrobot:task3 \
  .
```

The caller-supplied source-revision argument binds the OCI image revision and
its in-container provenance stamp. A generic Docker build does not itself
enforce a clean context; only `scripts/capture_stage1.sh` makes that exact-commit
claim and refuses local changes. The first build downloads the Isaac Sim base
image, pinned benchmark and Task 3 room, and pinned Robotiq asset. It is
intentionally large. No cloud credentials, datasets, or model weights are
required.

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

### Retained physical Stage 1 diagnostic capture

Use the repository capture wrapper for any Stage 1 run that will be retained as
judge-facing evidence. It refuses a dirty worktree and an existing output path,
builds with the full source commit, records the Docker image inspection, runs
smoke and the GPU workload by immutable image ID, and preserves complete logs:

```bash
git status --short  # must print nothing
AIRSIGN_STAGE1_GATE=cup-preflight ./scripts/capture_stage1.sh
```

The default output is
`evidence/stage1-physical-development/<gate>-<revision>/`. Each new bundle
contains the controller's `manifest.json`, a host-side `capture.json` hashing
all retained run artifacts, the Docker `image-inspect.json`, smoke and
controller logs, `container.exit`, a bundle `README.md`, and an
`INDEX_ENTRY.md` bullet. Review that bullet and add it to
[`evidence/stage1-physical-development/README.md`](evidence/stage1-physical-development/README.md),
then run `python3 scripts/validate_submission.py`.

Commit a retained bundle in a separate descendant commit. Do not amend,
rebase, squash, or cherry-pick away the captured source commit: validation
requires that exact image/source revision to remain an ancestor of the
submitted revision.

The wrapper returns the controller's status after packaging. A failed
fail-closed diagnostic therefore returns nonzero while still preserving a
complete, explicitly non-official bundle. Missing controller artifacts,
inconsistent provenance, malformed or non-finite JSON, image/revision mismatch,
or failed packaging prevents publication of the staging directory.

### Development-only Stage 1 run

For local iteration where no evidence bundle will be retained, run the
fail-closed, articulation-only Table Setup controller manually:

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

This manual command does **not** retain the image inspection and smoke binding
required for a schema-2 judge-evidence bundle. Do not copy its output into
`evidence/`; rerun the exact committed revision through
`scripts/capture_stage1.sh` when the result is ready to retain.

Available gates are `inspect`, `cup-preflight`, `cup`, `tray-lift`,
`tray-transport`, and `all`. The `cup-preflight` gate is read-only after normal
Isaac asset initialization: it captures the robot's current full collision
geometry and validates that footprint over the planned route, without issuing a
base, arm, gripper, or task-object command. The controller avoids the official
ready-pose helper because that helper initializes joints with
`set_joint_positions`; motion gates instead use guarded articulation targets
after evidence recording begins. The controller constructs the pinned official
scene, records task-object poses read-only, and exits nonzero when a posture,
route, navigation, IK, joint-effort, or physical-outcome gate fails. It writes
`trajectory.json`, `metrics.json`, and `manifest.json` with controller, scene,
and image provenance.

Nonzero base motion is currently disabled fail-closed. The controller does not
yet have an Isaac-measured, conservative swept stopping bound that accounts for
robot, payload, and environment motion and is cryptographically bound to each
route certificate. Fixed gaps between operational and certificate envelopes are
diagnostic containment margins, not a braking proof. Consequently,
`cup-preflight` is the only current evidence-capture gate intended to run to
completion; motion gates must report
`base_motion_stopping_envelope_uncertified` before issuing a nonzero wheel
command. This limitation must be closed and GPU-validated before claiming a
full Stage 1 attempt.

The tray gates are payload-stability experiments only. The tray is not one of
the four scored Stage 1 objects and a tray-gate pass is not a rulebook
placement.

This is explicitly a development workload. The public benchmark snapshot does
not expose the organizer's randomized Stage 1 target-provider or live scorer
contract, so even a passing grasp, lift, release, or transport gate keeps
`official_stage_complete: false` and `official_stage_score: null`.
The retained exact-entrypoint diagnostics are indexed under
[`evidence/stage1-physical-development/`](evidence/stage1-physical-development/);
all currently retained runs predate the schema-2 controller and are labeled
non-canonical. The current controller resolves the unique enabled dynamic body
below each task-object asset, replay-verifies route and proxy inputs, and binds
every enabled collider to exact local-geometry witnesses. Before, after, and
during zero-command settling for every base physics step, it recaptures full
robot/payload transforms and fails closed outside smaller operational
SE(2), non-planar-base, or collider envelopes. A wider certificate allowance
provides a diagnostic containment gap only; it is not presented as a reaction
or stopping guarantee. None of those safeguards is presented as empirically
validated until a fresh clean-revision schema-2 GPU bundle passes the repository
validator.

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
