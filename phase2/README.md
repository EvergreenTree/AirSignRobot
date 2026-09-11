# AirSign — EBiM Phase II

**On site:** start with [MIGRATION.md](MIGRATION.md) to fetch the exported weights,
install Python dependencies and connect the policy service to the current robot.

Fresh implementation for the **real-robot Phase II** competition. Task 1 is the
priority; Tasks 2 and 3 have independent implementations. The earlier
[AirSignRobot repository](https://github.com/EvergreenTree/AirSignRobot) is a
historical reference, not the policy being continued here.

Read [the official-rule research](docs/OFFICIAL_RULES.md),
[method selection and primary papers](docs/RELATED_RESEARCH.md), and
[Task 2/3 controller details](docs/TASKS23.md). Phase I scene geometry and
development scorers are not authoritative Phase II problem definitions.

## Implemented paths

| Task | Implementation | Evidence needed for robot deployment |
|---|---|---|
| 1: cable routing | Three-camera ACT imitation learning from both real releases; native 62-state/23-action schema; optional assigned-route prefix/checkpoint supervisor | Calibrated robot interface, route-progress perception, current-route generalization and closed-loop routing trials |
| 2: thermal pad | Independent ACT model using native 42-state/17-action data; geometric feedback controller with optional peeling | Verified native spine conversion, pad face/grasp/placement observations and physical contact trials |
| 3: table service | New measured-feedback skill program: assigned seats, stabilized scoop, loaded-spoon hold/return, bean recovery, cleanup | Site perception, tool calibration, collision observations and physical trials; no demonstration checkpoint is claimed |

The implementations are runnable baselines. Software tests and offline action
prediction do not establish a competition success rate. The task programs
consume calibrated observations. An optional [Grounding DINO/SAM2 component](docs/VISION.md)
supplies RGB detections and masks, plus visible-surface geometry when registered
depth and camera calibration are available. It does not infer grasp success,
pad face, or loaded-spoon predicates. The released trajectories do not supply
a verified real robot transport adapter.

## Results and review bundle

[Offline results and learning curves](docs/results/RESULTS.md) report the
selected checkpoints and complete reserved-test evaluation. The accompanying
[software validation record](docs/results/software-validation.json) identifies
the exact exports checked through the policy and HTTP interfaces, with all
physical-validation fields left unset. The report includes the frozen dataset
manifests, training configurations and metrics.

The review archive contains `source/`, `wheel/`, and the two inference exports
at `policies/task1/policy.pt` and `policies/task2/policy.pt`. From its `source/`
directory, after environment installation, a shadow service runs with:

```bash
python -m ebim_phase2.runtime --checkpoint ../policies/task1/policy.pt --port 8080
```

Task 2 uses the corresponding Task 2 path. Task 3 uses the measured-feedback
program below and has no demonstration weights. `MANIFEST.json` records every
bundled file's hash; the archive also has a separate SHA256 file. Optional
Grounding DINO/SAM2 weights are retrieved with the pinned download script.

## Environment

Python 3.12 is used on this machine. Its GPU reports `L20D`, but CUDA exposes
compute capability **10.3** and roughly 274 GB usable memory. Use a PyTorch/CUDA
build that actually supports that device; do not select an Ada-only build from
the display name. The tested local environment inherits NVIDIA PyTorch
2.12.0a0 / CUDA 13.2. CUDA BF16 forward/backward and inference were exercised.

```bash
python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[sim,test]'
python -m pytest -q
```

The prepared workspace uses `/mnt/nas/evergreen/ebim-phase2/.venv`, with this
source directory beside it. MuJoCo 3.3.7 provides kinematics and actuator
diagnostics. Task 2's official PhysX deformable model is an Isaac Sim 5.1
environment; a rigid MuJoCo object is not a validated replacement.

For optional perception, install `.[vision]` and fetch the exact model snapshots
with `python scripts/fetch_vision_models.py --output /weights/vision`. The loader
checks the download manifests and local hashes before loading safetensors.
RGB-only demonstration videos yield masks with unavailable 3D geometry; they
must not be assigned synthetic depth for robot control.

## Download and audit real demonstrations

```bash
python scripts/seed_data_locks.py --output /data/raw
python -m ebim_phase2.download --output /data/raw --workers 4
python -m ebim_phase2.data \
  --raw /data/raw/task1_part1 /data/raw/task1_part2 \
  --output /data/task1 --image-size 320 --stride 2 --workers 4
python -m ebim_phase2.data --raw /data/raw/task2_munich \
  --output /data/task2 --image-size 320 --stride 2 --workers 4 \
  --max-unaligned-fraction .20
```

The seed command selects the exact audited revisions listed below and refuses
to replace a different existing revision. Without seeded locks, the first
download resolves each repository's current revision into `source.json`. Reruns
preserve that revision, resume partial files, and validate sizes and available LFS SHA256
hashes. Finished manifests include a SHA256 for every file. No uploaded cache
directories or old Task 3 weights are downloaded. `--metadata-only` audits the
release metadata without fetching videos.

| Release | Episodes | Robot frames | Audited revision |
|---|---:|---:|---|
| Task 1 part 1 | 24 | 207,176 | `40cf7a2048a4c0e17cef0c8f5b0603c5181bd97b` |
| Task 1 part 2 | 26 | 145,479 | `c59cd308f92a2caa42dfeecaa9b086a416c8023d` |
| Task 2 Munich | 238 | 144,965 | `495ebb7b56fb9e2f3952398a63d86f08cacb9531` |

All robot data are 20 Hz. Video frames are matched using actual presentation
timestamps, not nominal video FPS or frame index. Images are letterboxed to
preserve geometry. Observations exceeding 75 ms camera skew are removed and
counted. The default permits at most 5% loss per episode. Three Task 2 clips
(66, 91, 157) have head-video rates near 5.8 Hz and lose 11–14% at that same
skew bound. The audited Task 2 command explicitly permits up to 20% discarded
observations; it does not widen the 75 ms synchronization bound. The audit
policy and discarded counts are retained with the prepared data.
Actions retain 20 Hz resolution even when observations are sampled every other
frame. The Task 2 episode index has holes; it is never reconstructed using
`range(238)`. Its `modality.json` describes a stale 56-value state, whereas the
actual parquet and `info.json` agree on 42 state values and 17 action values.

Splits use complete episodes: 80% training, 10% validation, 10% test per task,
with seed 42. Task 1's original episode identifiers are checked for duplicates
across parts. Normalization uses training episodes only. These are within-
release splits; missing site identities prevent a cross-site claim.

`scripts/prepare_incoming.py` can prepare completed individual episodes while
downloads continue. It publishes the full training manifest only after all
release manifests exist and all audits pass. `scripts/snapshot_prepared.py`
can freeze an explicitly partial smoke dataset while preserving the same
full-release train/validation/test assignment.

## Train and evaluate

Train separate models; the state orders and action units are not interchangeable.

```bash
python -m ebim_phase2.train --task 1 --dataset /data/task1 \
  --output /runs/task1 --steps 10000 --batch-size 64 --workers 4
python -m ebim_phase2.train --task 2 --dataset /data/task2 \
  --output /runs/task2 --steps 10000 --batch-size 64 --workers 4
```

The default visual backbone is ImageNet ResNet18. For a verified local copy,
pass `--backbone-checkpoint /weights/resnet18.pth`; its hash is recorded. The
local copy came from the official timm `resnet18.tv_in1k` mirror at revision
`bbd144b3e5565108aad885f145491d11bc6ce807`, SHA256
`19b09237bf0e4694f5807b91b082bd026d9c932ae21174b39583b6aaf2c80901`.

The current training path predicts joint corrections around the measured
pose, transformed into the native normalized action coordinates. Future
action targets never enter this anchor. `--absolute-actions` reproduces the
earlier absolute-regression baseline. Initial smoke experiments used 160-pixel
images; full-release experiments use 320 pixels to retain cable/pad detail.

Checkpoints include native schema, training statistics, optimizer and random
states, dataset hash, source hashes, and measured validation error. Resume with
`--resume /runs/task1/last.pt`, keeping the original model, seed and batch-size
flags. Sampling is indexed by optimizer step, so worker prefetch does not
silently change the resumed training sequence. `best.pt` is chosen only on
validation error. Evaluation never receives target actions in the CVAE encoder.

After selecting a checkpoint, evaluate the reserved test episodes:

```bash
python -m ebim_phase2.train --task 1 --dataset /data/task1 \
  --output /runs/task1-test --resume /runs/task1/best.pt --evaluate test \
  --eval-samples 1000000
```

Reported metrics include per-dimension native action MAE, first-action MAE and
the training-mean and measured-joint persistence baselines. These metrics measure offline prediction, not
route completion, pad IoU or table-service points.

Export just the inference weights and metadata with
`python -m ebim_phase2.export --checkpoint /runs/task1/best.pt --output /exports/task1`.
The export omits optimizer/RNG state, retains native schemas and source hashes,
and records whether the dataset was complete or an explicitly partial smoke run.
Training resumes require the original training checkpoint.

## Run inference

```bash
./run.sh --checkpoint /runs/task1/best.pt < observations.jsonl > proposals.jsonl
./run.sh --checkpoint /runs/task1/best.pt --calibration /site/policy.json --port 8080
./run.sh --checkpoint /runs/task1/best.pt --calibration /site/policy.json \
  --route /site/current-route.json < observations-with-progress.jsonl
python -m ebim_phase2.tasks --help
```

`run.sh` uses `AIRSIGN_PYTHON` if set. The HTTP endpoint binds to loopback and
accepts the same JSON observation as JSONL. See [the input/output contract](docs/INTERFACE.md).
Without site calibration, inference emits shadow output. Task 2 retains native
spine units unless a calibrated scale and offset are provided. A caller must
never forward shadow output directly to a robot.

The calibrated path validates state age, camera skew, collision evidence,
wrench limits, gripper endpoints, joint/velocity bounds and spine limits.
Rejected observations emit no command. The actual robot transport must retain
its own watchdog and low-level hold/collision controller.

## Packaging and validation scope

```bash
docker build -t airsign-ebim-phase2 .
docker run --rm --gpus all -i -v /runs:/runs:ro airsign-ebim-phase2 \
  --checkpoint /runs/task1/best.pt
docker run --rm --entrypoint python airsign-ebim-phase2 \
  -m ebim_phase2.tasks --task task3 --describe
```

The Dockerfile describes a CUDA 13 runtime. The local Docker build attempt was
blocked while the daemon pulled the base image from Docker Hub; this is not a
tested container artifact. The Python environment is independently usable.

`python -m ebim_phase2.mujoco_backend --help` exposes a robot-component probe.
The supplied mapping names actuators in a historical official scene solely
for that diagnostic. The scene's default keyframe has two right-arm joints
outside their limits, explaining the first probe's large drift. The diagnostic
now rejects that initial state. With the explicit simulation-only initial
condition in `config/mujoco_probe_initial_joints.json`, the 5-second probe's
maximum final joint error was 0.00302 radians. Only actuator controls are changed
during rollout. This is a robot-component result, not a Phase II task score.

Raw demonstrations and large model artifacts stay outside this source tree.
Check the original dataset licenses before redistributing them. Submission
materials must identify the team as **AirSign** and describe actual validated
capabilities and unresolved site integration accurately.
