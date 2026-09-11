# AirSign: Task 3 Phase II submission package

**Status: technical progress package, not a verified autonomous task solution.**
During supervised hardware work on September 11-12, 2026, the right arm moved
17.64 cm net toward the plate. Earlier independent scan registration estimated
57.96 degrees of cumulative right turn. Plate contact, grasp and lift were not
verified; no Task 3 stage or benchmark score is claimed.

Start with [the technical report](REPORT.md), [retained evidence](evidence/summary.json),
and [the reusable hardware skill](../../skills/ebim-task3-phase2/SKILL.md).
The [on-site controller sources](../hardware/task3_onsite/README.md)
preserve working primitives and their dependencies.

## What runs

The entrypoint has four modes, all without direct robot transport:

| Mode | Behavior |
| --- | --- |
| `review` (default) | Verify hashes; recompute arm displacement and accumulated turn from evidence |
| `self-test` | Run 31 controller/scoring tests plus 3 synthetic Task 3 JSONL failure cases |
| `describe` | Emit required poses and signals for every Task 3 waypoint |
| `policy` | Consume calibrated observations on stdin and emit feedback decisions/actions on stdout |

The existing Task 3 state machine covers the service-cycle sequence in software.
It requires externally supplied live object/TCP poses, Jacobians, obstacle maps,
task predicates, calibration and assigned seats. This package does not provide a
complete real-testbed perception/transport bridge. The supervised native arm
session is separate from the JSONL state machine. No turnkey external-agent runner
or autonomous grasp detector is claimed.

## Local review (verified separately from Docker)

From the repository root, with Python 3.12:

```bash
python3 -m venv /tmp/airsign-task3-review
/tmp/airsign-task3-review/bin/python -m pip install -r phase2/task3_submission/requirements-review.txt
/tmp/airsign-task3-review/bin/python phase2/task3_submission/review.py review
/tmp/airsign-task3-review/bin/python phase2/task3_submission/review.py self-test
/tmp/airsign-task3-review/bin/python phase2/task3_submission/review.py describe
```

See [validation status](validation/STATUS.md) for the exact environment, scope and
remaining checks. A software fixture is not a replayed successful robot trial.

## Docker commands (build/run not yet verified)

Run from a clean checkout of the eventual pinned commit. Use this dedicated
Dockerfile, not the root historical Isaac Sim image or the Phase II ACT image.

```bash
docker build -f phase2/Dockerfile.task3 -t airsign-task3:phase2-review .
docker run --rm --network none --read-only --tmpfs /tmp airsign-task3:phase2-review review
docker run --rm --network none --read-only --tmpfs /tmp airsign-task3:phase2-review self-test
docker run --rm --network none --read-only --tmpfs /tmp airsign-task3:phase2-review describe
```

Base: `python:3.12-slim-bookworm`; NumPy 2.0.2 and pytest 8.4.2. The image runs
as UID/GID 65532. No GPU, CUDA, ROS, robot devices, host networking or model
weights are required for review. Allow roughly 1 GB RAM for this small review
workload; this is a planning allowance, not a measured peak. Build requires
Internet for the base image and packages; runtime review is offline. The base
tag and transitive dependencies are not digest locked, so byte-identical image
reproduction is not asserted.

To exercise the JSONL boundary with **real measured calibration and observations**:

```bash
docker run --rm -i --network none --read-only --tmpfs /tmp \
  -v /absolute/path/to/site-inputs:/inputs:ro \
  airsign-task3:phase2-review policy \
  --calibration /inputs/calibration.json --assignments /inputs/seats.json \
  < /absolute/path/to/observations.jsonl > decisions.jsonl
```

See [the observation contract](../docs/TASKS23.md). Default freshness checks use
the receiver's wall clock. `--replay-clock` is for offline recorded/synthetic
fixtures only. The files named `*.synthetic.*` are deliberately unsuitable for
real hardware. Commands on stdout are not proof of physical execution. A site
bridge would still need to enforce a deadman timeout and the site's command schema.

## On-site execution assumptions

The tested finite arm helper ran at the libfranka control-callback rate on the
existing real-time Linux host with a wired link to the robot. Wrist RGB-D was
approximately 30 Hz; base scan relay 20 Hz. SSH was supervisory, not the servo
loop. Current-pose anchors replaced any fixed starting joint pose. During the
approach, an operator had moved the desk in front of the robot. Standard indoor
lighting and the existing wrist camera mounts were used; no illumination study
or placement tolerance was measured.

Before another attended trial, the site must establish valid arm/gripper state,
fresh intrinsics and hand-eye/contact calibration, object locations, clear swept
space and explicit seat assignments. Reset poses are not automated. The native
session is limited to individual operator-reviewed translations; it does not
perform a grasp. Read the [hardware README](../hardware/task3_onsite/README.md)
and skill before use. Credentials come from the site, never this repository.

## Submission route and missing information

The [current Phase II form](https://github.com/EBiM-Benchmark/submissions/blob/main/.github/ISSUE_TEMPLATE/phase2-submission.yml)
asks for an exact public commit, working build/run commands and the deadline in
the team's advancement email. The repository README also advertises a
technical-report route, but the current issue-template listing exposes only the
policy form and config. Confirm with organizers whether this progress package
can use the report route; do not tick runnable-policy declarations prematurely.

The [issue draft](ISSUE_DRAFT.md) records the unresolved fields and a concise
organizer question. Registered contact email, assigned site, team-specific
deadline, route acceptance and Docker/real-testbed integration remain to be
confirmed. Publishing the repository commit and opening the issue are separate
steps. No organizer message is sent by any code here.
