# Draft only - do not submit unchanged

Suggested issue title: `[Phase II] AirSign - Task 3 - technical progress`

The available issue form is for runnable policies. This package records partial
hardware progress, and its Docker build and full testbed launch are not verified.
Confirm the technical-report route before using the form. Required declarations
below intentionally remain unchecked where evidence is incomplete.

## Team name

AirSign

## Point of Contact email

NEEDS REGISTERED MEMBER 1 EMAIL

## Task

Task 3 - Assisted Living & Feeding

## Assigned Phase II pathway

NEEDS ASSIGNED SITE FROM ADVANCEMENT EMAIL

## Team-specific deadline

NEEDS DATE FROM ADVANCEMENT EMAIL

## Public repository and pinned commit

https://github.com/EvergreenTree/AirSignRobot

PIN EXACT 40-CHARACTER SHA AFTER FINAL PACKAGING; verify it is public and reachable.
The preparation handoff outside this tracked file records the candidate SHA,
avoiding a self-referential commit identifier.

## Build and run commands

See `phase2/task3_submission/README.md`. Proposed review commands:

```bash
docker build -f phase2/Dockerfile.task3 -t airsign-task3:phase2-review .
docker run --rm --network none --read-only --tmpfs /tmp airsign-task3:phase2-review review
docker run --rm --network none --read-only --tmpfs /tmp airsign-task3:phase2-review self-test
docker run --rm --network none --read-only --tmpfs /tmp airsign-task3:phase2-review describe
```

These launch evidence review and the feedback-interface description, not an
autonomous robot trial. Docker build/run remains unverified. The README also
documents the JSONL `policy` entrypoint and the supervised native primitives;
the live perception/transport bridge is incomplete.

## Environment and hardware assumptions

CPU review: Python 3.12 Debian Bookworm slim, NumPy 2.0.2, pytest 8.4.2; no CUDA,
GPU, ROS, credentials or runtime Internet. Build downloads the image and packages.
On site: Mobile FR3 Duo/TMR; libfranka 0.20.4 on the existing real-time host;
separate ROS Jazzy sensor host; two 640x480 D405 wrist RGB-D streams around 30 Hz;
20 Hz LiDAR relay; native arm loop on the RT host. Current measured pose is
required before every finite translation. Scene/grasp calibration and live
predicates are external prerequisites. No autonomous resetting is provided.

## Object-pose declaration

Partly - see Notes.

The supervised physical run used agent-selected image regions, measured depth,
nominal mounting geometry and reviewed waypoints. The feedback program consumes
externally supplied calibrated poses and outcome predicates. It does not provide
end-to-end autonomous object perception.

## Trajectory-data declaration (Task 3 only)

No - we did not use trajectory data for Task 3 training. No Task 3 weights.

## Changes since Phase I

Added measured-state real-hardware integration, wrist mapping/registration
analysis, bounded native arm/base primitives, a reusable operational skill,
retained telemetry and a technical progress report. The Task 3 feedback state
machine has software validation; the on-site trial demonstrated partial approach.

## Results and assistance

Measured 17.64 cm net right-arm displacement; estimated 57.96 degrees cumulative
right turn from independent scan registration. Plate contact/grasp/lift and full
Task 3 stages were not verified. An operator moved the desk into place and handled
hardware initialization and emergency-stop supervision. A remote agent reviewed
each finite step. No benchmark score is claimed.

## Requirement audit

- [ ] Candidate commit is public and reachable (publication pending).
- [ ] Docker image builds and clean-checkout commands have been run successfully.
- [ ] Full autonomous policy can launch on the real testbed from these instructions.
- [x] Source, Dockerfile, README, report and evidence are included in the preparation package.
- [ ] Team has reviewed and accepted the official form's acknowledgements.

## Organizer question prepared for approval

We are AirSign, competing in Task 3 Phase II. We have a technical progress report,
supervised real-hardware motion evidence and the working low-level sources, but
no verified autonomous plate grasp or complete deployable policy. Your README
advertises a technical-report route, while the issue chooser currently lists only
the Phase II policy form. May we submit this package as a Phase II technical
report, and through which form? Our registered contact and assigned site will be
provided with the entry.

This question has not been sent. No checkbox should be changed merely to bypass
the policy form's required build/run declarations.
