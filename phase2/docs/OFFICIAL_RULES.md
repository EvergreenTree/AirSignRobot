# Official Phase II research and interface record

Research checked 10 September 2026 for **AirSign**, already participating in Phase II. This implementation targets real-robot policies. Earlier simulation scenes and demonstration plans are development tools, not a specification of the current physical evaluation arrangement.

## Authority and evidence

The [official competition page](https://ebim-benchmark.github.io/competition.html) defines Phase II as real-world validation on Mobile FR3 Duo at Hamburg, Munich and Shanghai. Organizer-run testing is 4–12 September; the submission deadline is **12 September 2026 AoE**, equivalent to 13 September 11:59:59 UTC. Tasks rank independently. Official scoring is run by organizers; the public simulator evaluation scripts are explicitly described as development aids.

Read all five pages of the **Autonomous Robot Benchmark Rulebook 1.0**, including its scoring tables. The [immutable official PDF](https://github.com/EBiM-Benchmark/ebim-benchmark.github.io/blob/f92510f87c71ed4c975e9a571bcbbd288913cf5d/src/docs/Autonomous_Robot_Benchmark_Rulebook_1.0.pdf) is archived in [sources/Autonomous_Robot_Benchmark_Rulebook_1.0.pdf](sources/Autonomous_Robot_Benchmark_Rulebook_1.0.pdf). Its Git blob is `f847af65c5b38849e8378cb4c77ea6ccc66efeec`. The PDF does **not** contain an explicit Phase-II-only version label, physical fixture coordinates, or a robot API. It therefore supports task objectives and ranking, not an inference that old simulation coordinates match the current real scene.

Evidence snapshots:

| Source | Revision |
| --- | --- |
| Official website repository | `f92510f87c71ed4c975e9a571bcbbd288913cf5d` |
| Official benchmark repository | `161db49e9eafd34be31e5bdada72d8e2f796c16a` |
| Task 1 real data, part 1 | `40cf7a2048a4c0e17cef0c8f5b0603c5181bd97b` |
| Task 1 real data, part 2 | `c59cd308f92a2caa42dfeecaa9b086a416c8023d` |
| Task 2 real data | `495ebb7b56fb9e2f3952398a63d86f08cacb9531` |

Resolve conflicts by source and scope: organizer instructions for the actual Phase II run; current published rulebook/site; the relevant real dataset's `info.json` plus actual Parquet/video records; then simulator implementation details. A dataset feature name documents a recorded field, not necessarily its transport topic, message type, unit or calibration.

## Task definitions and ranking

The following scoring details come from the archived rulebook; page numbers refer to that PDF.

| Task | Ranked outcome | Tie breaking |
| --- | --- | --- |
| 1: Cable routing | Percentage of fixtures in the continuous correct prefix of the assigned route | Completion time |
| 2: Thermal pad | Pick success × orientation success × placement IoU | Completion time; raw IoU is also reported |
| 3: Service cycle | Highest completed stage, then total score out of 16 | Completion time |

**Task 1 (page 1).** Follow the supplied routing configuration, fixture sequence and routing direction autonomously. The time limit is 30 minutes. Completion is `100 × correct_prefix_fixture_count / total_fixture_count`. A later correct routing cannot repair an earlier incorrect fixture for prefix scoring. The public [task description](https://ebim-benchmark.github.io/competition.html#tasks) additionally describes O-props, C-props and Y-checkpoints that validate the preceding route. The concrete sequence and positions must come from the current evaluation scene; they are not prescribed by the PDF.

**Task 2 (pages 1–2).** Pick the thermal pad and place it at a randomly assigned target with the correct orientation. Both pick and orientation success are binary. Wrong orientation makes the score zero regardless of overlap. IoU lies in `[0,1]`. The PDF does not specify an angular tolerance, numerical tear threshold, a task timeout, or a separate score for liner removal. The site's broader task description mentions liner peeling; the rulebook calls this track pick-and-place. The runtime must use the actual Phase II starting condition and assigned target rather than assume a peeling stage or an old simulator target.

**Task 3 (pages 2–5).** Four stages are worth at most four points each:

1. Table setup: transport a plate, cup, bowl containing coffee beans, and spoon from the kitchen to the assigned dining locations. Three locations are assigned: plate, cup, and bowl-with-spoon next to the head. All four objects must reach their assigned locations for full stage completion. The PDF gives the stage maximum but does not explicitly tabulate one point per object here; any such partial-credit implementation is a local interpretation.
2. Feeding: scoop beans using the spoon, present it in front of the head for at least three seconds with beans present, then return the beans to the bowl. An empty spoon held at the head does not satisfy the feeding condition.
3. Bean recovery: transfer beans into the designated recycling container with its scale. Score is proportional to recovered bean mass divided by original bean mass: `4 × recovered_mass / original_mass`, capped at the stage maximum. This is a continuous ratio, not a coarse threshold schedule. Counting simulated beans is only a development proxy for measured mass.
4. Cleanup: place the plate, cup, bowl and spoon in the designated sink region, marked as a black rectangle on the kitchen counter. Each object earns one point; all four complete the stage. An additional tray is not a fifth scored utensil.

The PDF does not define how a skipped stage affects the phrase “highest completed stage”, or numerical completion thresholds for partial recovery. Keep measured stage evidence and explicit completion flags, and document local choices rather than attribute them to organizers. It specifies no Task 2 or Task 3 total time limit. The website mentions force-related safety gates, but these sources provide no numerical force limits suitable for an actuator configuration.

## Real demonstration contracts

The user-provided [Task 1 part 1](https://huggingface.co/datasets/ebim-benchmark/ebim-task1-realrobotdata-lerobot-part1) and [part 2](https://huggingface.co/datasets/ebim-benchmark/ebim-task1-realrobotdata-lerobot-part2) releases contain 24 and 26 episodes. Treat `(release, episode_index)` as the identity; each release has its own numbering. Exact metadata is archived alongside this report.

Task 1 uses a **62-value state** and **23-value action**. All ranges below use Python's end-exclusive indexing:

| Task 1 state indices | Recorded quantity |
| --- | --- |
| `0:7`, `7:14`, `14:20`, `20` | Left arm joints, external joint torques, stiffness-frame wrench, measured gripper knuckle joint |
| `21:28`, `28:35`, `35:41`, `41` | Right arm equivalents |
| `42:49` | Base position XYZ and quaternion XYZW |
| `49:55` | Measured base linear XYZ and angular XYZ velocity |
| `55:61` | Base command-output linear XYZ and angular XYZ velocity |
| `61` | Spine measured joint state |

| Task 1 action indices | Recorded quantity |
| --- | --- |
| `0:7`, `7` | Left GELLO joint targets and gripper width-percent command |
| `8:15`, `15` | Right GELLO joint targets and gripper width-percent command |
| `16:22` | Base linear XYZ and angular XYZ command |
| `22` | Spine target-height command |

Three RGB videos are supplied: `head` at 376×672 and `wrist_left`/`wrist_right` at 480×640. Metadata identifies AV1, 20 fps. Their feature keys are `observation.images.<camera>`. Measured gripper knuckle positions and width commands are different quantities. The implementation team's inspection found commands within a 0–1 scale despite the word `percent`; multiplying them by 100 or interpreting them as radians would corrupt the contract. Opening direction still requires the actual deployment calibration.

The subsequently located official [Task 2 real dataset](https://huggingface.co/datasets/ebim-benchmark/ebim_task2_realrobotdata) has a `task2_munich` release with **42-value state**, **17-value action** and 238 successful conversion episodes according to `meta/info.json`. Its metadata lists conversion failures, so episode filenames can have gaps. Enumerate actual files rather than assume a contiguous range.

| Task 2 state indices | Recorded quantity |
| --- | --- |
| `0:7`, `7`, `8:15`, `15:21` | Left arm joints, measured gripper knuckle, external torques, stiffness-frame wrench |
| `21:28`, `28`, `29:36`, `36:42` | Right arm equivalents |

Task 2 action indices `0:7`, `7`, `8:15`, `15` have the analogous joint/gripper meaning; index `16` is the spine target. There are no base state/action fields and no measured spine in this release. Do not pad missing robot feedback with zeros. The accompanying `modality.json` is inconsistent with the actual 42-value state and must not override verified `info.json`/Parquet shape and ordering.

Task 2 declares 20 Hz synchronized records, while individual camera metadata declares head 22 fps and wrists 29 fps. Head images are 720×1280 and wrists 480×640. Use actual record timestamps and video presentation times; nominal frame rate alone is not enough to establish alignment. The converter metadata identifies linear interpolation.

**Unresolved spine units:** the implementation team's raw Task 2 episode inspection found index `16` around `434.0`, whereas Task 1 index `22` is around `0.27–0.31`. Millimetres versus metres is plausible but **not verified** by the sources checked. Preserve each release's native training values. Require a documented calibration/conversion before applying model outputs to a physical spine. The old [simulator recording specification](https://github.com/EBiM-Benchmark/benchmark/blob/161db49e9eafd34be31e5bdada72d8e2f796c16a/task2_isaacsim/services/recording/DATASET.md) uses a different 20-value action and 29-value state; its metre label does not establish the real dataset's units.

The user reports that organizers will not release Task 3 trajectories. This implementation must not depend on a future Task 3 demonstration download. The older public [dataset tracking issue](https://github.com/EBiM-Benchmark/benchmark/issues/17) promises incremental releases generically; it is not evidence of available Task 3 trajectories and does not supersede the user's current organizer information.

## Simulator and deployment implications

The [official status matrix](https://github.com/EBiM-Benchmark/benchmark/blob/161db49e9eafd34be31e5bdada72d8e2f796c16a/STATUS.md) identifies Task 1 MuJoCo and Task 2 Isaac Sim as usable end-to-end. Task 3 MuJoCo has scenes and cameras but lacks maintainer verification of full task execution. Task 2's released Isaac implementation needs PhysX GPU deformables; a native MuJoCo Task 2 implementation is not in that release. The [FAQ](https://ebim-benchmark.github.io/faq.html) permits a participant-built MuJoCo Task 2 environment. These readiness statements describe development environments, not Phase II qualification or measured policy performance.

No inspected source establishes a stable, complete Phase II policy RPC or ROS contract. Dataset names suggest GELLO joint commands, gripper-width commands, base velocity and spine target, but do not prove exact live topic spelling, ROS message classes, timestamps, controller rates, TF frames, gripper direction, or calibration. The simulator ROS bridge is not sufficient evidence of the real interface. Keep transport and calibrated unit conversion explicit. Likewise obtain actual camera intrinsics/extrinsics and current target assignments from the evaluation integration instead of embedding old scene coordinates.

## Submission and known documentation traps

The current [official submissions repository](https://github.com/EBiM-Benchmark/submissions) requests a public repository with Dockerfile and run README, plus integration instructions for separately hosted weights. Each task has a separate issue; team name and point-of-contact email must match registration. Our team name is **AirSign**; a repository name is not the team identity. Do not publish submissions or message organizers merely as a side effect of implementation.

The repository also describes a technical-report path weighted at 0.65 of the policy scale and a declaration of whether simulator ground-truth poses were used. These are submission-route descriptions, not permission to replace measured real observations with simulated state. The exact numerical policy weighting was not established here.

Watch for stale snapshots: `docs/participant_readme.md` is explicitly a draft and older cached copies retain previous August/September deadlines. The status matrix's “200 episodes” roadmap predates the actual releases. The official issue submission is separate from the Task 1 ManipulationNet evaluation client. Do not treat its local submission/evaluation command as the Phase II entry procedure.

Remaining integration facts are concrete: current target/fixture assignment and calibration, exact physical command transport, gripper conversion, Task 2 spine units, live spine/base feedback where absent from training data, and numerical organizer safety limits. Model training and reversible offline verification can proceed while these are represented explicitly. Offline loss, simulator actuator checks and local scoring do not establish real-world task success.

## Annotation audit

Task 1's released numeric episodes contain only `1` in
`annotation.human.validity`. Task 2 contains whole episodes marked `0` and
others marked `1`: the complete release has 149 episodes / 86,444 frames marked
`1`, and 89 episodes / 58,521 frames marked `0`. Both groups can end with
`next.reward = 1`. Its
[pinned dataset card](https://huggingface.co/datasets/ebim-benchmark/ebim_task2_realrobotdata/blob/495ebb7b56fb9e2f3952398a63d86f08cacb9531/README.md)
contains a license declaration, and the inspected metadata supplies no enum
mapping or physical success definition. The baseline retains all released
episodes and records the raw annotation values. It does not silently remove
one group or treat the final reward as proof of successful pad placement.
These fields require an organizer/exporter definition before success-weighted
imitation learning can be justified.
