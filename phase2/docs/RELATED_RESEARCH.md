# Method selection for AirSign Phase II

Research date: 2026-09-10. The implementation was started after reading the official
competition page, the five-page ranking PDF, the release schemas, and the primary
papers below. This is a new Phase II project. The older AirSign Task 3 repository
supplies team identity and historical failure information, not a successful policy.

| Primary source | Relevant result/design | Decision in this project |
|---|---|---|
| [ACT / ALOHA, Zhao et al., RSS 2023](https://arxiv.org/abs/2304.13705) | Bimanual visual imitation with action chunks, a conditional variational model, and temporal aggregation. | First trained baseline: three-camera ResNet18 ACT, masked action chunks, target-free inference, short feedback interval. |
| [Diffusion Policy, Chi et al., RSS 2023](https://arxiv.org/abs/2303.04137) | Conditional action denoising handles multimodal manipulation; receding-horizon execution limits open-loop drift. | Use short-horizon re-observation. A diffusion comparison is a future experiment, not implemented or claimed. |
| [Multi-Stage Cable Routing, Luo et al., 2023](https://arxiv.org/html/2307.08927v2) | Learned local routing plus a high-level primitive selector and recovery demonstrations handles accumulating failures. | Route order and checkpoint verification belong outside raw motor regression. No clip labels or recovery demonstrations are fabricated from episode timestamps. |
| [UMI, Chi et al., RSS 2024](https://real.stanford.edu/umi/) | Deployment interface design, latency matching, and relative trajectories matter for transferring human demonstrations. | Preserve native dataset units, use real video PTS, reject camera skew, and account for observation age. No demonstration-to-robot interface is assumed from a matching dimension. |
| [RoboTwin 2.0, Chen et al., 2025](https://arxiv.org/abs/2506.18088) | Synthesized bimanual demonstrations and systematic scene variation can help real transfer. | Keep generated data distinct from organizer demonstrations. Do not report synthetic controller fixtures as robot task completion. |
| [Goal-Conditioned Transporter Networks / DeformableRavens](https://arxiv.org/abs/2012.03385) | Spatially structured perception supports manipulation of cables and other deformables. | Retain spatial image tokens and image geometry; avoid flattening each camera into a single global feature. |
| [CARBS, Grannen et al., 2022](https://arxiv.org/abs/2211.14652) | Reactive stabilization with the second arm improves scooping. | Task 3 explicitly stabilizes the bowl and verifies loaded-spoon feeding and return. |
| [VAPORS, Sundaresan et al., 2023](https://arxiv.org/abs/2309.05197) | Sequential food acquisition combines high-level planning with visually parameterized skills. | Task 3 uses a feedback-driven skill program, with fresh perception and bounded geometric control. |
| [FLAIR, Jenamani et al., 2024](https://arxiv.org/abs/2407.07561) | A parameterized skill library supports long-horizon feeding. | Independent skill completion predicates and recovery paths are useful; an LLM planner is not required for a fixed four-stage contest. |
| [OpenPI official implementation](https://github.com/Physical-Intelligence/openpi) | Pretrained VLA fine-tuning is an alternative to training task-specific imitation policies. | Keep this as an alternative; the first implementation has a small, auditable native action interface and uses the released data directly. No untested VLA performance is claimed. |

## Current implementation boundaries

The ACT implementation is a baseline, not a reproduction of every ACT experiment.
It uses a shared pretrained ResNet18, spatial and camera embeddings, a transformer
CVAE posterior with episode padding masks, a visual transformer encoder/decoder,
training-only normalization, and modest joint photometric augmentation. Tasks 1
and 2 have separate models because their state order, action dimensions, and spine
representations differ. Task 1 has 50 released episodes; Task 2 has 238, with holes
in the original episode indices.

The final model includes a measured-joint residual anchor. A controlled 1,000-step
Task 2 smoke comparison used the same partial dataset, seed and batch size:
normalized validation MAE was 0.17757 for absolute regression and 0.08890 for the
residual model. Only one complete validation episode (141 sampled observations)
was available in that snapshot. This selects an implementation baseline, not a
competition result or a general architecture superiority claim. Full-release
training uses 320-pixel views; the initial comparison used 160 pixels.

The dataset's task-description labels are generic ("cable routing" and pad
placement), not route-step annotations. Recorded action names identify GELLO
targets; conversion to actual servo coordinates remains a separate calibrated
interface. Even an identity joint/spine mapping must be explicit at deployment.

The real Task 1 images are used directly, including base approach and manipulation.
The model does not receive old Phase I fixture coordinates or simulator object
poses. It currently has no learned task-progress labels. This limits claims about
generalizing to unseen route instructions, recovering from unfamiliar errors, or
solving an entire long episode. Offline action MAE is not a competition score.

MuJoCo supplies robot kinematics and actuator diagnostics. Its old Task 1 example
scene is only a component test and does not define the Phase II environment.
Task 2's official soft-body simulator requires Isaac Sim 5.1 PhysX; the real-data
training path does not depend on that simulator. Task 3 is a new calibrated
feedback-controller implementation; no organizer Task 3 trajectories are assumed.
