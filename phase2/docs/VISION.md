# AirSign optional RGB-D object perception

`ebim_phase2.vision` runs a local Grounding DINO detector followed by SAM2
segmentation. It deprojects each segmented visible surface through measured,
registered depth and calibrated camera extrinsics. It adds object candidates
and measured surface geometry to the Task 3 stack; it does not supply grasp
verification, loaded-spoon detection, seat assignments or task-success signals.

## Primary research and checked APIs

[Grounding DINO](https://arxiv.org/abs/2303.05499) combines language grounding
with an object detector to accept category names and referring expressions.
The [official Grounding DINO model](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
provides a practical detector checkpoint. Here each named object prompt is
processed separately, retaining both the requested label and decoded phrase.
Several candidates are retained instead of pretending the largest score proves
which cup or bowl is the competition object.

[SAM2](https://arxiv.org/abs/2408.00714) provides promptable segmentation in
images and videos. Its [official implementation](https://github.com/facebookresearch/sam2)
and [Transformers integration](https://huggingface.co/docs/transformers/model_doc/sam2)
support bounding-box prompts. This component uses single-image box prompts;
it does not claim SAM2 video tracking or persistent object identity. The
authors' [Grounded SAM2 project](https://github.com/IDEA-Research/Grounded-SAM-2)
also combines grounding and segmentation, motivating the two-model interface.

Research and local API inspection preceded implementation. Installed
Transformers **5.12.0** source was checked at
`models/grounding_dino/processing_grounding_dino.py`,
`models/sam2/processing_sam2.py` and `models/sam2/modeling_sam2.py`.
The implementation uses `threshold=` for DINO postprocessing and
`Sam2Processor.post_process_masks(pred_masks, original_sizes)`. SAM2 box input
is `[image, object, xyxy]`, with returned single masks `[object, 1, H, W]`.

## Exact model artifacts

| Role | Official repository | Immutable revision |
|---|---|---|
| Detector | `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| Segmenter | `facebook/sam2.1-hiera-tiny` | `de431c4043854a71d8101e17995dfe596bf101a5` |

Both require `model.safetensors`, `config.json`, and preprocessing files. DINO
also needs its tokenizer JSON/configuration/vocabulary assets. SAM2 includes
its processor configuration. `scripts/fetch_vision_models.py` downloads the
selected safetensors and processor assets into
`../cache/vision/grounding-dino-tiny` and `../cache/vision/sam2.1-hiera-tiny`.
That script is maintained by the main project workflow.

Local model directories must contain `download_manifest.json` with `repo_id`,
`revision`, and `files: [{path, size, sha256}, ...]`. The loader verifies the
expected revision, required assets, every listed size/hash, and rejects
unlisted JSON/text/weight assets that could override verified configuration.
Models load using safetensors with `trust_remote_code=False` and no network
access during model loading. Alternatively, a normal Hugging Face cache can
resolve the same immutable revisions. Missing snapshots fail by default;
`--allow-download` explicitly permits fetching those pinned snapshots.

Optional model dependencies are `transformers==5.12.0`, `huggingface_hub`,
`safetensors`, and the project's compatible PyTorch/torchvision installation.
NumPy and Pillow are used for images and geometry. OpenCV accelerates connected
components when installed; the geometry code has a NumPy/Python fallback.
Do not replace the project's B300-compatible Torch build as part of installing
the optional Transformers dependency.

## Running one captured frame

```bash
python -m ebim_phase2.vision \
  --rgb /path/to/head_rgb.png \
  --depth /path/to/registered_depth.npy \
  --calibration /path/to/camera_frame.json \
  --prompts /path/to/prompts.json \
  --timestamp 100.0 --depth-timestamp 100.0 --extrinsics-timestamp 100.0 \
  --camera-id head \
  --detector-dir ../cache/vision/grounding-dino-tiny \
  --segmenter-dir ../cache/vision/sam2.1-hiera-tiny \
  --output /path/to/objects.json --masks-output /path/to/masks.npz
```

`prompts.json` maps task object identifiers to descriptions, for example:

```json
{"plate": "a dinner plate", "cup": "a drinking cup", "bowl": "a bowl", "spoon": "a metal spoon"}
```

Camera JSON contains `fx`, `fy`, `cx`, `cy`, `depth_scale`, `calibration_id` and
`world_from_camera: {xyz: [x,y,z], wxyz: [w,x,y,z]}`. `depth_scale` converts the
stored depth units into metres. Depth must be **optical-axis Z**, registered
to the RGB pixels, with matching image dimensions; unregistered depth or ray
distance is not interchangeable. Intrinsics must correspond to those pixels
after any resizing. For head or wrist cameras on moving robot links, supply
the extrinsic transform measured at the image timestamp, not an old mounting
pose. RGB, depth and extrinsic timestamps must agree within the configured
skew tolerance, default 50 ms.

For a released RGB-only frame, omit `--depth`, `--calibration` and the two
geometry timestamps. Detection and segmentation still run; each candidate
reports `geometry: null` and `geometry_reason: "no_registered_depth"`. The
Python API accepts `depth=None, calibration=None` for the same mode. Do not
supply synthetic depth to make a release look geometrically complete. Color
does not certify thermal-pad material face or placement orientation.

For direct integration:

```python
from pathlib import Path
from ebim_phase2.vision import RGBDObjectPerception, select_unique_geometry

perception = RGBDObjectPerception(
    detector_dir=Path("../cache/vision/grounding-dino-tiny"),
    segmenter_dir=Path("../cache/vision/sam2.1-hiera-tiny"),
)
report, masks = perception.infer(
    rgb, registered_depth, camera_calibration, {"spoon": "a metal spoon"},
    timestamp=t_rgb, depth_timestamp=t_depth,
    extrinsics_timestamp=t_camera_pose, camera_id="head",
)
surface = select_unique_geometry(report, "spoon")
```

`select_unique_geometry` rejects multiple same-label candidates, unresolved
depth, and conflicting labels with strongly overlapping masks. Instance IDs
are per-frame indices; they are not tracker IDs. Run perception asynchronously
from the 20 Hz actuator loop and preserve acquisition timestamps. A slow model
result must become stale, not receive a new capture timestamp.

## What the geometry means

The estimator keeps the largest connected visible mask component, erodes its
boundary, rejects invalid depth, removes large median-absolute-deviation depth
outliers, and samples a bounded point set. It returns a robust **visible-surface
center**, PCA axes/variances, observed extents, depth support, point-cloud
scatter and an uncertainty record.

- A visible-surface center is not the hidden object's center of mass and is
  not automatically a safe grasp point.
- PCA axes retain a **180-degree direction ambiguity**. A spoon's handle and
  spoon bowl cannot be distinguished from an unsigned principal axis.
  Circular/near-symmetric surfaces explicitly mark their major axis unresolved.
- A camera-facing normal only fixes a representational sign. It does not
  identify the correct thermal-pad material face or semantic object orientation.
- A single view cannot prove absence of occlusion. `occlusion_state` is
  `unknown`, or `suspected` for border truncation, fragmented masks, missing
  depth or substantial outliers; it is never reported as clear.
- Detector scores, predicted mask IoU and the combined `quality_score` are
  **not calibrated probabilities**. Point-cloud scatter is not a pose-error
  covariance. `centroid_error_bound_m` is null because calibration error and
  hidden-surface bias are not bounded by this estimator.

No full `wxyz` pose is inferred from a box or ambiguous PCA axes. To pass a
surface point into the existing controller's `Estimate(Pose, ...)` contract,
`position_with_calibrated_orientation(...)` requires an externally calibrated
orientation, its calibration identifier, confidence and current time. This
does not turn the point into a grasp or a task predicate. Suspected occlusion
is rejected by default. The resulting estimate retains source `rgbd` and the
original capture timestamp, so existing freshness checks remain effective.

## Validation limits

`tests/test_vision.py` tests projection/extrinsic rotation, depth rejection,
axis ambiguity, symmetric objects, fragmentation and border handling,
freshness, controller conversion requirements, semantic ambiguity and model
manifest integrity without pretrained downloads. A processor/model test double
exercises the complete detector→segmenter→geometry contract.

Both pinned pretrained models were also loaded on the local B300 and run on
native-resolution head and wrist-right frames from the released Task 2 Munich
episode 35. The loader verified all required tensors, with no missing,
mismatched or unused checkpoint tensors. Transformers warns that the SAM2
snapshot's configuration says `sam2_video`; the image-model load was checked
explicitly and the image-mask inference path executed successfully. This does
not implement video tracking.

The combined three-prompt wrist pass took about 0.19 seconds after a roughly
2-second first head-image pass. This two-image check is not a throughput or
accuracy benchmark. Visual inspection found correct memory-module regions and
one wrist gray-strip candidate, alongside false positives covering the table
and competing semantic labels. The component reports those conflicts; it does
not silently select a grasp target. All 3D geometry fields were null because
the released clips supplied no registered depth for this check.

Reproduce with `scripts/check_vision_models.py`; reports and masks are in the
workspace's `runs/vision-pretrained-verified`, and the qualitative visualization
is in `runs/vision-pretrained-check/qualitative-mask-check.png`. These checks
validate execution and expose failure cases. Site accuracy, threshold calibration,
contact skills, task outcomes, and Task 3 physical success remain unverified.
