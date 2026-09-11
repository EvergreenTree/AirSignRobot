"""Optional local Grounding DINO + SAM2 perception from registered RGB-D.

Outputs describe the visible surface, not a complete object pose or a grasp.
Model imports are lazy so geometry tests do not require model downloads.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import hashlib
import math
from pathlib import Path
import re
import time
from typing import Mapping

import numpy as np

from .tasks.control import Estimate
from .tasks.geometry import Pose, vector
from .tasks.perception import RGBDCalibration


@dataclass(frozen=True)
class ModelSnapshot:
    repo_id: str
    revision: str

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Model revision must be an immutable 40-character commit SHA")


DETECTOR = ModelSnapshot("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71")
SEGMENTER = ModelSnapshot("facebook/sam2.1-hiera-tiny", "de431c4043854a71d8101e17995dfe596bf101a5")


def verify_model_directory(directory: Path, snapshot: ModelSnapshot) -> Path:
    """Verify the pinned download manifest and every listed file before loading."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"download_manifest.json").read_text())
    if manifest.get("repo_id") != snapshot.repo_id or manifest.get("revision") != snapshot.revision:
        raise ValueError("Model directory does not match the pinned repository/revision")
    files = manifest.get("files", [])
    names = {entry["path"] for entry in files}
    required = {"config.json", "preprocessor_config.json", "model.safetensors"}
    if snapshot.repo_id == DETECTOR.repo_id:
        required |= {"tokenizer.json", "tokenizer_config.json"}
    if not required <= names:
        raise ValueError(f"Model manifest is missing required files: {sorted(required-names)}")
    for entry in files:
        path = (directory/entry["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Invalid model file path in manifest")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Model file size mismatch: {entry['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Model file hash mismatch: {entry['path']}")
    # Never let an unlisted local model/processor configuration shadow the
    # verified snapshot. HF config loading can resolve these auxiliary files.
    for path in directory.rglob("*"):
        if not path.is_file() or ".cache" in path.relative_to(directory).parts:
            continue
        if path.suffix in (".json", ".safetensors", ".txt") and path.name != "download_manifest.json" and str(path.relative_to(directory)) not in names:
            raise ValueError(f"Unverified model asset in directory: {path.name}")
    return directory


@dataclass(frozen=True)
class GeometrySettings:
    min_points: int = 32
    max_points: int = 10000
    erode_pixels: int = 1
    min_depth_m: float = 0.05
    max_depth_m: float = 5.
    depth_noise_floor_m: float = 0.002
    depth_mad_multiplier: float = 4.
    major_axis_ratio: float = 1.5
    normal_axis_ratio: float = 3.

    def __post_init__(self):
        if self.min_points < 3 or self.max_points < self.min_points or not 0 <= self.erode_pixels <= 5:
            raise ValueError("Invalid geometry sampling/erosion settings")
        scalars = (self.min_depth_m, self.max_depth_m, self.depth_noise_floor_m,
                   self.depth_mad_multiplier, self.major_axis_ratio, self.normal_axis_ratio)
        if not all(math.isfinite(x) and x > 0 for x in scalars) or self.max_depth_m <= self.min_depth_m:
            raise ValueError("Geometry thresholds must be positive and finite")


class GeometryUnavailable(ValueError):
    """A detected region has insufficient trustworthy depth for 3D geometry."""


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """8-connected visible component; optional OpenCV acceleration."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return np.zeros_like(mask), 0.
        selected = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == selected, float(stats[selected, cv2.CC_STAT_AREA]/max(1, int(mask.sum())))
    height, width = mask.shape
    remaining = mask.copy()
    largest = []
    for y, x in np.argwhere(mask):
        if not remaining[y, x]:
            continue
        queue, component = deque([(int(y), int(x))]), []
        remaining[y, x] = False
        while queue:
            row, col = queue.popleft()
            component.append((row, col))
            for yy in range(max(0, row-1), min(height, row+2)):
                for xx in range(max(0, col-1), min(width, col+2)):
                    if remaining[yy, xx]:
                        remaining[yy, xx] = False
                        queue.append((yy, xx))
        if len(component) > len(largest):
            largest = component
    selected = np.zeros_like(mask)
    if largest:
        yy, xx = np.asarray(largest).T
        selected[yy, xx] = True
    return selected, float(len(largest)/max(1, int(mask.sum())))


def _erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(pixels):
        padded = np.pad(result, 1, constant_values=False)
        result = np.logical_and.reduce([padded[y:y+mask.shape[0], x:x+mask.shape[1]]
                                        for y in range(3) for x in range(3)])
    return result


def _rotation_matrix(pose: Pose) -> np.ndarray:
    w, x, y, z = pose.wxyz
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass(frozen=True)
class VisibleObjectGeometry:
    label: str
    timestamp: float
    calibration_id: str
    camera_id: str
    visible_surface_center_world: tuple[float, float, float]
    principal_axes_world: tuple[tuple[float, float, float], ...]
    principal_variances_m2: tuple[float, float, float]
    visible_extent_m: tuple[float, float, float]
    surface_scatter_world_m: tuple[float, float, float]
    robust_depth_sigma_m: float
    mask_pixels: int
    depth_points: int
    valid_depth_fraction: float
    retained_depth_fraction: float
    largest_component_fraction: float
    detection_score: float
    segmentation_score: float
    quality_score: float
    major_axis_reliable: bool
    surface_normal_reliable: bool
    occlusion_state: str
    flags: tuple[str, ...]
    axis_sign_ambiguous: bool = True
    full_pose_observed: bool = False
    centroid_error_bound_m: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def position_with_calibrated_orientation(self, wxyz, *, orientation_calibration_id: str,
                                             orientation_confidence: float, now: float,
                                             max_age_s: float = .25,
                                             allow_suspected_occlusion: bool = False) -> Estimate:
        """Adapt a surface point to existing Pose consumers using external orientation.

        This remains a visible-surface point, not a tool/grasp target. The site
        must supply tool offsets, contact planning and an independent outcome
        detector. ``quality_score`` is a heuristic, not a calibrated probability.
        """
        if not orientation_calibration_id or not 0 <= orientation_confidence <= 1:
            raise ValueError("Externally calibrated orientation and confidence are required")
        if not math.isfinite(now) or not 0 <= now-self.timestamp <= max_age_s:
            raise ValueError("Object geometry is stale or from the future")
        if self.occlusion_state == "suspected" and not allow_suspected_occlusion:
            raise ValueError("Suspected occlusion requires another view before using this point")
        pose = Pose(self.visible_surface_center_world, tuple(vector(wxyz, 4, "calibrated orientation")))
        return Estimate(pose, self.timestamp, min(self.quality_score, orientation_confidence), "rgbd")


def estimate_mask_geometry(mask, depth, calibration: RGBDCalibration, *, label: str,
                           timestamp: float, camera_id: str, detection_score: float,
                           segmentation_score: float, settings: GeometrySettings | None = None) -> VisibleObjectGeometry:
    """Robust geometry of a segmented, visible surface in a calibrated frame.

    Depth must already be registered to RGB and represent optical-axis Z.
    PCA axes have no semantic direction; a spoon handle cannot be distinguished
    from its bowl end by this geometric calculation alone.
    """
    settings = settings or GeometrySettings()
    mask, raw = np.asarray(mask), np.asarray(depth)
    if mask.dtype != bool or mask.ndim != 2 or raw.shape != mask.shape:
        raise ValueError("Expected a 2D boolean mask and same-shaped registered depth")
    if not label or not camera_id or not math.isfinite(timestamp):
        raise ValueError("Label, camera identity and finite timestamp are required")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (detection_score, segmentation_score)):
        raise ValueError("Detector and segmentation scores must be finite and in [0,1]")
    count = int(mask.sum())
    if count < settings.min_points:
        raise GeometryUnavailable("segmentation mask is too small")
    touches_border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    component, component_fraction = _largest_component(mask)
    interior = _erode(component, settings.erode_pixels)
    if interior.sum() < settings.min_points:
        raise GeometryUnavailable("mask interior is too narrow after boundary erosion")
    yy, xx = np.nonzero(interior)
    z = np.asarray(raw[yy, xx], dtype=float)*calibration.depth_scale
    valid = np.isfinite(z) & (z >= settings.min_depth_m) & (z <= settings.max_depth_m)
    valid_fraction = float(valid.mean())
    xx, yy, z = xx[valid], yy[valid], z[valid]
    if len(z) < settings.min_points:
        raise GeometryUnavailable("insufficient valid registered depth")
    median = float(np.median(z))
    sigma = max(settings.depth_noise_floor_m, 1.4826*float(np.median(np.abs(z-median))))
    retained = np.abs(z-median) <= settings.depth_mad_multiplier*sigma
    retained_fraction = float(retained.mean())
    xx, yy, z = xx[retained], yy[retained], z[retained]
    if len(z) < settings.min_points:
        raise GeometryUnavailable("too few depth inliers after robust filtering")
    available_points = len(z)
    if len(z) > settings.max_points:
        indices = np.linspace(0, len(z)-1, settings.max_points, dtype=int)
        xx, yy, z = xx[indices], yy[indices], z[indices]
    camera_points = np.stack([(xx-calibration.cx)*z/calibration.fx,
                              (yy-calibration.cy)*z/calibration.fy, z], axis=1)
    rotation = _rotation_matrix(calibration.world_from_camera)
    points = camera_points @ rotation.T + np.asarray(calibration.world_from_camera.xyz)
    center = np.median(points, axis=0)
    covariance = np.cov(points-center, rowvar=False)
    eigenvalues, axes = np.linalg.eigh(covariance)
    eigenvalues, axes = np.maximum(eigenvalues[::-1], 0), axes[:, ::-1]
    # Fix representational signs for repeatability, while explicitly preserving
    # the physical 180-degree ambiguity in the output contract.
    if axes[np.argmax(np.abs(axes[:, 0])), 0] < 0:
        axes[:, 0] *= -1
    camera_direction = np.asarray(calibration.world_from_camera.xyz)-center
    if np.dot(axes[:, 2], camera_direction) < 0:
        axes[:, 2] *= -1
    axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
    coordinates = (points-center) @ axes
    extent = np.quantile(coordinates, .95, axis=0)-np.quantile(coordinates, .05, axis=0)
    floor = settings.depth_noise_floor_m**2
    major = bool(eigenvalues[0] > settings.major_axis_ratio*max(eigenvalues[1], floor))
    normal = bool(eigenvalues[1] > settings.normal_axis_ratio*max(eigenvalues[2], floor))
    flags = ["single_view_hidden_surface_unknown", "axis_direction_180_degree_ambiguity",
             "quality_score_not_calibrated_probability", "centroid_error_bound_unknown"]
    if touches_border:
        flags.append("image_border_truncation")
    if component_fraction < .9:
        flags.append("fragmented_mask")
    if valid_fraction < .8:
        flags.append("missing_depth")
    if retained_fraction < .9:
        flags.append("depth_outliers")
    if not major:
        flags.append("major_axis_unresolved")
    if not normal:
        flags.append("surface_normal_unresolved")
    suspect = touches_border or component_fraction < .9 or valid_fraction < .8 or retained_fraction < .9
    quality = min(detection_score, segmentation_score)*valid_fraction*retained_fraction*component_fraction
    return VisibleObjectGeometry(label, float(timestamp), calibration.calibration_id, camera_id,
                                 tuple(center), tuple(tuple(axes[:, i]) for i in range(3)),
                                 tuple(eigenvalues), tuple(extent), tuple(np.sqrt(np.diag(covariance))), sigma,
                                 count, available_points, valid_fraction, retained_fraction, component_fraction,
                                 float(detection_score), float(segmentation_score), float(quality), major, normal,
                                 "suspected" if suspect else "unknown", tuple(flags))


def _box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    size = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(size))
    union = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    return intersection/union if union > 0 else 0.


class RGBDObjectPerception:
    """Local pinned detector and box-prompted segmenter; no remote model code."""

    def __init__(self, detector: ModelSnapshot = DETECTOR, segmenter: ModelSnapshot = SEGMENTER, *,
                 cache_dir: Path | None = None, device: str = "cuda",
                 detector_dir: Path | None = None, segmenter_dir: Path | None = None,
                 allow_download: bool = False, detection_threshold: float = .3,
                 text_threshold: float = .25, mask_threshold: float = .5,
                 max_detections_per_prompt: int = 4, max_skew_s: float = .05,
                 geometry_settings: GeometrySettings | None = None):
        if not all(0 < x < 1 for x in (detection_threshold, text_threshold, mask_threshold)):
            raise ValueError("Model thresholds must be in (0,1)")
        if not 1 <= max_detections_per_prompt <= 8 or not 0 < max_skew_s <= .25:
            raise ValueError("Invalid detection budget or timestamp skew limit")
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.torch, self.device = torch, torch.device(device)
        self.models = {"detector": asdict(detector), "segmenter": asdict(segmenter)}
        self.transformers_version = transformers.__version__
        self.detection_threshold, self.text_threshold = detection_threshold, text_threshold
        self.mask_threshold, self.max_detections_per_prompt = mask_threshold, max_detections_per_prompt
        self.max_skew_s, self.geometry_settings = max_skew_s, geometry_settings or GeometrySettings()
        def resolve(spec, directory):
            if directory is not None:
                return str(verify_model_directory(directory, spec))
            return snapshot_download(spec.repo_id, revision=spec.revision, cache_dir=cache_dir,
                                     local_files_only=not allow_download,
                                     allow_patterns=["*.json", "*.safetensors", "vocab.txt", "merges.txt"])
        dpath, spath = resolve(detector, detector_dir), resolve(segmenter, segmenter_dir)
        self.detector_processor = AutoProcessor.from_pretrained(dpath, local_files_only=True, trust_remote_code=False)
        self.detector, detector_loading = AutoModelForZeroShotObjectDetection.from_pretrained(
            dpath, local_files_only=True, trust_remote_code=False, use_safetensors=True, output_loading_info=True)
        self.segmenter_processor = Sam2Processor.from_pretrained(spath, local_files_only=True, trust_remote_code=False)
        self.segmenter, segmenter_loading = Sam2Model.from_pretrained(
            spath, local_files_only=True, trust_remote_code=False, use_safetensors=True, output_loading_info=True)
        self.loading_report={}
        for name, model, info in (('detector',self.detector,detector_loading),('segmenter',self.segmenter,segmenter_loading)):
            if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
                raise RuntimeError(f'{name} checkpoint did not load every required model tensor: {info}')
            self.loading_report[name]={'required_tensors_loaded':True,'unused_checkpoint_tensors':len(info.get('unexpected_keys',[]))}
            model.to(self.device).eval()

    def infer(self, rgb, depth, calibration: RGBDCalibration | None, prompts: Mapping[str, str], *,
              timestamp: float, depth_timestamp: float | None = None, extrinsics_timestamp: float | None = None,
              camera_id: str) -> tuple[dict, dict[str, np.ndarray]]:
        rgb = np.asarray(rgb)
        depth = None if depth is None else np.asarray(depth)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or (depth is not None and depth.shape != rgb.shape[:2]):
            raise ValueError("Expected uint8 RGB and, when supplied, registered same-resolution depth")
        if not math.isfinite(timestamp):
            raise ValueError("Finite RGB timestamp is required")
        geometry_missing = "no_registered_depth" if depth is None else "missing_camera_calibration" if calibration is None else None
        if geometry_missing is None:
            if depth_timestamp is None or extrinsics_timestamp is None or not all(math.isfinite(x) for x in (depth_timestamp, extrinsics_timestamp)):
                raise ValueError("Finite depth/extrinsics timestamps are required for geometry")
            if max(abs(timestamp-depth_timestamp), abs(timestamp-extrinsics_timestamp)) > self.max_skew_s:
                raise ValueError("RGB, depth and camera pose are not synchronized")
        if not 1 <= len(prompts) <= 16 or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value.strip() or not 1 <= len(value) <= 120
                                             for key, value in prompts.items()):
            raise ValueError("Supply 1–16 named object prompts of 1–120 characters")
        from PIL import Image
        started = time.monotonic()
        image = Image.fromarray(rgb)
        detections = []
        with self.torch.inference_mode():
            # One category per grounding call preserves category association
            # even when the text decoder returns only part of a noun phrase.
            for label, prompt in prompts.items():
                text = prompt.strip().rstrip(".").lower()+"."
                inputs = self.detector_processor(images=image, text=text, return_tensors="pt").to(self.device)
                outputs = self.detector(**inputs)
                decoded = self.detector_processor.post_process_grounded_object_detection(
                    outputs, input_ids=inputs["input_ids"], threshold=self.detection_threshold,
                    text_threshold=self.text_threshold, target_sizes=[rgb.shape[:2]])[0]
                selected = []
                order = decoded["scores"].argsort(descending=True).tolist()
                for index in order:
                    box = decoded["boxes"][index].detach().cpu().numpy().astype(float)
                    box[[0, 2]], box[[1, 3]] = np.clip(box[[0, 2]], 0, rgb.shape[1]), np.clip(box[[1, 3]], 0, rgb.shape[0])
                    if not np.isfinite(box).all() or np.any(box[2:]-box[:2] < 2):
                        continue
                    if any(_box_iou(box, x["box_xyxy"]) > .6 for x in selected):
                        continue
                    item = {"label": label, "prompt": prompt, "box_xyxy": box.tolist(),
                            "detector_phrase": decoded["text_labels"][index],
                            "detection_score": float(decoded["scores"][index])}
                    selected.append(item)
                    if len(selected) == self.max_detections_per_prompt:
                        break
                detections.extend(selected)
            masks_by_id, objects = {}, []
            # Segment in bounded batches to avoid allocating all full-resolution
            # masks simultaneously when multiple prompts produce many matches.
            for offset in range(0, len(detections), 8):
                batch = detections[offset:offset+8]
                inputs = self.segmenter_processor(images=image, input_boxes=[[x["box_xyxy"] for x in batch]],
                                                   return_tensors="pt").to(self.device)
                outputs = self.segmenter(**inputs, multimask_output=False)
                masks = self.segmenter_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                                    binarize=True)[0]
                scores = outputs.iou_scores.detach().float().cpu().numpy()
                if masks.ndim != 4 or masks.shape[:2] != (len(batch), 1):
                    raise RuntimeError(f"Unexpected SAM2 mask layout: {tuple(masks.shape)}")
                for local, detection in enumerate(batch):
                    instance = f"object_{offset+local:03d}"
                    mask = masks[local, 0].numpy().astype(bool)
                    predicted_iou = float(scores.reshape(len(batch), -1)[local, 0])
                    item = {**detection, "instance_id": instance, "segmentation_score": predicted_iou,
                            "geometry": None, "geometry_status": "unavailable"}
                    masks_by_id[instance] = mask
                    if geometry_missing:
                        item["geometry_reason"] = geometry_missing
                    elif not math.isfinite(predicted_iou) or predicted_iou < self.mask_threshold:
                        item["geometry_reason"] = "low_segmentation_quality"
                    else:
                        try:
                            geometry = estimate_mask_geometry(mask, depth, calibration, label=detection["label"],
                                                              timestamp=timestamp, camera_id=camera_id,
                                                              detection_score=detection["detection_score"],
                                                              segmentation_score=float(np.clip(predicted_iou, 0, 1)),
                                                              settings=self.geometry_settings)
                            item.update(geometry=geometry.to_dict(), geometry_status="visible_surface_only")
                        except GeometryUnavailable as error:
                            item["geometry_reason"] = str(error)
                    objects.append(item)
        # Cross-label overlap is semantic ambiguity, not an excuse to silently
        # choose whichever description happened to receive the larger score.
        for item in objects:
            mask = masks_by_id[item["instance_id"]]
            conflicts = []
            for other in objects:
                if other["label"] == item["label"]:
                    continue
                other_mask = masks_by_id[other["instance_id"]]
                union = np.logical_or(mask, other_mask).sum()
                if union and np.logical_and(mask, other_mask).sum()/union > .7:
                    conflicts.append(other["instance_id"])
            item["semantic_conflicts"] = conflicts
        return {"format": "airsign_rgbd_objects_v1", "timestamp": timestamp, "camera_id": camera_id,
                "calibration_id": calibration.calibration_id if calibration else None, "models": self.models,
                "transformers_version": self.transformers_version, "objects": objects,
                "weight_loading":self.loading_report,
                "missing_labels": sorted(set(prompts)-{x["label"] for x in objects}),
                "elapsed_seconds": time.monotonic()-started,
                "scope": "single-frame visible-surface geometry; no grasp or task-success predicates"}, masks_by_id


def select_unique_geometry(report: dict, label: str) -> VisibleObjectGeometry:
    """Do not silently assign one of several detected cups to an official seat."""
    if report.get("format") != "airsign_rgbd_objects_v1":
        raise ValueError("Unsupported perception report format")
    matches = [item for item in report["objects"] if item["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"Expected one {label!r}; observed {len(matches)} candidates")
    item = matches[0]
    if item.get("semantic_conflicts") or item.get("geometry") is None:
        raise ValueError("Object identity or 3D geometry is unresolved")
    return VisibleObjectGeometry(**item["geometry"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, help="Optional registered optical-Z depth as .npy; absent means no 3D geometry")
    parser.add_argument("--calibration", type=Path, help="RGBDCalibration JSON with world_from_camera pose")
    parser.add_argument("--prompts", type=Path, required=True, help="JSON object label→description")
    parser.add_argument("--timestamp", type=float, required=True)
    parser.add_argument("--depth-timestamp", type=float)
    parser.add_argument("--extrinsics-timestamp", type=float)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--detector-dir", type=Path)
    parser.add_argument("--segmenter-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly download the pinned HF snapshots if absent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--masks-output", type=Path, help="Optional compressed .npz of instance masks")
    args = parser.parse_args(argv)
    if args.output.exists() or (args.masks_output is not None and args.masks_output.exists()):
        parser.error("Output exists; choose a new report/mask path")
    from PIL import Image
    calibration = None
    if args.calibration:
        config = json.loads(args.calibration.read_text())
        config["world_from_camera"] = Pose(**config["world_from_camera"])
        calibration = RGBDCalibration(**config)
    perception = RGBDObjectPerception(cache_dir=args.cache_dir, device=args.device,
                                      detector_dir=args.detector_dir, segmenter_dir=args.segmenter_dir,
                                      allow_download=args.allow_download)
    report, masks = perception.infer(np.asarray(Image.open(args.rgb).convert("RGB")),
                                     np.load(args.depth, allow_pickle=False) if args.depth else None, calibration,
                                     json.loads(args.prompts.read_text()), timestamp=args.timestamp,
                                     depth_timestamp=args.depth_timestamp,
                                     extrinsics_timestamp=args.extrinsics_timestamp, camera_id=args.camera_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.masks_output:
        args.masks_output.parent.mkdir(parents=True, exist_ok=True)
        with args.masks_output.open("wb") as target:
            np.savez_compressed(target, **masks)
    print(json.dumps({"report": str(args.output), "objects": len(report["objects"]),
                      "missing_labels": report["missing_labels"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
