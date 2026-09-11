from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest

from ebim_phase2.tasks.geometry import Pose
from ebim_phase2.tasks.perception import RGBDCalibration
from ebim_phase2.vision import (DETECTOR, GeometrySettings, GeometryUnavailable,
                               ModelSnapshot, estimate_mask_geometry,
                               select_unique_geometry, verify_model_directory)


def calibration(pose=None):
    return RGBDCalibration(100., 100., 50., 50., .001,
                            pose or Pose((1., 2., 3.), (1., 0., 0., 0.)), "synthetic-test-camera")


def surface(mask=None, depth=None, camera=None, **kwargs):
    if mask is None:
        mask = np.zeros((101, 101), bool)
        mask[40:61, 20:81] = True
    if depth is None:
        depth = np.full(mask.shape, 1000., dtype=float)
    return estimate_mask_geometry(mask, depth, camera or calibration(), label="spoon", timestamp=1.,
                                  camera_id="head", detection_score=.9, segmentation_score=.95, **kwargs)


def test_visible_surface_center_axes_and_uncertainty_are_explicit():
    estimate = surface()
    np.testing.assert_allclose(estimate.visible_surface_center_world, [1., 2., 4.])
    assert estimate.major_axis_reliable and estimate.surface_normal_reliable
    assert abs(estimate.principal_axes_world[0][0]) == pytest.approx(1.)
    assert estimate.axis_sign_ambiguous and not estimate.full_pose_observed
    assert estimate.centroid_error_bound_m is None
    assert estimate.occlusion_state == "unknown"
    assert "quality_score_not_calibrated_probability" in estimate.flags
    assert estimate.robust_depth_sigma_m > 0
    assert "wxyz" not in estimate.to_dict()


def test_camera_rotation_transforms_surface_axes_to_world():
    q = (math.sqrt(.5), 0., 0., math.sqrt(.5))
    estimate = surface(camera=calibration(Pose((1., 2., 3.), q)))
    np.testing.assert_allclose(estimate.visible_surface_center_world, [1., 2., 4.], atol=1e-10)
    assert abs(estimate.principal_axes_world[0][1]) == pytest.approx(1.)
    axes = np.asarray(estimate.principal_axes_world).T
    np.testing.assert_allclose(axes.T@axes, np.eye(3), atol=1e-10)
    assert np.linalg.det(axes) == pytest.approx(1.)


def test_circular_surface_does_not_claim_unique_yaw():
    yy, xx = np.mgrid[:101, :101]
    mask = (xx-50)**2+(yy-50)**2 <= 20**2
    estimate = surface(mask=mask)
    assert not estimate.major_axis_reliable
    assert "major_axis_unresolved" in estimate.flags
    assert estimate.axis_sign_ambiguous


def test_depth_outliers_are_removed_and_marked_as_suspect():
    mask = np.zeros((101, 101), bool); mask[30:71, 20:81] = True
    depth = np.full(mask.shape, 1000.)
    depth[30:71:4, 20:81] = 1800.
    estimate = surface(mask=mask, depth=depth)
    assert estimate.visible_surface_center_world[2] == pytest.approx(4.)
    assert estimate.retained_depth_fraction < .9
    assert estimate.occlusion_state == "suspected" and "depth_outliers" in estimate.flags


def test_missing_registered_depth_never_falls_back_to_box_center():
    with pytest.raises(GeometryUnavailable, match="registered depth"):
        surface(depth=np.zeros((101, 101)))
    with pytest.raises(ValueError, match="same-shaped"):
        surface(depth=np.zeros((20, 20)))


def test_fragmented_and_border_masks_report_suspected_occlusion():
    mask = np.zeros((101, 101), bool)
    mask[0:30, 0:30] = True
    mask[70:95, 70:95] = True
    estimate = surface(mask=mask)
    assert estimate.largest_component_fraction < .9
    assert {"fragmented_mask", "image_border_truncation"} <= set(estimate.flags)
    assert estimate.occlusion_state == "suspected"
    with pytest.raises(ValueError, match="occlusion"):
        estimate.position_with_calibrated_orientation((1, 0, 0, 0), orientation_calibration_id="site",
                                                      orientation_confidence=.9, now=1.1)


def test_pose_adapter_requires_external_orientation_and_current_observation():
    estimate = surface()
    result = estimate.position_with_calibrated_orientation((1, 0, 0, 0), orientation_calibration_id="site-tool",
                                                           orientation_confidence=.85, now=1.1)
    assert result.source == "rgbd" and result.timestamp == 1.
    assert result.value.xyz == estimate.visible_surface_center_world
    assert result.confidence <= .85
    with pytest.raises(ValueError, match="Externally"):
        estimate.position_with_calibrated_orientation((1, 0, 0, 0), orientation_calibration_id="",
                                                      orientation_confidence=.9, now=1.1)
    with pytest.raises(ValueError, match="stale"):
        estimate.position_with_calibrated_orientation((1, 0, 0, 0), orientation_calibration_id="site",
                                                      orientation_confidence=.9, now=3.)


def test_instance_and_cross_label_ambiguity_are_not_silently_resolved():
    item = {"label": "spoon", "geometry": surface().to_dict(), "semantic_conflicts": []}
    report = {"format": "airsign_rgbd_objects_v1", "objects": [item]}
    assert select_unique_geometry(report, "spoon").label == "spoon"
    with pytest.raises(ValueError, match="2 candidates"):
        select_unique_geometry({**report, "objects": [item, item]}, "spoon")
    with pytest.raises(ValueError, match="unresolved"):
        select_unique_geometry({**report, "objects": [{**item, "semantic_conflicts": ["object_002"]}]}, "spoon")


def test_model_pin_and_manifest_reject_mismatch_or_modified_assets(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        ModelSnapshot(DETECTOR.repo_id, "main")
    names = ("config.json", "preprocessor_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    entries = []
    for name in names:
        data = b"unit-test manifest fixture; not model weights"
        (tmp_path/name).write_bytes(data)
        entries.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"repo_id": DETECTOR.repo_id, "revision": DETECTOR.revision, "files": entries}
    (tmp_path/"download_manifest.json").write_text(json.dumps(manifest))
    assert verify_model_directory(tmp_path, DETECTOR) == tmp_path.resolve()
    (tmp_path/"model.safetensors").write_bytes(b"x"*entries[2]["size"])
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_model_directory(tmp_path, DETECTOR)


def test_geometry_settings_bound_sampling_and_do_not_require_transformers():
    estimate = surface(settings=GeometrySettings(max_points=100))
    assert np.isfinite(estimate.visible_surface_center_world).all()
    with pytest.raises(ValueError):
        GeometrySettings(min_points=100, max_points=10)


def test_inference_checks_sensor_time_before_model_use():
    from ebim_phase2.vision import RGBDObjectPerception
    backend = RGBDObjectPerception.__new__(RGBDObjectPerception)
    backend.max_skew_s = .05
    with pytest.raises(ValueError, match="synchronized"):
        backend.infer(np.zeros((101, 101, 3), np.uint8), np.zeros((101, 101)), calibration(),
                      {"cup": "a cup"}, timestamp=1., depth_timestamp=.5,
                      extrinsics_timestamp=1., camera_id="head")


def test_detector_segmenter_contract_produces_only_measured_surface_geometry():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from ebim_phase2.vision import RGBDObjectPerception
    class Inputs(dict):
        def to(self, device):
            return self
    class DetectorProcessor:
        def __call__(self, **kwargs):
            assert kwargs["text"].endswith(".")
            return Inputs(input_ids=torch.tensor([[1, 2, 3]]))
        def post_process_grounded_object_detection(self, outputs, **kwargs):
            assert kwargs["threshold"] == .3 and kwargs["target_sizes"] == [(101, 101)]
            return [{"scores": torch.tensor([.9]), "boxes": torch.tensor([[20., 40., 81., 61.]]),
                     "text_labels": ["spoon"]}]
    class SegmenterProcessor:
        def __call__(self, **kwargs):
            assert np.shape(kwargs["input_boxes"]) == (1, 1, 4)
            return Inputs(original_sizes=torch.tensor([[101, 101]]))
        def post_process_masks(self, masks, original_sizes, **kwargs):
            assert kwargs["binarize"] is True
            return [masks[0]]
    mask = torch.zeros((1, 1, 1, 101, 101), dtype=torch.bool)
    mask[:, :, :, 40:61, 20:81] = True
    backend = RGBDObjectPerception.__new__(RGBDObjectPerception)
    backend.torch, backend.device = torch, torch.device("cpu")
    backend.detector_processor, backend.segmenter_processor = DetectorProcessor(), SegmenterProcessor()
    backend.detector = lambda **kwargs: None
    backend.segmenter = lambda **kwargs: SimpleNamespace(pred_masks=mask, iou_scores=torch.tensor([[[.95]]]))
    backend.detection_threshold, backend.text_threshold, backend.mask_threshold = .3, .25, .5
    backend.max_detections_per_prompt, backend.max_skew_s = 4, .05
    backend.geometry_settings = GeometrySettings()
    backend.models, backend.transformers_version = {}, "test-double"
    backend.loading_report={}
    report, masks = backend.infer(np.zeros((101, 101, 3), np.uint8), np.full((101, 101), 1000.),
                                  calibration(), {"spoon": "a metal spoon"}, timestamp=1., depth_timestamp=1.,
                                  extrinsics_timestamp=1., camera_id="head")
    assert report["objects"][0]["geometry_status"] == "visible_surface_only"
    assert masks["object_000"].dtype == bool and masks["object_000"].sum() > 0
    assert report["objects"][0]["geometry"]["full_pose_observed"] is False
    assert "signals" not in report and report["missing_labels"] == []
    rgb_only, rgb_masks = backend.infer(np.zeros((101, 101, 3), np.uint8), None, None,
                                        {"spoon": "a metal spoon"}, timestamp=1., camera_id="head")
    assert rgb_only["objects"][0]["geometry"] is None
    assert rgb_only["objects"][0]["geometry_reason"] == "no_registered_depth"
    assert rgb_only["calibration_id"] is None and len(rgb_masks) == 1
    assert "pad_orientation_correct" not in json.dumps(rgb_only)
