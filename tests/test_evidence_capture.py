#!/usr/bin/env python3
"""CPU-only tests for strict evidence capture packaging and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import validate_submission
from scripts.evidence_json import EvidenceJSONError, strict_json_loads
from scripts.stage1_capture_bundle import (
    package_capture,
    sha256_file,
    validate_schema2_capture,
)


class StrictEvidenceJSONTests(unittest.TestCase):
    def test_rejects_nonfinite_duplicate_and_wrong_root_json(self) -> None:
        for text in (
            '{"value": NaN}',
            '{"value": Infinity}',
            '{"value": -Infinity}',
            '{"value": 1e999}',
            '{"value": 1, "value": 2}',
        ):
            with self.subTest(text=text):
                with self.assertRaises(EvidenceJSONError):
                    strict_json_loads(text, root_type=dict)
        with self.assertRaises(EvidenceJSONError):
            strict_json_loads("[]", root_type=dict)
        with self.assertRaises(EvidenceJSONError):
            strict_json_loads("[" * 2000 + "0" + "]" * 2000)

    def test_accepts_finite_nested_strict_json(self) -> None:
        self.assertEqual(
            strict_json_loads(
                '{"items": [0, 1.5, {"ok": true}]}',
                root_type=dict,
            ),
            {"items": [0, 1.5, {"ok": True}]},
        )


class Stage1CaptureBundleTests(unittest.TestCase):
    revision = "a" * 40
    benchmark_commit = "cb5184574f33611f943ff42aae461678ccb538e9"
    image_id = "sha256:" + "c" * 64
    image = "airsignrobot:test"

    def make_unpacked_bundle(self, bundle: Path) -> dict[str, object]:
        provenance: dict[str, object] = {
            "airsign_source_revision": self.revision,
            "container_image": self.image,
            "container_image_id": self.image_id,
            "official_benchmark_commit": self.benchmark_commit,
        }
        metrics = {
            "schema_version": 2,
            "gate_requested": "cup-preflight",
            "classification": "participant_physical_development_gate",
            "passed": False,
            "official_stage_completion_claimed": False,
            "official_stage_complete": False,
            "official_score_claimed": False,
            "official_stage_score": None,
            "provenance": provenance,
        }
        trajectory = {
            "schema_version": 2,
            "official_stage_completion_claimed": False,
            "official_score_claimed": False,
            "task_objects_teleported": False,
            "robot_links_teleported": False,
            "task_object_mutation_api_used": False,
            "provenance": provenance,
        }
        image_inspect = [
            {
                "Id": self.image_id,
                "Config": {
                    "Labels": {
                        "org.opencontainers.image.revision": self.revision,
                        "org.airsignrobot.ebim.revision": (
                            self.benchmark_commit
                        ),
                    },
                    "Env": [
                        f"AIRSIGN_REVISION={self.revision}",
                        f"EBIM_COMMIT={self.benchmark_commit}",
                    ],
                },
            }
        ]
        for name, value in (
            ("metrics.json", metrics),
            ("trajectory.json", trajectory),
            ("image-inspect.json", image_inspect),
        ):
            (bundle / name).write_text(
                json.dumps(value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        manifest = {
            "schema_version": 2,
            "provenance": provenance,
            "files": {
                name: {
                    "bytes": (bundle / name).stat().st_size,
                    "sha256": sha256_file(bundle / name),
                }
                for name in ("metrics.json", "trajectory.json")
            },
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (bundle / "smoke.log").write_text(
            "AIRSIGN_SMOKE_RESULT "
            + json.dumps(
                {
                    "passed": True,
                    "airsign_revision": self.revision,
                    "benchmark_commit": self.benchmark_commit,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        controller_result = {
            "passed": False,
            "classification": metrics["classification"],
            "gate_requested": metrics["gate_requested"],
            "official_stage_completion_claimed": False,
            "official_score_claimed": False,
            "benchmark_score": None,
            "files": manifest["files"],
        }
        (bundle / "controller.log").write_text(
            "STAGE1_TABLE_SETUP_RESULT "
            + json.dumps(controller_result, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (bundle / "container.exit").write_text("1\n", encoding="utf-8")
        return provenance

    def test_packages_and_validates_complete_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            provenance = self.make_unpacked_bundle(bundle)
            package_capture(
                bundle,
                bundle_name="diagnostic-a",
                gate="cup-preflight",
                revision=self.revision,
                image=self.image,
                image_id=self.image_id,
            )
            self.assertEqual(
                validate_schema2_capture(
                    bundle,
                    provenance,
                    expected_bundle_name="diagnostic-a",
                ),
                [],
            )
            capture = json.loads(
                (bundle / "capture.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(capture["artifacts"]),
                {
                    "INDEX_ENTRY.md",
                    "container.exit",
                    "controller.log",
                    "image-inspect.json",
                    "manifest.json",
                    "metrics.json",
                    "smoke.log",
                    "trajectory.json",
                },
            )
            self.assertIn(
                "not official",
                (bundle / "README.md").read_text(encoding="utf-8"),
            )

    def test_image_and_nested_shape_tampering_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            provenance = self.make_unpacked_bundle(bundle)
            package_capture(
                bundle,
                bundle_name="diagnostic-b",
                gate="cup-preflight",
                revision=self.revision,
                image=self.image,
                image_id=self.image_id,
            )
            (bundle / "image-inspect.json").write_text(
                json.dumps(
                    [
                        {
                            "Id": "sha256:" + "d" * 64,
                            "Config": [],
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            failures = validate_schema2_capture(
                bundle,
                provenance,
                expected_bundle_name="diagnostic-b",
            )
            self.assertTrue(
                any("Id does not match" in failure for failure in failures)
            )
            self.assertTrue(
                any("Config must be an object" in failure for failure in failures)
            )

    def test_packaging_rejects_host_controller_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            self.make_unpacked_bundle(bundle)
            metrics = json.loads(
                (bundle / "metrics.json").read_text(encoding="utf-8")
            )
            metrics["provenance"]["container_image_id"] = (
                "sha256:" + "d" * 64
            )
            (bundle / "metrics.json").write_text(
                json.dumps(metrics) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(EvidenceJSONError):
                package_capture(
                    bundle,
                    bundle_name="diagnostic-c",
                    gate="cup-preflight",
                    revision=self.revision,
                    image=self.image,
                    image_id=self.image_id,
                )


class SubmissionValidatorShapeTests(unittest.TestCase):
    def test_schema2_source_inventories_require_exact_unique_paths(self) -> None:
        inventory_cases = (
            (
                validate_submission.STAGE1_EXECUTED_SOURCES,
                "repository_path",
                "",
            ),
            (
                validate_submission.STAGE1_EXECUTED_PYTHON_SOURCES,
                "path",
                "participant/",
            ),
            (
                validate_submission.STAGE1_HOST_PACKAGING_SOURCES,
                "repository_path",
                "",
            ),
        )
        for expected_paths, path_key, prefix in inventory_cases:
            with self.subTest(path_key=path_key, prefix=prefix):
                records = [
                    {
                        path_key: (
                            path.removeprefix(prefix)
                            if prefix
                            else path
                        )
                    }
                    for path in sorted(expected_paths)
                ]
                self.assertTrue(
                    validate_submission.inventory_paths_are_exact(
                        records,
                        expected_paths=expected_paths,
                        path_key=path_key,
                        path_prefix=prefix,
                    )
                )
                self.assertFalse(
                    validate_submission.inventory_paths_are_exact(
                        [*records, dict(records[0])],
                        expected_paths=expected_paths,
                        path_key=path_key,
                        path_prefix=prefix,
                    )
                )

    def test_nested_nonobjects_fail_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_root = (
                root / "evidence" / "stage1-physical-development"
            )
            bundle = evidence_root / "malformed-nested"
            bundle.mkdir(parents=True)
            (evidence_root / "README.md").write_text(
                "- malformed-nested\n",
                encoding="utf-8",
            )
            documents = {
                "manifest.json": {
                    "schema_version": 1,
                    "files": [],
                    "controller_sha256": None,
                    "official_benchmark_commit": None,
                },
                "metrics.json": {
                    "schema_version": 1,
                    "provenance": [],
                    "mutation_guard": [],
                    "passed": False,
                    "official_stage_complete": False,
                    "official_stage_score": None,
                    "official_stage_completion_claimed": False,
                    "official_score_claimed": False,
                },
                "trajectory.json": {
                    "provenance": [],
                    "official_stage_completion_claimed": False,
                    "official_score_claimed": False,
                    "task_objects_teleported": False,
                    "robot_links_teleported": False,
                    "task_object_mutation_api_used": False,
                },
            }
            for name, value in documents.items():
                (bundle / name).write_text(
                    json.dumps(value) + "\n",
                    encoding="utf-8",
                )
            (bundle / "container.exit").write_text("1\n", encoding="utf-8")
            (bundle / "README.md").write_text(
                "not official diagnostic\n",
                encoding="utf-8",
            )

            old_root = validate_submission.ROOT
            old_stage1 = validate_submission.STAGE1_EVIDENCE
            old_failures = validate_submission.FAILURES
            old_passed = validate_submission.PASSED
            try:
                validate_submission.ROOT = root
                validate_submission.STAGE1_EVIDENCE = evidence_root
                validate_submission.FAILURES = []
                validate_submission.PASSED = []
                validate_submission.validate_stage1_evidence_bundles()
                failures = validate_submission.FAILURES
            finally:
                validate_submission.ROOT = old_root
                validate_submission.STAGE1_EVIDENCE = old_stage1
                validate_submission.FAILURES = old_failures
                validate_submission.PASSED = old_passed

            self.assertTrue(
                any(
                    "metrics.provenance must be an object" in failure
                    for failure in failures
                )
            )
            self.assertTrue(
                any(
                    "manifest.files must be an object" in failure
                    for failure in failures
                )
            )


class DockerContextContractTests(unittest.TestCase):
    def validate_dockerignore(self, dockerignore: str) -> list[str]:
        source_root = Path(validate_submission.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_text(
                (source_root / "Dockerfile").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / ".dockerignore").write_text(
                dockerignore,
                encoding="utf-8",
            )
            old_root = validate_submission.ROOT
            old_failures = validate_submission.FAILURES
            old_passed = validate_submission.PASSED
            try:
                validate_submission.ROOT = root
                validate_submission.FAILURES = []
                validate_submission.PASSED = []
                validate_submission.validate_docker_contract()
                return list(validate_submission.FAILURES)
            finally:
                validate_submission.ROOT = old_root
                validate_submission.FAILURES = old_failures
                validate_submission.PASSED = old_passed

    def test_current_context_allows_only_retained_stage1_logs(self) -> None:
        root = Path(validate_submission.__file__).resolve().parents[1]
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
        self.assertEqual(self.validate_dockerignore(dockerignore), [])

    def test_broad_log_reinclusion_is_rejected(self) -> None:
        root = Path(validate_submission.__file__).resolve().parents[1]
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
        unsafe = dockerignore.replace(
            "!evidence/stage1-physical-development/*/smoke.log",
            (
                "!evidence/stage1-physical-development/*/smoke.log\n"
                "!evidence/**/*.log"
            ),
        )
        failures = self.validate_dockerignore(unsafe)
        self.assertTrue(
            any(
                "may re-include only retained Stage 1" in failure
                for failure in failures
            )
        )

    def test_late_reinclusion_cannot_override_secret_exclusions(self) -> None:
        root = Path(validate_submission.__file__).resolve().parents[1]
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
        failures = self.validate_dockerignore(
            dockerignore + "\n!evidence/**\n"
        )
        self.assertTrue(
            any(
                "secret exclusions must follow every" in failure
                for failure in failures
            )
        )

    def test_broad_evidence_reinclusion_cannot_restore_other_logs(self) -> None:
        root = Path(validate_submission.__file__).resolve().parents[1]
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
        unsafe = dockerignore.replace(
            "**/.env",
            "!evidence/**\n**/.env",
        )
        failures = self.validate_dockerignore(unsafe)
        self.assertTrue(
            any(
                "Only the two retained Stage 1 log exceptions" in failure
                for failure in failures
            )
        )


if __name__ == "__main__":
    unittest.main()
