#!/usr/bin/env python3
"""Run AirSign's dependency-free, CPU-only submission readiness checks."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from evidence_json import (
        EvidenceJSONError,
        strict_json_load,
        strict_json_loads,
    )
    from stage1_capture_bundle import validate_schema2_capture
except ModuleNotFoundError:  # Imported as scripts.validate_submission in tests.
    from scripts.evidence_json import (
        EvidenceJSONError,
        strict_json_load,
        strict_json_loads,
    )
    from scripts.stage1_capture_bundle import validate_schema2_capture


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_EVIDENCE = ROOT / "evidence" / "four-stage-rehearsal"
STAGE1_EVIDENCE = ROOT / "evidence" / "stage1-physical-development"
PINNED_EBIM_COMMIT = "cb5184574f33611f943ff42aae461678ccb538e9"
PINNED_ROOM_ASSET_SHA256 = (
    "696c71577f1874d815fe29c6a58c65f0f1a0a0fb15c0d8adbb5105209f5ff883"
)
PINNED_ROBOT_ASSET_SHA256 = (
    "aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c"
)
DIRECT_UPSTREAM_HELPER_NAMES = {
    "dual_arm_lula",
    "gripper_profiles",
    "isaacsim_fr3duo_teleop_bridge_args",
    "isaacsim_fr3duo_teleop_bridge_core",
    "scene_robot_room_keyboard",
}
STAGE1_EXECUTED_PYTHON_SOURCES = {
    "participant/stage1_table_setup.py",
    "participant/base_motion_monitor.py",
    "participant/isaac_collision_geometry.py",
    "participant/joint_command_guard.py",
    "participant/se2_route_validator.py",
    "participant/executable_provenance.py",
}
STAGE1_EXECUTED_SOURCES = {
    *STAGE1_EXECUTED_PYTHON_SOURCES,
    "participant/upstream_source_manifest.json",
    "scripts/airsign",
}
STAGE1_HOST_PACKAGING_SOURCES = {
    ".dockerignore",
    "Dockerfile",
    "scripts/capture_stage1.sh",
    "scripts/evidence_json.py",
    "scripts/stage1_capture_bundle.py",
}
FAILURES: list[str] = []
PASSED: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def pass_check(message: str) -> None:
    PASSED.append(message)


def checked_object(value: Any, message: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    FAILURES.append(message)
    return {}


def inventory_paths_are_exact(
    records: Any,
    *,
    expected_paths: set[str],
    path_key: str,
    path_prefix: str = "",
) -> bool:
    """Return whether an inventory contains each expected path exactly once."""

    if not isinstance(records, list) or len(records) != len(expected_paths):
        return False
    paths: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            return False
        path = record.get(path_key)
        if not isinstance(path, str):
            return False
        paths.append(f"{path_prefix}{path}")
    return len(set(paths)) == len(paths) and set(paths) == expected_paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def expected_upstream_helpers(relative: Path) -> dict[str, dict[str, Any]]:
    manifest_path = ROOT / "participant" / "upstream_source_manifest.json"
    try:
        manifest = strict_json_load(manifest_path, root_type=dict)
    except EvidenceJSONError as error:
        FAILURES.append(
            f"{relative} upstream source manifest is invalid: {error}"
        )
        return {}
    check(
        type(manifest.get("schema_version")) is int
        and manifest.get("schema_version") == 1,
        f"{relative} upstream source manifest schema is invalid",
    )
    check(
        manifest.get("repository")
        == "https://github.com/EBiM-Benchmark/benchmark"
        and manifest.get("revision") == PINNED_EBIM_COMMIT,
        f"{relative} upstream source manifest revision is not pinned",
    )
    helpers = checked_object(
        manifest.get("helpers"),
        f"{relative} upstream source manifest helpers must be an object",
    )
    check(
        set(helpers) == DIRECT_UPSTREAM_HELPER_NAMES,
        f"{relative} upstream source manifest helper set is unexpected",
    )
    valid = all(
        isinstance(record, dict)
        and isinstance(record.get("repository_path"), str)
        and bool(record.get("repository_path"))
        and isinstance(record.get("bytes"), int)
        and not isinstance(record.get("bytes"), bool)
        and record.get("bytes") > 0
        and isinstance(record.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))
        is not None
        for record in helpers.values()
    )
    check(
        valid,
        f"{relative} upstream source manifest has invalid helper records",
    )
    return {
        name: record
        for name, record in helpers.items()
        if isinstance(record, dict)
    }


def tracked_and_untracked_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def compile_python_sources() -> None:
    paths = sorted((ROOT / "participant").glob("*.py"))
    paths.extend(sorted((ROOT / "scripts").glob("*.py")))
    for path in paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as error:
            FAILURES.append(f"Python syntax check failed for {path.relative_to(ROOT)}: {error}")
    if not any("Python syntax check failed" in failure for failure in FAILURES):
        pass_check(f"Python syntax ({len(paths)} files)")


def run_cpu_unit_tests() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        FAILURES.append(
            "CPU route/controller diagnostic tests failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
        return
    match = re.search(
        r"Ran\s+(\d+)\s+tests?", result.stderr + result.stdout
    )
    count = match.group(1) if match else "all"
    pass_check(f"CPU route/controller diagnostic tests ({count})")


def validate_canonical_replay() -> None:
    validator = ROOT / "participant" / "validate_four_stage_replay.py"
    result = subprocess.run(
        [sys.executable, str(validator), str(CANONICAL_EVIDENCE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        FAILURES.append(
            "Canonical replay validation failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
        return
    prefix = "FOUR_STAGE_REPLAY_VALIDATION "
    record_line = next(
        (line for line in result.stdout.splitlines() if line.startswith(prefix)),
        None,
    )
    if record_line is None:
        FAILURES.append("Canonical replay validator emitted no machine-readable result")
        return
    try:
        record = strict_json_loads(
            record_line.removeprefix(prefix),
            source="canonical replay validator result",
            root_type=dict,
        )
    except EvidenceJSONError as error:
        FAILURES.append(f"Canonical replay result is invalid JSON: {error}")
        return
    check(record.get("passed") is True, "Canonical replay validator did not pass")
    check(record.get("stage_ids") == [1, 2, 3, 4], "Canonical replay lacks stages 1–4")
    check(
        record.get("official_stage_completion_claimed") is False,
        "Canonical replay must reject official stage-completion claims",
    )
    check(
        record.get("official_score_claimed") is False
        and record.get("benchmark_score", "missing") is None,
        "Canonical replay must reject official score claims",
    )
    if not any("Canonical replay" in failure for failure in FAILURES):
        pass_check("Canonical replay schema, hashes, stage coverage, and claim boundary")


def walk_claims(value: Any, location: str, source: Path) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in {
                "official_score_claimed",
                "official_stage_completion_claimed",
                "official_stage_complete",
            }:
                check(
                    child is False,
                    f"{source.relative_to(ROOT)} {child_location} must be false",
                )
            if key in {"benchmark_score", "official_stage_score"}:
                check(
                    child is None,
                    f"{source.relative_to(ROOT)} {child_location} must be null",
                )
            walk_claims(child, child_location, source)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_claims(child, f"{location}[{index}]", source)


def validate_evidence_boundaries() -> None:
    paths = sorted((ROOT / "evidence").rglob("*.json"))
    before = len(FAILURES)
    documents: dict[Path, Any] = {}
    for path in paths:
        try:
            root_type = list if path.name == "image-inspect.json" else dict
            document = strict_json_load(path, root_type=root_type)
        except EvidenceJSONError as error:
            FAILURES.append(f"Invalid evidence JSON in {path.relative_to(ROOT)}: {error}")
            continue
        documents[path] = document
        walk_claims(document, "$", path)

    canonical_metrics_path = CANONICAL_EVIDENCE / "metrics.json"
    metrics = checked_object(
        documents.get(canonical_metrics_path),
        "Canonical metrics must be a strict JSON object",
    )
    canonical_provenance = checked_object(
        metrics.get("provenance"),
        "Canonical metrics provenance must be an object",
    )
    controller = ROOT / "participant" / "four_stage_rehearsal.py"
    check(
        canonical_provenance.get("controller_sha256") == sha256_file(controller),
        "Canonical metrics controller hash does not match four_stage_rehearsal.py",
    )
    try:
        canonical_exit = (
            (CANONICAL_EVIDENCE / "controller.exit")
            .read_text(encoding="utf-8")
            .strip()
        )
    except (OSError, UnicodeError) as error:
        canonical_exit = ""
        FAILURES.append(f"Canonical controller exit is unreadable: {error}")
    check(canonical_exit == "0", "Canonical controller exit status is not zero")
    development_path = (
        ROOT / "evidence" / "development_integration_validation.json"
    )
    development = checked_object(
        documents.get(development_path),
        "Development integration evidence must be a strict JSON object",
    )
    check(
        development.get("classification")
        == "development_grader_geometry_validation"
        and development.get("autonomous_robot_performance") is False
        and development.get("benchmark_score", "missing") is None,
        "Development grader evidence must remain explicitly non-autonomous and unscored",
    )
    if len(FAILURES) == before:
        pass_check(f"Evidence truth boundary ({len(paths)} JSON records)")


def validate_stage1_evidence_bundles() -> None:
    before = len(FAILURES)
    index_path = STAGE1_EVIDENCE / "README.md"
    check(index_path.is_file(), "Stage 1 diagnostic evidence index is missing")
    try:
        index = (
            index_path.read_text(encoding="utf-8")
            if index_path.is_file()
            else ""
        )
    except (OSError, UnicodeError) as error:
        index = ""
        FAILURES.append(f"Stage 1 evidence index is unreadable: {error}")
    bundle_dirs = sorted(
        path.parent for path in STAGE1_EVIDENCE.glob("*/manifest.json")
    )
    check(bundle_dirs, "No Stage 1 diagnostic evidence bundles were found")
    passing_schema2_bundles: list[Path] = []

    for bundle in bundle_dirs:
        relative = bundle.relative_to(ROOT)
        required = {
            name: bundle / name
            for name in (
                "README.md",
                "container.exit",
                "manifest.json",
                "metrics.json",
                "trajectory.json",
            )
        }
        for name, path in required.items():
            check(path.is_file(), f"{relative} is missing {name}")
        if not all(path.is_file() for path in required.values()):
            continue

        try:
            manifest = strict_json_load(
                required["manifest.json"],
                root_type=dict,
            )
            metrics = strict_json_load(
                required["metrics.json"],
                root_type=dict,
            )
            trajectory = strict_json_load(
                required["trajectory.json"],
                root_type=dict,
            )
            exit_text = (
                required["container.exit"]
                .read_text(encoding="utf-8")
                .strip()
            )
            if re.fullmatch(r"[01]", exit_text) is None:
                raise ValueError("container.exit must be exactly 0 or 1")
            exit_status = int(exit_text)
        except (EvidenceJSONError, OSError, UnicodeError, ValueError) as error:
            FAILURES.append(f"Invalid Stage 1 bundle {relative}: {error}")
            continue

        try:
            readme = required["README.md"].read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            FAILURES.append(f"Invalid Stage 1 bundle {relative}: {error}")
            continue
        computed_hashes = {
            name: sha256_file(path) for name, path in required.items()
        }
        manifest_files = checked_object(
            manifest.get("files"),
            f"{relative} manifest.files must be an object",
        )
        for name in ("metrics.json", "trajectory.json"):
            record = checked_object(
                manifest_files.get(name),
                f"{relative} manifest.files.{name} must be an object",
            )
            check(
                record.get("sha256") == computed_hashes[name],
                f"{relative} manifest hash mismatch for {name}",
            )
            check(
                isinstance(record.get("bytes"), int)
                and not isinstance(record.get("bytes"), bool)
                and record.get("bytes") == required[name].stat().st_size,
                f"{relative} manifest byte count mismatch for {name}",
            )

        provenance = checked_object(
            metrics.get("provenance"),
            f"{relative} metrics.provenance must be an object",
        )
        trajectory_provenance = checked_object(
            trajectory.get("provenance"),
            f"{relative} trajectory.provenance must be an object",
        )
        controller_sha = provenance.get("controller_sha256")
        check(
            isinstance(controller_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", controller_sha) is not None,
            f"{relative} has no valid controller provenance hash",
        )
        check(
            manifest.get("controller_sha256") == controller_sha
            and trajectory_provenance.get("controller_sha256") == controller_sha
            and checked_object(
                metrics.get("mutation_guard"),
                f"{relative} metrics.mutation_guard must be an object",
            ).get("controller_sha256")
            == controller_sha,
            f"{relative} controller provenance is internally inconsistent",
        )
        schema_version = metrics.get("schema_version")
        check(
            type(schema_version) is int and schema_version in {1, 2},
            f"{relative} metrics.schema_version must be integer 1 or 2",
        )
        if type(schema_version) is int and schema_version == 2:
            check(
                type(trajectory.get("schema_version")) is int
                and trajectory.get("schema_version") == 2
                and type(manifest.get("schema_version")) is int
                and manifest.get("schema_version") == 2,
                f"{relative} schema-2 records are not version-consistent",
            )
            manifest_provenance = checked_object(
                manifest.get("provenance"),
                f"{relative} manifest.provenance must be an object",
            )
            check(
                provenance == trajectory_provenance == manifest_provenance,
                f"{relative} schema-2 provenance objects differ",
            )
            check(
                manifest.get("provenance_sha256")
                == sha256_json(provenance),
                f"{relative} schema-2 provenance digest mismatch",
            )
            source_revision = provenance.get(
                "airsign_source_revision"
            )
            image_id = provenance.get("container_image_id")
            image_name = provenance.get("container_image")
            source_set_sha = provenance.get("source_set_sha256")
            python_source_set_sha = provenance.get(
                "python_source_set_sha256"
            )
            check(
                isinstance(source_revision, str)
                and re.fullmatch(r"[0-9a-f]{40}", source_revision)
                is not None,
                f"{relative} has no full AirSign source revision",
            )
            check(
                isinstance(image_id, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
                is not None,
                f"{relative} has no immutable container image ID",
            )
            check(
                isinstance(image_name, str) and bool(image_name.strip()),
                f"{relative} has no container image name",
            )
            check(
                isinstance(source_set_sha, str)
                and re.fullmatch(r"[0-9a-f]{64}", source_set_sha)
                is not None
                and manifest.get("source_set_sha256") == source_set_sha
                and manifest.get("airsign_source_revision")
                == source_revision
                and manifest.get("container_image_id") == image_id,
                f"{relative} source/image manifest binding is invalid",
            )
            check(
                provenance.get("official_benchmark_commit")
                == PINNED_EBIM_COMMIT,
                f"{relative} does not bind the pinned EBiM revision",
            )
            check(
                provenance.get("room_asset_sha256")
                == PINNED_ROOM_ASSET_SHA256,
                f"{relative} does not bind the pinned Task 3 room asset",
            )
            check(
                provenance.get("robot_asset_sha256")
                == PINNED_ROBOT_ASSET_SHA256,
                f"{relative} does not bind the pinned robot asset",
            )
            expected_helpers = expected_upstream_helpers(relative)
            observed_helpers = checked_object(
                provenance.get("direct_upstream_helpers"),
                f"{relative} direct_upstream_helpers must be an object",
            )
            check(
                set(observed_helpers) == set(expected_helpers)
                == DIRECT_UPSTREAM_HELPER_NAMES,
                f"{relative} direct upstream helper set is incomplete",
            )
            for helper_name, expected_helper in expected_helpers.items():
                observed_helper = checked_object(
                    observed_helpers.get(helper_name),
                    (
                        f"{relative} direct upstream helper "
                        f"{helper_name} must be an object"
                    ),
                )
                check(
                    observed_helper.get("official_repository_path")
                    == expected_helper.get("repository_path")
                    and observed_helper.get("bytes")
                    == expected_helper.get("bytes")
                    and observed_helper.get("sha256")
                    == expected_helper.get("sha256")
                    and observed_helper.get("pinned_manifest_revision")
                    == PINNED_EBIM_COMMIT
                    and observed_helper.get("pinned_manifest_path")
                    == "participant/upstream_source_manifest.json"
                    and observed_helper.get("matches_pinned_manifest")
                    is True
                    and isinstance(
                        observed_helper.get("runtime_path"),
                        str,
                    )
                    and bool(observed_helper.get("runtime_path")),
                    (
                        f"{relative} direct upstream helper "
                        f"{helper_name} differs from the pinned manifest"
                    ),
                )
            executed_sources = provenance.get("executed_sources")
            if not isinstance(executed_sources, list):
                check(
                    False,
                    f"{relative} executed source inventory is not a list",
                )
                executed_sources = []
            executed_records_valid = all(
                isinstance(record, dict)
                and isinstance(record.get("repository_path"), str)
                and isinstance(record.get("bytes"), int)
                and not isinstance(record.get("bytes"), bool)
                and isinstance(record.get("sha256"), str)
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    record.get("sha256", ""),
                )
                is not None
                for record in executed_sources
            )
            check(
                executed_records_valid,
                f"{relative} executed source inventory has invalid records",
            )
            check(
                inventory_paths_are_exact(
                    executed_sources,
                    expected_paths=STAGE1_EXECUTED_SOURCES,
                    path_key="repository_path",
                ),
                (
                    f"{relative} executed source inventory must contain "
                    f"exactly {len(STAGE1_EXECUTED_SOURCES)} unique "
                    "repository paths"
                ),
            )
            combined_source_records = sorted(
                (
                    {
                        "repository_path": record.get("repository_path"),
                        "bytes": record.get("bytes"),
                        "sha256": record.get("sha256"),
                    }
                    for record in executed_sources
                    if isinstance(record, dict)
                ),
                key=lambda record: str(record["repository_path"]),
            )
            check(
                sha256_json(combined_source_records)
                == source_set_sha,
                f"{relative} full executed source-set digest is invalid",
            )
            for record in executed_sources:
                if not isinstance(record, dict):
                    continue
                repository_path = record.get("repository_path")
                source_hash = record.get("sha256")
                if (
                    not isinstance(repository_path, str)
                    or repository_path not in STAGE1_EXECUTED_SOURCES
                ):
                    continue
                current_path = ROOT / str(repository_path)
                check(
                    current_path.is_file()
                    and source_hash == sha256_file(current_path),
                    (
                        f"{relative} source hash does not match current "
                        f"{repository_path}"
                    ),
                )
                if (
                    isinstance(source_revision, str)
                    and re.fullmatch(r"[0-9a-f]{40}", source_revision)
                    is not None
                ):
                    historical = subprocess.run(
                        [
                            "git",
                            "show",
                            f"{source_revision}:{repository_path}",
                        ],
                        cwd=ROOT,
                        capture_output=True,
                    )
                    check(
                        historical.returncode == 0
                        and hashlib.sha256(
                            historical.stdout
                        ).hexdigest()
                        == source_hash,
                        (
                            f"{relative} source hash does not match "
                            f"{source_revision}:{repository_path}"
                        ),
                    )
            mutation_guard = checked_object(
                metrics.get("mutation_guard"),
                f"{relative} metrics.mutation_guard must be an object",
            )
            raw_guard_records = mutation_guard.get("sources")
            guard_records_valid = (
                isinstance(raw_guard_records, list)
                and all(
                    isinstance(record, dict)
                    for record in raw_guard_records
                )
            )
            guard_records = (
                raw_guard_records
                if isinstance(raw_guard_records, list)
                else []
            )
            check(
                mutation_guard.get("source_set_sha256")
                == python_source_set_sha
                and inventory_paths_are_exact(
                    guard_records,
                    expected_paths=STAGE1_EXECUTED_PYTHON_SOURCES,
                    path_key="path",
                    path_prefix="participant/",
                ),
                (
                    f"{relative} mutation guard must contain exactly "
                    f"{len(STAGE1_EXECUTED_PYTHON_SOURCES)} unique "
                    "executed Python sources"
                ),
            )
            check(
                guard_records_valid
                and sha256_json(
                    sorted(
                        guard_records,
                        key=lambda record: str(record.get("path")),
                    )
                )
                == python_source_set_sha,
                f"{relative} source-set digest does not match its inventory",
            )
            if (
                isinstance(source_revision, str)
                and re.fullmatch(r"[0-9a-f]{40}", source_revision)
                is not None
            ):
                ancestor = subprocess.run(
                    [
                        "git",
                        "merge-base",
                        "--is-ancestor",
                        source_revision,
                        "HEAD",
                    ],
                    cwd=ROOT,
                )
                check(
                    ancestor.returncode == 0,
                    (
                        f"{relative} captured source revision is not an "
                        "ancestor of the submitted revision"
                    ),
                )
            for failure in validate_schema2_capture(bundle, provenance):
                FAILURES.append(f"{relative} {failure}")
            capture_path = bundle / "capture.json"
            if (
                capture_path.is_file()
                and isinstance(source_revision, str)
                and re.fullmatch(r"[0-9a-f]{40}", source_revision)
                is not None
            ):
                try:
                    capture_document = strict_json_load(
                        capture_path,
                        root_type=dict,
                    )
                except EvidenceJSONError:
                    capture_document = {}
                packaging_sources = capture_document.get(
                    "host_packaging_sources"
                )
                check(
                    inventory_paths_are_exact(
                        packaging_sources,
                        expected_paths=STAGE1_HOST_PACKAGING_SOURCES,
                        path_key="repository_path",
                    ),
                    (
                        f"{relative} host packaging inventory must contain "
                        f"exactly {len(STAGE1_HOST_PACKAGING_SOURCES)} "
                        "unique repository paths"
                    ),
                )
                if isinstance(packaging_sources, list):
                    for record in packaging_sources:
                        if not isinstance(record, dict):
                            continue
                        repository_path = record.get("repository_path")
                        source_hash = record.get("sha256")
                        if (
                            not isinstance(repository_path, str)
                            or repository_path
                            not in STAGE1_HOST_PACKAGING_SOURCES
                        ):
                            continue
                        historical = subprocess.run(
                            [
                                "git",
                                "show",
                                f"{source_revision}:{repository_path}",
                            ],
                            cwd=ROOT,
                            capture_output=True,
                        )
                        check(
                            historical.returncode == 0
                            and hashlib.sha256(
                                historical.stdout
                            ).hexdigest()
                            == source_hash,
                            (
                                f"{relative} host packaging source does not "
                                f"match {source_revision}:{repository_path}"
                            ),
                        )
        check(
            manifest.get("official_benchmark_commit")
            == provenance.get("official_benchmark_commit")
            == trajectory_provenance.get("official_benchmark_commit"),
            f"{relative} benchmark provenance is internally inconsistent",
        )

        passed = metrics.get("passed")
        check(
            isinstance(passed, bool),
            f"{relative} metrics.passed must be boolean",
        )
        check(
            metrics.get("official_stage_complete") is False
            and metrics.get("official_stage_score", "missing") is None
            and metrics.get("official_stage_completion_claimed") is False
            and metrics.get("official_score_claimed") is False,
            f"{relative} violates the Stage 1 official-claim boundary",
        )
        check(
            trajectory.get("official_stage_completion_claimed") is False
            and trajectory.get("official_score_claimed") is False
            and trajectory.get("task_objects_teleported") is False
            and trajectory.get("robot_links_teleported") is False
            and trajectory.get("task_object_mutation_api_used") is False,
            f"{relative} trajectory violates the physical-evidence boundary",
        )
        expected_exit = 0 if passed is True else 1
        check(
            exit_status == expected_exit,
            (
                f"{relative} container exit {exit_status} does not match "
                f"passed={passed!r}"
            ),
        )
        if (
            type(schema_version) is int
            and schema_version == 2
            and passed is True
            and exit_status == 0
            and metrics.get("gate_requested") == "cup-preflight"
        ):
            passing_schema2_bundles.append(bundle)
        check(
            bundle.name in index,
            f"Stage 1 evidence index does not list {bundle.name}",
        )
        check(
            "not official" in readme.lower()
            and "diagnostic" in readme.lower(),
            f"{relative} README lacks an explicit diagnostic claim boundary",
        )
        for name in (
            "metrics.json",
            "trajectory.json",
            "manifest.json",
            "container.exit",
        ):
            check(
                computed_hashes[name] in readme,
                f"{relative} README hash mismatch or omission for {name}",
            )

    check(
        bool(passing_schema2_bundles),
        (
            "No passing schema-2 Stage 1 cup-preflight binds all executed "
            "sources and the immutable container image"
        ),
    )
    if len(FAILURES) == before:
        pass_check(
            f"Stage 1 diagnostic manifests, hashes, provenance, and exits "
            f"({len(bundle_dirs)} bundles)"
        )


def validate_docker_contract() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    before = len(FAILURES)
    check(
        re.search(
            r"^ARG ISAAC_SIM_IMAGE=\S+@sha256:[0-9a-f]{64}$",
            dockerfile,
            flags=re.MULTILINE,
        )
        is not None,
        "Dockerfile base image must have a default SHA-256 digest pin",
    )
    check(
        "FROM ${ISAAC_SIM_IMAGE}" in dockerfile,
        "Dockerfile must build from the pinned ISAAC_SIM_IMAGE argument",
    )
    check(
        "ARG AIRSIGN_REVISION=unavailable" in dockerfile
        and (
            'LABEL org.opencontainers.image.revision="${AIRSIGN_REVISION}"'
            in dockerfile
        )
        and (
            'LABEL org.airsignrobot.ebim.revision="${EBIM_COMMIT}"'
            in dockerfile
        )
        and "/opt/airsign/source-revision" in dockerfile,
        (
            "Dockerfile must distinguish the AirSign OCI source revision "
            "from the pinned EBiM benchmark revision"
        ),
    )
    check(
        re.search(r"^\s*(?:ADD|COPY)\s+\.\s", dockerfile, flags=re.MULTILINE)
        is None,
        "Dockerfile must not copy the whole repository context",
    )
    check(
        re.search(r"^\s*ADD\s+", dockerfile, flags=re.MULTILINE) is None,
        "Dockerfile must not use ADD",
    )
    check(
        'ENTRYPOINT ["/usr/local/bin/airsign"]' in dockerfile
        and 'CMD ["smoke"]' in dockerfile,
        "Dockerfile must preserve the documented smoke-test default",
    )
    ordered_ignore_lines = [
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    ignore_lines = set(ordered_ignore_lines)
    check("*" in ignore_lines, ".dockerignore must default-deny the build context")
    sensitive_exclusions = {
        "**/.env",
        "**/.env.*",
        "**/*credential*",
        "**/*secret*",
        "**/*token*",
        "**/gcp.csv",
    }
    for required in sensitive_exclusions:
        check(required in ignore_lines, f".dockerignore is missing {required}")
    retained_stage1_log_exceptions = [
        "!evidence/stage1-physical-development/*/controller.log",
        "!evidence/stage1-physical-development/*/smoke.log",
    ]
    observed_log_exceptions = [
        line
        for line in ordered_ignore_lines
        if line.startswith("!") and ".log" in line
    ]
    check(
        "**/*.log" in ignore_lines,
        ".dockerignore must exclude logs by default",
    )
    check(
        observed_log_exceptions == retained_stage1_log_exceptions,
        (
            ".dockerignore may re-include only retained Stage 1 "
            "controller.log and smoke.log files"
        ),
    )
    if "**/*.log" in ordered_ignore_lines:
        blanket_log_index = ordered_ignore_lines.index("**/*.log")
        post_log_unignores = [
            line
            for line in ordered_ignore_lines[blanket_log_index + 1 :]
            if line.startswith("!")
        ]
        check(
            post_log_unignores == retained_stage1_log_exceptions,
            (
                "Only the two retained Stage 1 log exceptions may follow "
                "the blanket log exclusion"
            ),
        )
    unignore_indices = [
        index
        for index, line in enumerate(ordered_ignore_lines)
        if line.startswith("!")
    ]
    if unignore_indices:
        last_unignore_index = max(unignore_indices)
        check(
            all(
                exclusion in ordered_ignore_lines
                and ordered_ignore_lines.index(exclusion) > last_unignore_index
                for exclusion in sensitive_exclusions
            ),
            (
                "Credential and secret exclusions must follow every "
                ".dockerignore re-inclusion"
            ),
        )
    if len(FAILURES) == before:
        pass_check("Dockerfile pinning, entrypoint, and credential-safe build context")


def validate_stage1_development_contract() -> None:
    controller = ROOT / "participant" / "stage1_table_setup.py"
    joint_guard = ROOT / "participant" / "joint_command_guard.py"
    entrypoint = (ROOT / "scripts" / "airsign").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    before = len(FAILURES)
    check(controller.is_file(), "Stage 1 development controller is missing")
    check(joint_guard.is_file(), "Stage 1 joint command guard is missing")
    if not controller.is_file() or not joint_guard.is_file():
        return

    source = controller.read_text(encoding="utf-8")
    guard_source = joint_guard.read_text(encoding="utf-8")
    route_source = (
        ROOT / "participant" / "se2_route_validator.py"
    ).read_text(encoding="utf-8")
    collider_source = (
        ROOT / "participant" / "isaac_collision_geometry.py"
    ).read_text(encoding="utf-8")
    provenance_source = (
        ROOT / "participant" / "executable_provenance.py"
    ).read_text(encoding="utf-8")
    upstream_helpers = expected_upstream_helpers(
        Path("participant/upstream_source_manifest.json")
    )
    tree = ast.parse(source, filename=str(controller))
    forbidden_attributes = {
        "set_world_pose",
        "set_local_pose",
        "set_default_state",
        "set_linear_velocity",
        "set_angular_velocity",
        "set_joint_positions",
        "set_rigid_bodies_enabled_under",
        "set_rigid_bodies_kinematic_under",
        "translate_prim_preserving_rotation",
        "AddTranslateOp",
        "AddOrientOp",
        "AddTransformOp",
        "ClearXformOpOrder",
        "Set",
    }
    forbidden_direct = {"eval", "exec", "setattr", "delattr", "__import__"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_attributes
        ):
            violations.append(f"{node.func.attr}@{node.lineno}")
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in forbidden_direct
        ):
            violations.append(f"{node.func.id}@{node.lineno}")
    check(
        not violations,
        "Stage 1 controller contains direct mutation/dynamic calls: "
        + ", ".join(violations),
    )
    for required in {
        "BASE_MOTION_STOPPING_ENVELOPE_CERTIFIED = False",
        '"base_motion_stopping_envelope_uncertified"',
        '"base_motion_nonzero_commands_authorized"',
        '"official_stage_complete": False',
        '"official_stage_score": None',
        '"task_objects_teleported": False',
        '"robot_links_teleported": False',
        '"robot_link_initialization_boundary"',
        '"task_object_mutation_api_used": False',
        "raw Lula IK failed; no failed solution was applied",
        "steering_alignment_timeout",
        '"external_contact_telemetry": "unavailable"',
        'os.environ.get("EBIM_COMMIT"',
    }:
        check(
            required in source,
            f"Stage 1 evidence/safety contract is missing: {required}",
        )
    check(
        "planar = np.zeros(2, dtype=np.float64)" in source
        and "if distance > position_tolerance:" in source
        and "if planar_norm <= PLANAR_NORM_EPSILON:" in source
        and '"minimum_planar_command_floor_scope"' in source,
        (
            "Stage 1 minimum planar command must be gated outside the "
            "position tolerance, guard tiny norms, and expose telemetry"
        ),
    )
    check(
        "BaseMotionMonitor(" in source
        and '"steering_error_degrees"' in source
        and '"drive_target_rad_s"' in source
        and '"drive_measured_velocity_rad_s"' in source
        and '"stall_monitor_basis"' in source,
        (
            "Stage 1 base diagnostics must gate stalls on steering-aligned "
            "wheel motion and expose measured controller telemetry"
        ),
    )
    check(
        "joint_command_slew_record(" in source
        and "joint_tracking_error_dwell_record(" in source
        and "arm_and_spine_effort_record(" in source
        and '"spine_force_abort_threshold_newtons": None' in source
        and "--arm-max-command-slew-rad" in source
        and "--arm-tracking-error-threshold-rad" in source
        and "--arm-tracking-error-dwell-seconds" in source
        and "recorder.arm_last_command_by_name" in source
        and '"tracking_error_observation_gap_reset"' in source
        and '"command_clamping_used": False' in source
        and "target_to_previous_command_slew" in guard_source
        and "target_to_measured_tracking_error" in guard_source,
        (
            "Stage 1 arm commands must separately fail closed on adjacent-"
            "command slew and sustained target-to-measured tracking error, "
            "without clamping, while keeping prismatic spine force separate "
            "from revolute arm effort"
        ),
    )
    check(
        "route_execution_tube_record(" in source
        and "route_sha256(certified_waypoints)" in source
        and "--route-yaw-tracking-tolerance-deg" in source
        and "execution_translation_tolerance=(" in source
        and "execution_yaw_tolerance_rad=math.radians(" in source
        and '"route_certificate_binding_violation"' in source
        and '"route_yaw_violation"' in source
        and '"certificate_schema_version": 2' in route_source
        and '"certificate_inputs_sha256"' in route_source
        and '"robot_geometry_sha256"' in route_source
        and '"obstacle_geometry_sha256"' in route_source,
        (
            "Stage 1 base execution must remain inside an exact route- and "
            "geometry-bound SE(2) certificate envelope"
        ),
    )
    check(
        "attached_payload_root_paths=attached_payload_roots" in source
        and '"tray_loaded_collision_proxy_coverage"' in source
        and '"tray_loaded_return_route_preflight"' in source
        and "payload_reference_by_name=loaded_payload_reference" in source
        and '"payload_geometry_envelope_violation"' in source
        and "missing_attached_payload_root_paths" in collider_source
        and "attached_payload_proxy_paths" in collider_source,
        (
            "Loaded tray transport must recapture all payload colliders, "
            "preflight the return, and monitor the reserved compound envelope"
        ),
    )
    check(
        "resolve_enabled_dynamic_rigid_body_descendant(" in source
        and "live_collision_geometry_containment_record(" in source
        and "base_nonplanar_deviation_record(" in source
        and '"immediately_before_physics_step"' in source
        and '"immediately_after_physics_step"' in source
        and '"before_post_stop_physics_step"' in source
        and '"after_post_stop_physics_step"' in source
        and "--dynamic-geometry-certificate-allowance-metres" in source
        and "--dynamic-geometry-operational-limit-metres" in source
        and "--route-operational-cross-track-tolerance" in source
        and "--route-operational-yaw-tolerance-deg" in source
        and "CollisionGeometryWitness" in collider_source
        and "collision_geometry_containment_record(" in collider_source,
        (
            "Stage 1 base motion must bind unique dynamic task bodies and "
            "monitor exact articulated/payload collider witnesses plus "
            "non-planar base motion before, after, and through final settling"
        ),
    )
    check(
        "if not BASE_MOTION_STOPPING_ENVELOPE_CERTIFIED:" in source
        and "raise RuntimeError(BASE_MOTION_STOPPING_ENVELOPE_LIMITATION)"
        in source
        and "first_live_geometry_failure is None" in source
        and "post_stop_route_failure is None" in source,
        (
            "Stage 1 must keep nonzero base commands fail-closed until a "
            "certificate-bound stopping envelope exists and derive containment "
            "evidence directly from observed failures"
        ),
    )
    check(
        '"set_joint_positions"' in source
        and "core.setup_robot_control(" not in source
        and "core._find_drive_joint_ids(" in source
        and "capture_current_navigation_posture(" in source
        and (
            '"no base, arm, gripper, or task-object command was issued"'
            in source
        ),
        (
            "Stage 1 must avoid the upstream ready-pose teleport and keep the "
            "retained cup preflight read-only after asset initialization"
        ),
    )
    check(
        "executable_source_guard(" in source
        and "AIRSIGN_EXECUTED_SOURCE_NAMES" in source
        and "AIRSIGN_RUNTIME_DATA_NAMES" in source
        and "upstream_source_manifest.json" in source
        and "matches_pinned_manifest" in source
        and '"source_set_sha256"' in provenance_source
        and '"direct_upstream_helpers"' in source
        and '"airsign_source_revision"' in source
        and '"container_image_id"' in source,
        (
            "Stage 1 evidence must bind every executed AirSign source, direct "
            "upstream helpers, source revision, and container image"
        ),
    )
    check(
        set(upstream_helpers) == DIRECT_UPSTREAM_HELPER_NAMES,
        "Pinned direct-upstream helper manifest is incomplete",
    )
    check(
        "stage1-table-setup)" in entrypoint
        and "STAGE1_TABLE_SETUP_RESULT" in entrypoint,
        "Container entrypoint does not expose and verify stage1-table-setup",
    )
    check(
        "stage1_table_setup.py" in dockerfile,
        "Dockerfile does not package the Stage 1 development controller",
    )
    check(
        "executable_provenance.py" in entrypoint
        and "source-revision" in entrypoint
        and "airsign_revision" in entrypoint,
        (
            "Container smoke test does not bind the full executable source "
            "set and AirSign source revision"
        ),
    )
    if len(FAILURES) == before:
        pass_check(
            "Stage 1 articulation-only, fail-closed, truth-bounded development contract"
        )


def validate_workflow() -> None:
    workflow = ROOT / ".github" / "workflows" / "judge-readiness.yml"
    before = len(FAILURES)
    check(workflow.is_file(), "Judge-readiness GitHub Actions workflow is missing")
    if not workflow.is_file():
        return
    text = workflow.read_text(encoding="utf-8")
    check(
        re.search(r"^permissions:\s*\n\s+contents:\s+read\s*$", text, re.MULTILINE)
        is not None,
        "Workflow permissions must be read-only",
    )
    check("pull_request_target" not in text, "Workflow must not use pull_request_target")
    check("${{ secrets." not in text, "CPU-only workflow must not consume repository secrets")
    check(
        "persist-credentials: false" in text,
        "Checkout must not persist a GitHub credential",
    )
    check(
        "fetch-depth: 0" in text,
        (
            "Checkout must retain full history so captured source ancestry "
            "and historical host-packaging hashes can be verified"
        ),
    )
    for reference in re.findall(r"^\s*uses:\s+\S+@(\S+)\s*$", text, re.MULTILINE):
        check(
            re.fullmatch(r"[0-9a-f]{40}", reference) is not None,
            f"Workflow action reference must use a full commit SHA: {reference}",
        )
    if len(FAILURES) == before:
        pass_check("Read-only, secret-free, commit-pinned GitHub Actions workflow")


def scan_for_secrets() -> None:
    patterns = {
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "Google Cloud lab account": re.compile(r"(?i)[a-z0-9._%+-]+@gcplab\.me"),
        "Google OAuth authorization code": re.compile(r"\b4/0A[A-Za-z0-9_-]{20,}"),
        "Google OAuth access token": re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"),
        "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
        "GitHub token": re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
        ),
        "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
        "service-account private key": re.compile(r'"private_key"\s*:\s*"-----BEGIN'),
        "credential CSV header": re.compile(
            ",".join(
                ("Username", "Password", "Firstname", "Lastname", "Assigned Project")
            ),
            flags=re.IGNORECASE,
        ),
        "credential-bearing URL": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    }
    before = len(FAILURES)
    for path in tracked_and_untracked_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in patterns.items():
            if pattern.search(text):
                FAILURES.append(
                    f"Potential {label} in {path.relative_to(ROOT)}"
                )
    if len(FAILURES) == before:
        pass_check("Tracked and pending files contain no recognized credential patterns")


def validate_readme_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    before = len(FAILURES)
    for required in {
        "https://airsign-ebim-track3.chattytransformer.chatgpt.site/",
        "evidence/four-stage-rehearsal",
        "python3 scripts/validate_submission.py",
        "Official benchmark score claimed",
        "stage1-table-setup",
    }:
        check(required in readme, f"README judge contract is missing: {required}")
    if len(FAILURES) == before:
        pass_check("Judge quickstart, demo link, and canonical evidence pointers")


def main() -> int:
    compile_python_sources()
    run_cpu_unit_tests()
    validate_canonical_replay()
    validate_evidence_boundaries()
    validate_stage1_evidence_bundles()
    validate_docker_contract()
    validate_stage1_development_contract()
    validate_workflow()
    scan_for_secrets()
    validate_readme_contract()

    for message in PASSED:
        print(f"PASS  {message}")
    if FAILURES:
        for message in FAILURES:
            print(f"FAIL  {message}", file=sys.stderr)
        print(
            "AIRSIGN_SUBMISSION_CHECK "
            + json.dumps(
                {"passed": False, "checks_passed": len(PASSED), "failures": FAILURES},
                sort_keys=True,
            )
        )
        return 1
    print(
        "AIRSIGN_SUBMISSION_CHECK "
        + json.dumps(
            {"passed": True, "checks_passed": len(PASSED), "failures": []},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
