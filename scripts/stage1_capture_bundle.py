#!/usr/bin/env python3
"""Package and verify a retained Stage 1 host/container evidence capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from evidence_json import (
        EvidenceJSONError,
        strict_json_load,
        strict_json_loads,
    )
except ModuleNotFoundError:  # Imported as scripts.stage1_capture_bundle in tests.
    from scripts.evidence_json import (
        EvidenceJSONError,
        strict_json_load,
        strict_json_loads,
    )


CAPTURE_ARTIFACT_NAMES = {
    "INDEX_ENTRY.md",
    "container.exit",
    "controller.log",
    "image-inspect.json",
    "manifest.json",
    "metrics.json",
    "smoke.log",
    "trajectory.json",
}
HOST_PACKAGING_SOURCES = {
    ".dockerignore",
    "Dockerfile",
    "scripts/capture_stage1.sh",
    "scripts/evidence_json.py",
    "scripts/stage1_capture_bundle.py",
}
PINNED_EBIM_COMMIT = "cb5184574f33611f943ff42aae461678ccb538e9"
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: Any, location: str, failures: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    failures.append(f"{location} must be an object")
    return {}


def _string_list(
    value: Any,
    location: str,
    failures: list[str],
) -> list[str]:
    if isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        return value
    failures.append(f"{location} must be a list of strings")
    return []


def _load_prefixed_record(path: Path, prefix: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvidenceJSONError(f"{path}: {error}") from error
    records = [
        line.removeprefix(prefix)
        for line in lines
        if line.startswith(prefix)
    ]
    if len(records) != 1:
        raise EvidenceJSONError(
            f"{path}: expected exactly one {prefix.strip()} record, "
            f"found {len(records)}"
        )
    return strict_json_loads(
        records[0],
        source=f"{path} machine-readable result",
        root_type=dict,
    )


def _load_smoke_record(path: Path) -> dict[str, Any]:
    return _load_prefixed_record(path, "AIRSIGN_SMOKE_RESULT ")


def validate_schema2_capture(
    bundle: Path,
    provenance: Any,
    *,
    expected_bundle_name: str | None = None,
) -> list[str]:
    """Return all host/image binding failures for one schema-2 bundle."""

    failures: list[str] = []
    provenance_object = _object(provenance, "provenance", failures)
    source_revision = provenance_object.get("airsign_source_revision")
    image_id = provenance_object.get("container_image_id")
    image_name = provenance_object.get("container_image")
    benchmark_commit = provenance_object.get("official_benchmark_commit")

    required = CAPTURE_ARTIFACT_NAMES | {"capture.json", "README.md"}
    for name in sorted(required):
        if not (bundle / name).is_file():
            failures.append(f"missing retained capture artifact {name}")
    if failures:
        return failures

    try:
        capture = strict_json_load(bundle / "capture.json", root_type=dict)
        metrics = strict_json_load(bundle / "metrics.json", root_type=dict)
        manifest = strict_json_load(bundle / "manifest.json", root_type=dict)
        image_inspect = strict_json_load(
            bundle / "image-inspect.json",
            root_type=list,
        )
        smoke = _load_smoke_record(bundle / "smoke.log")
        controller_result = _load_prefixed_record(
            bundle / "controller.log",
            "STAGE1_TABLE_SETUP_RESULT ",
        )
        readme = (bundle / "README.md").read_text(encoding="utf-8")
    except (EvidenceJSONError, OSError, UnicodeError) as error:
        failures.append(str(error))
        return failures

    if (
        type(capture.get("schema_version")) is not int
        or capture.get("schema_version") != 1
    ):
        failures.append("capture.json schema_version must be 1")
    if capture.get("kind") != "airsign_stage1_host_capture_manifest":
        failures.append("capture.json has an unexpected kind")
    if capture.get("classification") != "retained_non_official_diagnostic":
        failures.append("capture.json has an unexpected classification")
    intended_bundle_name = expected_bundle_name or bundle.name
    if capture.get("bundle_name") != intended_bundle_name:
        failures.append(
            "capture.json bundle_name does not match its directory"
        )
    if capture.get("gate") != metrics.get("gate_requested"):
        failures.append(
            "capture.json gate does not match metrics.gate_requested"
        )
    for key, expected in (
        ("airsign_source_revision", source_revision),
        ("container_image", image_name),
        ("container_image_id", image_id),
        ("official_benchmark_commit", benchmark_commit),
    ):
        if capture.get(key) != expected:
            failures.append(
                f"capture.json {key} does not match controller provenance"
            )

    packaging_sources = capture.get("host_packaging_sources")
    if not isinstance(packaging_sources, list):
        failures.append(
            "capture.json host_packaging_sources must be a list"
        )
        packaging_sources = []
    packaging_records_valid = all(
        isinstance(record, dict)
        and isinstance(record.get("repository_path"), str)
        and isinstance(record.get("bytes"), int)
        and not isinstance(record.get("bytes"), bool)
        and isinstance(record.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))
        is not None
        for record in packaging_sources
    )
    if not packaging_records_valid:
        failures.append(
            "capture.json host packaging source records are invalid"
        )
    packaging_paths = {
        record.get("repository_path")
        for record in packaging_sources
        if isinstance(record, dict)
        and isinstance(record.get("repository_path"), str)
    }
    if packaging_paths != HOST_PACKAGING_SOURCES:
        failures.append(
            "capture.json host packaging source inventory is incomplete"
        )
    if (
        capture.get("host_packaging_source_set_sha256")
        != sha256_json(
            sorted(
                (
                    record
                    for record in packaging_sources
                    if isinstance(record, dict)
                ),
                key=lambda record: str(record.get("repository_path")),
            )
        )
    ):
        failures.append(
            "capture.json host packaging source-set digest is invalid"
        )
    repository_root = Path(__file__).resolve().parents[1]
    for record in packaging_sources:
        if not isinstance(record, dict):
            continue
        repository_path = record.get("repository_path")
        if not isinstance(repository_path, str):
            continue
        current_path = repository_root / repository_path
        if (
            not current_path.is_file()
            or record.get("bytes") != current_path.stat().st_size
            or record.get("sha256") != sha256_file(current_path)
        ):
            failures.append(
                f"host packaging source does not match current {repository_path}"
            )

    artifacts = _object(
        capture.get("artifacts"),
        "capture.json $.artifacts",
        failures,
    )
    if set(artifacts) != CAPTURE_ARTIFACT_NAMES:
        failures.append(
            "capture.json artifact inventory is incomplete or unexpected"
        )
    for name in sorted(CAPTURE_ARTIFACT_NAMES):
        record = _object(
            artifacts.get(name),
            f"capture.json $.artifacts.{name}",
            failures,
        )
        path = bundle / name
        if not path.is_file():
            continue
        if record.get("sha256") != sha256_file(path):
            failures.append(f"capture.json hash mismatch for {name}")
        if (
            not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record.get("bytes") != path.stat().st_size
        ):
            failures.append(f"capture.json byte count mismatch for {name}")
        if sha256_file(path) not in readme:
            failures.append(f"README omits retained artifact hash for {name}")
    if sha256_file(bundle / "capture.json") not in readme:
        failures.append("README omits capture.json hash")

    if (
        not isinstance(image_inspect, list)
        or len(image_inspect) != 1
        or not isinstance(image_inspect[0], dict)
    ):
        failures.append(
            "image-inspect.json must contain exactly one image object"
        )
        image_record: dict[str, Any] = {}
    else:
        image_record = image_inspect[0]
    if image_record.get("Id") != image_id:
        failures.append(
            "image-inspect.json Id does not match controller provenance"
        )
    config = _object(
        image_record.get("Config"),
        "image-inspect.json $[0].Config",
        failures,
    )
    labels = _object(
        config.get("Labels"),
        "image-inspect.json $[0].Config.Labels",
        failures,
    )
    if labels.get("org.opencontainers.image.revision") != source_revision:
        failures.append(
            "image OCI source-revision label does not match provenance"
        )
    if labels.get("org.airsignrobot.ebim.revision") != benchmark_commit:
        failures.append(
            "image EBiM revision label does not match provenance"
        )
    environment = _string_list(
        config.get("Env"),
        "image-inspect.json $[0].Config.Env",
        failures,
    )
    if f"AIRSIGN_REVISION={source_revision}" not in environment:
        failures.append(
            "image environment lacks the matching AirSign revision"
        )
    if f"EBIM_COMMIT={benchmark_commit}" not in environment:
        failures.append(
            "image environment lacks the matching EBiM revision"
        )

    if smoke.get("passed") is not True:
        failures.append("smoke result did not pass")
    if smoke.get("airsign_revision") != source_revision:
        failures.append(
            "smoke AirSign revision does not match provenance"
        )
    if smoke.get("benchmark_commit") != benchmark_commit:
        failures.append(
            "smoke benchmark revision does not match provenance"
        )
    if controller_result.get("passed") is not metrics.get("passed"):
        failures.append(
            "controller result passed field does not match metrics"
        )
    if (
        controller_result.get("classification")
        != metrics.get("classification")
    ):
        failures.append(
            "controller result classification does not match metrics"
        )
    if (
        controller_result.get("gate_requested")
        != metrics.get("gate_requested")
    ):
        failures.append(
            "controller result gate does not match metrics"
        )
    if (
        controller_result.get("official_stage_completion_claimed")
        is not False
        or controller_result.get("official_score_claimed") is not False
        or controller_result.get("benchmark_score", "missing") is not None
    ):
        failures.append(
            "controller result violates the non-official claim boundary"
        )
    if controller_result.get("files") != manifest.get("files"):
        failures.append(
            "controller result artifact inventory is inconsistent"
        )
    return failures


def package_capture(
    bundle: Path,
    *,
    bundle_name: str,
    gate: str,
    revision: str,
    image: str,
    image_id: str,
) -> None:
    """Generate a host capture manifest and human-readable bundle artifacts."""

    if REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("revision must be a full lowercase Git SHA")
    if SHA256_PATTERN.fullmatch(image_id) is None:
        raise ValueError("image_id must be an immutable sha256 image ID")
    if re.fullmatch(r"[A-Za-z0-9._-]+", bundle_name) is None:
        raise ValueError("bundle_name must be one safe path component")

    metrics = strict_json_load(bundle / "metrics.json", root_type=dict)
    manifest = strict_json_load(bundle / "manifest.json", root_type=dict)
    trajectory = strict_json_load(bundle / "trajectory.json", root_type=dict)
    provenance = metrics.get("provenance")
    if not isinstance(provenance, dict):
        raise EvidenceJSONError(
            f"{bundle / 'metrics.json'}: $.provenance must be an object"
        )
    if (
        type(metrics.get("schema_version")) is not int
        or metrics.get("schema_version") != 2
    ):
        raise EvidenceJSONError("capture packaging requires schema-2 metrics")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 2
    ):
        raise EvidenceJSONError("capture packaging requires schema-2 manifest")
    if (
        type(trajectory.get("schema_version")) is not int
        or trajectory.get("schema_version") != 2
    ):
        raise EvidenceJSONError(
            "capture packaging requires schema-2 trajectory"
        )
    if (
        manifest.get("provenance") != provenance
        or trajectory.get("provenance") != provenance
    ):
        raise EvidenceJSONError(
            "metrics, trajectory, and manifest provenance must be identical"
        )
    expected = {
        "airsign_source_revision": revision,
        "container_image": image,
        "container_image_id": image_id,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise EvidenceJSONError(
                f"controller provenance {key} does not match host capture"
            )
    benchmark_commit = provenance.get("official_benchmark_commit")
    if benchmark_commit != PINNED_EBIM_COMMIT:
        raise EvidenceJSONError(
            "controller provenance does not match the pinned benchmark revision"
        )

    passed = metrics.get("passed")
    if not isinstance(passed, bool):
        raise EvidenceJSONError("metrics.passed must be boolean")
    if metrics.get("gate_requested") != gate:
        raise EvidenceJSONError(
            "metrics.gate_requested does not match the host capture gate"
        )
    if (
        metrics.get("official_stage_complete") is not False
        or metrics.get("official_stage_score", "missing") is not None
        or metrics.get("official_stage_completion_claimed") is not False
        or metrics.get("official_score_claimed") is not False
    ):
        raise EvidenceJSONError(
            "metrics violates the non-official Stage 1 claim boundary"
        )
    if (
        trajectory.get("official_stage_completion_claimed") is not False
        or trajectory.get("official_score_claimed") is not False
        or trajectory.get("task_objects_teleported") is not False
        or trajectory.get("robot_links_teleported") is not False
        or trajectory.get("task_object_mutation_api_used") is not False
    ):
        raise EvidenceJSONError(
            "trajectory violates the physical evidence claim boundary"
        )
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise EvidenceJSONError("manifest.files must be an object")
    for name in ("metrics.json", "trajectory.json"):
        record = manifest_files.get(name)
        path = bundle / name
        if not isinstance(record, dict):
            raise EvidenceJSONError(
                f"manifest.files.{name} must be an object"
            )
        if (
            record.get("sha256") != sha256_file(path)
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record.get("bytes") != path.stat().st_size
        ):
            raise EvidenceJSONError(
                f"manifest file binding is invalid for {name}"
            )
    try:
        exit_text = (bundle / "container.exit").read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, UnicodeError) as error:
        raise EvidenceJSONError(
            f"container.exit is unreadable: {error}"
        ) from error
    if re.fullmatch(r"[01]", exit_text) is None:
        raise EvidenceJSONError("container.exit must be exactly 0 or 1")
    if int(exit_text) != (0 if passed else 1):
        raise EvidenceJSONError(
            "container.exit does not match metrics.passed"
        )
    status = "passed" if passed else "failed"
    index_entry = (
        f"- [`{bundle_name}`]({bundle_name}/README.md): retained schema-2 "
        f"`{gate}` {status} development diagnostic, bound to AirSign "
        f"`{revision}` and immutable image `{image_id}`. It is not official "
        "completion or score evidence.\n"
    )
    (bundle / "INDEX_ENTRY.md").write_text(index_entry, encoding="utf-8")

    artifacts = {
        name: artifact_record(bundle / name)
        for name in sorted(CAPTURE_ARTIFACT_NAMES)
    }
    repository_root = Path(__file__).resolve().parents[1]
    host_packaging_sources = [
        {
            "repository_path": repository_path,
            **artifact_record(repository_root / repository_path),
        }
        for repository_path in sorted(HOST_PACKAGING_SOURCES)
    ]
    capture = {
        "schema_version": 1,
        "kind": "airsign_stage1_host_capture_manifest",
        "classification": "retained_non_official_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_name": bundle_name,
        "gate": gate,
        "airsign_source_revision": revision,
        "official_benchmark_commit": benchmark_commit,
        "container_image": image,
        "container_image_id": image_id,
        "host_packaging_sources": host_packaging_sources,
        "host_packaging_source_set_sha256": sha256_json(
            host_packaging_sources
        ),
        "artifacts": artifacts,
    }
    capture_path = bundle / "capture.json"
    capture_path.write_text(
        json.dumps(capture, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    readme_records = {
        **artifacts,
        "capture.json": artifact_record(capture_path),
    }
    rows = "\n".join(
        f"- `{name}`: `{record['sha256']}` ({record['bytes']} bytes)"
        for name, record in sorted(readme_records.items())
    )
    readme = f"""# `{bundle_name}` — retained Stage 1 diagnostic

Status: **{status} development diagnostic; not official completion or score evidence**.

This bundle was captured by `scripts/capture_stage1.sh` from a clean AirSign
worktree. The host built the image with the full source revision, ran the
container smoke test and `{gate}` gate by immutable image ID, and retained the
image inspection plus complete logs. `capture.json` binds the host-retained
artifacts; the controller's `manifest.json` separately binds its metrics,
trajectory, executed sources, benchmark revision, and image identity.

- AirSign revision: `{revision}`
- Official benchmark revision: `{benchmark_commit}`
- Container image label: `{image}`
- Immutable container image ID: `{image_id}`
- `official_stage_complete`: `false`
- `official_stage_score`: `null`

This is a participant development diagnostic. It is not an organizer evaluation
and must not be presented as a rulebook task outcome or benchmark score.

Artifact SHA-256 values:

{rows}

To add the bundle to the repository evidence index, review and copy the single
bullet from `INDEX_ENTRY.md` into the parent
`evidence/stage1-physical-development/README.md`, then run
`python3 scripts/validate_submission.py`.
"""
    (bundle / "README.md").write_text(readme, encoding="utf-8")

    failures = validate_schema2_capture(
        bundle,
        provenance,
        expected_bundle_name=bundle_name,
    )
    if failures:
        raise EvidenceJSONError(
            "packaged capture failed validation: " + "; ".join(failures)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        package_capture(
            args.bundle,
            bundle_name=args.bundle_name,
            gate=args.gate,
            revision=args.revision,
            image=args.image,
            image_id=args.image_id,
        )
    except (EvidenceJSONError, OSError, UnicodeError, ValueError) as error:
        print(f"CAPTURE_PACKAGING_ERROR {error}")
        return 2
    print(f"CAPTURE_PACKAGING_OK {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
