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


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_EVIDENCE = ROOT / "evidence" / "four-stage-rehearsal"
STAGE1_EVIDENCE = ROOT / "evidence" / "stage1-physical-development"
FAILURES: list[str] = []
PASSED: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def pass_check(message: str) -> None:
    PASSED.append(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        record = json.loads(record_line.removeprefix(prefix))
    except json.JSONDecodeError as error:
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
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            FAILURES.append(f"Invalid evidence JSON in {path.relative_to(ROOT)}: {error}")
            continue
        walk_claims(document, "$", path)

    metrics = json.loads(
        (CANONICAL_EVIDENCE / "metrics.json").read_text(encoding="utf-8")
    )
    controller = ROOT / "participant" / "four_stage_rehearsal.py"
    check(
        metrics.get("provenance", {}).get("controller_sha256")
        == sha256_file(controller),
        "Canonical metrics controller hash does not match four_stage_rehearsal.py",
    )
    check(
        (CANONICAL_EVIDENCE / "controller.exit").read_text(encoding="utf-8").strip()
        == "0",
        "Canonical controller exit status is not zero",
    )
    development = json.loads(
        (ROOT / "evidence" / "development_integration_validation.json").read_text(
            encoding="utf-8"
        )
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
    index = (
        index_path.read_text(encoding="utf-8")
        if index_path.is_file()
        else ""
    )
    bundle_dirs = sorted(
        path.parent for path in STAGE1_EVIDENCE.glob("*/manifest.json")
    )
    check(bundle_dirs, "No Stage 1 diagnostic evidence bundles were found")

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
            manifest = json.loads(
                required["manifest.json"].read_text(encoding="utf-8")
            )
            metrics = json.loads(
                required["metrics.json"].read_text(encoding="utf-8")
            )
            trajectory = json.loads(
                required["trajectory.json"].read_text(encoding="utf-8")
            )
            exit_status = int(
                required["container.exit"]
                .read_text(encoding="utf-8")
                .strip()
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            FAILURES.append(f"Invalid Stage 1 bundle {relative}: {error}")
            continue

        readme = required["README.md"].read_text(encoding="utf-8")
        computed_hashes = {
            name: sha256_file(path) for name, path in required.items()
        }
        manifest_files = manifest.get("files", {})
        for name in ("metrics.json", "trajectory.json"):
            record = manifest_files.get(name, {})
            check(
                record.get("sha256") == computed_hashes[name],
                f"{relative} manifest hash mismatch for {name}",
            )
            check(
                record.get("bytes") == required[name].stat().st_size,
                f"{relative} manifest byte count mismatch for {name}",
            )

        provenance = metrics.get("provenance", {})
        controller_sha = provenance.get("controller_sha256")
        check(
            isinstance(controller_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", controller_sha) is not None,
            f"{relative} has no valid controller provenance hash",
        )
        check(
            manifest.get("controller_sha256") == controller_sha
            and trajectory.get("provenance", {}).get("controller_sha256")
            == controller_sha
            and metrics.get("mutation_guard", {}).get("controller_sha256")
            == controller_sha,
            f"{relative} controller provenance is internally inconsistent",
        )
        if bundle.name == "cup-image-b":
            check(
                controller_sha
                == sha256_file(
                    ROOT / "participant" / "stage1_table_setup.py"
                ),
                (
                    f"{relative} final diagnostic controller hash does not "
                    "match participant/stage1_table_setup.py"
                ),
            )
        check(
            manifest.get("official_benchmark_commit")
            == provenance.get("official_benchmark_commit")
            == trajectory.get("provenance", {}).get(
                "official_benchmark_commit"
            ),
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
    ignore_lines = {
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    check("*" in ignore_lines, ".dockerignore must default-deny the build context")
    for required in {
        "**/.env",
        "**/.env.*",
        "**/*credential*",
        "**/*secret*",
        "**/*token*",
        "**/gcp.csv",
    }:
        check(required in ignore_lines, f".dockerignore is missing {required}")
    if len(FAILURES) == before:
        pass_check("Dockerfile pinning, entrypoint, and credential-safe build context")


def validate_stage1_development_contract() -> None:
    controller = ROOT / "participant" / "stage1_table_setup.py"
    entrypoint = (ROOT / "scripts" / "airsign").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    before = len(FAILURES)
    check(controller.is_file(), "Stage 1 development controller is missing")
    if not controller.is_file():
        return

    source = controller.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(controller))
    forbidden_attributes = {
        "set_world_pose",
        "set_local_pose",
        "set_default_state",
        "set_linear_velocity",
        "set_angular_velocity",
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
        '"official_stage_complete": False',
        '"official_stage_score": None',
        '"task_objects_teleported": False',
        '"robot_links_teleported": False',
        '"task_object_mutation_api_used": False',
        "raw Lula IK failed; no failed solution was applied",
        "collision/obstruction safety abort",
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
        "stage1-table-setup)" in entrypoint
        and "STAGE1_TABLE_SETUP_RESULT" in entrypoint,
        "Container entrypoint does not expose and verify stage1-table-setup",
    )
    check(
        "stage1_table_setup.py" in dockerfile,
        "Dockerfile does not package the Stage 1 development controller",
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
        "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
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
