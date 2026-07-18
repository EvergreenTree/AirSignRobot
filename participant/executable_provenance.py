#!/usr/bin/env python3
"""Import-safe hashing and AST guards for executed participant sources."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable_source_guard(
    paths: Iterable[Path],
    *,
    root: Path,
    forbidden_attribute_calls: Iterable[str],
    forbidden_direct_calls: Iterable[str],
) -> dict[str, Any]:
    """Hash and AST-scan the complete participant source set."""

    resolved_root = root.resolve()
    attributes = frozenset(str(value) for value in forbidden_attribute_calls)
    direct = frozenset(str(value) for value in forbidden_direct_calls)
    records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for supplied_path in paths:
        path = supplied_path.resolve()
        try:
            relative = path.relative_to(resolved_root).as_posix()
        except ValueError:
            raise ValueError(
                f"executed source is outside the participant root: {path}"
            ) from None
        if relative in seen:
            raise ValueError(f"duplicate executed source path: {relative}")
        seen.add(relative)
        if path.suffix != ".py" or not path.is_file():
            raise ValueError(
                f"executed source must be an existing Python file: {relative}"
            )
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        call_count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_count += 1
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in attributes
            ):
                violations.append(
                    {
                        "path": relative,
                        "line": int(node.lineno),
                        "call": node.func.attr,
                        "kind": "forbidden_attribute_call",
                    }
                )
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id in direct
            ):
                violations.append(
                    {
                        "path": relative,
                        "line": int(node.lineno),
                        "call": node.func.id,
                        "kind": "forbidden_dynamic_call",
                    }
                )
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "ast_call_count": call_count,
            }
        )
    if not records:
        raise ValueError("executed source set must not be empty")
    records.sort(key=lambda record: str(record["path"]))
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "passed": not violations,
        "source_set_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "source_count": len(records),
        "sources": records,
        "ast_call_count": sum(
            int(record["ast_call_count"]) for record in records
        ),
        "forbidden_attribute_calls": sorted(attributes),
        "forbidden_direct_calls": sorted(direct),
        "violations": sorted(
            violations,
            key=lambda record: (
                str(record["path"]),
                int(record["line"]),
                str(record["call"]),
            ),
        ),
    }
