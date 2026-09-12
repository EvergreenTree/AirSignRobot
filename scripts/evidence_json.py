#!/usr/bin/env python3
"""Strict, dependency-free JSON helpers for retained evidence artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class EvidenceJSONError(ValueError):
    """Raised when an evidence JSON document is not strict, finite JSON."""


def _reject_constant(value: str) -> None:
    raise EvidenceJSONError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceJSONError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _require_finite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceJSONError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite(child, f"{location}[{index}]")


def strict_json_loads(
    text: str,
    *,
    source: str = "<string>",
    root_type: type | tuple[type, ...] | None = None,
) -> Any:
    """Decode strict JSON, rejecting duplicates and every non-finite number."""

    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        _require_finite(value)
    except (json.JSONDecodeError, EvidenceJSONError, RecursionError) as error:
        raise EvidenceJSONError(f"{source}: {error}") from error
    if root_type is not None and not isinstance(value, root_type):
        if isinstance(root_type, tuple):
            expected = " or ".join(item.__name__ for item in root_type)
        else:
            expected = root_type.__name__
        raise EvidenceJSONError(
            f"{source}: root must be {expected}, got {type(value).__name__}"
        )
    return value


def strict_json_load(
    path: Path,
    *,
    root_type: type | tuple[type, ...] | None = dict,
) -> Any:
    """Read and decode one strict evidence JSON document."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EvidenceJSONError(f"{path}: {error}") from error
    return strict_json_loads(
        text,
        source=str(path),
        root_type=root_type,
    )
