# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Canonical JSON serialization for LDS conflict results."""

from __future__ import annotations

import json
from typing import Any

from .model import LdsConflictResult, ModelValidationError


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def dumps(result: LdsConflictResult) -> str:
    """Return a deterministic, compact JSON representation of *result*."""

    if not isinstance(result, LdsConflictResult):
        raise TypeError("result must be an LdsConflictResult")
    return json.dumps(
        result.as_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def loads(document: str | bytes | bytearray) -> LdsConflictResult:
    """Parse and strictly validate an LDS conflict result JSON document."""

    if not isinstance(document, (str, bytes, bytearray)):
        raise TypeError("document must be str, bytes, or bytearray")
    try:
        data = json.loads(document, object_pairs_hook=_reject_duplicate_keys)
    except ModelValidationError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelValidationError(f"invalid JSON document: {exc}") from exc
    return LdsConflictResult.from_dict(data)
