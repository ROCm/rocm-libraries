# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Common JSON helpers for rocKE client AOT sidecars."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes for hashing sidecar sub-documents."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Return a SHA256 hash over canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


SIDECAR_SCHEMA = "rocke.aot.sidecar/v1"


def make_sidecar(
    *,
    cache_key: str,
    artifact: Mapping[str, Any],
    selection: Mapping[str, Any],
    launch: Mapping[str, Any],
    args_signature: Sequence[Mapping[str, Any]],
    schema: str = SIDECAR_SCHEMA,
) -> dict[str, Any]:
    """Return the common sidecar envelope with operation-specific entries."""

    return {
        "schema": schema,
        "cache_key": cache_key,
        "artifact": dict(artifact),
        "selection": dict(selection),
        "launch": dict(launch),
        "args_signature": [dict(arg) for arg in args_signature],
    }


__all__ = ["SIDECAR_SCHEMA", "canonical_hash", "canonical_json_bytes", "make_sidecar"]
