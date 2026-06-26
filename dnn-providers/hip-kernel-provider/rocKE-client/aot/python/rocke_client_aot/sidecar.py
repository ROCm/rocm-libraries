# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sidecar metadata for provider-owned rocKE AOT artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from rocke.core.arch import ArchTarget
from rocke.instances.common.fmha_mfma import fmha_fwd_mfma_signature

SIDECAR_SCHEMA = "ck.rocke.aot.sidecar/v1"
ABI_VERSION = "hipkg-sdpa-fwd-fmha-mfma/v1"
CANDIDATE_FMHA_FWD_MFMA = "fmha_fwd_mfma"
ALGORITHM_DENSE_FMHA_FWD = "dense_fmha_fwd"

_POINTER_SIZE_BYTES = 8
_POINTER_ALIGNMENT = 8
_SCALAR_ABI = {
    "f32": (4, 4),
    "i32": (4, 4),
}

__all__ = [
    "SIDECAR_SCHEMA",
    "canonical_hash",
    "emit_sidecar",
    "enrich_args_signature",
]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Return a SHA256 hash over canonical JSON."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{context} must be a mapping")


def _instance_data(instance: Any) -> Mapping[str, Any]:
    if isinstance(instance, Mapping):
        return instance
    data = getattr(instance, "data", None)
    if data is None:
        data = getattr(instance, "instance", None)
    return _as_mapping(data, "instance data")


def _external_dtype(dtype: Any) -> str:
    if dtype in ("f16", "fp16", "half"):
        return "fp16"
    raise ValueError(f"unsupported dtype for SDPA AOT sidecar: {dtype!r}")


def _spec_id(dtype: str, layout: str, block_size_q: int, block_size_k: int) -> str:
    return f"{dtype}_{layout.lower()}_blockq{block_size_q}_blockk{block_size_k}"


def _cache_key(kernel_id: Mapping[str, Any]) -> str:
    return ":".join(
        str(kernel_id[field])
        for field in (
            "op",
            "family",
            "candidate",
            "algorithm",
            "spec_id",
            "arch",
            "request_hash",
            "spec_hash",
        )
    )


def _signature_kind(type_text: str) -> str:
    return "pointer" if type_text.startswith("ptr<") else "scalar"


def _signature_size_and_alignment(type_text: str) -> tuple[int, int]:
    if type_text.startswith("ptr<"):
        return _POINTER_SIZE_BYTES, _POINTER_ALIGNMENT
    try:
        return _SCALAR_ABI[type_text]
    except KeyError as exc:
        raise ValueError(f"unsupported scalar ABI type {type_text!r}") from exc


def enrich_args_signature(spec: Any) -> list[dict[str, Any]]:
    """Add ABI kind, size, and alignment to ``fmha_fwd_mfma_signature``."""

    enriched: list[dict[str, Any]] = []
    for item in fmha_fwd_mfma_signature(spec):
        entry = _as_mapping(item, "signature entry")
        name = entry.get("name")
        type_text = entry.get("type")
        if not isinstance(name, str) or not isinstance(type_text, str):
            raise ValueError("signature entries must contain string name and type")
        kind = _signature_kind(type_text)
        size_bytes, alignment = _signature_size_and_alignment(type_text)
        enriched.append(
            {
                "name": name,
                "type": type_text,
                "kind": kind,
                "size_bytes": size_bytes,
                "alignment": alignment,
            }
        )

    expected_prefix = ["Q", "K", "V", "O"]
    actual_prefix = [entry["name"] for entry in enriched[:4]]
    if actual_prefix != expected_prefix:
        raise ValueError(
            "FMHA sidecar ABI must start with Q/K/V/O; " f"got {actual_prefix!r}"
        )
    for entry in enriched[:4]:
        if entry["type"] != "ptr<f16, global>":
            raise ValueError(
                f"FMHA tensor pointer {entry['name']} has unexpected type "
                f"{entry['type']!r}"
            )
    return enriched


def emit_sidecar(
    instance: Any,
    spec: Any,
    artifact: Any,
    hsaco_filename: str,
) -> dict[str, Any]:
    """Build sidecar metadata for one compiled SDPA FMHA AOT artifact."""

    data = _instance_data(instance)
    compile_spec = _as_mapping(data.get("compile_spec"), "compile_spec")
    selection = _as_mapping(data.get("selection", {}), "selection")
    shape = spec.common.shape

    op = data.get("op")
    family = data.get("family")
    arch = data.get("arch")
    if not all(isinstance(value, str) for value in (op, family, arch)):
        raise ValueError("instance op, family, and arch must be strings")
    if op != "sdpa_fwd" or family != CANDIDATE_FMHA_FWD_MFMA:
        raise ValueError(f"unsupported sidecar kernel id {op!r}/{family!r}")

    dtype = _external_dtype(compile_spec.get("dtype"))
    layout = str(compile_spec.get("canonical_layout"))
    block_size_q = int(compile_spec.get("block_size_q", shape.block_size_q))
    block_size_k = int(compile_spec.get("block_size_k", shape.block_size_k))
    head_size = int(compile_spec.get("head_size", shape.head_size))
    seqlen_q = int(compile_spec.get("seqlen_q", spec.seqlen_q))
    seqlen_k = int(compile_spec.get("seqlen_k", spec.seqlen_k))
    num_query_heads = int(compile_spec.get("num_query_heads", shape.num_query_heads))
    num_kv_heads = int(compile_spec.get("num_kv_heads", shape.num_kv_heads))
    mask_mode = str(compile_spec.get("mask_mode", spec.common.mask_mode))

    request_document = dict(data)
    spec_document = {
        "dtype": dtype,
        "canonical_layout": layout,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "block_size_q": block_size_q,
        "block_size_k": block_size_k,
        "mask_mode": mask_mode,
    }
    request_hash = canonical_hash(request_document)
    spec_hash = canonical_hash(spec_document)

    kernel_id = {
        "op": op,
        "family": family,
        "candidate": CANDIDATE_FMHA_FWD_MFMA,
        "algorithm": ALGORITHM_DENSE_FMHA_FWD,
        "spec_id": _spec_id(dtype, layout, block_size_q, block_size_k),
        "arch": arch,
        "abi_version": ABI_VERSION,
        "request_hash": request_hash,
        "spec_hash": spec_hash,
    }
    kernel_id["cache_key"] = _cache_key(kernel_id)

    hsaco = getattr(artifact, "hsaco")
    symbol = getattr(artifact, "kernel_name")
    wave_size = ArchTarget.from_gfx(arch).wave_size
    batch_constraint = selection.get("batch", {})
    attribute_constraints = selection.get("attribute_constraints", {})

    return {
        "schema": SIDECAR_SCHEMA,
        "kernel_id": kernel_id,
        "artifact": {
            "hsaco_filename": hsaco_filename,
            "symbol": symbol,
            "hsaco_sha256": hashlib.sha256(hsaco).hexdigest(),
            "hsaco_size": len(hsaco),
        },
        "selection": {
            "op": op,
            "arch": arch,
            "dtypes": {
                "q": dtype,
                "k": dtype,
                "v": dtype,
                "o": dtype,
                "acc": "fp32",
            },
            "canonical_layout": layout,
            "shape_constraints": {
                "batch": dict(_as_mapping(batch_constraint, "selection.batch")),
                "seqlen_q": {"equals": seqlen_q, "multiple_of": 16},
                "seqlen_k": {"equals": seqlen_k, "multiple_of": 16},
                "num_query_heads": {"equals": num_query_heads},
                "num_kv_heads": {"equals": num_kv_heads},
                "head_size": {"equals": head_size},
            },
            "attribute_constraints": dict(
                _as_mapping(attribute_constraints, "selection.attribute_constraints")
            ),
        },
        "launch": {
            "shared_mem_bytes": 0,
            "grid_formula": {
                "x": {"ceil_div": ["seqlen_q", block_size_q]},
                "y": "num_query_heads",
                "z": "batch",
            },
            "block": [wave_size, 1, 1],
            "tile_sizes": {
                "block_q": block_size_q,
                "block_k": block_size_k,
                "head_size": head_size,
                "wave_size": wave_size,
            },
        },
        "args_signature": enrich_args_signature(spec),
    }
