# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""AOT instance field parser and artifact metadata for SDPA FMHA MFMA."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rocke.core.arch import ArchTarget
from kernels.common.fmha_mfma import fmha_fwd_mfma_signature
from rocke_client_aot.instance_schema import (
    InstanceError,
    require_int,
    require_mapping,
    require_string,
)
from rocke_client_aot.sidecar import canonical_hash, make_sidecar

OP = "sdpa_fwd"
FAMILY = "fmha_fwd_mfma"
LAYOUT_BSHD = "BSHD"
MASK_MODE_NONE = "none"
DTYPE_FP16 = "fp16"
DTYPE_F16 = "f16"
ABI_VERSION = "hipkg-sdpa-fwd-fmha-mfma/v1"
ALGORITHM_DENSE_FMHA_FWD = "dense_fmha_fwd"

_DTYPE_ALIASES = {
    "fp16": DTYPE_F16,
    "f16": DTYPE_F16,
    "half": DTYPE_F16,
}

_EXTERNAL_DTYPE = {
    DTYPE_F16: DTYPE_FP16,
}

_COMPILE_FIELDS = (
    "dtype",
    "canonical_layout",
    "seqlen_q",
    "seqlen_k",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
    "block_size_q",
    "block_size_k",
    "mask_mode",
)
_POINTER_SIZE_BYTES = 8
_POINTER_ALIGNMENT = 8
_SCALAR_ABI = {
    "f32": (4, 4),
    "i32": (4, 4),
}
_FIXED_BLOCK_SIZE_Q = 16


def parse_instance_fields(
    instance: Mapping[str, Any], source: Path
) -> tuple[dict[str, Any], Any, str]:
    """Validate SDPA FMHA MFMA fields and build the rocKE spec."""

    compile_spec = _validate_compile_spec(
        instance.get("compile_spec", {}), context="compile_spec"
    )
    _validate_shape_constraints(compile_spec)
    _validate_instance_name(instance, compile_spec)
    normalized = {
        "compile_spec": compile_spec,
        "selection": dict(require_mapping(instance.get("selection"), "selection")),
        "test_profiles": list(instance.get("test_profiles", [])),
    }

    spec = build_fmha_mfma_spec(compile_spec)

    from kernels.common.fmha_mfma import is_valid_spec

    arch = require_string(instance["arch"], "instance arch")
    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        raise InstanceError(f"invalid FmhaMfmaSpec for {arch}: {reason}")
    return normalized, spec, reason


def build_kernel(spec: Any, *, arch: str) -> Any:
    """Build the rocKE kernel for this checked-in SDPA FMHA MFMA spec."""

    from kernels.common.fmha_mfma import build_fmha_fwd_mfma

    return build_fmha_fwd_mfma(spec, arch=arch)


def build_fmha_mfma_spec(compile_spec: Mapping[str, Any]) -> Any:
    """Build the rocKE ``FmhaMfmaSpec`` for a normalized compile spec."""

    from kernels import FmhaCommonSpec, FmhaShape
    from kernels.common.fmha_mfma import FmhaMfmaSpec

    dtype = normalize_dtype(compile_spec["dtype"])
    common = FmhaCommonSpec(
        FmhaShape(
            head_size=require_int(compile_spec["head_size"], "compile_spec.head_size"),
            num_query_heads=require_int(
                compile_spec["num_query_heads"], "compile_spec.num_query_heads"
            ),
            num_kv_heads=require_int(
                compile_spec["num_kv_heads"], "compile_spec.num_kv_heads"
            ),
            block_size_q=require_int(
                compile_spec["block_size_q"], "compile_spec.block_size_q"
            ),
            block_size_k=require_int(
                compile_spec["block_size_k"], "compile_spec.block_size_k"
            ),
        ),
        dtype=dtype,
        mask_mode=require_string(compile_spec["mask_mode"], "compile_spec.mask_mode"),
    )
    return FmhaMfmaSpec(
        common=common,
        seqlen_q=require_int(compile_spec["seqlen_q"], "compile_spec.seqlen_q"),
        seqlen_k=require_int(compile_spec["seqlen_k"], "compile_spec.seqlen_k"),
    )


def normalize_dtype(dtype: Any) -> str:
    """Return the rocKE-internal dtype spelling accepted by FMHA specs."""

    text = require_string(dtype, "dtype").lower()
    try:
        return _DTYPE_ALIASES[text]
    except KeyError as exc:
        raise InstanceError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_DTYPE_ALIASES)}"
        ) from exc


def external_dtype(dtype: Any) -> str:
    """Return the provider-facing dtype spelling used by checked-in instances."""

    return _EXTERNAL_DTYPE[normalize_dtype(dtype)]


def instance_name(envelope: Mapping[str, Any], compile_spec: Mapping[str, Any]) -> str:
    """Return the canonical artifact basename for an SDPA FMHA MFMA instance."""

    dtype = external_dtype(compile_spec["dtype"])
    layout = require_string(
        compile_spec["canonical_layout"], "compile_spec.canonical_layout"
    ).lower()
    return (
        f"{envelope['op']}_{envelope['family']}_{dtype}_{layout}_{envelope['arch']}"
        f"_q{require_int(compile_spec['seqlen_q'], 'compile_spec.seqlen_q')}"
        f"_k{require_int(compile_spec['seqlen_k'], 'compile_spec.seqlen_k')}"
        f"_hq{require_int(compile_spec['num_query_heads'], 'compile_spec.num_query_heads')}"
        f"_hkv{require_int(compile_spec['num_kv_heads'], 'compile_spec.num_kv_heads')}"
        f"_d{require_int(compile_spec['head_size'], 'compile_spec.head_size')}"
        f"_{require_string(compile_spec['mask_mode'], 'compile_spec.mask_mode')}"
    )


def _validate_instance_name(
    envelope: Mapping[str, Any], compile_spec: Mapping[str, Any]
) -> None:
    """Ensure the declared instance name matches the canonical basename."""
    name = require_string(envelope.get("name"), "instance name")
    expected = instance_name(envelope, compile_spec)
    if name != expected:
        raise InstanceError(
            "instance name " f"{name!r} must match SDPA FMHA MFMA basename {expected!r}"
        )


def _validate_compile_spec(spec: Any, *, context: str) -> dict[str, Any]:
    """Normalize and validate compile-time fields for an SDPA FMHA instance."""

    data = dict(require_mapping(spec, context))
    data["dtype"] = external_dtype(data.get("dtype"))
    layout = require_string(data.get("canonical_layout"), f"{context}.canonical_layout")
    if layout != LAYOUT_BSHD:
        raise InstanceError(
            f"{context}.canonical_layout must be {LAYOUT_BSHD!r}, got {layout!r}"
        )
    data["canonical_layout"] = layout
    mask_mode = require_string(data.get("mask_mode"), f"{context}.mask_mode")
    if mask_mode != MASK_MODE_NONE:
        raise InstanceError(
            f"{context}.mask_mode must be {MASK_MODE_NONE!r}, got {mask_mode!r}"
        )
    data["mask_mode"] = mask_mode

    for field in (
        "seqlen_q",
        "seqlen_k",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "block_size_q",
        "block_size_k",
    ):
        value = require_int(data.get(field), f"{context}.{field}")
        if value <= 0:
            raise InstanceError(f"{context}.{field} must be > 0, got {value}")
        data[field] = value

    missing = [field for field in _COMPILE_FIELDS if field not in data]
    if missing:
        raise InstanceError(
            f"{context} is missing required fields: {', '.join(missing)}"
        )
    return {field: data[field] for field in _COMPILE_FIELDS}


def _validate_shape_constraints(compile_spec: Mapping[str, Any]) -> None:
    """Validate shape relationships required by SDPA FMHA MFMA kernels."""

    seqlen_q = require_int(compile_spec["seqlen_q"], "compile_spec.seqlen_q")
    seqlen_k = require_int(compile_spec["seqlen_k"], "compile_spec.seqlen_k")
    head_size = require_int(compile_spec["head_size"], "compile_spec.head_size")
    num_query_heads = require_int(
        compile_spec["num_query_heads"], "compile_spec.num_query_heads"
    )
    num_kv_heads = require_int(
        compile_spec["num_kv_heads"], "compile_spec.num_kv_heads"
    )
    if seqlen_q % 16:
        raise InstanceError(
            f"compile_spec.seqlen_q ({seqlen_q}) must be divisible by 16"
        )
    if seqlen_k % 16:
        raise InstanceError(
            f"compile_spec.seqlen_k ({seqlen_k}) must be divisible by 16"
        )
    if head_size not in (32, 64, 128, 192, 256):
        raise InstanceError(
            "compile_spec.head_size "
            f"({head_size}) must be one of 32, 64, 128, 192, 256"
        )
    block_size_q = require_int(
        compile_spec["block_size_q"], "compile_spec.block_size_q"
    )
    if block_size_q != _FIXED_BLOCK_SIZE_Q:
        raise InstanceError(
            f"compile_spec.block_size_q must be {_FIXED_BLOCK_SIZE_Q}, "
            f"got {block_size_q}"
        )
    if num_query_heads % num_kv_heads:
        raise InstanceError(
            f"compile_spec.num_query_heads ({num_query_heads}) must be divisible by "
            f"compile_spec.num_kv_heads ({num_kv_heads})"
        )


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Return a value as a mapping or fail with contextual type information."""

    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{context} must be a mapping")


def _instance_data(instance: Any) -> Mapping[str, Any]:
    """Extract the normalized instance document from parser output."""

    if isinstance(instance, Mapping):
        return instance
    data = getattr(instance, "data", None)
    if data is None:
        data = getattr(instance, "instance", None)
    return _as_mapping(data, "instance data")


def _spec_id(dtype: str, layout: str, block_size_q: int, block_size_k: int) -> str:
    """Build the stable spec identifier used in sidecar kernel IDs."""

    return f"{dtype}_{layout.lower()}_blockq{block_size_q}_blockk{block_size_k}"


def _cache_key(kernel_id: Mapping[str, Any]) -> str:
    """Build the deterministic cache key for a sidecar kernel ID."""

    return ":".join(
        str(kernel_id[field])
        for field in (
            "op",
            "family",
            "candidate",
            "algorithm",
            "spec_id",
            "arch",
            "abi_version",
            "request_hash",
            "spec_hash",
        )
    )


def _signature_kind(type_text: str) -> str:
    """Classify a signature type as a pointer or scalar argument."""

    return "pointer" if type_text.startswith("ptr<") else "scalar"


def _signature_size_and_alignment(type_text: str) -> tuple[int, int]:
    """Return ABI size and alignment metadata for a signature type."""

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
    selection = _as_mapping(data.get("selection"), "selection")
    shape = spec.common.shape

    op = require_string(data.get("op"), "instance op")
    family = require_string(data.get("family"), "instance family")
    arch = require_string(data.get("arch"), "instance arch")
    if op != OP or family != FAMILY:
        raise ValueError(f"unsupported sidecar kernel id {op!r}/{family!r}")

    dtype = external_dtype(compile_spec.get("dtype"))
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
        "candidate": FAMILY,
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
    attribute_constraints = _as_mapping(
        selection.get("attribute_constraints"), "selection.attribute_constraints"
    )

    return make_sidecar(
        cache_key=kernel_id["cache_key"],
        artifact={
            "hsaco_filename": hsaco_filename,
            "symbol": symbol,
            "hsaco_sha256": hashlib.sha256(hsaco).hexdigest(),
            "hsaco_size": len(hsaco),
        },
        selection={
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
            "attribute_constraints": dict(attribute_constraints),
        },
        launch={
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
        args_signature=enrich_args_signature(spec),
    )
