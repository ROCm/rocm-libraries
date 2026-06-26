# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parse checked-in rocKE AOT instance descriptions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSTANCE_SCHEMA = "ck.rocke.aot.instance/v1"
OP_SDPA_FWD = "sdpa_fwd"
FAMILY_FMHA_FWD_MFMA = "fmha_fwd_mfma"
LAYOUT_BSHD = "BSHD"
MASK_MODE_NONE = "none"
DTYPE_FP16 = "fp16"
DTYPE_F16 = "f16"

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


class InstanceError(ValueError):
    """Raised when a checked-in AOT instance is invalid."""


@dataclass(frozen=True)
class ParsedInstance:
    """Normalized checked-in instance plus the rocKE FMHA MFMA spec."""

    path: Path
    data: Mapping[str, Any]
    spec: Any
    validation_reason: str


def parse_instance(path: str | Path) -> ParsedInstance:
    """Load one checked-in ``.instance.json`` and build ``FmhaMfmaSpec``."""

    instance_path = Path(path)
    data = _load_instance(instance_path)
    _validate_instance_header(data, instance_path)
    compile_spec = _validate_compile_spec(
        data.get("compile_spec", {}), context="compile_spec"
    )
    _validate_shape_constraints(compile_spec)
    _validate_instance_name(data, compile_spec, instance_path)
    normalized = dict(data)
    normalized["compile_spec"] = compile_spec
    spec = build_fmha_mfma_spec(compile_spec)

    from rocke.instances.common.fmha_mfma import is_valid_spec

    arch = require_string(normalized["arch"], "instance arch")
    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        raise InstanceError(f"invalid FmhaMfmaSpec for {arch}: {reason}")
    return ParsedInstance(
        path=instance_path,
        data=normalized,
        spec=spec,
        validation_reason=reason,
    )


def build_fmha_mfma_spec(compile_spec: Mapping[str, Any]) -> Any:
    """Build the rocKE ``FmhaMfmaSpec`` for a normalized compile spec."""

    from rocke.instances import FmhaCommonSpec, FmhaShape
    from rocke.instances.common.fmha_mfma import FmhaMfmaSpec

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


def instance_name(
    op: str, family: str, arch: str, compile_spec: Mapping[str, Any]
) -> str:
    """Return the canonical artifact basename for a checked-in instance."""

    dtype = external_dtype(compile_spec["dtype"])
    layout = require_string(
        compile_spec["canonical_layout"], "compile_spec.canonical_layout"
    ).lower()
    return (
        f"{op}_{family}_{dtype}_{layout}_{arch}"
        f"_q{require_int(compile_spec['seqlen_q'], 'compile_spec.seqlen_q')}"
        f"_k{require_int(compile_spec['seqlen_k'], 'compile_spec.seqlen_k')}"
        f"_hq{require_int(compile_spec['num_query_heads'], 'compile_spec.num_query_heads')}"
        f"_hkv{require_int(compile_spec['num_kv_heads'], 'compile_spec.num_kv_heads')}"
        f"_d{require_int(compile_spec['head_size'], 'compile_spec.head_size')}"
        f"_{require_string(compile_spec['mask_mode'], 'compile_spec.mask_mode')}"
    )


def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InstanceError(f"{context} must be an object")
    return value


def require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise InstanceError(f"{context} must be a non-empty string")
    return value


def require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InstanceError(f"{context} must be an integer")
    return value


def _load_instance(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as exc:
        raise InstanceError(f"failed to read instance {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InstanceError(f"failed to parse instance {path}: {exc}") from exc
    return dict(require_mapping(value, "instance file"))


def _validate_instance_header(data: Mapping[str, Any], path: Path) -> None:
    schema = data.get("schema")
    if schema != INSTANCE_SCHEMA:
        raise InstanceError(
            f"instance schema must be {INSTANCE_SCHEMA!r}, got {schema!r}"
        )
    op = data.get("op")
    if op != OP_SDPA_FWD:
        raise InstanceError(f"instance op must be {OP_SDPA_FWD!r}, got {op!r}")
    family = data.get("family")
    if family != FAMILY_FMHA_FWD_MFMA:
        raise InstanceError(
            f"instance family must be {FAMILY_FMHA_FWD_MFMA!r}, got {family!r}"
        )
    require_string(data.get("name"), "instance name")
    require_string(data.get("arch"), "instance arch")


def _validate_instance_name(
    data: Mapping[str, Any], compile_spec: Mapping[str, Any], path: Path
) -> None:
    op = require_string(data.get("op"), "instance op")
    family = require_string(data.get("family"), "instance family")
    arch = require_string(data.get("arch"), "instance arch")
    name = require_string(data.get("name"), "instance name")
    expected = instance_name(op, family, arch, compile_spec)
    if name != expected:
        raise InstanceError(
            f"instance name {name!r} must match compile spec basename {expected!r}"
        )
    if path.name != f"{expected}.instance.json":
        raise InstanceError(
            f"instance file basename {path.name!r} must match instance name {expected!r}"
        )


def _validate_compile_spec(spec: Any, *, context: str) -> dict[str, Any]:
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
            f"compile_spec.head_size ({head_size}) must be one of 32, 64, 128, 192, 256"
        )
    if num_query_heads % num_kv_heads:
        raise InstanceError(
            f"compile_spec.num_query_heads ({num_query_heads}) must be divisible by "
            f"compile_spec.num_kv_heads ({num_kv_heads})"
        )
