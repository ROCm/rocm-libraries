# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Type admission for attention inputs, independent of selection policy."""

from math import isfinite
from numbers import Real


def require_attention_int(value, name, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer excluding bool")


def require_attention_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def require_attention_string(value, name, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _field(obj, name, label):
    try:
        return getattr(obj, name)
    except AttributeError as exc:
        raise ValueError(f"{label}.{name} is required") from exc


def _fields(obj, names, check, label, *, optional=False, allow_missing=False):
    for name in names:
        if allow_missing and not hasattr(obj, name):
            continue
        value = _field(obj, name, label)
        if optional:
            check(value, f"{label}.{name}", optional=True)
        else:
            check(value, f"{label}.{name}")


def validate_attention_request_types(request):
    _fields(
        request,
        (
            "batch",
            "nhead_q",
            "nhead_k",
            "seqlen_q",
            "seqlen_k",
            "hdim_q",
            "hdim_v",
            "sliding_window",
            "kv_block_size",
            "num_cus",
            "target_ctas",
            "right_bound",
        ),
        require_attention_int,
        "request",
    )
    _fields(
        request,
        ("window_left", "window_right", "diagonal_alignment"),
        require_attention_int,
        "request",
        optional=True,
    )
    _fields(
        request,
        (
            "use_sinks",
            "use_fp8",
            "fp8_fnuz",
            "use_alibi",
            "use_qq_bias",
            "use_softcap",
            "use_attn_bias",
            "emit_lse",
            "runtime_head_dims",
            "window_bottom_right",
        ),
        require_attention_bool,
        "request",
    )
    _fields(
        request,
        (
            "arch",
            "op",
            "dtype",
            "algorithm",
            "spec_id",
            "tuning_id",
            "kv_layout",
            "layout",
            "tensor_layout",
            "attn_bias_dtype",
        ),
        require_attention_string,
        "request",
    )
    softcap = _field(request, "softcap", "request")
    if (
        not isinstance(softcap, Real)
        or isinstance(softcap, bool)
        or not isfinite(softcap)
        or softcap < 0
    ):
        raise ValueError(
            "request.softcap must be finite nonnegative real excluding bool"
        )


def validate_attention_problem_types(problem):
    _fields(
        problem,
        (
            "total_q",
            "num_seqs",
            "num_query_heads",
            "num_kv_heads",
            "head_size",
            "block_size",
            "max_seqlen_q",
            "max_seqlen_k",
            "sliding_window",
            "num_cus",
            "target_ctas",
            "right_bound",
            "num_kv_blocks",
        ),
        require_attention_int,
        "problem",
    )
    _fields(problem, ("waves_per_eu",), require_attention_int, "problem", optional=True)
    _fields(
        problem,
        (
            "use_sinks",
            "use_alibi",
            "use_qq_bias",
            "use_fp8",
            "fp8_fnuz",
            "use_additive_bias",
            "causal_top_left",
            "emit_lse",
        ),
        require_attention_bool,
        "problem",
    )
    _fields(problem, ("dtype",), require_attention_string, "problem")
    _fields(
        problem,
        ("q_dtype", "compile_backend", "clamp_arch"),
        require_attention_string,
        "problem",
        optional=True,
    )
    softcap = _field(problem, "softcap", "problem")
    if not isinstance(softcap, Real) or isinstance(softcap, bool):
        raise ValueError("problem.softcap must be real numeric excluding bool")


def validate_attention_3d_spec_types(spec, *, reducer=False, complete=False):
    """Check consumed structural fields; full builders require declared fields."""
    label = "reduce_spec" if reducer else "kernel_spec"
    integers = ("head_size", "num_query_heads", "num_kv_heads", "num_segments")
    if not reducer:
        integers += ("block_size", "sliding_window", "num_seqs")
    _fields(spec, integers, require_attention_int, label)
    _fields(spec, ("dtype",), require_attention_string, label)
    _fields(
        spec,
        ("waves_per_eu",),
        require_attention_int,
        label,
        optional=True,
        allow_missing=not complete,
    )
    if complete or hasattr(spec, "metadata_containment"):
        require_attention_bool(
            _field(spec, "metadata_containment", label), "metadata_containment"
        )
    if getattr(spec, "metadata_containment", False):
        if not 1 <= spec.num_segments <= 128 or spec.dtype not in ("fp16", "bf16"):
            raise ValueError("bounded decode requires fp16/bf16 and 1..128 segments")
        if spec.head_size not in (64, 128, 256) or spec.num_kv_heads <= 0:
            raise ValueError("bounded decode has invalid head geometry")
        ratio = spec.num_query_heads // spec.num_kv_heads
        if spec.num_query_heads != ratio * spec.num_kv_heads or ratio not in (
            1,
            2,
            4,
            8,
            16,
        ):
            raise ValueError("bounded decode has invalid GQA ratio")
        if spec.num_query_heads * spec.head_size * spec.num_segments * 4 > 0x7FFF0000:
            raise ValueError("bounded decode workspace geometry exceeds i32")
    if reducer:
        if complete or hasattr(spec, "emit_lse"):
            require_attention_bool(_field(spec, "emit_lse", label), "emit_lse")
        return
    gfx950 = (
        type(spec).__module__.endswith("gfx950.attention_tiled_3d")
        or hasattr(spec, "use_additive_bias")
        or hasattr(spec, "causal_top_left")
        or hasattr(spec, "right_bound")
    )
    _fields(
        spec,
        ("use_additive_bias", "causal_top_left"),
        require_attention_bool,
        label,
        allow_missing=not (complete and gfx950),
    )
    _fields(
        spec,
        ("right_bound",),
        require_attention_int,
        label,
        allow_missing=not (complete and gfx950),
    )
    _fields(
        spec,
        ("kv_cache_policy",),
        require_attention_string,
        label,
        allow_missing=not complete,
    )
    _fields(
        spec,
        ("use_sinks", "has_softcap", "use_alibi", "use_qq_bias"),
        require_attention_bool,
        label,
    )
    _fields(
        spec,
        (
            "use_invariant_hoist",
            "use_wide_kv_load",
            "use_i64_kv_addr",
            "use_decode_grid",
            "use_kv_lds_swizzle",
            "use_wave_page_id_load",
        ),
        require_attention_bool,
        label,
        allow_missing=not complete,
    )
    _fields(
        spec,
        ("tile_size_override",),
        require_attention_int,
        label,
        optional=True,
        allow_missing=not complete,
    )
    _fields(spec, ("kv_storage_dtype",), require_attention_string, label, optional=True)
    _fields(
        spec,
        ("kv_layout",),
        require_attention_string,
        label,
        allow_missing=not complete,
    )
    # Validate raw fields first: these properties can themselves use geometry.
    _fields(
        spec, ("tile_size",), require_attention_int, label, allow_missing=not complete
    )
    _fields(
        spec,
        ("uses_paged_gather",),
        require_attention_bool,
        label,
        allow_missing=not complete,
    )

    if getattr(spec, "metadata_containment", False):
        if (
            spec.kv_layout != "paged"
            or spec.tile_size != 32
            or spec.block_size not in (1, 16, 32, 64)
            or spec.num_seqs <= 0
            or spec.num_seqs
            * spec.num_query_heads
            * spec.head_size
            * spec.num_segments
            * 4
            > 0x7FFF0000
            or spec.sliding_window < 0
            or spec.sliding_window > 0x7FFFFFFF
            or not spec.use_decode_grid
            or spec.use_i64_kv_addr
            or spec.kv_storage_dtype is not None
            or spec.use_wide_kv_load
            or spec.use_wave_page_id_load
            or getattr(spec, "use_additive_bias", False)
            or getattr(spec, "causal_top_left", False)
            or getattr(spec, "right_bound", 0) != 0
        ):
            raise ValueError("bounded decode requires paged T32 async decode grid")
