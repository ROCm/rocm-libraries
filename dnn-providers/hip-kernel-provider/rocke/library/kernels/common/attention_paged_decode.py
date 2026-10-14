# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host admission for tiled paged decode. No device metadata reads."""

from ._attention_kv_loader import AttentionKVLoaderKind, resolve_attention_kv_loader
from ._attention_validation import (
    require_attention_int,
    validate_attention_problem_types,
    validate_attention_3d_spec_types,
)

from math import isfinite
from numbers import Real

BUFFER_LIMIT = 0x7FFF0000


def is_unit_scale(value) -> bool:
    """Unsupported scaling accepts real numeric unit values, excluding bool."""
    return isinstance(value, Real) and not isinstance(value, bool) and value == 1.0


def validate_paged_3d_features(
    problem,
    tuning_spec,
    *,
    softcap,
    sinks,
    alibi_slopes,
    qq_bias,
    k_scale,
    v_scale,
    out_scale,
    qq_bias_stride_0=0,
):
    """Match explicit CDNA paged 3D features to the inputs actually launched.

    Read the compiled spec rather than optional candidate metadata or tile size.
    Geometry and gather-only address bounds have separate admission checks.
    """
    validate_attention_problem_types(problem)
    require_attention_int(qq_bias_stride_0, "qq_bias_stride_0")
    if qq_bias_stride_0 < 0:
        raise ValueError("qq_bias_stride_0 must be nonnegative")
    expected_flags = (
        ("has_softcap", problem.softcap > 0),
        ("use_sinks", problem.use_sinks),
        ("use_alibi", problem.use_alibi),
        ("use_qq_bias", problem.use_qq_bias),
    )
    storage = "fp8e4m3" if problem.use_fp8 else None
    if tuning_spec is not None:
        segment = tuning_spec.kernel_spec
        for name, expected in expected_flags:
            if not hasattr(segment, name):
                raise ValueError(f"paged 3D kernel_spec.{name} is required")
            value = getattr(segment, name)
            if not isinstance(value, bool) or value is not expected:
                raise ValueError(
                    f"paged 3D kernel_spec.{name} must be a bool agreeing with problem"
                )
        if not hasattr(segment, "kv_storage_dtype"):
            raise ValueError("paged 3D kernel_spec.kv_storage_dtype is required")
        if segment.kv_storage_dtype != storage:
            raise ValueError(
                "paged 3D kernel_spec.kv_storage_dtype disagrees with problem"
            )
    if (
        not isinstance(softcap, Real)
        or isinstance(softcap, bool)
        or not isfinite(softcap)
        or (softcap <= 0 if problem.softcap > 0 else softcap != 0)
    ):
        raise ValueError("paged 3D softcap disagrees with compiled enablement")
    for name, value, enabled in (
        ("sinks", sinks, problem.use_sinks),
        ("alibi_slopes", alibi_slopes, problem.use_alibi),
        ("qq_bias", qq_bias, problem.use_qq_bias),
    ):
        if (value is not None) != enabled:
            raise ValueError(f"paged 3D {name} disagrees with compiled enablement")
    # The segment consumes K/V scales only in its FP8 dequantization loader;
    # the 3D adapter never forwards output scaling to either kernel.
    scales = [("out_scale", out_scale)]
    if storage is None:
        scales.extend((("k_scale", k_scale), ("v_scale", v_scale)))
    else:
        for name, value in (("k_scale", k_scale), ("v_scale", v_scale)):
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not isfinite(value)
            ):
                raise ValueError(
                    f"paged 3D {name} requires finite real numeric scaling"
                )
    for name, value in scales:
        if not is_unit_scale(value):
            raise ValueError(f"paged 3D {name} requires numeric unit scaling")


def validate_paged_loader_geometry(
    problem, tuning_spec
) -> tuple[int, int, bool] | None:
    """Validate the explicit CDNA paged loader's geometry before admission.

    Structural specs must describe the same loader as the concrete builders.
    The returned immutable geometry also guards address-width retargeting.
    Other paths and architectures have independent spec contracts.
    """
    if (
        tuning_spec is None
        or tuning_spec.path != "3d"
        or tuning_spec.arch not in ("gfx942", "gfx950")
    ):
        return None
    validate_attention_problem_types(problem)
    spec = tuning_spec.kernel_spec
    for name in ("block_size", "tile_size_override"):
        try:
            value = getattr(spec, name)
        except AttributeError as exc:
            raise ValueError(
                f"3D kernel_spec.{name} is required on gfx942/gfx950"
            ) from exc
        require_attention_int(
            value, f"kernel_spec.{name}", optional=name == "tile_size_override"
        )
    values = {}
    for name in (
        "uses_paged_gather",
        "kv_layout",
        "block_size",
        "tile_size",
        "tile_size_override",
    ):
        try:
            values[name] = getattr(spec, name)
        except AttributeError as exc:
            raise ValueError(
                f"3D kernel_spec.{name} is required on gfx942/gfx950"
            ) from exc
        if name == "uses_paged_gather" and not isinstance(values[name], bool):
            raise ValueError("3D kernel_spec.uses_paged_gather must be a bool")
    if values["kv_layout"] != "paged":
        raise ValueError("paged loader requires kernel_spec.kv_layout='paged'")
    for name, allowed in (
        ("block_size", (1, 16, 32, 64)),
        ("tile_size", (16, 32, 64)),
        ("tile_size_override", (16, 32, 64)),
    ):
        value = values[name]
        if name == "tile_size_override" and value is None:
            continue
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value not in allowed
        ):
            raise ValueError(f"3D kernel_spec.{name} must be an integer in {allowed}")
    page, tile = values["block_size"], values["tile_size"]
    if page != problem.block_size:
        raise ValueError("3D kernel_spec.block_size disagrees with problem")
    override = values["tile_size_override"]
    expected_tile = override if override is not None else (32 if page == 1 else page)
    if tile != expected_tile:
        raise ValueError("3D kernel_spec.tile_size disagrees with tile_size_override")
    loader_kind = resolve_attention_kv_loader(
        tuning_spec.arch, values["kv_layout"], page, tile
    )
    gather = loader_kind is AttentionKVLoaderKind.PAGED_TOKEN_GATHER
    if values["uses_paged_gather"] != gather:
        raise ValueError(
            "3D kernel_spec.uses_paged_gather disagrees with loader geometry"
        )
    if gather:
        _validate_gather_wide_load(spec)
    validate_attention_3d_spec_types(spec)
    reduce = getattr(tuning_spec, "reduce_spec", None)
    if reduce is None:
        raise ValueError("paged 3D reduce_spec is required")
    validate_attention_3d_spec_types(reduce, reducer=True)
    for name in (
        "head_size",
        "num_query_heads",
        "num_kv_heads",
        "dtype",
        "num_segments",
    ):
        if getattr(reduce, name) != getattr(spec, name):
            raise ValueError(f"paged reduce_spec.{name} disagrees with segment")
    if gather and tile != 32:
        raise ValueError("paged gather requires tile_size=32")
    return page, tile, gather


def _validate_gather_wide_load(spec):
    if getattr(spec, "use_wide_kv_load", None) is not False:
        raise ValueError("paged gather requires kernel_spec.use_wide_kv_load=False")


def validate_paged_decode(problem, tensors, tuning_spec=None):
    """Validate the new loader ABI without synchronization or graph allocation.

    Caller-owned values: lengths lie in [0,max_seqlen_k], query offsets are
    [0,1,...,B], and referenced page IDs are in [0,physical_pages). Inactive
    table entries need not be initialized. K and V use the same page mapping.
    """
    validate_attention_problem_types(problem)
    p = problem
    if p.max_seqlen_q != 1 or p.total_q != p.num_seqs:
        raise ValueError("paged gather requires one query per sequence")
    if p.dtype not in ("fp16", "bf16") or p.use_fp8:
        raise ValueError("paged gather requires fp16/bf16")
    if not 0 <= p.max_seqlen_k <= BUFFER_LIMIT:
        raise ValueError("paged gather sequence length exceeds token-offset bound")
    if tuning_spec is not None:
        segment = tuning_spec.kernel_spec
        _validate_gather_wide_load(segment)
        validate_attention_3d_spec_types(segment)
        if tuning_spec.path != "3d" or tuning_spec.reduce_spec is None:
            raise ValueError("paged gather requires segment and reduce specs")
        validate_attention_3d_spec_types(tuning_spec.reduce_spec, reducer=True)
        for name, expected in (
            ("kv_layout", "paged"),
            ("head_size", p.head_size),
            ("block_size", p.block_size),
            ("num_query_heads", p.num_query_heads),
            ("num_kv_heads", p.num_kv_heads),
            ("num_seqs", p.num_seqs),
            ("dtype", p.dtype),
            ("sliding_window", p.sliding_window),
            ("kv_storage_dtype", None),
            ("use_i64_kv_addr", False),
            ("use_sinks", p.use_sinks),
            ("use_alibi", p.use_alibi),
            ("use_qq_bias", p.use_qq_bias),
            ("has_softcap", p.softcap > 0),
        ):
            if not hasattr(segment, name) or getattr(segment, name) != expected:
                raise ValueError(
                    f"paged kernel_spec.{name} missing or disagrees with problem"
                )
        for name in (
            "head_size",
            "num_query_heads",
            "num_kv_heads",
            "dtype",
            "num_segments",
        ):
            if getattr(tuning_spec.reduce_spec, name, None) != getattr(segment, name):
                raise ValueError(f"paged reduce_spec.{name} disagrees with segment")
    # The segment workspace uses bounded scalar offsets too. Default routing
    # selects at most 128 segments; use that ceiling before building its spec.
    segments = (
        128
        if tuning_spec is None
        else getattr(tuning_spec.kernel_spec, "num_segments", 0)
    )
    if not 1 <= segments <= 128:
        raise ValueError("paged gather requires 1..128 segments")
    if p.num_seqs * p.num_query_heads * p.head_size * segments * 4 > BUFFER_LIMIT:
        raise ValueError("paged gather workspace exceeds buffer-offset bound")
    q, k, v, out, cu, used, table = tensors
    dtype = "torch.float16" if p.dtype == "fp16" else "torch.bfloat16"
    q_shape = (p.num_seqs, p.num_query_heads, p.head_size)
    tail = (p.block_size, p.num_kv_heads, p.head_size)
    if tuple(q.shape) != q_shape or tuple(out.shape) != q_shape:
        raise ValueError("paged decode Q/O shape disagrees with problem")
    if (
        len(k.shape) != 4
        or tuple(k.shape[1:]) != tail
        or tuple(v.shape) != tuple(k.shape)
        or k.shape[0] <= 0
    ):
        raise ValueError("paged decode K/V must be matching [pages,P,Hkv,D] pools")
    # Check both actual allocations; num_kv_blocks supplied by callers is not
    # proof of representability. Element counts use Python's unbounded integer.
    for tensor in (k, v):
        if int(tensor.numel()) * 2 > BUFFER_LIMIT:
            raise ValueError("paged gather KV pool exceeds buffer-offset bound")
    if any(str(t.dtype) != dtype for t in (q, k, v, out)):
        raise ValueError("paged decode Q/K/V/O dtype disagrees with problem")
    for tensor, shape in ((cu, (p.num_seqs + 1,)), (used, (p.num_seqs,))):
        if str(tensor.dtype) != "torch.int32" or tuple(tensor.shape) != shape:
            raise ValueError("paged decode lengths/offsets must be int32 vectors")
    pages = max(1, (p.max_seqlen_k + p.block_size - 1) // p.block_size)
    if (
        table is None
        or len(table.shape) != 2
        or table.shape[0] != p.num_seqs
        or table.shape[1] < pages
        or str(table.dtype) != "torch.int32"
    ):
        raise ValueError(
            "paged decode block table must be int32 [B,ceil(capacity/P)] or wider"
        )
    if int(table.numel()) * 4 > BUFFER_LIMIT:
        raise ValueError("paged gather page table exceeds index bound")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("paged decode tensors must be contiguous")
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError("paged decode tensors must be on the same device")
    if any(int(t.data_ptr()) % 16 for t in (q, k, v, out)):
        raise ValueError("paged decode Q/K/V/O must be 16-byte aligned")
