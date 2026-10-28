# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host admission for tiled paged decode. No device metadata reads."""

from ._attention_kv_loader import AttentionKVLoaderKind, resolve_attention_kv_loader
from ._attention_validation import (
    require_attention_bool,
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


def _feature_tensor_metadata(tensor, name, q, dtype, rank, alignment):
    """Check the metadata used by raw feature-pointer loads, without host reads."""
    try:
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
        pointer = tensor.data_ptr()
        device = tensor.device
        device_type = getattr(device, "type", str(device).split(":")[0])
        if str(tensor.dtype) != dtype:
            raise ValueError(f"{name} dtype must be {dtype}")
        if str(device_type) != "cuda" or device != q.device:
            raise ValueError(f"{name} must be on Q's HIP/CUDA device")
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"{name} must expose tensor metadata") from exc
    if len(shape) != rank or len(strides) != rank:
        raise ValueError(f"{name} must be rank-{rank}")
    for value in (*shape, *strides, pointer):
        require_attention_int(value, f"{name} metadata")
    if any(value <= 0 for value in shape) or any(value < 0 for value in strides):
        raise ValueError(f"{name} shape/strides must describe accessible storage")
    if pointer <= 0 or pointer % alignment:
        raise ValueError(f"{name} must be {alignment}-byte aligned")
    return shape, strides


def _validate_feature_tensors(
    problem, tuning_spec, q, sinks, alibi_slopes, qq_bias, stride, defer
):
    scalar_dtype = "torch.float16" if problem.dtype == "fp16" else "torch.bfloat16"
    for tensor, name, dtype, alignment in (
        (sinks, "sinks", scalar_dtype, 16),
        (alibi_slopes, "alibi_slopes", "torch.float32", 4),
    ):
        if tensor is None:
            continue
        shape, strides = _feature_tensor_metadata(tensor, name, q, dtype, 1, alignment)
        if shape != (problem.num_query_heads,) or strides != (1,):
            raise ValueError(f"{name} must be a contiguous [num_query_heads] vector")
    if qq_bias is None:
        return
    shape, strides = _feature_tensor_metadata(
        qq_bias, "qq_bias", q, "torch.float32", 2, 4
    )
    # qp and krp are local query positions. The kernel also uses the row stride
    # as its column bound, so padding is legal when all active columns exist.
    extent = problem.max_seqlen_q
    if shape[0] < extent:
        raise ValueError("qq_bias must cover max_seqlen_q rows")
    if strides[1] != 1 or strides[0] < shape[1]:
        raise ValueError("qq_bias requires contiguous columns and nonoverlapping rows")
    if shape[1] < extent and strides[0] != shape[1]:
        raise ValueError("partial-width qq_bias requires unpadded rows")
    tile = 32 if tuning_spec is None else tuning_spec.kernel_spec.tile_size
    # Bias loads precede the causal/tail score mask. Padded tile columns can
    # therefore read up to stride-1, even when they do not contribute to output.
    last_column = min(strides[0] - 1, extent + tile - 2)
    last_index = (extent - 1) * strides[0] + last_column
    if strides[0] > 0x7FFF_FFFF or last_index > 0x7FFF_FFFF:
        raise ValueError("qq_bias addressed extent must fit i32 indexing")
    declared_last = (shape[0] - 1) * strides[0] + shape[1] - 1
    if last_index > declared_last:
        # A legal strided Torch view can expose padding beyond its last declared
        # element. Require actual backing-storage metadata for that case.
        try:
            remaining = (
                qq_bias.untyped_storage().nbytes() - qq_bias.storage_offset() * 4
            )
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                "qq_bias backing storage must cover masked tail reads"
            ) from exc
        require_attention_int(remaining, "qq_bias storage extent")
        if remaining < (last_index + 1) * 4:
            raise ValueError("qq_bias backing storage must cover masked tail reads")
    if stride == 0 and defer:
        return
    if stride <= 0 or stride != strides[0]:
        raise ValueError(
            "qq_bias_stride_0 must be positive and match qq_bias.stride(0)"
        )


def validate_paged_3d_features(
    problem,
    tuning_spec,
    *,
    q,
    softcap,
    sinks,
    alibi_slopes,
    qq_bias,
    k_scale,
    v_scale,
    out_scale,
    qq_bias_stride_0=0,
    defer_qq_bias_stride=False,
):
    """Match explicit CDNA paged 3D features to the inputs actually launched.

    Read the compiled spec rather than optional candidate metadata or tile size.
    Geometry and gather-only address bounds have separate admission checks.
    """
    validate_attention_problem_types(problem)
    require_attention_int(qq_bias_stride_0, "qq_bias_stride_0")
    if not 0 <= qq_bias_stride_0 <= 0x7FFF_FFFF:
        raise ValueError("qq_bias_stride_0 must be a nonnegative i32")
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
        require_attention_int(segment.sliding_window, "kernel_spec.sliding_window")
        if segment.sliding_window != problem.sliding_window:
            raise ValueError(
                "paged 3D kernel_spec.sliding_window disagrees with problem"
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
    _validate_feature_tensors(
        problem,
        tuning_spec,
        q,
        sinks,
        alibi_slopes,
        qq_bias,
        qq_bias_stride_0,
        defer_qq_bias_stride,
    )
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
    tiles = tuple(range(16, 129, 16)) if tuning_spec.arch == "gfx950" else (16, 32, 64)
    for name, allowed in (
        ("block_size", (1, 16, 32, 64)),
        ("tile_size", tiles),
        ("tile_size_override", tiles),
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
    if hasattr(spec, "use_i64_kv_addr"):
        require_attention_bool(spec.use_i64_kv_addr, "kernel_spec.use_i64_kv_addr")
    loader_kind = resolve_attention_kv_loader(
        tuning_spec.arch,
        values["kv_layout"],
        page,
        tile,
        use_i64_kv_addr=getattr(spec, "use_i64_kv_addr", False),
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
    if gather and not has_paged_metadata_containment(tuning_spec):
        raise ValueError("paged gather requires segment/reducer metadata containment")
    return page, tile, gather


def _validate_gather_wide_load(spec):
    if getattr(spec, "use_wide_kv_load", None) is not False:
        raise ValueError("paged gather requires kernel_spec.use_wide_kv_load=False")


def validate_paged_decode(problem, tensors, tuning_spec=None, *, specs=None):
    """Validate the new loader ABI without synchronization or graph allocation.

    Caller-owned values: lengths lie in [0,max_seqlen_k], query offsets are
    [0,1,...,B], and referenced page IDs are in [0,physical_pages). Inactive
    table entries need not be initialized. K and V use the same page mapping.
    """
    validate_attention_problem_types(problem)
    p = problem
    if (
        p.num_seqs <= 0
        or p.num_query_heads <= 0
        or p.num_kv_heads <= 0
        or p.head_size not in (64, 128, 256)
        or p.block_size not in (1, 16, 32, 64)
    ):
        raise ValueError("paged decode has invalid geometry")
    if p.max_seqlen_q != 1 or p.total_q != p.num_seqs:
        raise ValueError("paged gather requires one query per sequence")
    if p.dtype not in ("fp16", "bf16") or p.use_fp8:
        raise ValueError("paged gather requires fp16/bf16")
    if not 0 <= p.max_seqlen_k <= BUFFER_LIMIT:
        raise ValueError("paged gather sequence length exceeds token-offset bound")
    if tuning_spec is not None:
        if tuning_spec.path != "3d" or tuning_spec.reduce_spec is None:
            raise ValueError("paged gather requires segment and reduce specs")
        specs = tuning_spec.kernel_spec, tuning_spec.reduce_spec
    if specs is not None:
        segment, reducer = specs
        _validate_gather_wide_load(segment)
        validate_attention_3d_spec_types(segment)
        validate_attention_3d_spec_types(reducer, reducer=True)
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
            if getattr(reducer, name, None) != getattr(segment, name):
                raise ValueError(f"paged reduce_spec.{name} disagrees with segment")
    # The segment workspace uses bounded scalar offsets too. Default routing
    # selects at most 128 segments; use that ceiling before building its spec.
    segments = 128 if specs is None else getattr(specs[0], "num_segments", 0)
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


def paged_metadata_bounds(problem, tensors, tuning_spec=None, *, specs=None):
    """Trusted scalar limits for bounded specs, without reading device contents."""
    validate_paged_decode(problem, tensors, tuning_spec, specs=specs)
    if tuning_spec is not None:
        specs = tuning_spec.kernel_spec, tuning_spec.reduce_spec
    if specs is None:
        raise ValueError("bounded decode requires a selected segment/reducer pair")
    segment, reducer = specs
    if not (
        getattr(segment, "metadata_containment", False)
        and getattr(reducer, "metadata_containment", False)
    ):
        raise ValueError("bounded decode requires matching segment and reducer")
    if not segment.use_decode_grid or segment.tile_size != 32:
        raise ValueError("bounded decode requires the one-query T32 grid")
    q, k, v, out, cu, used, table = tensors
    # Every intermediate, including rounded inactive segment starts, stays i32.
    segments = segment.num_segments
    tile_span = segments * 32
    rounded = ((problem.max_seqlen_k + tile_span - 1) // tile_span) * tile_span
    if rounded + max(32, segment.sliding_window, segment.block_size) > BUFFER_LIMIT:
        raise ValueError("bounded decode rounded token arithmetic exceeds i32")
    for tensor, itemsize in ((q, 2), (out, 2), (cu, 4), (used, 4)):
        if int(tensor.numel()) * itemsize > BUFFER_LIMIT:
            raise ValueError("bounded decode tensor span exceeds i32")
    # Noalias input/output contracts also forbid metadata writes overlapping a
    # kernel read. Check view intervals; internal scratch is separately allocated.
    spans = [
        (int(t.data_ptr()), int(t.numel()) * size)
        for t, size in zip(tensors, (2, 2, 2, 2, 4, 4, 4))
    ]
    for i, (base, size) in enumerate(spans):
        for other, other_size in spans[i + 1 :]:
            if base < other + other_size and other < base + size:
                raise ValueError("bounded decode tensor views must not overlap")
    status_bytes = problem.num_seqs * problem.num_kv_heads * segments * 4
    if status_bytes > BUFFER_LIMIT:
        raise ValueError("bounded decode status span exceeds i32")
    return {
        "physical_pages": int(k.shape[0]),
        "max_seqlen_k": int(problem.max_seqlen_k),
        "table_cols": int(table.shape[1]),
        "k_span_bytes": int(k.numel()) * 2,
        "v_span_bytes": int(v.numel()) * 2,
    }


def _default_paged_metadata_containment(problem, arch: str) -> bool:
    return arch in ("gfx942", "gfx950") and problem.block_size == 1


def has_paged_metadata_containment(tuning_spec, *, problem=None, arch=None):
    if tuning_spec is None:
        return (
            problem is not None
            and arch is not None
            and _default_paged_metadata_containment(problem, arch)
        )
    segment = getattr(tuning_spec.kernel_spec, "metadata_containment", False)
    reducer = getattr(
        getattr(tuning_spec, "reduce_spec", None), "metadata_containment", False
    )
    require_attention_bool(segment, "kernel_spec.metadata_containment")
    require_attention_bool(reducer, "reduce_spec.metadata_containment")
    if segment != reducer:
        raise ValueError("segment/reducer metadata containment must match")
    return segment
