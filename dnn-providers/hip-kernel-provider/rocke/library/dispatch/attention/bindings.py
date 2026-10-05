# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch bindings shared by executable attention dispatch candidates.

The dispatcher owns tensor-to-runner adaptation.  Torch remains a lazy runtime
dependency: this module only closes over caller-owned tensors and imports
architecture kernel runners inside binding calls.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Mapping, Tuple

from rocke.dispatch.core import TorchBinding


def _shape(tensor, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in tensor.shape)
    except Exception as exc:
        raise ValueError(f"{name} must expose an integer shape") from exc


def _dtype_kind(tensor, name: str) -> str:
    value = str(getattr(tensor, "dtype", "")).lower()
    if "float8_e4m3fnuz" in value or value.endswith("fp8e4m3fnuz"):
        return "fp8_fnuz"
    if "float8_e4m3fn" in value or value.endswith("fp8e4m3"):
        return "fp8"
    if "bfloat16" in value or value.endswith("bf16"):
        return "bf16"
    if "float16" in value or value.endswith("fp16"):
        return "fp16"
    if "int32" in value or value.endswith("i32"):
        return "int32"
    raise ValueError(f"{name} has unsupported dtype {getattr(tensor, 'dtype', None)!r}")


def _require_contiguous(tensor, name: str) -> None:
    predicate = getattr(tensor, "is_contiguous", None)
    if not callable(predicate) or not bool(predicate()):
        raise ValueError(f"{name} must be contiguous")


def _tolist(tensor, name: str):
    value = tensor
    for method in ("detach", "cpu"):
        fn = getattr(value, method, None)
        if callable(fn):
            value = fn()
    fn = getattr(value, "tolist", None)
    if not callable(fn):
        raise ValueError(f"{name} must support tolist() for validation")
    return fn()


def validate_tuning_attention_contract(request, problem, spec) -> None:
    """Ensure request, runtime problem, and explicit specs describe one launch."""
    request_fields = (
        ("batch", "num_seqs"),
        ("nhead_q", "num_query_heads"),
        ("nhead_k", "num_kv_heads"),
        ("hdim_q", "head_size"),
        ("kv_block_size", "block_size"),
        ("seqlen_q", "max_seqlen_q"),
        ("seqlen_k", "max_seqlen_k"),
    )
    for request_name, problem_name in request_fields:
        requested = int(getattr(request, request_name))
        actual = int(getattr(problem, problem_name))
        if requested != actual:
            raise ValueError(
                f"request.{request_name}={requested} disagrees with "
                f"problem.{problem_name}={actual}"
            )
    if str(request.dtype).lower() != str(problem.dtype).lower():
        raise ValueError("request dtype disagrees with attention problem dtype")
    if bool(request.use_fp8) != bool(problem.use_fp8) or bool(request.fp8_fnuz) != bool(
        problem.fp8_fnuz
    ):
        raise ValueError("request FP8 encoding disagrees with attention problem")

    kernel_spec = spec.kernel_spec
    for name, expected in (
        ("head_size", problem.head_size),
        ("block_size", problem.block_size),
        ("num_query_heads", problem.num_query_heads),
        ("num_kv_heads", problem.num_kv_heads),
        ("dtype", problem.dtype),
        ("num_seqs", problem.num_seqs),
    ):
        if getattr(kernel_spec, name) != expected:
            raise ValueError(
                f"kernel_spec.{name}={getattr(kernel_spec, name)!r} disagrees "
                f"with problem value {expected!r}"
            )
    expected_kv_dtype = "fp8e4m3" if problem.use_fp8 else None
    if kernel_spec.kv_storage_dtype != expected_kv_dtype:
        raise ValueError("kernel spec K/V storage dtype disagrees with problem")
    if bool(spec.fp8_fnuz) != bool(problem.fp8_fnuz):
        raise ValueError("tuning wrapper FP8 encoding disagrees with problem")

    reduce_spec = spec.reduce_spec
    if reduce_spec is not None:
        for name in ("head_size", "num_query_heads", "num_kv_heads", "dtype"):
            if getattr(reduce_spec, name) != getattr(kernel_spec, name):
                raise ValueError(
                    f"reduce_spec.{name} disagrees with segment kernel spec"
                )
        if int(reduce_spec.num_segments) != int(kernel_spec.num_segments):
            raise ValueError("reduce spec segment count disagrees with kernel spec")


def validate_tuning_attention_tensors(
    problem,
    tensors: Mapping[str, Any],
    *,
    validate_contents: bool = True,
) -> int:
    """Validate the paged ABI once before binding an explicit tuning launch.

    ``validate_contents=False`` skips device-to-host metadata checks for trusted
    graph/hot paths, but structural shape/dtype/layout checks remain mandatory.
    Returns the physical K/V block count used to refresh address-width state.
    """
    required = (
        "q",
        "k",
        "v",
        "out",
        "cu_seqlens_q",
        "seqused_k",
        "block_table",
    )
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError("missing attention tensors: " + ", ".join(missing))

    q, k, v, out = (tensors[name] for name in ("q", "k", "v", "out"))
    cu = tensors["cu_seqlens_q"]
    used = tensors["seqused_k"]
    table = tensors["block_table"]

    q_shape = (
        int(problem.total_q),
        int(problem.num_query_heads),
        int(problem.head_size),
    )
    kv_tail = (
        int(problem.block_size),
        int(problem.num_kv_heads),
        int(problem.head_size),
    )
    if _shape(q, "q") != q_shape:
        raise ValueError(f"q shape must be {q_shape}, got {_shape(q, 'q')}")
    if _shape(out, "out") != q_shape:
        raise ValueError(f"out shape must be {q_shape}, got {_shape(out, 'out')}")
    k_shape = _shape(k, "k")
    v_shape = _shape(v, "v")
    if len(k_shape) != 4 or k_shape[1:] != kv_tail:
        raise ValueError(f"k shape must be [blocks, {kv_tail}], got {k_shape}")
    if v_shape != k_shape:
        raise ValueError(f"v shape must match k shape {k_shape}, got {v_shape}")
    num_blocks = int(k_shape[0])
    if num_blocks <= 0:
        raise ValueError("paged K/V cache must contain at least one physical block")

    q_kind = "bf16" if str(problem.dtype).lower() == "bf16" else "fp16"
    kv_kind = (
        "fp8_fnuz"
        if problem.use_fp8 and problem.fp8_fnuz
        else "fp8" if problem.use_fp8 else q_kind
    )
    for tensor, name, expected in (
        (q, "q", q_kind),
        (out, "out", q_kind),
        (k, "k", kv_kind),
        (v, "v", kv_kind),
        (cu, "cu_seqlens_q", "int32"),
        (used, "seqused_k", "int32"),
        (table, "block_table", "int32"),
    ):
        actual = _dtype_kind(tensor, name)
        if actual != expected:
            raise ValueError(f"{name} dtype must be {expected}, got {actual}")
        if name != "block_table":
            _require_contiguous(tensor, name)

    device_values = [
        getattr(tensor, "device", None) for tensor in (q, k, v, out, cu, used, table)
    ]
    devices = {str(device) for device in device_values}
    if len(devices) != 1:
        raise ValueError(f"all attention tensors must share one device, got {devices}")
    device_type = getattr(device_values[0], "type", str(device_values[0]).split(":")[0])
    if str(device_type).lower() != "cuda":
        raise ValueError("explicit attention tensors must be on a HIP/CUDA device")

    table_shape = _shape(table, "block_table")
    if len(table_shape) != 2 or table_shape[0] != int(problem.num_seqs):
        raise ValueError(
            "block_table shape must be "
            f"[{problem.num_seqs}, max_blocks], got {table_shape}"
        )
    if table_shape[1] <= 0:
        raise ValueError("block_table must have at least one block column")
    stride = getattr(table, "stride", None)
    if not callable(stride):
        raise ValueError("block_table must expose strides")
    inner_stride = int(stride(1))
    row_stride = int(stride(0))
    if inner_stride != 1:
        raise ValueError("block_table innermost stride must be 1")
    if row_stride < table_shape[1] or row_stride > 0x7FFF_FFFF:
        raise ValueError("block_table row stride must be non-overlapping and fit int32")
    if _shape(cu, "cu_seqlens_q") != (int(problem.num_seqs) + 1,):
        raise ValueError("cu_seqlens_q must have shape [num_seqs + 1]")
    if _shape(used, "seqused_k") != (int(problem.num_seqs),):
        raise ValueError("seqused_k must have shape [num_seqs]")

    if validate_contents:
        cu_values = [int(v) for v in _tolist(cu, "cu_seqlens_q")]
        used_values = [int(v) for v in _tolist(used, "seqused_k")]
        table_values = _tolist(table, "block_table")
        if (
            cu_values[0] != 0
            or cu_values[-1] != int(problem.total_q)
            or any(a > b for a, b in zip(cu_values, cu_values[1:]))
        ):
            raise ValueError(
                "cu_seqlens_q must start at 0, end at total_q, and be monotonic"
            )
        query_lengths = [b - a for a, b in zip(cu_values, cu_values[1:])]
        if any(
            length < 0 or length > int(problem.max_seqlen_q) for length in query_lengths
        ):
            raise ValueError("cu_seqlens_q contains an unsupported per-sequence length")
        for seq, kv_len in enumerate(used_values):
            if kv_len < 0 or kv_len > int(problem.max_seqlen_k):
                raise ValueError(
                    f"seqused_k[{seq}]={kv_len} is outside [0, {problem.max_seqlen_k}]"
                )
            pages = (kv_len + int(problem.block_size) - 1) // int(problem.block_size)
            if pages > table_shape[1]:
                raise ValueError(
                    f"block_table row {seq} has {table_shape[1]} entries but "
                    f"seqused_k requires {pages}"
                )
            row = table_values[seq]
            if len(row) != table_shape[1]:
                raise ValueError(
                    f"block_table row {seq} has {len(row)} values, "
                    f"expected {table_shape[1]}"
                )
            for page, physical in enumerate(row[:pages]):
                physical = int(physical)
                if physical < 0 or physical >= num_blocks:
                    raise ValueError(
                        f"block_table[{seq}, {page}]={physical} is outside "
                        f"[0, {num_blocks})"
                    )
    return num_blocks


# Declared runner contracts. Inferring them with inspect.signature drops
# arguments the runner does not list and launches a different kernel.
_DENSE_OPTIONAL_INPUTS = {
    "gfx942": frozenset({"cu_seqlens_q", "cu_seqlens_kv"}),
    "gfx950": frozenset(
        {"cu_seqlens_q", "cu_seqlens_kv", "block_tables", "kv_lens", "sinks"}
    ),
}


def _dense_runner(arch: str):
    if arch == "gfx942":
        from kernels.gfx942.attention_dense import (
            attention_dense_block,
            attention_dense_grid,
            run_attention_dense_torch,
        )
    elif arch == "gfx950":
        from kernels.gfx950.attention_dense import (
            attention_dense_block,
            attention_dense_grid,
            run_attention_dense_torch,
        )
    else:
        raise ValueError(f"no dense attention Torch runner for arch {arch!r}")
    return run_attention_dense_torch, attention_dense_grid, attention_dense_block


def bind_dense_attention_torch(
    request, spec, tensors: Mapping[str, Any], **kwargs
) -> TorchBinding:
    """Bind a concrete dense spec to ``q``/``k``/``v``/``out`` tensors."""
    run, grid_fn, block_fn = _dense_runner(str(getattr(request, "arch", "")))
    scale = kwargs.get("scale")
    if scale is None:
        scale = 1.0 / math.sqrt(int(getattr(request, "hdim_q", spec.head_size)))
    stream = kwargs.get("stream", 0)

    def launch(**_kw):
        call = {
            "spec": spec,
            "q": tensors["q"],
            "k": tensors["k"],
            "v": tensors["v"],
            "out": tensors["out"],
            "scale": float(_kw.get("scale", scale)),
            "stream": int(_kw.get("stream", stream)),
            "arch": str(request.arch),
        }
        optional = {
            "cu_seqlens_q": _kw.get("cu_seqlens_q", tensors.get("cu_seqlens_q")),
            "cu_seqlens_kv": _kw.get("cu_seqlens_kv", tensors.get("cu_seqlens_kv")),
            "block_tables": _kw.get("block_tables", tensors.get("block_tables")),
            "kv_lens": _kw.get("kv_lens", tensors.get("kv_lens")),
            "sinks": _kw.get("sinks", tensors.get("sinks")),
        }
        accepted = _DENSE_OPTIONAL_INPUTS[str(request.arch)]
        for name, value in optional.items():
            if value is None:
                continue
            if name not in accepted:
                raise NotImplementedError(
                    f"{request.arch} dense runner does not accept {name!r}"
                )
            call[name] = value
        return run(**call)

    return TorchBinding(launch=launch, grid=grid_fn(spec), block=block_fn(spec))


def bind_tuning_attention_torch(
    request, spec, tensors: Mapping[str, Any], **kwargs
) -> TorchBinding:
    """Bind an explicit unified 2D/3D tuning spec to paged tensors.

    Metadata values are snapshotted once; callers must rebind after mutating
    sequence lengths or block tables. Trusted callers that enforce those
    invariants externally may pass
    ``unsafe_skip_paged_value_validation=True`` to skip the synchronized value
    check; structural ABI validation is never skipped.
    """
    from kernels.common.attention_unified import (
        UnifiedAttentionProblem,
        run_unified_attention_torch,
    )

    problem = tensors.get("problem")
    if problem is None:
        raise ValueError(
            "bind_tuning_attention_torch requires tensors['problem'] "
            "(a UnifiedAttentionProblem); dispatch injects it before calling"
        )
    if not isinstance(problem, UnifiedAttentionProblem):
        raise TypeError(
            "tensors['problem'] must be a UnifiedAttentionProblem, got "
            f"{type(problem).__name__}"
        )
    validate_tuning_attention_contract(request, problem, spec)
    unsafe_skip = bool(kwargs.pop("unsafe_skip_paged_value_validation", False))
    num_blocks = validate_tuning_attention_tensors(
        problem,
        tensors,
        validate_contents=not unsafe_skip,
    )
    problem = replace(problem, num_kv_blocks=num_blocks)
    if hasattr(spec, "with_num_kv_blocks"):
        spec = spec.with_num_kv_blocks(num_blocks)
    stream = kwargs.get("stream", 0)
    path = str(getattr(spec, "path", "2d"))
    backend = "tiled" if path == "2d" else path
    scale = kwargs.get("softmax_scale")
    if scale is None:
        scale = 1.0 / math.sqrt(int(problem.head_size))

    def launch(**_kw):
        return run_unified_attention_torch(
            problem=problem,
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            out=tensors["out"],
            cu_seqlens_q=tensors["cu_seqlens_q"],
            seqused_k=tensors["seqused_k"],
            softmax_scale=float(_kw.get("softmax_scale", scale)),
            block_table=tensors["block_table"],
            softcap=float(_kw.get("softcap", tensors.get("softcap", 0.0))),
            sinks=_kw.get("sinks", tensors.get("sinks")),
            alibi_slopes=_kw.get("alibi_slopes", tensors.get("alibi_slopes")),
            qq_bias=_kw.get("qq_bias", tensors.get("qq_bias")),
            backend=backend,
            stream=int(_kw.get("stream", stream)),
            tuning_spec=spec,
        )

    grid = kwargs.get("grid") or (0, 0, 0)
    block = kwargs.get("block") or (0, 0, 0)
    return TorchBinding(launch=launch, grid=tuple(grid), block=tuple(block))


def bind_wmma_attention_torch(
    request, spec, tensors: Mapping[str, Any], **kwargs
) -> TorchBinding:
    """Bind a gfx1250 WMMA spec to dense ``q``/``k``/``v``/``out`` tensors."""
    import struct

    from kernels.gfx1250.wmma_attention_fwd import (
        build_wmma_attention_fwd,
        wmma_attention_fwd_grid,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    grid = wmma_attention_fwd_grid(
        spec, seqlen_q=int(request.seqlen_q), batch=int(request.batch)
    )
    block = (int(spec.block_size), 1, 1)
    scale_log2 = float(
        kwargs.get(
            "scale_log2",
            1.0 / math.sqrt(int(spec.head_size)) * math.log2(math.e),
        )
    )

    def launch(**_kw):
        q, k, v, out = tensors["q"], tensors["k"], tensors["v"], tensors["out"]
        kernel = build_wmma_attention_fwd(spec, arch=str(request.arch))
        art = compile_kernel(kernel, arch=str(request.arch))
        rt = Runtime()
        module = rt.load_module(art.hsaco)
        fn = module.get_function(art.kernel_name)
        hq = int(spec.num_query_heads)
        hk = int(spec.num_kv_heads)
        d = int(spec.head_size)
        packed = struct.pack(
            "<QQQQfiiiiiiiiii",
            int(q.data_ptr()),
            int(k.data_ptr()),
            int(v.data_ptr()),
            int(out.data_ptr()),
            float(_kw.get("scale_log2", scale_log2)),
            int(request.seqlen_q),
            int(request.seqlen_k),
            hq * d,
            d,
            hk * d,
            d,
            hk * d,
            d,
            hq * d,
            d,
        )
        rt.launch(fn, grid, block, packed)
        rt.sync()
        module.unload()
        return out

    return TorchBinding(launch=launch, grid=grid, block=block)


# Cached modules stay alive across bindings; asynchronous launches retain the
# supplied tensor owners through KernelLauncher until the caller's stream drain.

_GFX1151_LAUNCHERS: Dict[Tuple[str, Any], Any] = {}

_LOG2E = math.log2(math.e)


def _gfx1151_launcher(spec, arch: str):
    """Build-or-fetch the cached ``KernelLauncher`` for one resolved spec."""
    key = (str(arch), spec)
    launcher = _GFX1151_LAUNCHERS.get(key)
    if launcher is not None:
        return launcher
    from kernels.gfx1151.wmma_fmha_fwd import (
        build_wmma_fmha_fwd,
        wmma_fmha_fwd_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    kernel = build_wmma_fmha_fwd(spec, arch=arch)
    artifact = compile_kernel(kernel, arch=arch, capture_ir_text=False)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=wmma_fmha_fwd_signature(spec),
        cache_key=key,
    )
    _GFX1151_LAUNCHERS[key] = launcher
    return launcher


def _gfx1151_q_kind(spec) -> str:
    return "bf16" if str(spec.dtype).lower() == "bf16" else "fp16"


def _gfx1151_kv_kind(spec) -> str:
    return "fp8" if spec.kv_dtype else _gfx1151_q_kind(spec)


def _fits_i32(value: Any, name: str) -> int:
    value = int(value)
    if value < 0 or value > 0x7FFF_FFFF:
        raise ValueError(f"{name}={value} does not fit an I32 kernel argument")
    return value


def _check_max_element_offset_i32(tensor, name: str) -> None:
    """Reject tensors whose highest addressable element index exceeds I32.

    The kernel forms element offsets in I32 (``token * stride + head * stride``),
    so a large KV cache would otherwise wrap silently.
    """
    shape = _shape(tensor, name)
    stride = getattr(tensor, "stride", None)
    if not callable(stride):
        raise ValueError(f"{name} must expose strides")
    if any(int(extent) <= 0 for extent in shape):
        return
    highest = sum((int(e) - 1) * int(stride(i)) for i, e in enumerate(shape))
    if highest > 0x7FFF_FFFF:
        raise ValueError(
            f"{name} spans element offset {highest}, which does not fit the "
            "I32 offset arithmetic of the gfx1151 kernel; split the tensor "
            "(e.g. per batch or per page range) before launching"
        )


def _check_last_dim_contiguous(tensor, name: str) -> None:
    shape = _shape(tensor, name)
    stride = getattr(tensor, "stride", None)
    if not callable(stride):
        raise ValueError(f"{name} must expose strides")
    if int(stride(len(shape) - 1)) != 1:
        raise ValueError(f"{name} innermost dimension must be contiguous")


def _check_gfx1151_device(tensors: Mapping[str, Any]) -> None:
    present = {name: t for name, t in tensors.items() if t is not None}
    if not present:
        return
    devices = {str(getattr(t, "device", None)) for t in present.values()}
    if len(devices) != 1:
        raise ValueError(
            f"all gfx1151 attention tensors must share one device, got {devices}"
        )
    device = getattr(next(iter(present.values())), "device", None)
    device_type = getattr(device, "type", str(device).split(":")[0])
    if str(device_type).lower() != "cuda":
        raise ValueError("gfx1151 attention tensors must be on a HIP/CUDA device")


def _check_gfx1151_qo(
    tensor, name: str, kind: str, num_heads: int, head_size: int
) -> None:
    actual = _dtype_kind(tensor, name)
    if actual != kind:
        raise ValueError(f"{name} dtype must be {kind}, got {actual}")
    shape = _shape(tensor, name)
    if len(shape) < 2 or shape[-1] != head_size or shape[-2] != num_heads:
        raise ValueError(
            f"{name} trailing dims must be [..., {num_heads}, {head_size}], got {shape}"
        )
    _check_last_dim_contiguous(tensor, name)


def _check_gfx1151_vector_alignment(
    tensor, name: str, itemsize: int, width: int
) -> None:
    alignment = max(16, itemsize * width)
    if int(tensor.data_ptr()) % alignment:
        raise ValueError(f"{name} base address must be aligned to {alignment} bytes")
    if width > 1:
        for axis in range(len(tensor.shape) - 1):
            if int(tensor.stride(axis)) % width:
                raise ValueError(
                    f"{name} strides must preserve {width}-element vector alignment"
                )


def _gfx1151_dense_needs_per_batch(request, tensors: Mapping[str, Any]) -> bool:
    """True when a dense tensor is not batch-folded (batch stride != S*token stride).

    The kernel derives each batch's base as ``batch * seqlen * token_stride``;
    any other batch stride (broadcast, padded, ``[B, H, S, D]`` permuted views)
    is served by one launch per batch on sliced views.
    """
    batch = int(request.batch)
    if batch <= 1:
        return False
    for name, seqlen in (
        ("q", request.seqlen_q),
        ("out", request.seqlen_q),
        ("k", request.seqlen_k),
        ("v", request.seqlen_k),
    ):
        tensor = tensors[name]
        if int(tensor.stride(0)) != int(seqlen) * int(tensor.stride(1)):
            return True
    return False


def _gfx1151_validate_and_collect(
    request, spec, tensors: Mapping[str, Any]
) -> Dict[str, Any]:
    """Structural (shape/dtype/stride) gate for a gfx1151 WMMA launch.

    Deliberately never reads tensor *contents* (``cu_seqlens_q``/``cu_seqlens_k``/
    ``seqused_k``/``block_table`` values are loaded on-device by the kernel):
    only shapes, dtypes, and strides -- all host-visible metadata -- are
    checked, so this never triggers a GPU length readback or host
    densification. Returns the base kernel-arg values (everything except the
    runtime-overridable scalars ``scale_log2``/``softcap``/``k_scale``/
    ``v_scale``, which the caller merges in per bind/launch).
    """
    layout = spec.layout
    if layout not in ("dense", "ragged", "paged"):
        raise ValueError(f"unsupported gfx1151 attention layout {layout!r}")
    if spec.transposed_qk and (
        int(request.seqlen_q) % spec.q_rows_per_cta
        or int(request.seqlen_k) % spec.block_n
    ):
        raise ValueError("transposed QK requires complete query and KV tiles")
    required = ("q", "k", "v", "out")
    missing = [name for name in required if tensors.get(name) is None]
    if missing:
        raise ValueError("missing gfx1151 attention tensors: " + ", ".join(missing))
    q, k, v, out = (tensors[name] for name in required)
    q_kind = _gfx1151_q_kind(spec)
    kv_kind = _gfx1151_kv_kind(spec)
    if kv_kind == "fp8" and _dtype_kind(k, "k") == "fp8_fnuz":
        raise ValueError(
            "gfx1151 WMMA FP8 KV storage is OCP fp8e4m3 (not FNUZ); "
            "got fp8_fnuz for k"
        )
    for name, tensor, rank in (
        ("q", q, 4 if layout == "dense" else 3),
        ("out", out, 4 if layout == "dense" else 3),
        ("k", k, 3 if layout == "ragged" else 4),
        ("v", v, 3 if layout == "ragged" else 4),
    ):
        if len(_shape(tensor, name)) != rank:
            raise ValueError(f"{name} must be rank-{rank} for {layout} layout")
    _check_gfx1151_qo(q, "q", q_kind, spec.num_query_heads, spec.head_size)
    _check_gfx1151_qo(out, "out", q_kind, spec.num_query_heads, spec.v_dim)
    _check_gfx1151_qo(k, "k", kv_kind, spec.kv_heads, spec.head_size)
    _check_gfx1151_qo(v, "v", kv_kind, spec.kv_heads, spec.v_dim)
    if _shape(out, "out")[:-1] != _shape(q, "q")[:-1]:
        raise ValueError("out shape must match q shape except the head dim")
    if _shape(v, "v")[:-1] != _shape(k, "k")[:-1]:
        raise ValueError("v shape must match k shape except the head dim")
    _check_gfx1151_vector_alignment(q, "q", 2, 16)
    _check_gfx1151_vector_alignment(k, "k", 1 if spec.kv_dtype else 2, 16)
    _check_gfx1151_vector_alignment(
        v, "v", 1 if spec.kv_dtype else 2, 8 if spec.v_lds_stage else 1
    )
    _check_gfx1151_vector_alignment(out, "out", 2, 1)
    out_shape = _shape(out, "out")
    inner_span = spec.v_dim
    for stride, extent in sorted(
        (
            (int(out.stride(-2)), spec.num_query_heads),
            (int(out.stride(-3)), int(out_shape[-3])),
        )
    ):
        if extent <= 1:
            continue
        if stride < inner_span:
            raise ValueError("out token/head strides must not overlap")
        inner_span = stride * extent
    _check_gfx1151_device(tensors)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        _check_max_element_offset_i32(tensor, name)

    values: Dict[str, Any] = {
        "Q": q,
        "K": k,
        "V": v,
        "O": out,
        "seqlen_q": _fits_i32(request.seqlen_q, "seqlen_q"),
        "seqlen_k": _fits_i32(request.seqlen_k, "seqlen_k"),
    }

    if layout == "dense":
        batch = int(request.batch)
        seqlen_q = int(request.seqlen_q)
        seqlen_k = int(request.seqlen_k)
        for tensor, name, shape0, seqlen in (
            (q, "q", batch, seqlen_q),
            (out, "out", batch, seqlen_q),
            (k, "k", batch, seqlen_k),
            (v, "v", batch, seqlen_k),
        ):
            shape = _shape(tensor, name)
            if len(shape) != 4:
                raise ValueError(
                    f"{name} must be rank-4 [B, S, H, D] for dense layout, got {shape}"
                )
            if shape[0] != shape0 or shape[1] != seqlen:
                raise ValueError(
                    f"{name} shape[0:2] must be [{shape0}, {seqlen}], got {shape[:2]}"
                )
            if batch > 1 and int(tensor.stride(0)) < 0:
                raise ValueError(f"{name} batch stride must be non-negative")
            if batch > 1 and name == "out":
                span = (
                    (seqlen - 1) * int(tensor.stride(1))
                    + (spec.num_query_heads - 1) * int(tensor.stride(2))
                    + spec.v_dim
                )
                if int(tensor.stride(0)) < span:
                    raise ValueError("out batch stride must not overlap")
        values.update(
            {
                "stride_q_token": _fits_i32(q.stride(1), "stride_q_token"),
                "stride_q_head": _fits_i32(q.stride(2), "stride_q_head"),
                "stride_k_token": _fits_i32(k.stride(1), "stride_k_token"),
                "stride_k_head": _fits_i32(k.stride(2), "stride_k_head"),
                "stride_v_token": _fits_i32(v.stride(1), "stride_v_token"),
                "stride_v_head": _fits_i32(v.stride(2), "stride_v_head"),
                "stride_o_token": _fits_i32(out.stride(1), "stride_o_token"),
                "stride_o_head": _fits_i32(out.stride(2), "stride_o_head"),
            }
        )
    else:
        for tensor, name in ((q, "q"), (out, "out")):
            shape = _shape(tensor, name)
            if len(shape) != 3:
                raise ValueError(
                    f"{name} must be rank-3 [tokens, H, D] for a packed layout, got {shape}"
                )
        values.update(
            {
                "stride_q_token": _fits_i32(q.stride(0), "stride_q_token"),
                "stride_q_head": _fits_i32(q.stride(1), "stride_q_head"),
                "stride_o_token": _fits_i32(out.stride(0), "stride_o_token"),
                "stride_o_head": _fits_i32(out.stride(1), "stride_o_head"),
            }
        )
        cu_seqlens_q = tensors.get("cu_seqlens_q")
        if cu_seqlens_q is None:
            raise ValueError(f"{layout} layout requires tensors['cu_seqlens_q']")
        batch = int(request.batch)
        if _dtype_kind(cu_seqlens_q, "cu_seqlens_q") != "int32":
            raise ValueError("cu_seqlens_q dtype must be int32")
        if _shape(cu_seqlens_q, "cu_seqlens_q") != (batch + 1,):
            raise ValueError(f"cu_seqlens_q must have shape [{batch + 1}]")
        _check_last_dim_contiguous(cu_seqlens_q, "cu_seqlens_q")
        values["cu_seqlens_q"] = cu_seqlens_q

        if layout == "ragged":
            cu_seqlens_k = tensors.get("cu_seqlens_k")
            if cu_seqlens_k is None:
                raise ValueError("ragged layout requires tensors['cu_seqlens_k']")
            if _dtype_kind(cu_seqlens_k, "cu_seqlens_k") != "int32":
                raise ValueError("cu_seqlens_k dtype must be int32")
            if _shape(cu_seqlens_k, "cu_seqlens_k") != (batch + 1,):
                raise ValueError(f"cu_seqlens_k must have shape [{batch + 1}]")
            _check_last_dim_contiguous(cu_seqlens_k, "cu_seqlens_k")
            k_shape = _shape(k, "k")
            if len(k_shape) != 3:
                raise ValueError(
                    f"k must be rank-3 [tokens, H, D] for ragged layout, got {k_shape}"
                )
            values["cu_seqlens_k"] = cu_seqlens_k
            values["stride_k_token"] = _fits_i32(k.stride(0), "stride_k_token")
            values["stride_k_head"] = _fits_i32(k.stride(1), "stride_k_head")
            values["stride_v_token"] = _fits_i32(v.stride(0), "stride_v_token")
            values["stride_v_head"] = _fits_i32(v.stride(1), "stride_v_head")
        else:  # paged
            seqused_k = tensors.get("seqused_k")
            block_table = tensors.get("block_table")
            if seqused_k is None or block_table is None:
                raise ValueError(
                    "paged layout requires tensors['seqused_k'] and tensors['block_table']"
                )
            if _dtype_kind(seqused_k, "seqused_k") != "int32":
                raise ValueError("seqused_k dtype must be int32")
            if _shape(seqused_k, "seqused_k") != (batch,):
                raise ValueError(f"seqused_k must have shape [{batch}]")
            _check_last_dim_contiguous(seqused_k, "seqused_k")
            if _dtype_kind(block_table, "block_table") != "int32":
                raise ValueError("block_table dtype must be int32")
            table_shape = _shape(block_table, "block_table")
            if len(table_shape) != 2 or table_shape[0] != batch or table_shape[1] <= 0:
                raise ValueError(
                    f"block_table shape must be [{batch}, max_pages], got {table_shape}"
                )
            table_stride = getattr(block_table, "stride", None)
            if not callable(table_stride):
                raise ValueError("block_table must expose strides")
            if int(table_stride(1)) != 1:
                raise ValueError("block_table innermost stride must be 1")
            row_stride = _fits_i32(table_stride(0), "block_table_stride")
            if row_stride < table_shape[1]:
                raise ValueError("block_table row stride must be non-overlapping")
            k_shape = _shape(k, "k")
            v_shape = _shape(v, "v")
            if len(k_shape) != 4 or k_shape[1] != int(spec.page_block_size):
                raise ValueError(
                    f"k cache must be rank-4 [pages, {spec.page_block_size}, H, D], got {k_shape}"
                )
            if v_shape != k_shape:
                raise ValueError(
                    f"v cache shape {v_shape} must match k cache shape {k_shape}"
                )
            values["seqused_k"] = seqused_k
            values["block_table"] = block_table
            values["block_table_stride"] = row_stride
            values["stride_k_block"] = _fits_i32(k.stride(0), "stride_k_block")
            values["stride_v_block"] = _fits_i32(v.stride(0), "stride_v_block")
            values["stride_k_token"] = _fits_i32(k.stride(1), "stride_k_token")
            values["stride_k_head"] = _fits_i32(k.stride(2), "stride_k_head")
            values["stride_v_token"] = _fits_i32(v.stride(1), "stride_v_token")
            values["stride_v_head"] = _fits_i32(v.stride(2), "stride_v_head")

    if layout == "dense" and _gfx1151_dense_needs_per_batch(request, tensors):
        for name, tensor, itemsize in (
            ("q", q, 2),
            ("k", k, 1 if spec.kv_dtype else 2),
            ("v", v, 1 if spec.kv_dtype else 2),
            ("out", out, 2),
        ):
            if int(tensor.stride(0)) * itemsize % 16:
                raise ValueError(
                    f"{name} batch stride must keep per-batch views 16-byte aligned"
                )

    if spec.use_sinks:
        sinks = tensors.get("sinks")
        if sinks is None:
            raise ValueError("spec.use_sinks requires tensors['sinks']")
        if _dtype_kind(sinks, "sinks") != q_kind:
            raise ValueError(
                f"sinks dtype must be {q_kind}, got {_dtype_kind(sinks, 'sinks')}"
            )
        if _shape(sinks, "sinks") != (spec.num_query_heads,):
            raise ValueError(f"sinks must have shape [{spec.num_query_heads}]")
        _check_last_dim_contiguous(sinks, "sinks")
        values["sink_ptr"] = sinks
    if spec.store_lse:
        lse = tensors.get("lse")
        if lse is None:
            raise ValueError("spec.store_lse requires tensors['lse']")
        if "float32" not in str(getattr(lse, "dtype", "")).lower():
            raise ValueError("lse dtype must be float32")
        lse_shape = _shape(lse, "lse")
        if layout == "dense":
            expected = (int(request.batch), spec.num_query_heads, int(request.seqlen_q))
            if lse_shape != expected:
                raise ValueError(f"lse must have shape {list(expected)}, got {lse_shape}")
            strides = (
                spec.num_query_heads * int(request.seqlen_q),
                int(request.seqlen_q),
                1,
            )
            for axis, (extent, want) in enumerate(zip(lse_shape, strides)):
                if extent > 1 and int(lse.stride(axis)) != want:
                    raise ValueError("lse must be contiguous [B, H, Sq]")
        else:
            expected = (spec.num_query_heads, int(_shape(q, "q")[0]))
            if lse_shape != expected:
                raise ValueError(f"lse must have shape {list(expected)}, got {lse_shape}")
            _check_last_dim_contiguous(lse, "lse")
            if expected[0] > 1 and int(lse.stride(0)) < expected[1]:
                raise ValueError("lse head stride must not overlap")
            values["stride_lse_head"] = _fits_i32(lse.stride(0), "stride_lse_head")
        _check_max_element_offset_i32(lse, "lse")
        values["lse"] = lse
    if spec.use_alibi:
        alibi = tensors.get("alibi_slopes")
        if alibi is None:
            raise ValueError("spec.use_alibi requires tensors['alibi_slopes']")
        if "float32" not in str(getattr(alibi, "dtype", "")).lower():
            raise ValueError("alibi_slopes dtype must be float32")
        if _shape(alibi, "alibi_slopes") != (spec.num_query_heads,):
            raise ValueError(f"alibi_slopes must have shape [{spec.num_query_heads}]")
        _check_last_dim_contiguous(alibi, "alibi_slopes")
        values["alibi_slopes_ptr"] = alibi
    if spec.use_qq_bias:
        qq_bias = tensors.get("qq_bias")
        if qq_bias is None:
            raise ValueError("spec.use_qq_bias requires tensors['qq_bias']")
        if "float32" not in str(getattr(qq_bias, "dtype", "")).lower():
            raise ValueError("qq_bias dtype must be float32")
        bias_shape = _shape(qq_bias, "qq_bias")
        if len(bias_shape) != 2:
            raise ValueError(f"qq_bias must be rank-2 [rows, cols], got {bias_shape}")
        _check_last_dim_contiguous(qq_bias, "qq_bias")
        _check_max_element_offset_i32(qq_bias, "qq_bias")
        values["qq_bias_ptr"] = qq_bias
        values["qq_bias_rows"] = _fits_i32(bias_shape[0], "qq_bias_rows")
        values["qq_bias_cols"] = _fits_i32(bias_shape[1], "qq_bias_cols")
        values["qq_bias_stride"] = _fits_i32(qq_bias.stride(0), "qq_bias_stride")

    return values


def bind_gfx1151_attention_torch(
    request, spec, tensors: Mapping[str, Any], **kwargs
) -> TorchBinding:
    """Bind a resolved gfx1151 ``WmmaFmhaFwdSpec`` to caller-owned tensors.

    Handles all three layouts (``spec.layout`` ``dense``/``ragged``/``paged``)
    and FP16/BF16 or OCP fp8e4m3 KV storage (``spec.kv_dtype``). Tensor keys:
    ``q``/``k``/``v``/``out`` (required), plus layout metadata
    (``cu_seqlens_q``, ``cu_seqlens_k``, ``seqused_k``, ``block_table``) and
    optional score inputs (``sinks``, ``alibi_slopes``, ``qq_bias``) as
    declared by ``spec``. When ``spec.store_lse`` is set, ``lse`` is a required
    FP32 output: ``[B, H, Sq]`` (dense) or ``[H, total_q]`` (packed), holding the
    natural-log softmax statistic (``-inf`` for a fully masked row). Runtime scalar kwargs: ``softmax_scale`` (default
    ``1/sqrt(D)``), ``softcap`` (required, positive, when ``spec.use_softcap``),
    ``k_scale``/``v_scale`` (required FP32 dequant scales when
    ``spec.kv_dtype`` is set), ``stream`` (HIP stream handle; ``0``/omitted
    resolves to torch's current stream), and ``fence`` (per-call
    ``LaunchConfig.fence``; default ``True`` -- a stream-scoped
    ``hipStreamSynchronize``, never a device-wide sync). All of these may be
    overridden again on the returned binding's ``launch(**kwargs)`` call.

    Metadata validation (shapes/dtypes/strides) happens once here, at bind
    time; it never reads ``cu_seqlens*``/``seqused_k``/``block_table``
    *contents* (that would require a GPU->host readback on every rebind).
    The compiled ``KernelLauncher`` is cached by ``(arch, spec)``, so binding
    the same spec again reuses the already-loaded module.
    """
    from kernels.gfx1151.wmma_fmha_fwd import wmma_fmha_fwd_grid

    arch = str(request.arch)
    base_values = _gfx1151_validate_and_collect(request, spec, tensors)
    per_batch = spec.layout == "dense" and _gfx1151_dense_needs_per_batch(
        request, tensors
    )
    grid = wmma_fmha_fwd_grid(
        spec,
        seqlen_q=int(request.seqlen_q),
        batch=1 if per_batch else int(request.batch),
    )
    block = (int(spec.block_size), 1, 1)

    kv_dtype = spec.kv_dtype
    softmax_scale_default = float(
        kwargs.get("softmax_scale", 1.0 / math.sqrt(int(spec.head_size)))
    )
    softcap_default = kwargs.get("softcap")
    if spec.use_softcap and softcap_default is None:
        raise ValueError("spec.use_softcap requires kwargs['softcap']")
    if spec.use_softcap:
        softcap_default = float(softcap_default)
        if not math.isfinite(softcap_default) or softcap_default <= 0:
            raise ValueError("softcap must be finite and positive")
    if kv_dtype:
        if "k_scale" not in kwargs or "v_scale" not in kwargs:
            raise ValueError(
                "spec.kv_dtype='fp8e4m3' requires explicit kwargs['k_scale'] "
                "and kwargs['v_scale']"
            )
    k_scale_default = float(kwargs.get("k_scale", 1.0))
    v_scale_default = float(kwargs.get("v_scale", 1.0))
    stream_default = kwargs.get("stream", 0)
    fence_default = bool(kwargs.get("fence", True))
    # Cheap structural/kwargs validation above is complete; only now does a
    # cache miss compile+load a module (never on any of the error paths above).
    launcher = _gfx1151_launcher(spec, arch)

    def launch(**_kw):
        from rocke.runtime.launcher import LaunchConfig

        values = dict(base_values)
        values["scale_log2"] = (
            float(_kw.get("softmax_scale", softmax_scale_default)) * _LOG2E
        )
        if spec.use_softcap:
            softcap = float(_kw.get("softcap", softcap_default))
            if not math.isfinite(softcap) or softcap <= 0:
                raise ValueError("softcap must be finite and positive")
            values["softcap"] = softcap
        if kv_dtype:
            values["k_scale"] = float(_kw.get("k_scale", k_scale_default))
            values["v_scale"] = float(_kw.get("v_scale", v_scale_default))
        stream = _kw.get("stream", stream_default)
        config = LaunchConfig(
            grid=grid,
            block=block,
            stream=0 if stream is None else int(stream),
            fence=bool(_kw.get("fence", fence_default)),
        )
        if not per_batch:
            launcher(values, config=config)
            return tensors["out"]
        last = int(request.batch) - 1
        for b in range(last + 1):
            batch_values = dict(values)
            for key, name in (("Q", "q"), ("K", "k"), ("V", "v"), ("O", "out")):
                batch_values[key] = tensors[name][b : b + 1]
            if spec.store_lse:
                batch_values["lse"] = tensors["lse"][b : b + 1]
            launcher(
                batch_values,
                config=LaunchConfig(
                    grid=grid,
                    block=block,
                    stream=config.stream,
                    fence=config.fence and b == last,
                ),
            )
        return tensors["out"]

    return TorchBinding(launch=launch, grid=grid, block=block)
