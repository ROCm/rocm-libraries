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
from typing import Any, Mapping

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
