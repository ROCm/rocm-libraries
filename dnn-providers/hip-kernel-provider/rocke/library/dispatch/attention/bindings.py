# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch bindings shared by executable attention dispatch candidates.

The dispatcher owns tensor-to-runner adaptation.  Torch remains a lazy runtime
dependency: this module only closes over caller-owned tensors and imports
architecture kernel runners inside binding calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

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
    if "float32" in value or value.endswith("fp32"):
        return "fp32"
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
    if bool(request.emit_lse) != problem.emit_lse:
        raise ValueError(
            f"request.emit_lse={bool(request.emit_lse)} disagrees with "
            f"problem.emit_lse={problem.emit_lse}"
        )

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
    "gfx942": frozenset({"cu_seqlens_q", "cu_seqlens_kv", "sinks", "lse"}),
    "gfx950": frozenset(
        {"cu_seqlens_q", "cu_seqlens_kv", "block_tables", "kv_lens", "sinks", "lse"}
    ),
}


def _dense_runner(arch: str):
    if arch == "gfx942":
        from kernels.gfx942.attention_dense import run_attention_dense_torch
    elif arch == "gfx950":
        from kernels.gfx950.attention_dense import run_attention_dense_torch
    else:
        raise ValueError(f"no dense attention Torch runner for arch {arch!r}")
    return run_attention_dense_torch


def bind_dense_attention_torch(
    request, tuning_spec, tensors: Mapping[str, Any], **kwargs
) -> TorchBinding:
    """Bind a dense ``AttentionTuningSpec`` to ``q``/``k``/``v``/``out`` tensors.

    Grid and block come from the spec, the same values the candidate reports.
    An ``emit_lse`` spec also needs ``tensors["lse"]``: a caller-owned FP32
    buffer the kernel fills in place (the runner validates it).
    """
    run = _dense_runner(str(getattr(request, "arch", "")))
    spec = tuning_spec.kernel_spec
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
            "lse": _kw.get("lse", tensors.get("lse")),
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

    return TorchBinding(
        launch=launch,
        grid=tuple(tuning_spec.launch_grid()),
        block=tuple(tuning_spec.launch_block()),
    )


_ADDITIVE_BIAS_INPUTS = (
    "additive_bias",
    "additive_bias_batch_stride",
    "additive_bias_head_stride",
    "additive_bias_sq_stride",
)


def _reject_additive_bias_inputs(inputs: Mapping[str, Any], where: str) -> None:
    supplied = [name for name in _ADDITIVE_BIAS_INPUTS if name in inputs]
    if supplied:
        raise ValueError(
            "explicit tuning bindings do not support additive bias; got "
            + ", ".join(supplied)
            + f" in {where}"
        )


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
    # The caller-owned LSE buffer must be bound exactly when the problem asks
    # for it, so misuse fails here and not at the first launch. Its shape,
    # dtype, device and overlap are validated by the runner, which needs torch.
    has_lse = tensors.get("lse") is not None
    if problem.emit_lse and not has_lse:
        raise ValueError("emit_lse=True requires tensors['lse']")
    if has_lse and not problem.emit_lse:
        raise ValueError("tensors['lse'] was given but emit_lse is off")
    # Explicit specs are built without additive bias (the spec makers reject
    # it), so refuse a bias-on problem or bias inputs here instead of launching
    # a kernel that would silently ignore them.
    if problem.use_additive_bias:
        raise ValueError("explicit tuning bindings do not support additive bias")
    _reject_additive_bias_inputs(tensors, "tensors")
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
        _reject_additive_bias_inputs(_kw, "launch kwargs")
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
            lse=_kw.get("lse", tensors.get("lse")),
        )

    return TorchBinding(
        launch=launch,
        grid=tuple(spec.launch_grid(problem)),
        block=tuple(spec.launch_block()),
    )


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


# ---------------------------------------------------------------- RDNA WMMA
# Cached modules stay alive across bindings; asynchronous launches retain the
# supplied tensor owners through KernelLauncher until the caller's stream drain.

_RDNA_LAUNCHERS: Dict[Tuple[Any, ...], Any] = {}

_LOG2E = math.log2(math.e)


def _rdna_launcher(key: Tuple[Any, ...], build, signature):
    """Build-or-fetch the cached ``KernelLauncher`` under ``key``."""
    launcher = _RDNA_LAUNCHERS.get(key)
    if launcher is not None:
        return launcher
    from rocke.helpers import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    arch = key[0]
    artifact = compile_kernel(build(), arch=arch, capture_ir_text=False)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=signature,
        cache_key=key,
    )
    _RDNA_LAUNCHERS[key] = launcher
    return launcher


def _prefill_launcher(spec, arch: str):
    from kernels.gfx1151.wmma_fmha_fwd_v2 import (
        build_wmma_fmha_fwd_v2,
        wmma_fmha_fwd_v2_signature,
    )

    return _rdna_launcher(
        (str(arch), spec),
        lambda: build_wmma_fmha_fwd_v2(spec, arch=arch),
        wmma_fmha_fwd_v2_signature(spec),
    )


def _decode_launcher(spec, arch: str):
    from kernels.gfx1151.wmma_fmha_decode import (
        build_wmma_fmha_decode,
        wmma_fmha_decode_signature,
    )

    return _rdna_launcher(
        (str(arch), spec),
        lambda: build_wmma_fmha_decode(spec, arch=arch),
        wmma_fmha_decode_signature(spec),
    )


def _merge_launcher(spec, arch: str):
    """The merge kernel: one per (arch, dtype, packed rows), shared by every
    decode object with those."""
    from kernels.gfx1151.wmma_fmha_decode import (
        build_wmma_fmha_decode_merge,
        wmma_fmha_decode_merge_signature,
    )

    return _rdna_launcher(
        (str(arch), "wmma_fmha_decode_merge", spec.dtype, spec.packed),
        lambda: build_wmma_fmha_decode_merge(spec, arch=arch),
        wmma_fmha_decode_merge_signature(spec),
    )


@dataclass(frozen=True)
class _RdnaView:
    """What the shared RDNA tensor validation reads from either kernel spec."""

    decode: bool
    dtype: str
    q_bucket: int
    v_bucket: int
    layout: str
    page_block_size: int
    kv_fp8: bool
    softcap: bool
    sinks: bool
    alibi: bool
    qq_bias: bool
    bias: Optional[str]  # None | "f32" | "q"


def _rdna_view(spec) -> _RdnaView:
    from kernels.gfx1151.wmma_fmha_decode import WmmaFmhaDecodeSpec

    if isinstance(spec, WmmaFmhaDecodeSpec):
        return _RdnaView(
            decode=True,
            dtype=spec.dtype,
            q_bucket=spec.head_size,
            v_bucket=spec.head_size,
            layout=spec.layout,
            page_block_size=spec.page_block_size,
            kv_fp8=bool(spec.kv_dtype),
            softcap=spec.softcap,
            sinks=spec.sinks,
            alibi=spec.alibi,
            qq_bias=spec.qq_bias,
            bias=spec.bias or None,
        )
    return _RdnaView(
        decode=False,
        dtype=spec.dtype,
        q_bucket=spec.head_dim,
        v_bucket=spec.dv,
        layout=spec.layout,
        page_block_size=spec.page_block_size,
        kv_fp8=bool(spec.kv_dtype),
        softcap=spec.use_softcap,
        sinks=spec.use_sinks,
        alibi=spec.use_alibi,
        qq_bias=spec.use_qq_bias,
        bias=spec.bias,
    )


def _fits_i32(value: Any, name: str) -> int:
    value = int(value)
    if value < 0 or value > 0x7FFF_FFFF:
        raise ValueError(f"{name}={value} does not fit an I32 kernel argument")
    return value


def _check_byte_extent_i32(tensor, name: str, element_bytes: int) -> None:
    """Reject tensors whose addressed byte extent exceeds I32.

    The kernels form element offsets in I32 (``token * stride + head *
    stride``) and scale them to I32 byte offsets for buffer loads against a
    descriptor of at most 2 GiB, so the bound is on bytes, not elements.
    Matches the native pack's ``addressFitsI32``.
    """
    shape = _shape(tensor, name)
    stride = getattr(tensor, "stride", None)
    if not callable(stride):
        raise ValueError(f"{name} must expose strides")
    if any(int(extent) <= 0 for extent in shape):
        return
    highest = sum((int(e) - 1) * int(stride(i)) for i, e in enumerate(shape))
    extent = (highest + 1) * int(element_bytes)
    if extent > 0x7FFF_FFFF:
        raise ValueError(
            f"{name} spans {extent} bytes, which does not fit the I32 "
            "offset arithmetic of the RDNA attention kernels; split the tensor "
            "(e.g. per batch or per page range) before launching"
        )


def _check_rows_aligned(tensor, name: str, element_bytes: int, align: int) -> None:
    """Reject a tensor whose rows can start off an ``align``-byte boundary
    (data pointer or any outer stride): the kernels load whole rows as
    ``align``-byte vectors."""
    shape = _shape(tensor, name)
    stride = getattr(tensor, "stride", None)
    if not callable(stride):
        raise ValueError(f"{name} must expose strides")
    if int(tensor.data_ptr()) % align:
        raise ValueError(
            f"{name} data pointer must be {align}-byte aligned for the RDNA "
            "attention row loads"
        )
    for axis in range(len(shape) - 1):
        if int(shape[axis]) > 1 and (int(stride(axis)) * element_bytes) % align:
            raise ValueError(
                f"{name} stride({axis})={int(stride(axis))} elements must be a "
                f"multiple of {align} bytes for the RDNA attention row loads"
            )


def _check_last_dim_contiguous(tensor, name: str) -> None:
    shape = _shape(tensor, name)
    stride = getattr(tensor, "stride", None)
    if not callable(stride):
        raise ValueError(f"{name} must expose strides")
    if int(stride(len(shape) - 1)) != 1:
        raise ValueError(f"{name} innermost dimension must be contiguous")


def _check_rdna_device(tensors: Mapping[str, Any]) -> None:
    present = {name: t for name, t in tensors.items() if t is not None}
    if not present:
        return
    devices = {str(getattr(t, "device", None)) for t in present.values()}
    if len(devices) != 1:
        raise ValueError(
            f"all RDNA attention tensors must share one device, got {devices}"
        )
    device = getattr(next(iter(present.values())), "device", None)
    device_type = getattr(device, "type", str(device).split(":")[0])
    if str(device_type).lower() != "cuda":
        raise ValueError("RDNA attention tensors must be on a HIP/CUDA device")


def _check_packed_rows(
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


def _fits_signed_i32(value: Any, name: str) -> int:
    value = int(value)
    if value < -0x8000_0000 or value > 0x7FFF_FFFF:
        raise ValueError(f"{name}={value} does not fit a signed I32 kernel argument")
    return value


def _check_storage(tensor, name: str, itemsize: int) -> None:
    shape = _shape(tensor, name)
    stride_fn = getattr(tensor, "stride", None)
    if not callable(stride_fn):
        raise ValueError(f"{name} must expose strides")
    strides = tuple(int(stride_fn(axis)) for axis in range(len(shape)))
    if strides[-1] != 1:
        raise ValueError(f"{name} innermost dimension must be contiguous")
    if int(tensor.data_ptr()) % itemsize:
        raise ValueError(f"{name} base address must be {itemsize}-byte aligned")
    span = 1
    for stride, extent in sorted(
        (stride, extent) for stride, extent in zip(strides, shape) if extent > 1
    ):
        if stride < span:
            raise ValueError(f"{name} strides overlap for shape {shape}: {strides}")
        span += (extent - 1) * stride


def _dense_axis_order(request, q) -> tuple[int, int, str]:
    batch = int(request.batch)
    seqlen = int(request.seqlen_q)
    heads = int(request.nhead_q)
    dim = int(request.hdim_q)
    requested = str(request.tensor_layout).strip().lower()
    candidates = []
    if _shape(q, "q") == (batch, seqlen, heads, dim):
        candidates.append((1, 2, "bshd"))
    if _shape(q, "q") == (batch, heads, seqlen, dim):
        candidates.append((2, 1, "bhsd"))
    if requested != "auto":
        candidates = [axes for axes in candidates if axes[2] == requested]
    elif len(candidates) > 1:
        candidates = [axes for axes in candidates if axes[2] == "bshd"]
    if len(candidates) != 1:
        expected = (
            f"[B,S,H,D]=[{batch},{seqlen},{heads},{dim}] or "
            f"[B,H,S,D]=[{batch},{heads},{seqlen},{dim}]"
        )
        raise ValueError(f"q shape must be {expected}")
    return candidates[0]


def _rdna_head_widths(request, view: _RdnaView) -> Tuple[int, int]:
    """The launch's Q/K and V/O head widths: the request's, multiples of 8
    inside the object's bucket (the decode bucket also bounds Q/K below)."""
    widths = []
    for name, value, bucket in (
        ("hdim_q", request.hdim_q, view.q_bucket),
        ("hdim_v", request.hdim_v, view.v_bucket),
    ):
        value = int(value)
        if value < 8 or value > bucket or value % 8:
            raise ValueError(
                f"{name}={value} must be a multiple of 8 in [8, {bucket}] for "
                "this RDNA attention object"
            )
        widths.append(value)
    if view.decode:
        from kernels.gfx1151.wmma_fmha_decode import wmma_fmha_decode_bucket

        # Bucket D serves Q/K widths from its own minimum up (narrower Q/K
        # would read K columns the kernel never masks).
        if (
            wmma_fmha_decode_bucket(head_dim_q=widths[0], head_dim_v=view.q_bucket)
            != view.q_bucket
        ):
            raise ValueError(
                f"hdim_q={widths[0]} is below what the decode bucket "
                f"{view.q_bucket} serves"
            )
    return widths[0], widths[1]


def _rdna_v_heads(num_v_heads: int, request, view: _RdnaView) -> int:
    hq, hk = int(request.nhead_q), int(request.nhead_k)
    if num_v_heads <= 0 or hq % num_v_heads:
        raise ValueError(
            f"V head count {num_v_heads} must divide the query head count {hq}"
        )
    if view.decode and max(hk, num_v_heads) % min(hk, num_v_heads):
        raise ValueError(
            f"the decode kernel needs one of the K head count {hk} and the V "
            f"head count {num_v_heads} to divide the other"
        )
    return _fits_i32(num_v_heads, "num_v_heads")


def _rdna_validate_and_collect(
    request, spec, tensors: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate one launch and collect the runtime arguments both RDNA kernels
    share (pointers, shape, mask, strides, feature operands)."""
    from .gfx1151 import mask_arguments

    view = _rdna_view(spec)
    layout = view.layout
    if layout not in ("dense", "ragged", "paged"):
        raise ValueError(f"unsupported RDNA attention layout {layout!r}")
    required = ["q", "k", "v", "out"]
    if bool(request.emit_lse):
        required.append("lse")
    missing = [name for name in required if tensors.get(name) is None]
    if missing:
        raise ValueError("missing RDNA attention tensors: " + ", ".join(missing))
    q, k, v, out = (tensors[name] for name in ("q", "k", "v", "out"))
    lse = tensors.get("lse")
    q_kind = "bf16" if view.dtype == "bf16" else "fp16"
    kv_kind = "fp8" if view.kv_fp8 else q_kind
    if view.kv_fp8 and _dtype_kind(k, "k") == "fp8_fnuz":
        raise ValueError(
            "RDNA WMMA FP8 KV storage is OCP fp8e4m3 (not FNUZ); got fp8_fnuz for k"
        )
    for name, tensor, rank in (
        ("q", q, 4 if layout == "dense" else 3),
        ("out", out, 4 if layout == "dense" else 3),
        ("k", k, 3 if layout == "ragged" else 4),
        ("v", v, 3 if layout == "ragged" else 4),
    ):
        if len(_shape(tensor, name)) != rank:
            raise ValueError(f"{name} must be rank-{rank} for {layout} layout")
    _check_rdna_device(tensors)
    kv_bytes = 1 if view.kv_fp8 else 2
    for name, tensor, element_bytes in (
        ("q", q, 2),
        ("k", k, kv_bytes),
        ("v", v, kv_bytes),
        ("out", out, 2),
    ):
        _check_byte_extent_i32(tensor, name, element_bytes)
        # 16-bit rows load as 16-byte vectors, FP8 K/V rows as 8-byte ones.
        _check_rows_aligned(
            tensor, name, element_bytes, 8 if element_bytes == 1 else 16
        )
    head_q, head_v = _rdna_head_widths(request, view)
    # A runtime_head_dims request takes its V head count from the v tensor.
    runtime_v_heads = bool(request.runtime_head_dims)
    num_v_heads = int(request.nhead_k)

    bottom_right, window_left, window_right = mask_arguments(request)
    values: Dict[str, Any] = {
        "Q": q,
        "K": k,
        "V": v,
        "O": out,
        "LSE": lse if lse is not None else out,
        "seqlen_q": _fits_i32(request.seqlen_q, "seqlen_q"),
        "seqlen_k": _fits_i32(request.seqlen_k, "seqlen_k"),
        "num_query_heads": _fits_i32(request.nhead_q, "num_query_heads"),
        "num_kv_heads": _fits_i32(request.nhead_k, "num_kv_heads"),
        "head_dim_q": head_q,
        "head_dim_v": head_v,
        "bottom_right": int(bottom_right),
        "window_left": _fits_signed_i32(window_left, "window_left"),
        "window_right": _fits_signed_i32(window_right, "window_right"),
        "write_lse": int(bool(request.emit_lse)),
        "stride_q_batch": 0,
        "stride_k_batch": 0,
        "stride_v_batch": 0,
        "stride_o_batch": 0,
        "stride_lse_batch": 0,
        "stride_lse_token": 0,
        "stride_lse_head": 0,
    }

    if layout == "dense":
        token_axis, head_axis, dense_layout = _dense_axis_order(request, q)
        batch = int(request.batch)
        seqlen_q = int(request.seqlen_q)
        seqlen_k = int(request.seqlen_k)

        def validate_dense(tensor, name, kind, seqlen, heads, head_dim):
            expected = (
                (batch, seqlen, heads, head_dim)
                if dense_layout == "bshd"
                else (batch, heads, seqlen, head_dim)
            )
            if _shape(tensor, name) != expected:
                raise ValueError(
                    f"{name} must have {dense_layout.upper()} shape {expected}, "
                    f"got {_shape(tensor, name)}"
                )
            actual = _dtype_kind(tensor, name)
            if actual != kind:
                raise ValueError(f"{name} dtype must be {kind}, got {actual}")
            _check_storage(tensor, name, 1 if kind == "fp8" else 2)

        if runtime_v_heads:
            num_v_heads = _shape(v, "v")[head_axis]
        num_v_heads = _rdna_v_heads(num_v_heads, request, view)
        validate_dense(q, "q", q_kind, seqlen_q, int(request.nhead_q), head_q)
        validate_dense(out, "out", q_kind, seqlen_q, int(request.nhead_q), head_v)
        validate_dense(k, "k", kv_kind, seqlen_k, int(request.nhead_k), head_q)
        validate_dense(v, "v", kv_kind, seqlen_k, num_v_heads, head_v)
        values.update(
            {
                "stride_q_batch": _fits_i32(q.stride(0), "stride_q_batch"),
                "stride_q_token": _fits_i32(q.stride(token_axis), "stride_q_token"),
                "stride_q_head": _fits_i32(q.stride(head_axis), "stride_q_head"),
                "stride_k_batch": _fits_i32(k.stride(0), "stride_k_batch"),
                "stride_k_token": _fits_i32(k.stride(token_axis), "stride_k_token"),
                "stride_k_head": _fits_i32(k.stride(head_axis), "stride_k_head"),
                "stride_v_batch": _fits_i32(v.stride(0), "stride_v_batch"),
                "stride_v_token": _fits_i32(v.stride(token_axis), "stride_v_token"),
                "stride_v_head": _fits_i32(v.stride(head_axis), "stride_v_head"),
                "stride_o_batch": _fits_i32(out.stride(0), "stride_o_batch"),
                "stride_o_token": _fits_i32(out.stride(token_axis), "stride_o_token"),
                "stride_o_head": _fits_i32(out.stride(head_axis), "stride_o_head"),
            }
        )
        if bool(request.emit_lse):
            shape = _shape(lse, "lse")
            if _dtype_kind(lse, "lse") != "fp32":
                raise ValueError("lse dtype must be FP32")
            if shape == (batch, int(request.nhead_q), seqlen_q):
                lse_token_axis, lse_head_axis = 2, 1
            else:
                expected = (
                    (batch, seqlen_q, int(request.nhead_q), 1)
                    if dense_layout == "bshd"
                    else (batch, int(request.nhead_q), seqlen_q, 1)
                )
                if shape != expected:
                    raise ValueError(
                        "lse must be FP32 [B,H,S] or the rank-4 "
                        f"{dense_layout.upper()} shape {expected}, got {shape}"
                    )
                lse_token_axis, lse_head_axis = token_axis, head_axis
            _check_storage(lse, "lse", 4)
            _check_byte_extent_i32(lse, "lse", 4)
            values.update(
                {
                    "stride_lse_batch": _fits_i32(lse.stride(0), "stride_lse_batch"),
                    "stride_lse_token": _fits_i32(
                        lse.stride(lse_token_axis), "stride_lse_token"
                    ),
                    "stride_lse_head": _fits_i32(
                        lse.stride(lse_head_axis), "stride_lse_head"
                    ),
                }
            )
    else:
        if runtime_v_heads:
            num_v_heads = _shape(v, "v")[-2]
        num_v_heads = _rdna_v_heads(num_v_heads, request, view)
        _check_packed_rows(q, "q", q_kind, int(request.nhead_q), head_q)
        _check_packed_rows(out, "out", q_kind, int(request.nhead_q), head_v)
        _check_packed_rows(k, "k", kv_kind, int(request.nhead_k), head_q)
        _check_packed_rows(v, "v", kv_kind, num_v_heads, head_v)
        if _shape(out, "out")[:-1] != _shape(q, "q")[:-1]:
            raise ValueError("out shape must match q shape except the head dim")
        if _shape(v, "v")[:-2] != _shape(k, "k")[:-2]:
            raise ValueError("v shape must match k shape except the head count and dim")
        for name, tensor, itemsize in (
            ("q", q, 2),
            ("k", k, kv_bytes),
            ("v", v, kv_bytes),
            ("out", out, 2),
        ):
            _check_storage(tensor, name, itemsize)
        values.update(
            {
                "stride_q_token": _fits_i32(q.stride(0), "stride_q_token"),
                "stride_q_head": _fits_i32(q.stride(1), "stride_q_head"),
                "stride_o_token": _fits_i32(out.stride(0), "stride_o_token"),
                "stride_o_head": _fits_i32(out.stride(1), "stride_o_head"),
            }
        )
        if bool(request.emit_lse):
            lse_shape = _shape(lse, "lse")
            total_q = _shape(q, "q")[0]
            heads = int(request.nhead_q)
            if _dtype_kind(lse, "lse") != "fp32":
                raise ValueError("lse dtype must be FP32")
            if lse_shape == (heads, total_q):
                lse_head_axis, lse_token_axis = 0, 1
            elif lse_shape in ((total_q, heads), (total_q, heads, 1)):
                lse_token_axis, lse_head_axis = 0, 1
            else:
                raise ValueError(
                    "lse must be packed [H,total_q], [total_q,H], or "
                    f"[total_q,H,1], got {lse_shape}"
                )
            _check_storage(lse, "lse", 4)
            _check_byte_extent_i32(lse, "lse", 4)
            values["stride_lse_token"] = _fits_i32(
                lse.stride(lse_token_axis), "stride_lse_token"
            )
            values["stride_lse_head"] = _fits_i32(
                lse.stride(lse_head_axis), "stride_lse_head"
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
            values["cu_seqlens_k"] = cu_seqlens_k
            values["stride_k_token"] = _fits_i32(k.stride(0), "stride_k_token")
            values["stride_k_head"] = _fits_i32(k.stride(1), "stride_k_head")
            values["stride_v_token"] = _fits_i32(v.stride(0), "stride_v_token")
            values["stride_v_head"] = _fits_i32(v.stride(1), "stride_v_head")
        else:
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
            if not callable(table_stride) or int(table_stride(1)) != 1:
                raise ValueError("block_table innermost stride must be 1")
            row_stride = _fits_i32(table_stride(0), "block_table_stride")
            if row_stride < table_shape[1]:
                raise ValueError("block_table row stride must be non-overlapping")
            if _shape(k, "k")[1] != int(view.page_block_size):
                raise ValueError(
                    f"k cache page length must be {view.page_block_size}, "
                    f"got {_shape(k, 'k')}"
                )
            values.update(
                {
                    "seqused_k": seqused_k,
                    "block_table": block_table,
                    "block_table_stride": row_stride,
                    "stride_k_block": _fits_i32(k.stride(0), "stride_k_block"),
                    "stride_v_block": _fits_i32(v.stride(0), "stride_v_block"),
                    # Every block_table entry is bounded to this page count on
                    # device (v's matches k's, checked above).
                    "num_blocks": _fits_i32(_shape(k, "k")[0], "num_blocks"),
                    "stride_k_token": _fits_i32(k.stride(1), "stride_k_token"),
                    "stride_k_head": _fits_i32(k.stride(2), "stride_k_head"),
                    "stride_v_token": _fits_i32(v.stride(1), "stride_v_token"),
                    "stride_v_head": _fits_i32(v.stride(2), "stride_v_head"),
                }
            )
    values["num_v_heads"] = num_v_heads

    if view.sinks:
        sinks = tensors.get("sinks")
        if sinks is None:
            raise ValueError("sinks require tensors['sinks']")
        if _dtype_kind(sinks, "sinks") != q_kind:
            raise ValueError(
                f"sinks dtype must be {q_kind}, got {_dtype_kind(sinks, 'sinks')}"
            )
        if _shape(sinks, "sinks") != (int(request.nhead_q),):
            raise ValueError(f"sinks must have shape [{request.nhead_q}]")
        _check_last_dim_contiguous(sinks, "sinks")
        values["sink_ptr"] = sinks
    if view.alibi:
        alibi = tensors.get("alibi_slopes")
        if alibi is None:
            raise ValueError("ALiBi requires tensors['alibi_slopes']")
        if _dtype_kind(alibi, "alibi_slopes") != "fp32":
            raise ValueError("alibi_slopes dtype must be float32")
        if _shape(alibi, "alibi_slopes") != (int(request.nhead_q),):
            raise ValueError(f"alibi_slopes must have shape [{request.nhead_q}]")
        _check_last_dim_contiguous(alibi, "alibi_slopes")
        values["alibi_slopes_ptr"] = alibi
    if view.qq_bias:
        qq_bias = tensors.get("qq_bias")
        if qq_bias is None:
            raise ValueError("QQ-bias requires tensors['qq_bias']")
        if _dtype_kind(qq_bias, "qq_bias") != "fp32":
            raise ValueError("qq_bias dtype must be float32")
        bias_shape = _shape(qq_bias, "qq_bias")
        if len(bias_shape) != 2:
            raise ValueError(f"qq_bias must be rank-2 [rows, cols], got {bias_shape}")
        _check_last_dim_contiguous(qq_bias, "qq_bias")
        _check_byte_extent_i32(qq_bias, "qq_bias", 4)
        values["qq_bias_ptr"] = qq_bias
        values["qq_bias_rows"] = _fits_i32(bias_shape[0], "qq_bias_rows")
        values["qq_bias_cols"] = _fits_i32(bias_shape[1], "qq_bias_cols")
        values["qq_bias_stride"] = _fits_i32(qq_bias.stride(0), "qq_bias_stride")
    if view.bias:
        attn_bias = tensors.get("attn_bias")
        if attn_bias is None:
            raise ValueError("an additive bias requires tensors['attn_bias']")
        want_kind = "fp32" if view.bias == "f32" else q_kind
        bias_kind = _dtype_kind(attn_bias, "attn_bias")
        if bias_kind != want_kind:
            raise ValueError(
                f"attn_bias dtype must be {want_kind} for this object's "
                f"{view.bias!r} bias, got {bias_kind}"
            )
        bias_shape = _shape(attn_bias, "attn_bias")
        if len(bias_shape) != 4:
            raise ValueError(
                f"attn_bias must be rank-4 [B|1, H|1, Sq|1, Sk], got {bias_shape}"
            )
        for axis, (extent, full, label) in enumerate(
            (
                (bias_shape[0], int(request.batch), "batch"),
                (bias_shape[1], int(request.nhead_q), "head"),
                (bias_shape[2], int(request.seqlen_q), "query"),
            )
        ):
            if extent not in (1, full):
                raise ValueError(
                    f"attn_bias dim {axis} ({label}) must be 1 or {full}, got {extent}"
                )
        if bias_shape[3] != int(request.seqlen_k):
            raise ValueError(
                f"attn_bias key dim must equal seqlen_k={int(request.seqlen_k)}, "
                f"got {bias_shape[3]}"
            )
        if bias_shape[3] > 1 and int(attn_bias.stride(3)) != 1:
            raise ValueError("attn_bias innermost (key) dimension must be contiguous")
        _check_byte_extent_i32(attn_bias, "attn_bias", 4 if bias_kind == "fp32" else 2)
        values["attn_bias_ptr"] = attn_bias
        for key, axis in (
            ("bias_stride_b", 0),
            ("bias_stride_h", 1),
            ("bias_stride_q", 2),
        ):
            values[key] = (
                0 if bias_shape[axis] == 1 else _fits_i32(attn_bias.stride(axis), key)
            )
    return values


def _rdna_softmax_scale(value) -> float:
    """The RDNA kernels fold ``softmax_scale * log2(e)`` into an online
    softmax that requires it finite and positive (zero or negative gives NaN
    or zero rows), so reject anything else."""
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            "softmax_scale must be finite and positive for the RDNA attention "
            f"kernels, got {scale}"
        )
    return scale


def _rdna_scalars(view: _RdnaView, head_q: int, kwargs: Mapping[str, Any]):
    """Bind-time runtime scalars and the per-launch resolver over them."""
    softmax_scale = _rdna_softmax_scale(
        kwargs.get("softmax_scale", 1.0 / math.sqrt(head_q))
    )
    softcap = kwargs.get("softcap")
    if view.softcap:
        if softcap is None:
            raise ValueError("softcap objects require kwargs['softcap']")
        softcap = float(softcap)
        if not math.isfinite(softcap) or softcap <= 0:
            raise ValueError("softcap must be finite and positive")
    if view.kv_fp8 and ("k_scale" not in kwargs or "v_scale" not in kwargs):
        raise ValueError(
            "FP8 K/V objects require explicit kwargs['k_scale'] and kwargs['v_scale']"
        )
    k_scale = float(kwargs.get("k_scale", 1.0))
    v_scale = float(kwargs.get("v_scale", 1.0))
    stream_default = kwargs.get("stream", 0)
    fence_default = bool(kwargs.get("fence", True))

    def resolve(values: Dict[str, Any], kw: Mapping[str, Any]):
        values["scale_log2"] = (
            _rdna_softmax_scale(kw.get("softmax_scale", softmax_scale)) * _LOG2E
        )
        if view.softcap:
            cap = float(kw.get("softcap", softcap))
            if not math.isfinite(cap) or cap <= 0:
                raise ValueError("softcap must be finite and positive")
            values["softcap"] = cap
        if view.kv_fp8:
            values["k_scale"] = float(kw.get("k_scale", k_scale))
            values["v_scale"] = float(kw.get("v_scale", v_scale))
        stream = kw.get("stream", stream_default)
        return (0 if stream is None else int(stream)), bool(
            kw.get("fence", fence_default)
        )

    return resolve


def bind_rdna_prefill_torch(
    request, spec, tensors: Mapping[str, Any], **kwargs
) -> TorchBinding:
    """Bind a ``WmmaFmhaFwdV2Spec`` (RDNA prefill) to caller-owned tensors.

    Handles dense BSHD/BHSD tensors with arbitrary non-overlapping outer
    strides, packed ragged/paged layouts, FP16/BF16, and OCP fp8e4m3 KV
    storage. Tensor keys are ``q``/``k``/``v``/``out`` plus caller-owned
    ``lse`` when ``request.emit_lse`` is true, layout metadata
    (``cu_seqlens_q``/``cu_seqlens_k``/``seqused_k``/``block_table``) and the
    enabled score inputs (``sinks``, ``alibi_slopes``, ``qq_bias``,
    ``attn_bias``). LSE accepts live-head ``[B,H,S]`` / ``[H,total_q]``
    layouts and layout-shaped forms.

    Head widths are the request's ``hdim_q``/``hdim_v``: multiples of 8 inside
    the object's buckets. Rows load as 16-byte vectors (8-byte for FP8 K/V),
    so data pointers and every outer stride must keep that alignment. The V
    head count is ``nhead_k``, or the ``v`` tensor's for a
    ``runtime_head_dims`` request (any count dividing ``nhead_q``).

    Runtime scalar kwargs are ``softmax_scale`` (default ``1/sqrt(hdim_q)``),
    ``softcap`` (required and positive when enabled), ``k_scale``/``v_scale``
    for FP8 KV, ``stream``, and ``fence``. Validation reads shapes, dtypes and
    strides only (never the contents of the layout metadata); the kernels bound
    every ``block_table`` entry to the cache's page count on device (an entry
    outside ``[0, k.shape[0])`` reads as zero K / V rows). The compiled
    ``KernelLauncher`` is cached by ``(arch, spec)``.
    """
    from kernels.gfx1151.wmma_fmha_fwd_v2 import wmma_fmha_fwd_v2_grid

    arch = str(request.arch)
    base_values = _rdna_validate_and_collect(request, spec, tensors)
    base_values["batch"] = _fits_i32(request.batch, "batch")
    grid, block = wmma_fmha_fwd_v2_grid(
        spec,
        batch=int(request.batch),
        seqlen_q=int(request.seqlen_q),
        num_query_heads=int(request.nhead_q),
    )
    resolve = _rdna_scalars(_rdna_view(spec), base_values["head_dim_q"], kwargs)
    # Validation is complete; only now does a cache miss compile+load a module.
    launcher = _prefill_launcher(spec, arch)

    def launch(**_kw):
        from rocke.runtime.launcher import LaunchConfig

        values = dict(base_values)
        stream, fence = resolve(values, _kw)
        launcher(
            values,
            config=LaunchConfig(grid=grid, block=block, stream=stream, fence=fence),
        )
        return tensors["out"]

    return TorchBinding(launch=launch, grid=grid, block=block)


def bind_rdna_decode_torch(
    request, spec, tensors: Mapping[str, Any], **kwargs
) -> TorchBinding:
    """Bind a resolved ``WmmaFmhaDecodeSpec`` (RDNA split-KV decode) to
    caller-owned tensors; the tensor, layout and scalar contract is
    :func:`bind_rdna_prefill_torch`'s (the decode bucket serves Q/K widths from
    its own minimum up, and with a V head count other than ``nhead_k`` one of
    the K/V head counts must divide the other).

    The split count comes from the shape and the CU count
    (``gfx1151.request_decode_plan``; the live device's on device). One split
    launches the decode kernel alone, writing O and LSE. More splits use a
    split workspace the binding allocates and owns (``launch.workspace``) for
    its lifetime: up to 4 splits merge in the last split to finish (its arrival
    counters are zeroed once here; the kernel re-arms them), more launch the
    merge kernel after the decode kernel on the same stream. Launches of one
    binding therefore share that workspace and must not overlap: issue them on
    one stream (or synchronize between streams), as for its output tensors.
    """
    from rocke.runtime.launcher import DeviceMem

    from .gfx1151 import request_decode_plan

    arch = str(request.arch)
    base_values = _rdna_validate_and_collect(request, spec, tensors)
    plan = request_decode_plan(request, spec, num_v_heads=base_values["num_v_heads"])
    base_values["num_splits"] = plan.num_splits
    resolve = _rdna_scalars(_rdna_view(spec), base_values["head_dim_q"], kwargs)
    layout = plan.workspace
    # Validation is complete; only now does a cache miss compile+load a module
    # or the workspace get allocated.
    launcher = _decode_launcher(spec, arch)
    merge = (
        _merge_launcher(spec, arch) if plan.num_splits > 1 and not plan.fused else None
    )
    workspace = DeviceMem(layout["bytes"])
    if layout["bytes"]:
        base = workspace.ptr()
        if plan.fused:
            from rocke.runtime.hip_module import Runtime

            # The arrival counters start at zero; the device-wide sync orders
            # the null-stream memset before launches on any stream.
            rt = Runtime()
            rt.memset(base + layout["ws_cnt"], 0, layout["bytes"] - layout["ws_cnt"])
            rt.sync()
        # WS_O starts the allocation: passing the owner keeps it alive for any
        # asynchronous launch still queued when the binding is dropped.
        base_values["WS_O"] = workspace
        base_values["WS_LSE"] = base + layout["ws_lse"]
        base_values["WS_CNT"] = base + layout["ws_cnt"] if plan.fused else 0
    else:
        base_values.update(WS_O=0, WS_LSE=0, WS_CNT=0)
    merge_keys = (
        "O",
        "LSE",
        "WS_O",
        "WS_LSE",
        "seqlen_q",
        "num_query_heads",
        "head_dim_v",
        "num_splits",
        "write_lse",
    ) + tuple(
        f"stride_{t}_{a}" for t in ("o", "lse") for a in ("batch", "token", "head")
    )
    if spec.packed:
        merge_keys += ("cu_seqlens_q",)

    def launch(**_kw):
        from rocke.runtime.launcher import LaunchConfig

        values = dict(base_values)
        stream, fence = resolve(values, _kw)
        launcher(
            values,
            config=LaunchConfig(
                grid=plan.grid,
                block=plan.block,
                stream=stream,
                fence=fence and merge is None,
            ),
        )
        if merge is not None:
            merge(
                {key: values[key] for key in merge_keys},
                config=LaunchConfig(
                    grid=plan.merge_grid,
                    block=plan.merge_block,
                    stream=stream,
                    fence=fence,
                ),
            )
        return tensors["out"]

    launch.workspace = workspace
    launch.plan = plan
    return TorchBinding(launch=launch, grid=plan.grid, block=plan.block)
