# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch bindings shared by executable attention dispatch candidates.

The dispatcher owns tensor-to-runner adaptation.  Torch remains a lazy runtime
dependency: this module only closes over caller-owned tensors and imports
architecture kernel runners inside binding calls.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Mapping

from rocke.dispatch.core import TorchBinding


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
        accepted = inspect.signature(run).parameters
        for name, value in optional.items():
            if name in accepted and value is not None:
                call[name] = value
        return run(**call)

    return TorchBinding(launch=launch, grid=grid_fn(spec), block=block_fn(spec))


def bind_tuning_attention_torch(
    request, spec, tensors: Mapping[str, Any], **kwargs
) -> TorchBinding:
    """Bind an explicit unified 2D/3D tuning spec to paged tensors."""
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
    assert isinstance(problem, UnifiedAttentionProblem)
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
