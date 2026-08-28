# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Two-stage deterministic backward-weight convolution launcher.

Assembles a :class:`~rocke.runtime.launcher.PipelineLauncher` from:

* **Stage 1** — an implicit-GEMM wgrad kernel (``two_stage=True``) that
  writes f32 partial sums into a workspace buffer instead of
  atomic-adding into ``dW``.
* **Stage 2** — a workspace-reduce kernel that sums the f32 workspace
  slices in a fixed sequential order and writes the result as ``dtype_d``
  to ``dW``.

Both stages are submitted on the same HIP stream, so HIP's in-order
execution guarantees Stage 2 begins only after Stage 1 has completed —
no explicit ``hipStreamSynchronize`` is needed between them.

Usage::

    from dataclasses import replace
    spec = WgradConvSpec(problem=..., split_k=4, two_stage=True)
    pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(spec, arch)

    ws = DeviceMem(ws_nbytes)
    runtime.memset(ws.ptr(), 0, ws_nbytes)  # optional: belt-and-suspenders; not required

    s1_vals = {"A": dY_ptr, "B": X_ptr, "D": dW_ptr,
               "A_bytes": dY_nb, "B_bytes": X_nb, "D_bytes": dW_nb,
               "ws_ptr": ws.ptr(), "ws_bytes": ws_nbytes}
    s2_vals = {"ws_ptr": ws.ptr(), "dw_ptr": dw_ptr,
               "wg_M": spec.wg_M, "wg_N": spec.wg_N,
               "split_k": spec.split_k,
               "ws_bytes": ws_nbytes, "dw_bytes": dw_nb}

    pipeline((s1_vals, s2_vals), (s1_cfg, s2_cfg), stream=stream)
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Tuple

from .conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    _wg_M,
    _wg_N,
    build_implicit_gemm_conv_wgrad,
)
from .conv_wgrad_workspace_reduce import (
    WgradReduceSpec,
    build_conv_wgrad_workspace_reduce,
    wgrad_reduce_grid,
    wgrad_reduce_signature,
)


def wgrad_two_stage_workspace_nbytes(spec: WgradConvSpec) -> int:
    """Return workspace bytes required for the two-stage deterministic path.

    Always f32 (4 bytes per element), shape ``[groups * split_k, wg_M, wg_N]``
    where ``wg_M = kpg`` and ``wg_N = Y*X*cpg`` are the per-group GEMM dimensions.
    ``blockIdx.z = group*split_k + k_id`` indexes directly into this flat array,
    giving each (group, split-K slice) pair a unique workspace region.
    """
    return spec.problem.groups * spec.split_k * spec.wg_M * spec.wg_N * 4


def _wgrad_stage1_signature(spec: WgradConvSpec) -> list:
    """Signature for the Stage 1 wgrad kernel (two_stage=True).

    Extends the standard conv ABI (A/B/D + byte sizes) with two extra
    parameters for the workspace: ``ws_ptr`` and ``ws_bytes``.

    A (dY), B (X), and D (dW) each carry their own element type so that
    mixed-dtype configurations (e.g. bf16 inputs with fp32 output) are
    described correctly.
    """
    _dtype_map = {
        "fp16": "f16",
        "bf16": "bf16",
        "fp32": "f32",
        "f16": "f16",
        "f32": "f32",
    }

    def _ir(dt: str) -> str:
        return _dtype_map.get(dt, dt)

    return [
        {
            "name": "A",
            "type": f"ptr<{_ir(spec.data.dtype_a)}, global>",
            "size_bytes": 8,
        },
        {
            "name": "B",
            "type": f"ptr<{_ir(spec.data.dtype_b)}, global>",
            "size_bytes": 8,
        },
        {
            "name": "D",
            "type": f"ptr<{_ir(spec.data.dtype_d)}, global>",
            "size_bytes": 8,
        },
        {"name": "A_bytes", "type": "i32", "size_bytes": 4},
        {"name": "B_bytes", "type": "i32", "size_bytes": 4},
        {"name": "D_bytes", "type": "i32", "size_bytes": 4},
        {"name": "ws_ptr", "type": "ptr<f32, global>", "size_bytes": 8},
        {"name": "ws_bytes", "type": "i32", "size_bytes": 4},
    ]


def build_implicit_gemm_conv_wgrad_two_stage(
    spec: WgradConvSpec,
    arch: str = "gfx950",
) -> Tuple["PipelineLauncher", int]:
    """Build a two-stage deterministic wgrad pipeline.

    Args:
        spec:   A :class:`WgradConvSpec` with ``split_k > 1``.  The
                ``two_stage`` flag is forced to ``True`` internally.
        arch:   Target GPU architecture string (e.g. ``"gfx942"``).

    Returns:
        A ``(pipeline, workspace_nbytes)`` tuple where ``pipeline`` is a
        :class:`~rocke.runtime.launcher.PipelineLauncher` over two stages and
        ``workspace_nbytes`` is the size (bytes) of the f32 scratch buffer the
        caller must allocate before each pipeline call.

        The workspace has shape ``[split_k, wg_M, wg_N]`` (f32).  Stage 1
        writes every element within ``[0, wg_M) × [0, wg_N)`` via plain
        stores (no atomics); OOB positions are skipped by a per-element
        ``scf_if`` guard.  Stage 2 wraps its entire reduction loop in the
        same OOB guard, so out-of-bounds threads perform no workspace loads
        at all.  Zero-initialising the workspace is therefore not required
        for correctness::

            pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(spec, arch)
            ws = DeviceMem(ws_nbytes)
            pipeline((s1_vals, s2_vals), (s1_cfg, s2_cfg), stream=stream)

        Both stages are submitted on the same HIP stream. HIP same-stream FIFO
        ordering guarantees Stage 2 observes Stage 1's stores — no explicit
        ``hipStreamSynchronize`` is needed between them.

    Raises:
        ValueError: if ``spec.split_k <= 1`` (two-stage requires split_k > 1).
    """
    if spec.split_k <= 1:
        raise ValueError(
            f"build_implicit_gemm_conv_wgrad_two_stage requires split_k > 1, "
            f"got split_k={spec.split_k}"
        )

    # Lazy imports: keep module import-time safe for static IR tests running
    # without a HIP runtime.
    from ...helpers.compile import compile_kernel
    from ...runtime.launcher import KernelLauncher, PipelineLauncher

    # ---- Stage 1: wgrad GEMM → f32 workspace --------------------------------
    s1_spec = dc_replace(spec, two_stage=True)
    s1_kernel = build_implicit_gemm_conv_wgrad(s1_spec, arch)
    s1_artifact = compile_kernel(s1_kernel, arch=arch, capture_ir_text=False)
    s1_sig = _wgrad_stage1_signature(s1_spec)
    s1_launcher = KernelLauncher(
        hsaco=s1_artifact.hsaco,
        kernel_name=s1_artifact.kernel_name,
        signature=s1_sig,
        cache_key=("conv_wgrad_two_stage_s1", s1_spec.kernel_name()),
    )

    # ---- Stage 2: workspace → dW (sequential reduce) ------------------------
    s2_spec = WgradReduceSpec(
        problem=spec.problem,
        dtype_d=spec.data.dtype_d,
    )
    s2_kernel = build_conv_wgrad_workspace_reduce(s2_spec, arch)
    s2_artifact = compile_kernel(s2_kernel, arch=arch, capture_ir_text=False)
    s2_sig = wgrad_reduce_signature(s2_spec)
    s2_launcher = KernelLauncher(
        hsaco=s2_artifact.hsaco,
        kernel_name=s2_artifact.kernel_name,
        signature=s2_sig,
        cache_key=("conv_wgrad_two_stage_s2", s2_spec.kernel_name()),
    )

    pipeline = PipelineLauncher([s1_launcher, s2_launcher])
    ws_nbytes = wgrad_two_stage_workspace_nbytes(s1_spec)
    return pipeline, ws_nbytes
