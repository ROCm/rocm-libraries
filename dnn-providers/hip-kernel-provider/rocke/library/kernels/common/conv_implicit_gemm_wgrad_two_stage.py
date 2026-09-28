# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Two-stage backward-weight convolution launcher.

Assembles a :class:`~rocke.runtime.launcher.PipelineLauncher` from:

* **Stage 1** — an implicit-GEMM wgrad kernel (``two_stage=True``) that
  f32-atomic-adds its partial sums into a scratch buffer instead of
  16-bit-atomic-adding into ``dW``.
* **Stage 2** — a cast kernel that converts the finished f32 scratch to
  ``dtype_d`` and writes ``dW``.

This is the route split-K takes when the packed ``<2 x dtype>`` atomic that
writes a 16-bit ``dW`` directly cannot address the problem -- it needs an even
``wg_N = Y*X*cpg``, while ``atomicrmw fadd f32`` has no alignment constraint.
The reduction over ``split_k`` is done by the hardware atomics, so Stage 2 is
one load / convert / store per element and does not scale with the degree.

Both stages are submitted on the same HIP stream, so HIP's in-order
execution guarantees Stage 2 begins only after Stage 1 has completed —
no explicit ``hipStreamSynchronize`` is needed between them.

**The caller must zero the scratch before every Stage 1 launch.** Stage 1
accumulates into it; stale content is added to the result.

Grouped convolutions (``groups > 1``) are fully supported.  Stage 1 uses grid
``z = groups * split_k`` and decodes the group from it, so every slice of a
group lands on that group's single scratch slab.  Stage 2 uses grid
``z = groups`` and casts all groups in one launch.

Usage::

    from dataclasses import replace
    spec = WgradConvSpec(problem=..., split_k=4, two_stage=True)
    pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(spec, arch=arch)

    ws = DeviceMem(ws_nbytes)
    ws.memset(0)                      # REQUIRED: Stage 1 accumulates

    s1_vals = {"A": dY_ptr, "B": X_ptr, "D": dW_ptr,
               "A_bytes": dY_nb, "B_bytes": X_nb, "D_bytes": dW_nb,
               "ws_ptr": ws.ptr(), "ws_bytes": ws_nbytes}
    s2_vals = {"ws_ptr": ws.ptr(), "dw_ptr": dw_ptr,
               "wg_M": spec.wg_M, "wg_N": spec.wg_N,
               "ws_bytes": ws_nbytes, "dw_bytes": dw_nb,
               "groups": spec.problem.groups}

    pipeline((s1_vals, s2_vals), (s1_cfg, s2_cfg), stream=stream)
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Tuple


from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    _wg_K,
    _wg_M,
    _wg_N,
    build_implicit_gemm_conv_wgrad,
)
from kernels.common.conv_wgrad_workspace_reduce import (
    WgradReduceSpec,
    build_conv_wgrad_workspace_reduce,
    wgrad_reduce_grid,
    wgrad_reduce_signature,
)


def wgrad_two_stage_workspace_nbytes(spec: WgradConvSpec) -> int:
    """Return scratch bytes required for the two-stage path.

    Always f32 (4 bytes per element), shape ``[groups * R, wg_M, wg_N]`` where
    ``R = spec.ws_replicas`` and ``wg_M = kpg`` / ``wg_N = Y*X*cpg`` are the
    per-group GEMM dimensions.

    There is **no** ``split_k`` factor: a group's K-slices atomic-add on top of
    each other across its ``R`` slabs, so the scratch is ``R`` copies of ``dW``
    and does not grow with the reduction degree. ``R`` trades scratch footprint
    against L2 atomic contention -- see the ``ws_replicas`` field docs on
    :class:`WgradConvSpec`.

    The caller must zero this buffer before each Stage 1 launch -- Stage 1
    accumulates into it rather than overwriting it.
    """
    return spec.problem.groups * spec.ws_replicas * spec.wg_M * spec.wg_N * 4


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
    *,
    arch: str = "gfx950",
) -> tuple:
    """Build a two-stage wgrad pipeline (f32 scratch atomics + cast).

    Args:
        spec:   A :class:`WgradConvSpec` with ``split_k > 1``.  The
                ``two_stage`` flag is forced to ``True`` internally.
        arch:   Target GPU architecture string (e.g. ``"gfx942"``).

    Returns:
        A ``(pipeline, workspace_nbytes)`` tuple where ``pipeline`` is a
        :class:`~rocke.runtime.launcher.PipelineLauncher` over two stages and
        ``workspace_nbytes`` is the size (bytes) of the f32 scratch buffer the
        caller must allocate before each pipeline call.

        The scratch has shape ``[groups, wg_M, wg_N]`` (f32) -- the shape of
        ``dW``, with no ``split_k`` factor.  Stage 1 f32-atomic-adds every
        element within ``[0, wg_M) × [0, wg_N)``; OOB positions are skipped by
        a per-element ``scf_if`` guard.

        **The scratch must be zeroed before each pipeline call.** Stage 1
        accumulates into it, so whatever is already there is added to the
        result -- including the previous call's output::

            pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(spec, arch=arch)
            ws = DeviceMem(ws_nbytes)
            ws.memset(0)
            pipeline((s1_vals, s2_vals), (s1_cfg, s2_cfg), stream=stream)

        Both stages are submitted on the same HIP stream. HIP same-stream FIFO
        ordering guarantees Stage 2 observes Stage 1's stores — no explicit
        ``hipStreamSynchronize`` is needed between them.

    Raises:
        ValueError: if ``spec.split_k <= 1`` after auto-resolution.
    """
    # Resolve split_k=-1 (auto sentinel) before the guard so callers can pass
    # split_k=-1 and get the heuristic value rather than a confusing rejection.
    if spec.split_k == -1:
        from rocke.helpers.split_k import select_split_k_wgrad

        resolved = select_split_k_wgrad(
            wg_M=spec.wg_M,
            wg_N=spec.wg_N,
            wg_K=_wg_K(spec.problem),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            arch=arch,
            # See the note in build_implicit_gemm_conv_wgrad: the merged
            # group count is the real CTA multiplier. Equal at gm == 1.
            groups=spec.grid_groups,
            block_size=spec.block_size,
        ).split_k
        spec = dc_replace(spec, split_k=resolved)

    if spec.split_k <= 1:
        raise ValueError(
            f"build_implicit_gemm_conv_wgrad_two_stage requires split_k > 1 "
            f"(or split_k=-1 for auto-selection); got split_k={spec.split_k}"
        )

    # Lazy imports: keep module import-time safe for static IR tests running
    # without a HIP runtime.
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, PipelineLauncher

    # ---- Stage 1: wgrad GEMM → f32 workspace --------------------------------
    s1_spec = dc_replace(spec, two_stage=True)
    s1_kernel = build_implicit_gemm_conv_wgrad(s1_spec, arch=arch)
    s1_artifact = compile_kernel(s1_kernel, arch=arch, capture_ir_text=False)
    s1_sig = _wgrad_stage1_signature(s1_spec)
    s1_launcher = KernelLauncher(
        hsaco=s1_artifact.hsaco,
        kernel_name=s1_artifact.kernel_name,
        signature=s1_sig,
        cache_key=("conv_wgrad_two_stage_s1", s1_spec.kernel_name()),
    )

    # ---- Stage 2: scratch → dW (fold the R replicas + cast, all groups in one launch) -
    s2_spec = WgradReduceSpec(
        problem=spec.problem,
        dtype_d=spec.data.dtype_d,
        groups=spec.problem.groups,
        # Must match Stage 1 or the fold covers the wrong number of slabs.
        ws_replicas=s1_spec.ws_replicas,
    )
    s2_kernel = build_conv_wgrad_workspace_reduce(s2_spec, arch=arch)
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
