# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Two-stage backward-weight convolution launcher.

Assembles a :class:`~rocke.runtime.launcher.PipelineLauncher` from:

* **Stage 1** — an implicit-GEMM wgrad kernel (``two_stage=True``) that
  f32-atomic-adds its partial sums into a scratch buffer instead of
  16-bit-atomic-adding into ``dW``.
* **Stage 2** — a fold/cast kernel that sums the scratch's ``ws_replicas``
  slabs per group, converts to ``dtype_d``, and writes ``dW``.

This is the route split-K takes when the packed ``<2 x dtype>`` atomic that
writes a 16-bit ``dW`` directly cannot address the problem -- it needs an even
``wg_N = Y*X*cpg``, while ``atomicrmw fadd f32`` has no alignment constraint.
The reduction over ``split_k`` is done by the hardware atomics; what is left
for Stage 2 is the fold over the ``R = ws_replicas`` slabs those atomics were
spread across, which is a compile-time-unrolled ``R`` loads / ``R-1`` adds /
convert / store per element.  ``R`` is independent of ``split_k``, so Stage 2's
cost does not scale with the split degree.

Both stages are submitted on the same HIP stream, so HIP's in-order
execution guarantees Stage 2 begins only after Stage 1 has completed —
no explicit ``hipStreamSynchronize`` is needed between them.

**The caller must zero the scratch before every Stage 1 launch.** Stage 1
accumulates into it; stale content is added to the result.

Grouped convolutions (``groups > 1``) are fully supported.  Stage 1 uses grid
``z = groups * split_k`` and decodes the group from it, so every slice of a
group lands on one of that group's ``R`` scratch slabs (picked by ``z % R``)
and never on another group's.  Stage 2 uses grid ``z = groups`` and folds and
casts all groups in one launch.

Usage::

    from dataclasses import replace
    spec = WgradConvSpec(problem=..., split_k=4, two_stage=True)
    pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(spec, arch=arch)

    ws = DeviceMem(ws_nbytes)
    ws.memset(0)                      # REQUIRED: Stage 1 accumulates

    # The AOT argument block (problem extents, magic numbers, ks/ks_count)
    # plus the workspace pair; split_k is the degree to launch at.
    s1_vals = wgrad_stage1_launch_values(
        spec, dY_ptr=dY_ptr, X_ptr=X_ptr, dW_ptr=dW_ptr,
        dY_bytes=dY_nb, X_bytes=X_nb, dW_bytes=dW_nb,
        ws_ptr=ws.ptr(), ws_bytes=ws_nbytes, split_k=4)
    s2_vals = {"ws_ptr": ws.ptr(), "dw_ptr": dw_ptr,
               "wg_M": spec.wg_M, "wg_N": spec.wg_N,
               "ws_bytes": ws_nbytes, "dw_bytes": dw_nb,
               "groups": spec.problem.groups}

    pipeline((s1_vals, s2_vals), (s1_cfg, s2_cfg), stream=stream)
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Optional, Tuple


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


def wgrad_workspace_nbytes(problem, ws_replicas: int) -> int:
    """Two-stage scratch bytes for ``problem`` at ``ws_replicas`` slabs per
    group: ``groups * R * wg_M * wg_N * 4``. The spec-free form of
    :func:`wgrad_two_stage_workspace_nbytes`, for hosts that launch a cached
    binary and know its replica count but have no spec."""
    if ws_replicas < 1:
        raise ValueError(f"ws_replicas must be >= 1 (got {ws_replicas})")
    return problem.groups * ws_replicas * _wg_M(problem) * _wg_N(problem) * 4


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
    return wgrad_workspace_nbytes(spec.problem, spec.ws_replicas)


def _wgrad_stage1_signature(spec: WgradConvSpec) -> list:
    """Launch signature for the Stage 1 wgrad kernel (``two_stage=True``).

    The Stage 1 kernel is an ordinary AOT wgrad kernel with the two-stage
    workspace pair appended, so the signature is the shared wgrad AOT one
    built with ``two_stage=True`` -- deriving it here rather than restating
    the argument list is what keeps it from drifting out of step with the
    builder (kernargs pack positionally, so a stale copy corrupts silently).

    A (dY), B (X), and D (dW) each carry their own element type so that
    mixed-dtype configurations (e.g. bf16 inputs with fp32 output) are
    described correctly.
    """
    from kernels.common.conv_abi import conv_args_signature

    return conv_args_signature(
        spec.data.dtype_a,
        direction="wgrad",
        dtype_b=spec.data.dtype_b,
        dtype_d=spec.data.dtype_d,
        is_3d=spec.problem.is_3d,
        two_stage=True,
    )


def wgrad_stage1_launch_values(
    spec: WgradConvSpec,
    *,
    dY_ptr: int,
    X_ptr: int,
    dW_ptr: int,
    dY_bytes: int,
    X_bytes: int,
    dW_bytes: int,
    ws_ptr: int,
    ws_bytes: int,
    split_k: Optional[int] = None,
) -> dict:
    """Host-side ``values`` dict for a Stage 1 launch.

    Mirrors :func:`_wgrad_stage1_signature`: the shared wgrad AOT arguments
    plus the workspace pair.

    ``split_k`` is the degree to launch at (any value > 1 -- the kernel takes
    it as a kernarg); it defaults to the spec's.
    """
    if split_k is None:
        split_k = spec.split_k
    if split_k <= 1:
        raise ValueError(f"two-stage Stage 1 needs split_k > 1 (got {split_k})")
    from kernels.common.conv_args import ConvArgs

    return ConvArgs.from_problem(
        spec.problem,
        direction="wgrad",
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
    ).to_launch_values(
        dY_ptr,
        X_ptr,
        dW_ptr,
        dY_bytes,
        X_bytes,
        dW_bytes,
        split_k=split_k,
        ws_ptr=ws_ptr,
        ws_bytes=ws_bytes,
    )


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

        The scratch has shape ``[groups * R, wg_M, wg_N]`` (f32), where
        ``R = spec.ws_replicas``: ``R`` copies of the per-group ``dW`` slab,
        with no ``split_k`` factor.  Stage 1 f32-atomic-adds every element
        within ``[0, wg_M) × [0, wg_N)`` of its slab; OOB positions are skipped
        by a per-element ``scf_if`` guard.  Stage 2 folds the ``R`` slabs.
        Size it with :func:`wgrad_two_stage_workspace_nbytes` -- the returned
        ``workspace_nbytes`` is exactly that value.

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
            f"or -1 (auto-selection); got split_k={spec.split_k}"
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
