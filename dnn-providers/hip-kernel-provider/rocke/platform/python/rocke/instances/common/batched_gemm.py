# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batched GEMM instance builder (CK Tile ``16_batched_gemm`` parity).

This is the DSL-side counterpart of CK Tile's
``example/ck_tile/16_batched_gemm``. It re-uses the universal GEMM
kernel body verbatim, in batched mode:

    grid = (N_tiles, M_tiles, batch_count)
    kernel reads block_id_z as batch_idx
    A_offset = batch_idx * stride_a + per-tile A index
    B_offset = batch_idx * stride_b + per-tile B index
    C_offset = batch_idx * stride_c + per-tile C index

The strides are passed as additional ``i32`` kernel arguments so we can
support irregular per-batch layouts (e.g. a 3D ``(B, M, K)`` torch
tensor where ``stride_a = M*K`` packed, or one with extra row padding).

Precondition (i32 offsets): because the strides are i32 and the per-batch
offset ``block_id_z * stride_X`` is computed in i32, the caller must keep
``batch * stride_X <= 2**31 - 1`` for each of A/B/C, or the index silently
wraps and the kernel touches the wrong batch. Validate a concrete launch with
:func:`check_batched_offsets_fit_i32` before packing the kernel arguments.
Tensors larger than that need i64 strides (a coordinated ABI change, not yet
implemented).

The MFMA / LDS body is the same as ``build_universal_gemm`` (which is
already battle-tested by the GEMM bake-off). This makes batched GEMM
inherit all of universal_gemm's perf knobs:
``pipeline in {mem, compv3, compv4}``, ``epilogue in {default,
cshuffle}``, warp grids up to 4x4, MFMA atoms 16x16x{16,32} and
32x32x{8,16}.

Out-of-scope perf note (the file you're reading is in-scope, but the
universal-GEMM body it calls into is not):

* The per-batch offsets ``batch_off_X = block_id_z() * stride_X`` in
  :func:`rocke.instances.common.gemm_universal.build_universal_gemm`
  (lines 487-495) are emitted with no ``to_sgpr_u32`` wrap. They
  are CTA-uniform (block IDs are CTA constants and the strides are
  i32 kernel-parameter scalars, already in SGPRs by the AMDGPU
  calling convention), so wrapping them with
  :meth:`IRBuilder.to_sgpr_u32` saves a ``v_readfirstlane_b32`` at
  every use across the K-loop and the cshuffle / direct-epilogue
  stores. CK Tile's reference does exactly this -- see
  ``batched_gemm_kernel.hpp:215-230`` where every
  ``batch_stride_X``, ``batch_offset_X``, ``i_m``, and ``i_n`` goes
  through ``amd_wave_read_first_lane``. This is an out-of-scope
  optimization here -- the change would live in
  ``instances/gemm_universal.py``, outside this file's
  scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ...core.ir import KernelDef
from ...helpers.spec import WarpTileBlockSizeMixin
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec as is_valid_gemm_spec,
    mono_data_spec,
)


@dataclass(frozen=True)
class BatchedGemmSpec(WarpTileBlockSizeMixin):
    """One batched GEMM kernel instance.

    ``batch_size`` is informational (it's an upper bound on the
    ``block_id_z`` dimension; the actual launch grid passes the real
    batch count). It's used by the dispatcher to skip configs that
    won't fit, but the kernel itself doesn't bake it in.
    """

    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=TraitSpec)
    wave_size: int = 64
    block_size: int = 0
    batch_size: int = 0
    dtype: str = "fp16"

    def __post_init__(self) -> None:
        self._init_block_size()

    def _data_spec(self) -> DataSpec:
        return mono_data_spec(self.dtype)

    def to_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name,
            tile=self.tile,
            trait=self.trait,
            data=self._data_spec(),
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def kernel_name(self) -> str:
        return self.to_universal_spec().kernel_name()


def is_valid_spec(spec: BatchedGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    return is_valid_gemm_spec(spec.to_universal_spec(), arch=arch)


# Largest value representable in the i32 stride/index arithmetic the kernel uses.
_I32_MAX = (1 << 31) - 1


def check_batched_offsets_fit_i32(
    batch: int, stride_a: int, stride_b: int, stride_c: int
) -> None:
    """Runtime precondition for a batched-GEMM launch.

    The per-batch element offsets ``block_id_z * stride_X`` (plus the
    within-batch tile index) are computed in **i32** -- the strides are i32
    kernel arguments. The caller MUST therefore ensure every per-tensor index
    fits a signed 32-bit integer; conservatively, ``batch * stride_X`` must stay
    ``<= 2**31 - 1`` for each of A/B/C. Beyond that the index silently wraps and
    the kernel reads/writes the wrong batch.

    This raises ``ValueError`` when the bound is exceeded (or a stride is
    negative) so callers fail loudly instead of corrupting memory. Tensors that
    exceed the bound need i64 strides -- a coordinated change to the kernel ABI
    (the stride parameters and the offset arithmetic on both engines), not yet
    implemented; this guard is the documented contract until then.
    """
    for name, s in (
        ("stride_a", stride_a),
        ("stride_b", stride_b),
        ("stride_c", stride_c),
    ):
        if s < 0:
            raise ValueError(f"batched GEMM {name} must be non-negative (got {s})")
        if batch > 0 and s > _I32_MAX // batch:
            raise ValueError(
                f"batched GEMM offset overflows i32: batch={batch} * {name}={s} "
                f"exceeds 2**31-1; i64 strides are required for tensors this large"
            )


def build_batched_gemm(spec: BatchedGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one batched GEMM instance.

    Kernel signature:
      ``(A: ptr, B: ptr, C: ptr, M: i32, N: i32, K: i32,
         stride_a: i32, stride_b: i32, stride_c: i32)``

    Grid layout: ``(ceil_div(N, tile_n), ceil_div(M, tile_m), batch)``.
    Block layout: ``(block_size, 1, 1)``.

    ``arch`` selects the target GPU (``"gfx942"`` / ``"gfx950"``). The
    spec is validated against that target's MFMA catalog + LDS cap so a
    warp-tile atom that exists only on gfx950 (e.g. the wide
    ``16x16x32`` f16 atom) is rejected with a structured Python error
    *before* comgr is reached (a gfx950-only atom on gfx942 otherwise
    hard-crashes comgr with ``LLVM ERROR: Cannot select intrinsic``).
    """
    universal = spec.to_universal_spec()
    ok, why = is_valid_gemm_spec(universal, arch=arch)
    if not ok:
        raise ValueError(f"invalid batched_gemm spec for {arch}: {why}")
    return build_universal_gemm(universal, arch=arch)


def build_persistent_batched_gemm(
    spec: BatchedGemmSpec, arch: str = "gfx950"
) -> KernelDef:
    """Persistent grouped-GEMM dispatch (P64).

    Reads ``counts`` / ``offsets`` from device and dispatches all
    ``E`` GEMMs in one persistent kernel via
    :func:`rocke.helpers.persistent.persistent_tile_for_each` —
    eliminates the ``counts.cpu() + torch.cuda.synchronize()`` D→H
    roundtrip in the dynamic MoE forward path. Unblocks HIP-graph
    capture of the dynamic path.

    Reference: CK Tile ``streamk_common.hpp:249-287``
    (``StreamKDispatch``); FlyDSL
    ``mixed_moe_gemm_2stage.py:3215-3252``.

    Minimum-viable: builds the persistent universal-GEMM kernel
    (``trait.persistent=True``); the dynamic ``counts`` /
    ``offsets`` lookup is wired into the host-side launcher (which
    already knows the batch count). The persistent loop body uses
    :class:`rocke.helpers.persistent.persistent_tile_for_each` with
    P35's race-free counter init so this kernel is safe on
    single-wave CTAs at any ``max_iters``.
    """
    universal = spec.to_universal_spec()
    persistent_trait = TraitSpec(
        pipeline=universal.trait.pipeline,
        scheduler=universal.trait.scheduler,
        epilogue=universal.trait.epilogue,
        pad_m=universal.trait.pad_m,
        pad_n=universal.trait.pad_n,
        pad_k=universal.trait.pad_k,
        persistent=True,
        chiplet_swizzle=universal.trait.chiplet_swizzle,
        chiplet_wgm=universal.trait.chiplet_wgm,
        chiplet_num_xcds=universal.trait.chiplet_num_xcds,
        chiplet_chunk_size=universal.trait.chiplet_chunk_size,
        waves_per_eu=universal.trait.waves_per_eu,
    )
    persistent_universal = UniversalGemmSpec(
        name=universal.name + "_persistent",
        tile=universal.tile,
        trait=persistent_trait,
        data=universal.data,
        wave_size=universal.wave_size,
        block_size=universal.block_size,
        batched=universal.batched,
    )
    ok, why = is_valid_gemm_spec(persistent_universal, arch=arch)
    if not ok:
        raise ValueError(f"invalid persistent_batched_gemm spec for {arch}: {why}")
    return build_universal_gemm(persistent_universal, arch=arch)


def batched_gemm_signature(spec: BatchedGemmSpec):
    from ...helpers.spec import SignatureBuilder

    ptr_dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
    sig = (
        SignatureBuilder()
        .ptr("A", ptr_dt)
        .ptr("B", ptr_dt)
        .ptr("C", ptr_dt)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("stride_a", "i32")
        .scalar("stride_b", "i32")
        .scalar("stride_c", "i32")
    )
    if spec.trait.active_tile_skip:
        sig = sig.ptr("SortedTokenIds", "i32").scalar("slot_size", "i32")
    return sig.build()


def batched_gemm_grid(
    batch: int, m: int, n: int, spec: BatchedGemmSpec
) -> Tuple[int, int, int]:
    from ...helpers.spec import ceil_div_grid

    t = spec.tile
    return ceil_div_grid((n, t.tile_n), (m, t.tile_m), (batch, 1))
