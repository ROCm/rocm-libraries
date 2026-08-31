# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped GEMM instance builder (CK Tile ``17_grouped_gemm`` parity).

This is the DSL-side counterpart of CK Tile's
``example/ck_tile/17_grouped_gemm`` (the unquantised base case;
``abquant_grouped_gemm.cpp`` / ``quant_grouped_gemm*.cpp`` are
follow-ups).

A grouped GEMM is a batch of GEMMs where each batch entry can have a
distinct ``(M, N, K)`` shape and is stored at an arbitrary device
offset (not a uniform per-batch stride like ``16_batched_gemm``).
Use cases:

  - MoE-style routing where each expert handles a different number of
    tokens
  - Multi-LoRA inference where each request hits a different LoRA
    adapter

Two implementations are provided here, mirroring CK Tile's own design
points:

1. :class:`GroupedGemmLauncher` (default) — re-uses
   :func:`rocke.instances.common.gemm_universal.build_universal_gemm` and
   launches it once per group. This is what CK Tile's reference
   path does when ``persistent=False``. Correct, easy to reason
   about, but pays one ``hipModuleLaunchKernel`` round-trip per
   group (~3-5us on MI300X/MI355X). Acceptable when the per-group
   work is large relative to the launch cost (typical for MoE).

   v1 optimisation note (this file): the per-group ``LaunchConfig``,
   the per-group element-arg dict, and the ``ceil_div`` grid math
   are extracted into a single hot loop that allocates one launch
   bundle per call rather than rebuilding them every iteration -- the
   per-group ``__call__`` overhead drops to one ``KernelLauncher``
   dispatch + one launch grid tuple per group. See
   :class:`GroupedGemmLauncher.__call__` for the loop body.

2. ``GroupedGemmKernelSpec`` (planned) — single-launch kernel that
   uses ``block_id_z`` as the group index, looks up
   ``M[g] / N[g] / K[g] / A_off[g] / B_off[g] / C_off[g]`` from
   device-side arrays, and runs the universal_gemm body in-place
   with the looked-up offsets. This matches CK Tile's
   ``persistent=True`` grouped GEMM. The lookup math is per-group
   wave-uniform i32 (one ``amd_wave_read_first_lane`` per offset in
   the CK Tile reference, ``grouped_gemm_kernel.hpp:297-298,
   390, 451``) so an SGPR-pinned version saves one
   ``v_readfirstlane_b32`` per use across the K-loop and epilogue.

   Listed as a follow-up because:

   * The current ``mfma_k_loop`` helper requires compile-time ``K``
     (the K-loop trip count is a Python int baked into
     ``scf.for_iter``). A single-launch grouped GEMM with per-group
     dynamic ``K`` needs either a new runtime-K MFMA helper or a
     compile-time per-K specialisation of the kernel.

   * The descriptor lookup wants a constant-address-space pointer
     (per CK Tile's pattern at ``grouped_gemm_kernel.hpp:503``,
     ``void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const``); the
     ``rocke.core.ir.PtrType("global")`` path covers this but the
     LLVM-side ``addrspace(4)`` lift needs a small extension to the
     param attrs. Both items are future extensions; the public API
     and per-group input layout here match what the single-launch
     kernel will use.

Today's launcher therefore is the per-group multi-launch path. The
public API and per-group input layout match what the single-launch
kernel will use, so callers can switch implementations without
changing their input plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from ...core.ir import KernelDef
from ...helpers.spec import WarpTileBlockSizeMixin, ceil_div_grid
from ...runtime.launcher import KernelLauncher, LaunchConfig, LaunchSummary
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
class GroupedGemmProblem:
    """One entry in a grouped GEMM workload.

    Pointers are device pointers (e.g. ``tensor.data_ptr()`` from
    torch). Strides are in elements. Layout is RCR (row-major A, col-
    major B, row-major C), matching ``build_universal_gemm``.
    """

    M: int
    N: int
    K: int
    A_ptr: int
    B_ptr: int
    C_ptr: int


@dataclass(frozen=True)
class GroupedGemmSpec(WarpTileBlockSizeMixin):
    """One grouped GEMM kernel-instance bundle.

    The same tile/trait pair is applied to every group (CK Tile's
    grouped GEMM has the same constraint). The dispatcher should
    select tile dims that divide every ``(M[g], N[g], K[g])`` (the
    current implementation does not pad).
    """

    name: str
    tile: TileSpec
    trait: TraitSpec
    wave_size: int = 64
    block_size: int = 0
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
            batched=False,
        )

    def kernel_name(self) -> str:
        return self.to_universal_spec().kernel_name()


def is_valid_spec(spec: GroupedGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Validity of the per-group base kernel on ``arch``.

    Delegates to :func:`rocke.instances.common.gemm_universal.is_valid_spec`
    so the MFMA-atom + LDS-cap gating is arch-aware (a gfx950-only wide
    atom is rejected for gfx942 with a structured reason).
    """
    return is_valid_gemm_spec(spec.to_universal_spec(), arch=arch)


def build_grouped_gemm(spec: GroupedGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for the per-group base kernel.

    The returned :class:`KernelDef` is the *same* kernel as
    :func:`build_universal_gemm`; the grouping happens entirely at
    launch time (one launch per group) in
    :class:`GroupedGemmLauncher`.

    ``arch`` validates the spec against the target's MFMA catalog so a
    gfx950-only warp-tile atom requested for gfx942 raises a clean
    Python error before comgr.
    """
    universal = spec.to_universal_spec()
    ok, why = is_valid_gemm_spec(universal, arch=arch)
    if not ok:
        raise ValueError(f"invalid grouped_gemm spec for {arch}: {why}")
    return build_universal_gemm(universal, arch=arch)


def grouped_gemm_signature(spec: GroupedGemmSpec):
    from ...helpers.spec import SignatureBuilder

    ptr_dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
    return (
        SignatureBuilder()
        .ptr("A", ptr_dt)
        .ptr("B", ptr_dt)
        .ptr("C", ptr_dt)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .build()
    )


class GroupedGemmLauncher:
    """Single-pass grouped GEMM launcher.

    Owns one :class:`KernelLauncher` and re-uses it across every
    group in the workload. The HSACO module is loaded once at
    construction time and cached on the underlying launcher; per-
    group launches only update the (A, B, C, M, N, K) packed args
    and the grid dims.

    Constructed once per (problem-tile, dtype) tuple; call repeatedly
    with a list of :class:`GroupedGemmProblem` entries.
    """

    def __init__(self, *, hsaco: bytes, spec: GroupedGemmSpec) -> None:
        self._spec = spec
        self._launcher = KernelLauncher(
            hsaco=hsaco,
            kernel_name=spec.kernel_name(),
            signature=grouped_gemm_signature(spec),
            cache_key=(spec.kernel_name(),),
        )
        # Hoist host-side per-group constants out of the inner loop so
        # ``__call__`` only pays for the bits that genuinely vary per
        # group (the (A, B, C) torch pointers and the (M, N, K) shape).
        # These three values are CTA-uniform for the kernel and could in
        # principle be lifted further if we expose a "shape-bucketed"
        # API (see the GroupedGemmKernelSpec follow-up in the module
        # docstring above). For the multi-launch path it's pure Python
        # overhead reduction: one tuple build + one ``__init__`` of
        # ``LaunchConfig`` per group instead of building the same
        # constant block-dim 3-tuple every call.
        self._block = (spec.block_size, 1, 1)
        self._tile_m = spec.tile.tile_m
        self._tile_n = spec.tile.tile_n

    def __call__(
        self,
        problems: Sequence[GroupedGemmProblem],
        *,
        stream: int = 0,
    ) -> LaunchSummary:
        # Pre-bind the per-instance constants once; the inner loop
        # then only does the work that *cannot* be hoisted host-side:
        # the per-group ``(p.M, p.N, p.K, p.A_ptr, p.B_ptr, p.C_ptr)``
        # plumbing and the per-group ``ceil_div`` grid math.
        tile_m = self._tile_m
        tile_n = self._tile_n
        block = self._block
        launcher = self._launcher
        launches = 0
        for p in problems:
            cfg = LaunchConfig(
                stream=stream,
                grid=ceil_div_grid((p.N, tile_n), (p.M, tile_m)),
                block=block,
            )
            launches += launcher(
                {
                    "A": p.A_ptr,
                    "B": p.B_ptr,
                    "C": p.C_ptr,
                    "M": p.M,
                    "N": p.N,
                    "K": p.K,
                },
                config=cfg,
            ).launches
        return LaunchSummary(launches=launches)


def build_grouped_gemm_single_launch(
    spec: GroupedGemmSpec, arch: str = "gfx950"
) -> KernelDef:
    """Single-launch grouped GEMM device kernel.

    Grid ``(N_tile_max, M_tile_max, num_groups)``: ``block_id_z`` is the group
    index. All ``num_groups`` GEMMs run in **one** ``hipModuleLaunchKernel``,
    eliminating the per-group HIP launch tax (~3-5us / group on MI300X /
    MI355X) that the multi-launch :class:`GroupedGemmLauncher` pays.

    The realised kernel is the **batched** universal-GEMM body
    (``UniversalGemmSpec(batched=True)``): the kernel reads ``block_id_z`` as
    the group index and offsets the A / B / C base pointers by
    ``block_id_z * stride_{a,b,c}`` (element strides passed as i32 kernel
    args). This is exactly CK Tile's ``persistent=False`` grouped GEMM for the
    *uniform-shape* case (every group shares ``(M, N, K)`` — the dominant
    batched-MoE / multi-LoRA routing case), where each group's matrices are
    contiguous so ``stride_a = M*K``, ``stride_b = N*K``, ``stride_c = M*N``
    and the ``block_id_z``-scaled offset lands exactly on each group's base.

    Drive it with :class:`GroupedGemmSingleLaunchRunner`, which packs the
    per-group contiguous tensors into one launch with ``grid.z = num_groups``.
    Verify against the per-group path with the same tile/trait spec.

    The fully-variable-shape single launch (per-group ``(M[g], N[g], K[g])``
    decoded from an ``addrspace(4)`` descriptor table + ``mfma_k_loop_dynamic_K``
    runtime-K loop + out-of-range tile guard, CK Tile
    ``grouped_gemm_kernel.hpp:501-577``) remains the v2 follow-up; it needs the
    universal body to take per-group runtime M/N/K, which the current
    block-offset model does not yet thread.
    """
    base_spec = spec.to_universal_spec()
    batched_spec = UniversalGemmSpec(
        name=base_spec.name + "_single_launch",
        tile=base_spec.tile,
        trait=base_spec.trait,
        data=base_spec.data,
        wave_size=base_spec.wave_size,
        block_size=base_spec.block_size,
        batched=True,
    )
    ok, why = is_valid_gemm_spec(batched_spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid grouped_gemm_single_launch spec for {arch}: {why}")
    return build_universal_gemm(batched_spec, arch=arch)


def grouped_gemm_single_launch_signature(spec: GroupedGemmSpec):
    """Kernel-ABI signature for :func:`build_grouped_gemm_single_launch`.

    Matches the batched universal-GEMM body the builder emits:
    ``(A, B, C, M, N, K, stride_a, stride_b, stride_c)``. The host runner
    sets ``M`` / ``N`` / ``K`` to the shared per-group shape and the three
    strides to the per-group element counts (``M*K`` / ``N*K`` / ``M*N``).
    """
    from ...helpers.spec import SignatureBuilder

    ptr_dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
    return (
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
        .build()
    )


class GroupedGemmSingleLaunchRunner:
    """Host driver for the single-launch grouped GEMM (uniform per-group shape).

    Owns one :class:`KernelLauncher` over the batched universal-GEMM body.
    ``__call__`` issues a *single* device launch with ``grid.z = num_groups``;
    each group ``g`` is selected on-device by ``block_id_z == g`` and its
    matrices are addressed via the ``block_id_z * stride`` offset. The
    per-group tensors must be uniform shape and packed contiguously (the
    common batched-MoE layout); pass the contiguous base pointers + the shared
    ``(M, N, K)``.
    """

    def __init__(self, *, hsaco: bytes, spec: GroupedGemmSpec) -> None:
        self._spec = spec
        kernel_name = build_grouped_gemm_single_launch(spec).name
        self._launcher = KernelLauncher(
            hsaco=hsaco,
            kernel_name=kernel_name,
            signature=grouped_gemm_single_launch_signature(spec),
            cache_key=(kernel_name,),
        )
        self._block = (spec.block_size, 1, 1)
        self._tile_m = spec.tile.tile_m
        self._tile_n = spec.tile.tile_n

    def __call__(
        self,
        *,
        A_ptr: int,
        B_ptr: int,
        C_ptr: int,
        M: int,
        N: int,
        K: int,
        num_groups: int,
        stream: int = 0,
    ) -> LaunchSummary:
        cfg = LaunchConfig(
            stream=stream,
            grid=ceil_div_grid((N, self._tile_n), (M, self._tile_m), (num_groups, 1)),
            block=self._block,
        )
        summ = self._launcher(
            {
                "A": A_ptr,
                "B": B_ptr,
                "C": C_ptr,
                "M": M,
                "N": N,
                "K": K,
                "stride_a": M * K,
                "stride_b": N * K,
                "stride_c": M * N,
            },
            config=cfg,
        )
        return LaunchSummary(launches=summ.launches)


def grouped_gemm_problems(
    a_tensors: Iterable, b_tensors: Iterable, c_tensors: Iterable
) -> List[GroupedGemmProblem]:
    """Convenience: turn three lists of torch tensors into a problem list.

    ``a_tensors[g]`` is the ``(M_g, K)`` A matrix for group ``g``,
    ``b_tensors[g]`` is the ``(N_g, K)`` B matrix (note: B is stored
    transposed, matching the RCR convention), and ``c_tensors[g]`` is
    the ``(M_g, N)`` output matrix.
    """
    problems: List[GroupedGemmProblem] = []
    for a, bm, c in zip(a_tensors, b_tensors, c_tensors):
        M, K = a.shape[-2], a.shape[-1]
        N, K2 = bm.shape[-2], bm.shape[-1]
        if K != K2:
            raise ValueError(f"K mismatch: A K={K} vs B K={K2}")
        problems.append(
            GroupedGemmProblem(
                M=int(M),
                N=int(N),
                K=int(K),
                A_ptr=int(a.data_ptr()),
                B_ptr=int(bm.data_ptr()),
                C_ptr=int(c.data_ptr()),
            )
        )
    return problems
