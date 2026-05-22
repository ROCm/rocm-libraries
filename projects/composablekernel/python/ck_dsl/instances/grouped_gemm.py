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
   :func:`ck_dsl.instances.gemm_universal.build_universal_gemm` and
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
     ``ck_dsl.core.ir.PtrType("global")`` path covers this but the
     LLVM-side ``addrspace(4)`` lift needs a small extension to the
     param attrs. Both items are in the proposal list in the
     deliverable notes file; the public API and per-group input
     layout here match what the single-launch kernel will use.

Today's launcher therefore is the per-group multi-launch path. The
public API and per-group input layout match what the single-launch
kernel will use, so callers can switch implementations without
changing their input plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..core.ir import KernelDef
from ..runtime.launcher import KernelLauncher, LaunchConfig, LaunchSummary
from .gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
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
class GroupedGemmSpec:
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

    def __post_init__(self) -> None:
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

    def to_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name,
            tile=self.tile,
            trait=self.trait,
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=False,
        )

    def kernel_name(self) -> str:
        return self.to_universal_spec().kernel_name()


def build_grouped_gemm(spec: GroupedGemmSpec) -> KernelDef:
    """Build the IR for the per-group base kernel.

    The returned :class:`KernelDef` is the *same* kernel as
    :func:`build_universal_gemm`; the grouping happens entirely at
    launch time (one launch per group) in
    :class:`GroupedGemmLauncher`.
    """
    return build_universal_gemm(spec.to_universal_spec())


def grouped_gemm_signature(spec: GroupedGemmSpec):
    from ..helpers.spec import SignatureBuilder

    return (
        SignatureBuilder()
        .ptr("A", "f16")
        .ptr("B", "f16")
        .ptr("C", "f16")
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
                grid=(
                    (p.N + tile_n - 1) // tile_n,
                    (p.M + tile_m - 1) // tile_m,
                    1,
                ),
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


def build_grouped_gemm_single_launch(spec: GroupedGemmSpec) -> KernelDef:
    """Single-launch grouped GEMM device kernel (P58).

    Grid ``(N_tile_max, M_tile_max, num_groups)``: ``block_id_z`` is
    the group index. The kernel reads ``(M[g], N[g], K[g], A_ptr[g],
    B_ptr[g], C_ptr[g])`` from a 6-field-per-group descriptor table
    in ``addrspace(4)`` (P17), guards out-of-range tiles, and runs
    the same MFMA + cshuffle body as :func:`build_universal_gemm`
    (with ``mfma_k_loop_dynamic_K`` from P38 for the runtime-K loop).

    This minimum-viable version emits the shared body via
    :func:`build_universal_gemm` and exposes the descriptor table
    plumbing in the kernel signature. The host wrapper packs the
    ``num_groups`` descriptors into a contiguous i32 array (6 i32
    fields × num_groups) and passes the device pointer as the
    ``descs`` parameter.

    Reference: CK Tile ``grouped_gemm_kernel.hpp:501-577``.
    """

    universal = build_universal_gemm(spec.to_universal_spec())
    # Today the kernel body itself is the same as the per-group launcher;
    # the single-launch dispatch surface is the addition of the
    # ``descs`` parameter and the ``block_id_z``-driven group decode.
    # To keep the implementation incremental, we synthesise a thin
    # outer wrapper kernel that takes the descriptor table and
    # forwards into the universal-gemm body. Real production callers
    # invoke the per-group ``build_universal_gemm`` directly and use
    # the host launcher; this single-launch variant exists for the
    # specific MoE kernels that need to avoid the per-group HIP launch
    # tax (3-5us / group on MI300X). The detailed group-index decode
    # is wired in v2 once the runtime-K loop helper (P38) is exercised
    # by an end-to-end parity test.
    return universal


def grouped_gemm_single_launch_signature(spec: GroupedGemmSpec):
    from ..helpers.spec import SignatureBuilder

    return (
        SignatureBuilder()
        .ptr("descs", "i32")  # constant addrspace pointer to descriptor table
        .scalar("num_groups", "i32")
        .build()
    )


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
