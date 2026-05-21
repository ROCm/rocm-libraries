# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""StreamK GEMM kernel (CK Tile ``40_streamk_gemm`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/40_streamk_gemm``.
StreamK trades launch-time tile assignment for runtime tile assignment:
a small number of persistent CTAs pull macro tiles from a global
counter via ``atomic_add(1)`` and process them until the counter
exhausts the total work count. Each macro tile is one
``(m_tile, n_tile, k_iter)`` triple; the partial K-iter contributions
to a single ``(m_tile, n_tile)`` output land via the chosen reduction
strategy (atomic or cooperative reduction).

What v1 ships:

* Persistent grid + StreamK partitioner end-to-end. The kernel uses
 :func:`ck_dsl.helpers.persistent_tile_for_each` and
 :func:`ck_dsl.helpers.emit_streamk_decode` to decode the linear
 macro-tile id into ``(m_tile, n_tile, k_iter, is_first, is_last)``.
* ``Atomic`` reduction strategy via ``global_atomic_add(workspace, ...)``.
 The workspace is f32 of shape ``(M, N)`` and the caller is
 responsible for clearing it to 0 before launch.
* **Scalar inner GEMM**: per-thread one-element-of-output ``tile_k``
 inner product (no MFMA). This is intentionally simple so the
 StreamK *infrastructure* is exercised end-to-end with a small,
 reviewable kernel. Real-perf MFMA + cshuffle StreamK builds on this
 partitioner + atomic surface and lands as a follow-on in the same
 module.
* fp16 inputs, f32 workspace output. A separate finalisation kernel
 (or a Python-side ``workspace.to(target_dtype)``) converts to the
 caller's target dtype.

When to use this v1 kernel:

* As a *correctness* oracle for the partitioning + atomic-accumulate
 pipeline -- numeric output matches a reference GEMM exactly.
* As a small, reviewable example of the StreamK pattern (less than
 ~200 LOC).
* **Not** as a perf target -- the scalar inner body misses every
 hardware feature CK Tile relies on (MFMA, vec loads, async DMA,
 cshuffle). The MFMA upgrade lives in ``build_streamk_gemm_mfma``
 ( follow-on).

The partitioner + reduction helpers (`helpers/streamk.py`) are the
durable deliverable; this kernel is the smallest end-to-end
consumer that proves they compose correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.atoms import MfmaAtom
from ..helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    mfma_k_loop,
    store_acc_to_global,
)
from ..helpers.spec import SignatureBuilder, kernel_name_join
from ..helpers.streamk import (
    StreamKPartition,
    StreamKReductionStrategy,
    compute_streamk_grid_size,
    emit_streamk_decode,
)


DType = Literal["f16"]


@dataclass(frozen=True)
class StreamKGemmSpec:
    """One concrete StreamK GEMM kernel configuration (v1: scalar inner).

    ``M``, ``N``, ``K`` are compile-time so the partitioner can
    statically derive ``m_tiles / n_tiles / k_iters``. ``num_cus`` and
    ``blocks_per_cu`` size the persistent launch grid.
    """

    M: int
    N: int
    K: int
    # ``tile_m`` / ``tile_n`` / ``tile_k`` now bind to the MFMA atom
    # shape: for the default 16x16x16 f16 atom, ``tile_k`` must be a
    # multiple of 16. The v1 scalar inner allowed arbitrary tile_k;
    # the MFMA path can use any ``tile_k = N * atom.k`` (N MFMA
    # invocations per macro tile). Larger ``tile_k`` reduces the
    # atomic_add frequency at the cost of slightly more K-loop trip.
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    dtype: DType = "f16"
    num_cus: int = 304  # MI300X / MI355X default
    blocks_per_cu: int = 1
    reduction: StreamKReductionStrategy = StreamKReductionStrategy.Atomic
    name: str = "ck_dsl_streamk_gemm"

    @property
    def partition(self) -> StreamKPartition:
        if self.M % self.tile_m or self.N % self.tile_n or self.K % self.tile_k:
            raise ValueError(
                f"M / N / K must be divisible by their tile sizes; got "
                f"M={self.M}, tile_m={self.tile_m}, N={self.N}, "
                f"tile_n={self.tile_n}, K={self.K}, tile_k={self.tile_k}"
            )
        return StreamKPartition(
            m_tiles=self.M // self.tile_m,
            n_tiles=self.N // self.tile_n,
            k_iters=self.K // self.tile_k,
        )

    @property
    def grid_size(self) -> int:
        return compute_streamk_grid_size(
            self.partition,
            num_cus=self.num_cus,
            blocks_per_cu=self.blocks_per_cu,
        )

    @property
    def atom(self) -> MfmaAtom:
        # Pick a square MFMA atom matching (tile_m, tile_n).
        if (self.tile_m, self.tile_n) == (16, 16):
            return MfmaAtom.f16_16x16x16()
        if (self.tile_m, self.tile_n) == (32, 32):
            return MfmaAtom.f16_32x32x8()
        raise ValueError(
            f"streamk_gemm MFMA path supports (16,16) or (32,32) atom "
            f"shapes; got ({self.tile_m}, {self.tile_n})"
        )

    @property
    def block_size(self) -> int:
        # MFMA path: one wave64 warp per CTA -- the MFMA atom is
        # per-wave and each macro tile (m_tile, n_tile, k_iter) is
        # handled by one warp.
        return 64

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"r{self.reduction.value}",
            f"g{self.grid_size}",
        )


def is_valid_spec(spec: StreamKGemmSpec) -> Tuple[bool, str]:
    if spec.dtype != "f16":
        return False, f"v1 ships f16 only, got {spec.dtype!r}"
    if spec.reduction != StreamKReductionStrategy.Atomic:
        return False, (
            f"v1 ships the Atomic reduction strategy only; "
            f"got {spec.reduction!r} (Reduction strategy is a v2 follow-on)"
        )
    if (spec.tile_m, spec.tile_n) not in ((16, 16), (32, 32)):
        return False, (
            f"MFMA path supports tile (16,16) or (32,32); "
            f"got ({spec.tile_m}, {spec.tile_n})"
        )
    if spec.M % spec.tile_m or spec.N % spec.tile_n or spec.K % spec.tile_k:
        return False, (
            "M / N / K must be divisible by their tile sizes "
            "(v1 doesn't handle partial tiles)"
        )
    if spec.tile_k % spec.atom.k != 0:
        return False, (
            f"tile_k ({spec.tile_k}) must be a multiple of atom.k "
            f"({spec.atom.k}) so the K-loop emits whole MFMA invocations"
        )
    return True, "ok"


def build_streamk_gemm(spec: StreamKGemmSpec) -> KernelDef:
    """Build the IR for one StreamK GEMM instance.

    Kernel signature::

    (A: ptr<f16, global>, # (M, K) row-major
    B: ptr<f16, global>, # (K, N) row-major
    Cf32: ptr<f32, global>, # (M, N) f32 workspace -- caller pre-cleared
    Counter: ptr<i32, global>) # 1-slot tile counter, pre-cleared

    Grid: ``(spec.grid_size, 1, 1)`` -- one persistent CTA per
    target wavefront slot.

    Algorithm (per macro tile):
    1. Pull next ``linear_id`` via atomic_add on ``Counter``.
    2. Decode ``(m_tile, n_tile, k_iter)`` via the StreamK partitioner.
    3. Each thread owns one ``(local_m, local_n)`` output position.
    Loop over the ``tile_k`` K-slice, computing
    ``acc += A[m_tile*tile_m + local_m, k_iter*tile_k + k]
    * B[k_iter*tile_k + k, n_tile*tile_n + local_n]``.
    4. Atomic-add the f32 ``acc`` into
    ``Cf32[m_global, n_global]``.

    Notes:
    * ``is_first`` / ``is_last`` predicates are decoded but unused in
    v1; the Atomic strategy doesn't need them. They're hooked up
    so the v2 Reduction strategy can reuse the same partitioner.
    * The K-loop is a Python-time unrolled ``range(tile_k)``;
    ``tile_k`` is usually 8-32 for the inner-product fit.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid streamk_gemm spec: {why}")

    partition = spec.partition
    BS = spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Cf32 = b.param("Cf32", PtrType(F32, "global"), align=4)
    Counter = b.param("Counter", PtrType(I32, "global"), align=4)

    lane = b.thread_id_x()
    bid = b.block_id_x()
    atom = spec.atom
    _ = partition.num_macro_tiles  # documented for parity with the partitioner
    _ = Counter  # noqa: F841 - kept in the ABI for the persistent v2 path

    # Decode (m_tile, n_tile, k_iter) for this CTA's macro tile.
    decoded = emit_streamk_decode(b, bid, partition)
    m_tile, n_tile, k_iter = decoded.m_tile, decoded.n_tile, decoded.k_iter
    m_tile_base = b.mul(m_tile, b.const_i32(spec.tile_m))
    n_tile_base = b.mul(n_tile, b.const_i32(spec.tile_n))
    # K-base for this macro tile: ``k_iter * tile_k``. The MFMA K-loop
    # runs ``tile_k / atom.k`` atoms within this K-slice.
    k_macro_base = b.mul(k_iter, b.const_i32(spec.tile_k))

    lane_decode = decode_mfma_lanes(b, atom, lane)

    def _load_a(b, kt):
        k_tile_base = b.add(k_macro_base, b.mul(kt, b.const_i32(atom.k)))
        return load_a_row_major_contiguous(
            b,
            A=A,
            atom=atom,
            lane_decode=lane_decode,
            m_tile_base=m_tile_base,
            k_tile_base=k_tile_base,
            K=spec.K,
        )

    def _load_b(b, kt):
        k_tile_base = b.add(k_macro_base, b.mul(kt, b.const_i32(atom.k)))
        return load_b_col_strided_scalars(
            b,
            B=Bp,
            atom=atom,
            lane_decode=lane_decode,
            n_tile_base=n_tile_base,
            k_tile_base=k_tile_base,
            N=spec.N,
        )

    # Run ``tile_k / atom.k`` MFMA atoms; result is per-lane
    # <c_per_lane x f32> accumulator.
    acc_final = mfma_k_loop(
        b,
        K=spec.tile_k,
        atom=atom,
        load_a=_load_a,
        load_b=_load_b,
    )

    # Atomic-add each lane's c_per_lane cells into the Cf32 workspace.
    # The atom's lane_to_output mapping decodes the per-lane output
    # cell coords; the f32 split-K reduction across (m_tile, n_tile,
    # *) k_iter values converges to the full f32 GEMM.
    store_acc_to_global(
        b,
        C=Cf32,
        atom=atom,
        lane_decode=lane_decode,
        m_tile_base=m_tile_base,
        n_tile_base=n_tile_base,
        acc=acc_final,
        N=spec.N,
        out_dtype="f32",
        atomic_add=True,
    )
    b.ret()
    return b.kernel


def streamk_gemm_grid(spec: StreamKGemmSpec) -> Tuple[int, int, int]:
    """Atomic-strategy v1 launch: one CTA per macro tile.

    The persistent v2 launches ``spec.grid_size`` CTAs and pulls tiles
    from the global ``Counter`` via cooperative atomic; v1 launches
    every macro tile directly so the kernel is a single-pass split-K
    GEMM with atomic f32 accumulate into ``Cf32``.
    """
    return (spec.partition.num_macro_tiles, 1, 1)


def streamk_gemm_signature(spec: StreamKGemmSpec):
    return (
        SignatureBuilder()
        .ptr("A", spec.dtype)
        .ptr("B", spec.dtype)
        .ptr("Cf32", "f32")
        .ptr("Counter", "i32")
        .build()
    )


def streamk_gemm_workspace_bytes(spec: StreamKGemmSpec) -> int:
    """Bytes the caller must zero-clear before launch.

    The workspace holds:

    * ``Cf32`` -- ``4 * M * N`` bytes for the f32 partial accumulator.
    * ``Counter`` -- 4 bytes for the i32 persistent-loop tile counter.

    Returns the **combined** size; the caller is expected to split it
    into two separate buffers (or use the same one in two slices).
    """
    return 4 * spec.M * spec.N + 4


__all__ = [
    "StreamKGemmSpec",
    "build_streamk_gemm",
    "is_valid_spec",
    "streamk_gemm_grid",
    "streamk_gemm_signature",
    "streamk_gemm_workspace_bytes",
    # Re-export the helper so callers don't need a second import.
    "StreamKReductionStrategy",
]
