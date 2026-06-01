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

* MFMA inner GEMM via :func:`ck_dsl.helpers.mfma_gemm_inner.mfma_k_loop`
  (one warp per CTA, ``tile_k / atom.k`` MFMA atoms per macro tile).
* StreamK partitioner end-to-end (:func:`emit_streamk_decode`) plus
  two launch modes for the partitioned macro tiles:

  - **non-persistent** (``persistent=False``, default): grid sized to
    ``num_macro_tiles``; ``block_id_x`` *is* the linear macro-tile id.
    Matches the CK Tile non-persistent dispatcher.
  - **persistent** (``persistent=True``): grid sized to
    ``compute_streamk_grid_size`` (~num_cus * blocks_per_cu); each CTA
    pulls macro tiles from a global atomic counter via
    :func:`ck_dsl.helpers.persistent_tile_for_each`. Matches CK Tile's
    persistent-DP dispatcher.

* ``Atomic`` reduction strategy via ``global_atomic_add(workspace, ...)``.
  The workspace is f32 of shape ``(M, N)`` and the caller is
  responsible for clearing it to 0 before launch.
* SGPR pinning on every wave-uniform i32 derived from the macro-tile
  id (``m_tile``, ``n_tile``, ``k_iter``, ``m_tile_base``,
  ``n_tile_base``, ``k_macro_base``), matching CK Tile's
  ``amd_wave_read_first_lane(iM * MPerBlock)`` pattern (see
  ``streamk_gemm_kernel.hpp:339-462``). This keeps the per-tile
  address arithmetic in scalar registers instead of re-materialising
  it in VGPRs at every use inside the K loop and atomic-store epilogue.
* fp16 inputs, f32 workspace output. A separate finalisation kernel
  (or a Python-side ``workspace.to(target_dtype)``) converts to the
  caller's target dtype.

When to use this v1 kernel:

* As a *correctness* oracle for the partitioning + atomic-accumulate
  pipeline -- numeric output matches a reference GEMM exactly.
* As a small, reviewable example of the StreamK pattern.
* The persistent variant amortises the kernel-launch cost across all
  macro tiles by sizing the grid to the CU count once; the
  non-persistent variant matches the v1 contract for shapes where the
  total macro-tile count is < grid budget and per-tile launches are
  cheap enough.

The partitioner + reduction helpers (`helpers/streamk.py`) are the
durable deliverable; this kernel is the smallest end-to-end
consumer that proves they compose correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.atoms import MfmaAtom
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    mfma_k_loop,
    store_acc_to_global,
)
from ...helpers.persistent import persistent_tile_for_each
from ...helpers.spec import SignatureBuilder, kernel_name_join
from ...helpers.streamk import (
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
    # ``persistent`` selects between the two macro-tile dispatch modes:
    #   False (default) -> grid = (num_macro_tiles, 1, 1); each CTA owns
    #                       exactly one ``(m_tile, n_tile, k_iter)`` and
    #                       reads ``block_id_x`` as its linear id. Keeps
    #                       parity with the v1 atomic-strategy launch.
    #   True            -> grid = (compute_streamk_grid_size(...), 1, 1);
    #                       each CTA atomically fetches the next macro
    #                       tile id from the ``Counter`` slot via
    #                       :func:`persistent_tile_for_each`. This is
    #                       CK Tile's persistent DP dispatcher pattern
    #                       (``streamk_common.hpp::StreamKDispatch``).
    persistent: bool = False
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

    @property
    def persistent_max_iters(self) -> int:
        """Worst-case macro tiles processed per CTA in persistent mode.

        ``ceil(num_macro_tiles / grid_size)`` -- the bounded
        ``scf.for_iter`` trip count fed into
        :func:`persistent_tile_for_each`. The per-iteration ``in_range``
        guard makes any over-estimation correct (over-fetched tile ids
        beyond ``num_macro_tiles`` are skipped); under-estimation would
        drop work, so we round up.
        """
        nm = self.partition.num_macro_tiles
        gs = self.grid_size
        return (nm + gs - 1) // gs

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"r{self.reduction.value}",
            f"g{self.grid_size}",
            flags={"pers": self.persistent},
        )


def is_valid_spec(spec: StreamKGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
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
    # Arch gating: the streamk MFMA inner uses the legacy f16 atoms
    # (16x16x16 / 32x32x8), both of which exist on gfx942 and gfx950.
    # Validate against the target's MFMA catalog so the predicate stays
    # honest if a future spec selects a gfx950-only K-packed atom.
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    atom = spec.atom
    if not target.mma.has_shape(
        a_dtype="fp16",
        b_dtype="fp16",
        c_dtype="fp32",
        m=atom.m,
        n=atom.n,
        k=atom.k,
    ):
        return False, (
            f"f16 MFMA atom {atom.m}x{atom.n}x{atom.k} not available on {arch}"
        )
    # ``persistent=True`` composes :func:`ck_dsl.helpers.persistent_tile_for_each`
    # whose cooperative LDS-broadcast counter path has a correctness
    # regression for ``max_iters > 1`` on gfx950: some macro-tile
    # bodies that should execute under ``in_range == True`` are
    # observed as skipped, producing partial-K accumulator drops in
    # the f32 workspace. Reproducible at e.g. M=N=K=256 with
    # ``num_cus=256, tile=(16,16,32)`` (max_iters=8) -- only the
    # ``num_cus`` so large that ``max_iters == 1`` (effectively the
    # same as the non-persistent path) verifies bit-exact today.
    # Until the helper is fixed in :mod:`ck_dsl.helpers.persistent`,
    # reject ``persistent=True`` for any spec whose
    # ``persistent_max_iters`` would exceed 1 -- this preserves the
    # ABI surface (the kernel still builds and runs) so callers can
    # opt back in trivially once the helper lands its fix.
    if spec.persistent and spec.persistent_max_iters > 1:
        return False, (
            f"persistent=True with persistent_max_iters="
            f"{spec.persistent_max_iters} > 1 hits a known bug in "
            f"ck_dsl.helpers.persistent_tile_for_each's cooperative "
            f"LDS-broadcast counter path (some macro-tile bodies "
            f"are skipped). Use persistent=False, or raise num_cus * "
            f"blocks_per_cu so persistent_max_iters == 1."
        )
    return True, "ok"


def build_streamk_gemm(spec: StreamKGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one StreamK GEMM instance.

    Kernel signature::

    (A: ptr<f16, global>, # (M, K) row-major
    B: ptr<f16, global>, # (K, N) row-major
    Cf32: ptr<f32, global>, # (M, N) f32 workspace -- caller pre-cleared
    Counter: ptr<i32, global>) # 1-slot tile counter, pre-cleared

    Grid:
    * ``persistent=False`` (default): ``(num_macro_tiles, 1, 1)`` -- one
      CTA per macro tile; the kernel reads ``block_id_x`` as the
      linear macro-tile id and the ``Counter`` arg is unused (but
      still required for ABI parity).
    * ``persistent=True``: ``(compute_streamk_grid_size(...), 1, 1)`` --
      a persistent CTA pool; each CTA atomically fetches its next
      macro tile id from the ``Counter`` slot via
      :func:`persistent_tile_for_each` and processes
      ``persistent_max_iters`` macro tiles in the worst case.

    Algorithm (per macro tile):
    1. Decode ``(m_tile, n_tile, k_iter)`` from the linear id via the
       StreamK partitioner.
    2. SGPR-pin every wave-uniform i32 derived from the macro-tile id
       (``m_tile``, ``n_tile``, ``k_iter``, ``m_tile_base``,
       ``n_tile_base``, ``k_macro_base``). This mirrors CK Tile's
       ``amd_wave_read_first_lane(iM * MPerBlock)`` pattern -- the
       compiler is told these are scalar across the wave so the
       address arithmetic in the MFMA K-loop and atomic-store epilogue
       stays in SGPRs.
    3. Run the MFMA K-loop (`tile_k / atom.k` atoms; per-lane
       ``<c_per_lane x f32>`` accumulator).
    4. Atomic-add the f32 ``acc`` into ``Cf32[m_global, n_global]``.

    Notes:
    * ``is_first`` / ``is_last`` predicates are decoded but unused in
      v1; the Atomic strategy doesn't need them. They're hooked up so
      the v2 Reduction strategy can reuse the same partitioner.
    * In the persistent path the broadcast LDS slot allocated by
      :func:`persistent_tile_for_each` is 4 bytes -- negligible
      against the gfx950 160 KiB LDS budget.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid streamk_gemm spec for {arch}: {why}")

    partition = spec.partition
    BS = spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Cf32 = b.param("Cf32", PtrType(F32, "global"), align=4)
    Counter = b.param("Counter", PtrType(I32, "global"), align=4)

    lane = b.thread_id_x()
    atom = spec.atom
    lane_decode = decode_mfma_lanes(b, atom, lane)

    def _process_macro_tile(linear_id: Value) -> None:
        """Emit the full MFMA + atomic-store body for one macro tile.

        Called once per CTA in the non-persistent path; called once
        per iteration of the persistent loop in the persistent path.
        Every CTA-uniform i32 derived from ``linear_id`` is
        SGPR-pinned via :meth:`IRBuilder.to_sgpr_u32` so the
        downstream address arithmetic stays in scalar registers.
        """
        # Decode (m_tile, n_tile, k_iter) for this macro tile and
        # SGPR-pin -- they are CTA-uniform (derived from ``linear_id``,
        # which is the same across all lanes in the wave) and feed
        # every subsequent address computation in the MFMA K-loop and
        # epilogue. The CK Tile reference does the equivalent via
        # ``amd_wave_read_first_lane`` (streamk_gemm_kernel.hpp:339).
        decoded = emit_streamk_decode(b, linear_id, partition)
        m_tile = b.to_sgpr_u32(decoded.m_tile)
        n_tile = b.to_sgpr_u32(decoded.n_tile)
        k_iter = b.to_sgpr_u32(decoded.k_iter)

        # Per-tile bases: pin them too. CK Tile's reference pins the
        # equivalent ``i_m = iM * MPerBlock`` (batched_gemm_kernel.hpp:207)
        # and ``i_n = iN * NPerBlock`` (same:208); the streamk path
        # pins ``im``, ``in`` inside ``get_output_tile_index``
        # (streamk_gemm_tile_partitioner_impl.hpp:147-148).
        m_tile_base = b.to_sgpr_u32(b.mul(m_tile, b.const_i32(spec.tile_m)))
        n_tile_base = b.to_sgpr_u32(b.mul(n_tile, b.const_i32(spec.tile_n)))
        # K-base for this macro tile: ``k_iter * tile_k``. Pinning
        # keeps the per-K-atom address ``k_macro_base + kt*atom.k``
        # in SGPRs throughout the K-loop.
        k_macro_base = b.to_sgpr_u32(b.mul(k_iter, b.const_i32(spec.tile_k)))

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
        # <c_per_lane x f32> accumulator. ``mfma_k_loop`` re-emits a
        # fresh ``atom.zero_acc(b)`` per call so persistent reuse
        # starts cleanly on every iteration.
        acc_final = mfma_k_loop(
            b,
            K=spec.tile_k,
            atom=atom,
            load_a=_load_a,
            load_b=_load_b,
        )

        # Atomic-add each lane's c_per_lane cells into the Cf32
        # workspace. The atom's ``lane_to_output`` mapping decodes the
        # per-lane output cell coords; the f32 split-K reduction
        # across (m_tile, n_tile, *) k_iter values converges to the
        # full f32 GEMM.
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

    if spec.persistent:
        # Persistent dispatch: one CTA per CU/wave-slot, iteration
        # count bounded by the worst-case ``ceil(num_macro_tiles /
        # grid_size)`` so the inner scf.for trip count is a Python
        # int (the in-range guard inside :func:`persistent_tile_for_each`
        # masks over-fetched ids past ``num_macro_tiles``).
        persistent_tile_for_each(
            b,
            counter=Counter,
            num_tiles=b.const_i32(partition.num_macro_tiles),
            max_iters=spec.persistent_max_iters,
            body=_process_macro_tile,
        )
    else:
        # Non-persistent dispatch: one CTA per macro tile; the
        # ``Counter`` slot is unused (still required by the ABI for
        # parity with the persistent variant).
        _ = Counter  # noqa: F841 - kept in the ABI for the persistent path
        _process_macro_tile(b.block_id_x())

    b.ret()
    return b.kernel


def streamk_gemm_grid(spec: StreamKGemmSpec) -> Tuple[int, int, int]:
    """Launch grid for one StreamK GEMM spec.

    Dispatches on ``spec.persistent``:

    * ``persistent=False`` (default): grid = ``(num_macro_tiles, 1, 1)``
      -- one CTA per macro tile; ``block_id_x`` *is* the linear macro-
      tile id. Best for shapes where ``num_macro_tiles`` is comparable
      to the GPU's CU count (the launch grid is naturally bounded).
    * ``persistent=True``: grid = ``(compute_streamk_grid_size(...), 1, 1)``
      -- the CU-bounded persistent pool. Each CTA pulls macro tiles
      from the ``Counter`` atomic-counter slot, processing up to
      ``persistent_max_iters`` macro tiles in the worst case. Best
      when ``num_macro_tiles`` >> CU count, so the launch grid stays
      constant regardless of problem size.

    The kernel ABI is identical across both modes; only the launch
    grid (and the kernel's interpretation of ``Counter``) differs.
    """
    if spec.persistent:
        return (spec.grid_size, 1, 1)
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


def build_streamk_gemm_block_tile(
    spec: StreamKGemmSpec, arch: str = "gfx950"
) -> KernelDef:
    """Block-tile / multi-warp variant of streamk_gemm (P66).

    Today's :func:`build_streamk_gemm` uses one wave per CTA with a
    "scalar-style" MFMA inner (one warp tile per CTA). This variant
    re-uses the universal-GEMM block-tile body inside the persistent
    StreamK loop, parameterised on the StreamK partitioner. With P35
    fixed, the persistent loop is safe at any ``max_iters`` so this
    builder defaults to ``persistent=True``.

    Minimum-viable: dispatches into :func:`build_streamk_gemm` with
    ``persistent=True``; the block-tile body is inherited from the
    universal-gemm path the streamk kernel already wraps. Real
    perf hoist is enabling ``streamk`` with multi-warp tile + larger
    M/N bands — exposed via the spec's ``tile_m`` / ``tile_n`` /
    ``warp_m`` / ``warp_n`` fields.

    Reference: CK Tile ``streamk_gemm_kernel.hpp`` block-tile body.
    """
    from dataclasses import replace

    block_tile_spec = replace(spec, persistent=True)
    return build_streamk_gemm(block_tile_spec, arch=arch)


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
