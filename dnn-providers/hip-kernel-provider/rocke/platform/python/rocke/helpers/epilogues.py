# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Epilogue helpers: accumulators -> global memory.

Two epilogues are exposed:

* `DirectEpilogue` — per-lane stores from the MFMA's accumulator at the
  position the atom dictates. Best when:
    - the atom's per-lane output layout already matches the global
      store layout (CK Tile's "no shuffle needed" case);
    - the per-lane output width matches a natural store vector
      (vec-1, vec-2, vec-4, vec-8 halves) and the in-lane elements
      are contiguous in the output;
    - the LDS bandwidth is the bottleneck and the cshuffle staging
      pass would add a barrier without much coalescing win.

* `CShuffleEpilogue` — the LDS-stage shuffle. Per the runbook §9.3:
    1. Each warp converts its `<c_per_lane x f32>` accumulator vector
       to `<c_per_lane x f16>` and writes the halves into a per-block
       LDS region at the *MFMA output layout* (consecutive lanes hold
       consecutive N-direction elements).
    2. `block_sync_lds` (s_barrier).
    3. A flat distribution of `STORE_VECS = (tile_m * tile_n) /
       store_vec` threads reads `<store_vec x f16>` from LDS in
       row-major order and issues one wide global store per thread.
  Best when:
    - the atom's per-lane output layout is *not* contiguous in N (the
      common case for 16x16 atoms; lane 0 holds N=0 but lane 1 holds
      N=1 which is 2 bytes away — fine — but lane 16 holds N=0 again,
      which is a different M row and *not* contiguous);
    - direct vector stores would issue `c_per_lane` scalar
      `buffer_store_short`s per atom, none of which coalesce.

Both helpers take three pieces of authoring input:

  - `atom`        : `MfmaAtom`, supplies the lane->output mapping
  - `grid`        : a *bound* `WarpGrid` for the per-warp offsets
  - `addr_fn`     : `(b, m_global, n_global) -> (off_elements, valid)`
                    callback the kernel provides to map (M, N) into
                    the output's linear offset. For plain GEMM this is
                    just `row * N + col`; for NHWK conv this is the
                    D descriptor's `offset(m, k_out)`.

`bounds` is the (M, N) tuple of i32 SSA values used to OOB-mask the
per-element validity. For tiles that always divide M and N evenly,
pass `None` to skip the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Union

from ..core.ir import BF16, F16, F32, IRBuilder, Value
from .atoms import MfmaAtom

if TYPE_CHECKING:
    from ..core.arch.target import MmaOp
from .distribution import (
    LoadStoreTraits,
    TileDistribution,
    TileDistributionEncoding,
    make_static_distributed_tensor,
    make_static_tile_distribution,
    store_tile_cshuffle,
)
from .geometry import WarpGrid
from .layouts import LdsLayout
from .tensor_view import make_lds_view


AddrFn = Callable[[IRBuilder, Value, Value], Tuple[Value, Optional[Value]]]


@dataclass(frozen=True)
class DirectEpilogue:
    """Per-lane direct global store using the atom's `lane_to_output` map.

    Authoring usage:

        epi = DirectEpilogue(atom=atom, grid=grid)
        epi.store(b, accs=accs, addr_fn=d_off,
                  d_rsrc=d_rsrc, bounds=(M, N))

    The store issues, per (mi, ni) MFMA-tile and per lane, one packed
    `<c_per_lane x half>` value. When the atom's `lane_to_output` puts
    those `c_per_lane` elements at consecutive positions along the *N*
    axis, we issue a single `buffer_store_vN_f16` per lane (the
    runbook §6.2 lever — `~95 -> ~213 TFLOPS` on the 16c direct conv).
    When they are not consecutive (16x16 / 32x32 atoms) we fall back to
    `c_per_lane` scalar `buffer_store_short`s; for that case use
    `CShuffleEpilogue` instead.
    """

    atom: MfmaAtom
    grid: WarpGrid
    out_dtype: str = "f16"

    @property
    def _row_stride_per_slot(self) -> int:
        """How many output ROWS the per-lane accumulator vector spans.

        - 16x16: `c_per_lane = 4` and `acc[i]` lives at `row = m_blk*4 + i`,
          so 4 distinct rows per lane (consecutive rows).
        - 32x32: 16 elements per lane spread over 8 distinct rows
          (rb=0..3 * 8 + ri=0..3 + m_blk*4); not row-contiguous.
        - 4x4:   `c_per_lane = 4` and `acc[i]` lives at `row = i`,
          col fixed; 4 distinct rows per lane.

        For 4x4 the 4 elements ARE contiguous in N (col = lane_in_batch
        is constant across i); for 16x16 the 4 elements are contiguous
        in *M* (rows i = 0..3 at fixed N); for 32x32 they are
        scattered.
        """
        if (self.atom.m, self.atom.n) == (16, 16):
            return 4  # 4 consecutive M rows per lane
        if (self.atom.m, self.atom.n) == (4, 4):
            return 4  # 4 consecutive M rows per lane
        return 0  # 32x32: scattered

    @property
    def _is_col_contiguous(self) -> bool:
        """True when acc[0..c_per_lane-1] are contiguous in *columns*.

        For 4x4 atom: lane has 4 elements at (row=i, col=lane_in_batch).
        Each lane's 4 elements are at the *same col*, different rows.
        Hmm — that's not col-contiguous. Let me re-read the
        `lane_to_output` mapping in `atoms.py`:
            4x4: row = i, col = lane_in_batch
        So the 4 elements per lane are at (0,c), (1,c), (2,c), (3,c)
        — same col c, different rows i. Not col-contiguous.

        For direct conv 4c, the *output* layout is NHWK with k_out
        sweeping fastest, so the consumer kernel maps these 4 acc
        elements to `D[n, h, w, g*kpg + i]` — *i* sweeps the fastest
        output index. So in the *consumer*'s addressing the 4 acc
        elements ARE consecutive in the fastest output dim — but
        that's a consumer-side property, not an atom-side property.

        For now we keep the helper simple: only emit a wide vec store
        when the user explicitly opts in by setting `vec_in_acc=True`
        (the 4c direct conv case).
        """
        return False

    def store(
        self,
        b: IRBuilder,
        *,
        accs: Sequence[Value],
        addr_fn: AddrFn,
        d_rsrc: Value,
        bounds: Optional[Tuple[Value, Value]] = None,
        vec_in_acc: bool = False,
    ) -> None:
        """Issue all per-(mi, ni)-tile stores.

        `accs` is the flat list of accumulator SSA values, in row-major
        order over (mi, ni). Length must equal
        `mfmas_per_warp_m * mfmas_per_warp_n`.

        `vec_in_acc=True`: the caller asserts that the
        `c_per_lane` accumulator elements are contiguous in the global
        output's fastest dim (true for the direct conv 4c case where
        `i` -> `k_out`). When true, we emit one `buffer_store_vN_f16`
        per atom-tile per lane instead of `c_per_lane` scalar stores.

        `bounds`: `(M, N)` i32 SSA values; if provided, per-element
        stores are guarded by `m < M && n < N`. Pass `None` for
        always-in-bounds tiles.
        """
        atom = self.atom
        grid = self.grid
        if not grid.is_bound:
            raise RuntimeError("DirectEpilogue: grid must be bound first")

        mfmas_m = grid.mfmas_per_warp_m
        mfmas_n = grid.mfmas_per_warp_n
        if len(accs) != mfmas_m * mfmas_n:
            raise ValueError(
                f"DirectEpilogue: expected {mfmas_m * mfmas_n} accs, got {len(accs)}"
            )

        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)
        _fp32_out = self.out_dtype == "fp32"
        _bf16_out = self.out_dtype == "bf16"
        c_elem_bytes = b.const_i32(4 if _fp32_out else 2)
        oob_sentinel = b.const_i32((1 << 31) - 1)

        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[mi * mfmas_n + ni]
                atom_m_off = b.add(
                    b.add(grid.block_m_off, warp_m_off),
                    b.const_i32(mi * atom.m),
                )
                atom_n_off = b.add(
                    b.add(grid.block_n_off, warp_n_off),
                    b.const_i32(ni * atom.n),
                )

                if vec_in_acc:
                    # Emit one wide vec store per lane (4 elements per
                    # 4x4 atom). The `addr_fn` is called at the
                    # *first* output element (i=0); the contiguous
                    # `c_per_lane - 1` more elements come from the
                    # accumulator's vector layout.
                    row_off, col_off = atom.lane_to_output(b, grid.lane, 0)
                    m_val = b.add(atom_m_off, row_off)
                    n_val = b.add(atom_n_off, col_off)
                    ok = self._bounds_check(
                        b, m_val, n_val, bounds, vec_n=atom.c_per_lane
                    )
                    off_elems, valid = addr_fn(b, m_val, n_val)
                    if ok is not None:
                        ok = b.land(ok, valid) if valid is not None else ok
                    else:
                        ok = valid
                    off_bytes = b.mul(off_elems, c_elem_bytes)
                    safe = (
                        b.select(ok, off_bytes, oob_sentinel)
                        if ok is not None
                        else off_bytes
                    )
                    if _fp32_out:
                        # fp32 output: store accumulator elements directly as f32.
                        # c_per_lane elements = c_per_lane dwords.
                        n_elems = atom.c_per_lane
                        if n_elems not in (1, 2, 4):
                            raise ValueError(
                                f"vec_in_acc=True fp32 with c_per_lane={n_elems} unsupported"
                            )
                        b.buffer_store_vN_f32(
                            d_rsrc, safe, b.const_i32(0), acc, n_elems
                        )
                    elif _bf16_out:
                        acc_bf = b.vec_trunc_f32_to_bf16(acc)
                        # 4 bfloats -> 2 dwords; 8 -> 4.
                        if atom.c_per_lane == 4:
                            b.buffer_store_vN_bf16(
                                d_rsrc, safe, b.const_i32(0), acc_bf, 2
                            )
                        elif atom.c_per_lane == 8:
                            b.buffer_store_vN_bf16(
                                d_rsrc, safe, b.const_i32(0), acc_bf, 4
                            )
                        else:
                            raise ValueError(
                                f"vec_in_acc=True bf16 with c_per_lane={atom.c_per_lane} unsupported"
                            )
                    else:
                        acc_h = b.vec_trunc_f32_to_f16(acc)
                        # Choose dword width based on c_per_lane.
                        # 4 halves -> 2 dwords; 8 -> 4; 16 -> not supported
                        # as a single store (the 32x32 atom is unreachable here).
                        if atom.c_per_lane == 4:
                            b.buffer_store_vN_f16(
                                d_rsrc, safe, b.const_i32(0), acc_h, 2
                            )
                        elif atom.c_per_lane == 8:
                            b.buffer_store_vN_f16(
                                d_rsrc, safe, b.const_i32(0), acc_h, 4
                            )
                        else:
                            raise ValueError(
                                f"vec_in_acc=True with c_per_lane={atom.c_per_lane} unsupported"
                            )
                else:
                    for i in range(atom.c_per_lane):
                        row_off, col_off = atom.lane_to_output(b, grid.lane, i)
                        m_val = b.add(atom_m_off, row_off)
                        n_val = b.add(atom_n_off, col_off)
                        ok = self._bounds_check(b, m_val, n_val, bounds, vec_n=1)
                        off_elems, valid = addr_fn(b, m_val, n_val)
                        if ok is not None:
                            ok = b.land(ok, valid) if valid is not None else ok
                        else:
                            ok = valid
                        v_f32 = b.vec_extract(acc, i)
                        off_bytes = b.mul(off_elems, c_elem_bytes)
                        safe = (
                            b.select(ok, off_bytes, oob_sentinel)
                            if ok is not None
                            else off_bytes
                        )
                        if _fp32_out:
                            b.buffer_store_f32(d_rsrc, safe, b.const_i32(0), v_f32)
                        elif _bf16_out:
                            v_bf16 = b.trunc_f32_to_bf16(v_f32)
                            b.buffer_store_bf16(d_rsrc, safe, b.const_i32(0), v_bf16)
                        else:
                            v_f16 = b.trunc_f32_to_f16(v_f32)
                            b.buffer_store_f16(d_rsrc, safe, b.const_i32(0), v_f16)

    @staticmethod
    def _bounds_check(
        b: IRBuilder,
        m: Value,
        n: Value,
        bounds: Optional[Tuple[Value, Value]],
        *,
        vec_n: int = 1,
    ) -> Optional[Value]:
        """Compute `m < M && (n + vec_n - 1) < N` (or None)."""
        if bounds is None:
            return None
        M, N = bounds
        m_ok = b.cmp_lt(m, M)
        if vec_n > 1:
            n_end = b.add(n, b.const_i32(vec_n))
            n_ok = b.cmp_le(n_end, N)
        else:
            n_ok = b.cmp_lt(n, N)
        return b.land(m_ok, n_ok)


def _cshuffle_acc_distribution(c_per_lane: int) -> TileDistribution:
    """Register-container distribution for one warp tile's accumulator.

    A single X dim is decomposed as ``(c_per_lane, 1)`` with one Y per
    level: the outer Y (length ``c_per_lane``) enumerates the per-lane
    accumulator register slots, and the inner Y (length 1) is the scalar
    publish unit. ``y_to_linear((i, 0)) == i`` so slot ``i`` maps to
    accumulator element ``i``. The distribution is used only as the
    :class:`StaticDistributedTensor` shape / iteration order; the actual
    LDS coordinate is supplied by the epilogue's ``coord_fn`` (the MFMA
    output layout is not a clean ``unmerge`` of the lane id).
    """
    enc = TileDistributionEncoding(
        Hs=((int(c_per_lane), 1),),
        Ys2RHs_major=(1, 1),
        Ys2RHs_minor=(0, 1),
    )
    return make_static_tile_distribution(enc)


@dataclass(frozen=True)
class CShuffleEpilogue:
    """LDS-staged C-shuffle epilogue (runbook §9.3).

    Three-stage pattern:
      1. Each lane converts its `<c_per_lane x f32>` accumulator to
         `<c_per_lane x f16>` and writes the halves into LDS at the
         *MFMA output layout* positions:
           - 16x16: `LDS[m_blk*4 + i + warp_m_off + mi*16, n_in_atom + warp_n_off + ni*16]`
           - 32x32: `LDS[(i//4)*8 + m_blk*4 + (i%4) + warp_m_off + mi*32, n_in_atom + warp_n_off + ni*32]`
         All threads write c_per_lane halves with one `ds_write_b16`
         per slot.
      2. `block_sync_lds` (`s_barrier`).
      3. A flat distribution of `block_size` threads reads
         `<store_vec x f16>` from LDS at consecutive row-major
         positions and issues one wide `buffer_store_vN_f16` per
         thread. Each thread handles
         `vecs_per_thread = tile_m * tile_n / store_vec / block_size`
         output rows.
    """

    grid: WarpGrid
    store_vec: int = 8  # halves per wide store
    smem_name_hint: str = "C_smem"
    # P41: output dtype the per-thread wide store produces. ``"f16"`` is
    # the default and uses the f16 ``buffer_store_vN_f16`` intrinsic;
    # ``"bf16"`` swaps the LDS staging element type to bf16 and
    # uses the matching store; ``"fp8e4m3"`` / ``"bf8e5m2"`` emit
    # 1-byte stores via ``global_store_vN`` (no buffer-store fp8
    # intrinsic exists today on the AMDGPU LLVM target).
    out_dtype: str = "f16"
    # cshuffle "no-alias" mode: when True the C staging tile gets its own
    # exclusive LDS bytes (not aliased onto the A/B pool) and the step-0 reuse
    # barrier is elided -> lower small-tile latency, more LDS. Default False
    # keeps the aliased/low-LDS behavior (byte-identical).
    no_alias: bool = False
    # Number of step-0 WAR barriers to emit when no_alias=False.
    # Single-role pipelines (all waves are math waves): 1.
    # Wavelet pipeline (split load/math waves): 2 —
    #   barrier 0: MFMAs done; load waves may exit.
    #   barrier 1: load waves exited; safe to overwrite A/B LDS with C writes.
    war_barriers: int = 1
    # MFMA path: set by from_grid(); drives lane_to_output in store().
    atom: Optional[MfmaAtom] = None
    # WMMA path: set by from_grid_op(); uses c_frag_len / c_layout().coord() instead.
    mma_op: Optional["MmaOp"] = None

    @property
    def barrier_count(self) -> int:
        """Total number of ``s_barrier`` ops emitted by :meth:`store`.

        Equals ``war_barriers`` (step-0 WAR reuse barriers, elided when
        ``no_alias=True``) plus 1 (step-2 RAW barrier — always emitted).
        Use this in place of hand-maintained constants when the load branch
        of a split-role pipeline must mirror the epilogue barrier count exactly.
        """
        return CShuffleEpilogue.compute_barrier_count(self.no_alias, self.war_barriers)

    @staticmethod
    def compute_barrier_count(no_alias: bool, war_barriers: int) -> int:
        """Return the number of barriers :meth:`store` will emit.

        This is the canonical formula — call it from K-loop setup code that needs
        to know the count *before* building the epilogue IR, so both the math
        branch and the load branch in a split-role pipeline use the same number.
        """
        return (0 if no_alias else war_barriers) + 1

    @classmethod
    def from_grid_op(
        cls,
        *,
        op: "MmaOp",
        grid: WarpGrid,
        max_store_vec: int = 8,
        out_dtype: str = "f16",
        no_alias: bool = False,
    ) -> "CShuffleEpilogue":
        """Construct for a WMMA (``MmaOp``) accumulator layout.

        Uses ``op.c_frag_len`` / ``op.c_layout()`` for the LDS scatter in
        place of the MFMA-specific ``atom.lane_to_output``.
        """
        _fp32_out = out_dtype == "fp32"
        _max_sv = min(max_store_vec, 4) if _fp32_out else min(max_store_vec, 8)
        v = _max_sv
        block_size = grid.block_size
        while v > 1:
            ok = (
                grid.tile_n % v == 0
                and (grid.tile_m * grid.tile_n) // v >= block_size
                and ((grid.tile_m * grid.tile_n) // v) % block_size == 0
            )
            if ok:
                break
            v //= 2
        return cls(
            grid=grid, store_vec=v, out_dtype=out_dtype, mma_op=op, no_alias=no_alias
        )

    @classmethod
    def from_grid(
        cls,
        *,
        atom: MfmaAtom,
        grid: WarpGrid,
        max_store_vec: int = 8,
        out_dtype: str = "f16",
        no_alias: bool = False,
    ) -> "CShuffleEpilogue":
        """Pick the widest `store_vec` that distributes the tile evenly.

        For fp32 output the LDS tile is stored in f32 elements (4 bytes
        each), so `store_vec` counts f32 elements and is capped at 4
        (= 16 bytes, the hardware buffer_store limit).
        """
        _fp32_out = out_dtype == "fp32"
        _max_sv = min(max_store_vec, 4) if _fp32_out else min(max_store_vec, 8)
        v = _max_sv
        block_size = grid.block_size
        while v > 1:
            ok = (
                grid.tile_n % v == 0
                and (grid.tile_m * grid.tile_n) // v >= block_size
                and ((grid.tile_m * grid.tile_n) // v) % block_size == 0
            )
            if ok:
                break
            v //= 2
        return cls(
            atom=atom, grid=grid, store_vec=v, out_dtype=out_dtype, no_alias=no_alias
        )

    def store(
        self,
        b: IRBuilder,
        *,
        accs: Sequence[Value],
        addr_fn: AddrFn,
        d_rsrc: Value,
        bounds: Optional[Tuple[Value, Value]] = None,
    ) -> None:
        atom = self.atom
        op = self.mma_op
        grid = self.grid
        if not grid.is_bound:
            raise RuntimeError("CShuffleEpilogue: grid must be bound first")
        if atom is None and op is None:
            raise RuntimeError(
                "CShuffleEpilogue: must set either atom (MFMA) or mma_op (WMMA)"
            )

        # Resolve per-lane count and atom tile size from whichever ISA is active.
        if op is not None:
            _c_per_lane = op.c_frag_len
            _atom_m = op.m
            _atom_n = op.n
            _c_layout = op.c_layout()
        else:
            _c_per_lane = atom.c_per_lane
            _atom_m = atom.m
            _atom_n = atom.n
            _c_layout = None

        mfmas_m = grid.mfmas_per_warp_m
        mfmas_n = grid.mfmas_per_warp_n
        if len(accs) != mfmas_m * mfmas_n:
            raise ValueError(
                f"CShuffleEpilogue: expected {mfmas_m * mfmas_n} accs, got {len(accs)}"
            )

        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)
        _fp32_out = self.out_dtype == "fp32"
        _bf16_out = self.out_dtype == "bf16"
        _lds_dtype = F32 if _fp32_out else (BF16 if _bf16_out else F16)

        # ---- step 1: publish accs to LDS at the MMA output layout. ----
        #
        # The LDS staging region is a plain ``[tile_m, tile_n]`` row-major
        # buffer (``LdsLayout.cshuffle``); each lane writes its
        # ``c_per_lane`` accumulator elements at the exact MMA *output*
        # coordinate the atom/op dictates, so the subsequent row-major
        # stage-3 read reconstructs the global tile. The per-warp-tile
        # accumulator is carried in a :class:`StaticDistributedTensor`
        # and published through :func:`store_tile_cshuffle`.
        #
        # The MMA output layout (``atom.lane_to_output`` / ``op.c_layout().coord()``)
        # is not a clean ``unmerge`` of the lane id, so the LDS coordinate is
        # supplied via an explicit ``coord_fn`` rather than ``calculate_x``.
        lds_layout = LdsLayout.cshuffle(tile_m=grid.tile_m, tile_n=grid.tile_n)
        lds_layout.validate()
        c_view = make_lds_view(
            b,
            dtype=_lds_dtype,
            shape=lds_layout.storage_shape(grid.tile_m),
            name_hint=self.smem_name_hint,
            exclusive=self.no_alias,
        )
        c_smem = c_view.base
        # A full-extent LDS window; per-(mi, ni) origins move within it.
        c_window = c_view.tile(
            list(lds_layout.storage_shape(grid.tile_m)),
            [b.const_i32(0), b.const_i32(0)],
        )

        # One distributed-tensor container per warp tile: a single X dim
        # decomposed as ``(c_per_lane, 1)`` with one Y per level, so the
        # outer Y enumerates the ``c_per_lane`` register slots and the
        # inner (vector) Y is the scalar publish unit. ``y_to_linear`` of
        # ``(i, 0)`` is exactly the accumulator index ``i``.
        dist = _cshuffle_acc_distribution(_c_per_lane)
        traits = LoadStoreTraits(distribution=dist, vector_dim_y=1, scalar_per_vector=1)

        # ---- step 0: reuse barrier(s). ----
        # The common-LDS packer aliases this C staging tile onto the A/B
        # staging bytes (non-interfering in program order).
        # war_barriers controls how many are emitted:
        #   1 (default) — single-role pipelines: one barrier drains last LDS
        #                 reads before C scatter writes begin.
        #   2 (wavelet) — split load/math waves need two:
        #                 barrier 0: MFMAs done; load waves may exit.
        #                 barrier 1: load waves exited; A/B LDS safe to overwrite.
        # With ``no_alias`` the C tile never overlaps A/B so all are elided.
        # The step-2 C-write->C-read barrier is a genuine RAW and always stays.
        if not self.no_alias:
            for _ in range(self.war_barriers):
                b.sync()

        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[mi * mfmas_n + ni]
                if _fp32_out:
                    # For fp32 output keep accumulator as f32 in LDS.
                    acc_staged = acc
                    dt = make_static_distributed_tensor(dist, dtype=F32)
                elif _bf16_out:
                    acc_staged = b.vec_trunc_f32_to_bf16(acc)
                    dt = make_static_distributed_tensor(dist, dtype=BF16)
                else:
                    acc_staged = b.vec_trunc_f32_to_f16(acc)
                    dt = make_static_distributed_tensor(dist, dtype=F16)
                for i in range(_c_per_lane):
                    dt.set([i, 0], b.vec_extract(acc_staged, i))

                tile_m_base = b.add(warp_m_off, b.const_i32(mi * _atom_m))
                tile_n_base = b.add(warp_n_off, b.const_i32(ni * _atom_n))

                if _c_layout is not None:
                    # WMMA path: use op.c_layout().coord() for the LDS scatter.
                    def coord_fn(
                        b_,
                        y_base,
                        k,
                        *,
                        _mb=tile_m_base,
                        _nb=tile_n_base,
                        _layout=_c_layout,
                        _lane=grid.lane,
                    ):
                        i = int(y_base[0])
                        row_in_atom, col_in_atom = _layout.coord(b_, _lane, i)
                        return [b_.add(_mb, row_in_atom), b_.add(_nb, col_in_atom)]

                else:
                    # MFMA path: use atom.lane_to_output().
                    def coord_fn(
                        b_,
                        y_base,
                        k,
                        *,
                        _mb=tile_m_base,
                        _nb=tile_n_base,
                        _atom=atom,
                        _lane=grid.lane,
                    ):
                        i = int(y_base[0])
                        row_in_atom, col_in_atom = _atom.lane_to_output(b_, _lane, i)
                        return [b_.add(_mb, row_in_atom), b_.add(_nb, col_in_atom)]

                store_tile_cshuffle(b, c_window, dt, traits=traits, coord_fn=coord_fn)

        # ---- step 2: barrier. ----
        b.sync()

        # ---- step 3: wide global stores from LDS. ----
        threads = grid.block_size
        sv = self.store_vec
        if (
            grid.tile_n % sv
            or (grid.tile_m * grid.tile_n) // sv < threads
            or ((grid.tile_m * grid.tile_n) // sv) % threads
        ):
            raise ValueError(
                f"store_vec {sv} does not distribute over tile "
                f"{grid.tile_m}x{grid.tile_n} and block_size {threads}"
            )
        vecs_per_thread = (grid.tile_m * grid.tile_n // sv) // threads
        c_threads = b.const_i32(threads)
        c_tile_n_div_vec = b.const_i32(grid.tile_n // sv)
        c_elem_bytes = b.const_i32(4 if _fp32_out else 2)
        oob_sentinel = b.const_i32((1 << 31) - 1)

        for e in range(vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), grid.tid)
            row = b.div(vec_idx, c_tile_n_div_vec)
            col_v = b.mod(vec_idx, c_tile_n_div_vec)
            col = b.mul(col_v, b.const_i32(sv)) if sv > 1 else col_v

            m_val = b.add(grid.block_m_off, row)
            n_val = b.add(grid.block_n_off, col)
            ok = DirectEpilogue._bounds_check(b, m_val, n_val, bounds, vec_n=sv)
            off_elems, valid = addr_fn(b, m_val, n_val)
            ok = (
                b.land(ok, valid)
                if (ok is not None and valid is not None)
                else (ok if ok is not None else valid)
            )

            off_bytes = b.mul(off_elems, c_elem_bytes)
            safe = (
                b.select(ok, off_bytes, oob_sentinel) if ok is not None else off_bytes
            )

            if _fp32_out:
                # fp32 output: load sv f32 elements from LDS, store as f32.
                if sv == 1:
                    vf32 = b.smem_load_vN(c_smem, row, col, dtype=F32, n=1)
                    b.buffer_store_f32(
                        d_rsrc, safe, b.const_i32(0), b.vec_extract(vf32, 0)
                    )
                else:
                    vf32 = b.smem_load_vN(c_smem, row, col, dtype=F32, n=sv)
                    b.buffer_store_vN_f32(d_rsrc, safe, b.const_i32(0), vf32, sv)
            elif _bf16_out:
                # bf16 output: load sv bf16 elements from LDS, store as bf16.
                if sv == 1:
                    vbf = b.smem_load_vN(c_smem, row, col, dtype=BF16, n=1)
                    b.buffer_store_bf16(
                        d_rsrc, safe, b.const_i32(0), b.vec_extract(vbf, 0)
                    )
                else:
                    vbf = b.smem_load_vN(c_smem, row, col, dtype=BF16, n=sv)
                    dwords = sv // 2
                    b.buffer_store_vN_bf16(d_rsrc, safe, b.const_i32(0), vbf, dwords)
            else:
                if sv == 1:
                    v = b.smem_load_vN_f16(c_smem, row, col, n=1)
                    h = b.vec_extract(v, 0)
                    b.buffer_store_f16(d_rsrc, safe, b.const_i32(0), h)
                else:
                    if sv == 4:
                        v = b.smem_load_v4_f16(c_smem, row, col)
                    else:
                        v = b.smem_load_vN_f16(c_smem, row, col, n=sv)
                    dwords = sv // 2
                    b.buffer_store_vN_f16(d_rsrc, safe, b.const_i32(0), v, dwords)
