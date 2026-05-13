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
from typing import Callable, Optional, Sequence, Tuple

from ..core.ir import F16, IRBuilder, Value
from .atoms import MfmaAtom
from .geometry import WarpGrid


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
        c_half_bytes = b.const_i32(2)
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
                    # Emit one wide vec store per lane (4 halves per
                    # 4x4 atom). The `addr_fn` is called at the
                    # *first* output element (i=0); the contiguous
                    # `c_per_lane - 1` more halves come from the
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
                    off_bytes = b.mul(off_elems, c_half_bytes)
                    safe = (
                        b.select(ok, off_bytes, oob_sentinel)
                        if ok is not None
                        else off_bytes
                    )
                    acc_h = b.vec_trunc_f32_to_f16(acc)
                    # Choose dword width based on c_per_lane.
                    # 4 halves -> 2 dwords; 8 -> 4; 16 -> not supported
                    # as a single store (the 32x32 atom is unreachable here).
                    if atom.c_per_lane == 4:
                        b.buffer_store_vN_f16(d_rsrc, safe, b.const_i32(0), acc_h, 2)
                    elif atom.c_per_lane == 8:
                        b.buffer_store_vN_f16(d_rsrc, safe, b.const_i32(0), acc_h, 4)
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
                        v_f16 = b.trunc_f32_to_f16(v_f32)
                        off_bytes = b.mul(off_elems, c_half_bytes)
                        safe = (
                            b.select(ok, off_bytes, oob_sentinel)
                            if ok is not None
                            else off_bytes
                        )
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

    atom: MfmaAtom
    grid: WarpGrid
    store_vec: int = 8  # halves per wide store
    smem_name_hint: str = "C_smem"

    @classmethod
    def from_grid(
        cls,
        *,
        atom: MfmaAtom,
        grid: WarpGrid,
        max_store_vec: int = 8,
    ) -> "CShuffleEpilogue":
        """Pick the widest `store_vec` that distributes the tile evenly."""
        v = max_store_vec
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
        return cls(atom=atom, grid=grid, store_vec=v)

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
        grid = self.grid
        if not grid.is_bound:
            raise RuntimeError("CShuffleEpilogue: grid must be bound first")

        mfmas_m = grid.mfmas_per_warp_m
        mfmas_n = grid.mfmas_per_warp_n
        if len(accs) != mfmas_m * mfmas_n:
            raise ValueError(
                f"CShuffleEpilogue: expected {mfmas_m * mfmas_n} accs, got {len(accs)}"
            )

        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)

        c_smem = b.smem_alloc(
            F16, [grid.tile_m, grid.tile_n], name_hint=self.smem_name_hint
        )

        # ---- step 1: write accs to LDS at the MFMA output layout. ----
        is_32x32 = (atom.m, atom.n) == (32, 32)
        c_atom_n = b.const_i32(atom.n)
        n_in_atom = b.mod(grid.lane, c_atom_n)
        m_blk = b.div(grid.lane, c_atom_n)

        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[mi * mfmas_n + ni]
                acc_h = b.vec_trunc_f32_to_f16(acc)
                ld_n = b.add(
                    b.add(warp_n_off, b.const_i32(ni * atom.n)),
                    n_in_atom,
                )
                for i in range(atom.c_per_lane):
                    if is_32x32:
                        rb = i // 4
                        ri = i % 4
                        m_off = b.add(
                            b.add(b.const_i32(rb * 8), b.mul(m_blk, b.const_i32(4))),
                            b.const_i32(ri),
                        )
                    else:
                        m_off = b.add(
                            b.mul(m_blk, b.const_i32(atom.c_per_lane)),
                            b.const_i32(i),
                        )
                    ld_m = b.add(
                        b.add(warp_m_off, b.const_i32(mi * atom.m)),
                        m_off,
                    )
                    h = b.vec_extract(acc_h, i)
                    b.smem_store_f16(c_smem, [ld_m, ld_n], h)

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
        c_half_bytes = b.const_i32(2)
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

            off_bytes = b.mul(off_elems, c_half_bytes)
            safe = (
                b.select(ok, off_bytes, oob_sentinel) if ok is not None else off_bytes
            )

            if sv == 1:
                v = b.smem_load_vN_f16(c_smem, row, col, n=2)
                h = b.vec_extract(v, 0)
                b.buffer_store_f16(d_rsrc, safe, b.const_i32(0), h)
            else:
                if sv == 4:
                    v = b.smem_load_v4_f16(c_smem, row, col)
                else:
                    v = b.smem_load_vN_f16(c_smem, row, col, n=sv)
                dwords = sv // 2
                b.buffer_store_vN_f16(d_rsrc, safe, b.const_i32(0), v, dwords)
