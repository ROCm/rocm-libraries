# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Speed-run GEMM authored from the basic tiling primitives + manual layouts, driven toward
rocWMMA `perf_hgemm`. Phases are built up in-place (P1 register-blocked -> P2 LDS -> P3 pipeline);
this file currently holds P1.

P1 -- register-blocked, no LDS. A per-CTA wave tile of ``TILE_M x TILE_N`` (default 32x32 = a 2x2
grid of 16x16x16 MFMA atoms), single wave (wave64), RCC layout (A row-major, B/C col-major),
canonical `make_tile_desc` layouts (load layout == MFMA layout, so no cross-lane relayout is needed
and no LDS is used). The 2x2 grid gives register reuse: each loaded A / B atom feeds two MMAs.

Everything is authored with the primitive surface: `make_tensor_desc`/`make_window` (memory),
`make_tile_desc` (per-atom distribution), `load/store/fill_fragment` (verbs), and `TileMma` only to
hand back the backend `mma` op. Bit-exact vs a numpy golden; timed via the CK-style
`make_kernel`/`launch_kernel` hipEvent harness.
"""

from __future__ import annotations

import numpy as np

from .. import (
    TileMma,
    Tiling,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
    transform_fragment,
)
from ..encoding import WarpDistributionEncoding
from ..fragments import TileDesc

_ATOM = 16  # the 16x16x16 MFMA atom


def _bank_swizzle(width_elems: int):
    """EXPERIMENTAL LDS bank swizzle for the interleaved M-innermost store, at a chosen access width.

    Maps (K, m) -> (K, m ^ ((kk ^ (kk<<2)) << shift)), kk = K>>2, shift = log2(width_elems). The XOR
    de-aliases the K-by-4 coop store while keeping each ``width_elems``-run contiguous (it moves whole
    b{width}-blocks), so it's a bijection (bit-exact) and the emit stays a valid ds_{write,read}_b{width}.
    Narrower width -> finer bank placement -> fewer conflicts, but more instructions:
      b32 (width 2): conflict-free (full 32-bank permutation);  b64 (width 4): 2-way;  b128 (width 8): 4-way.
    ``vw_elems`` tells store/load the access width. Pass e.g. ``lds_swizzle=b32_swizzle`` and MEASURE the net
    (fewer conflicts vs more instruction-issue) -- the best width is bottleneck-dependent, not always b32."""
    shift = width_elems.bit_length() - 1

    def swz(b, positions):
        K, m = positions[-2], positions[-1]
        kk = b.div(K, b.const_i32(4))
        s = b.xor(kk, b.shl(kk, b.const_i32(2)))
        return positions[:-1] + [b.xor(m, b.shl(s, b.const_i32(shift)))]

    swz.vw_elems = width_elems
    return swz


b32_swizzle = _bank_swizzle(2)   # conflict-free (0-way), 4x instructions vs b128
b64_swizzle = _bank_swizzle(4)   # 2-way, 2x instructions
full_perm_swizzle = b32_swizzle  # back-compat alias (the conflict-free one)

# sched_group_barrier instruction-class mask bits (AMDGPU).
_SGB_MFMA = 0x008
_SGB_VMEM_READ = 0x020
_SGB_VMEM_WRITE = 0x040
_SGB_DS_READ = 0x100
_SGB_DS_WRITE = 0x200


def _wave_descs_interleaved(m_sub: int = 1, n_sub: int = 1, k_sub: int = 1):
    """INTERLEAVED wave-tile layouts (our own `make_tile_desc`) -- the free-dim (M for A, N for B) is a
    CONTIGUOUS per-lane `thread_tile` vector instead of the canonical K-vec. This matches the reference
    sheet's OrthoInterleave (A-Block: lane owns M0K0-3,M1K0-3,... = free-dim contiguous). It has the
    IDENTICAL K-distribution to the canonical layout (so it is hardware-valid -- same contraction), with
    M/N placed contiguously per lane (free -- it only routes where each product lands in C). TileMma
    slices atom `mi` = ``regs[mi*4:+4]`` = one free-dim index's K-vec (a valid 16x16 atom, free M/N).
    k_sub currently 1 (tile_k == atom K); deeper K is a follow-up.
    """
    a_desc = make_tile_desc(
        shape=[m_sub * _ATOM, k_sub * _ATOM],
        thread_tile=[m_sub, 4 * k_sub],  # free-dim (M) CONTIGUOUS per lane + 4*k_sub K (k_iter folded in)
        thread_dist=[16, 4],             # same hardware K distribution as canonical
        thread_order=[1, 0],
        thread_broadcast=1,
        block_repeat=[1, 1],
        wave_dist=[1, 1],
        wave_order=None,
        wave_broadcast=1,
        wave_size=64,
    )
    b_desc = make_tile_desc(
        shape=[n_sub * _ATOM, k_sub * _ATOM],
        thread_tile=[n_sub, 4 * k_sub],
        thread_dist=[16, 4],
        thread_order=[1, 0],
        thread_broadcast=1,
        block_repeat=[1, 1],
        wave_dist=[1, 1],
        wave_order=None,
        wave_broadcast=1,
        wave_size=64,
    )
    # NATIVE (MMA) C -- hand-built WarpDistributionEncoding matching what TileMma actually produces from
    # the interleaved A/B (make_tile_desc can't express it: its register order is coupled to the
    # coordinate stride, but the native C needs mi at stride 1 while TileMma forces mi an outer register).
    # Derived map: reg = mi*(n_sub*4) + nj*4 + r  ->  M = mi + m_sub*(4*(L//16) + r), N = nj + n_sub*(L%16).
    c_desc = TileDesc(
        shape=(m_sub * _ATOM, n_sub * _ATOM),
        layout=WarpDistributionEncoding(
            replication_lengths=(),
            hierarchical_lengths=((4, 4, m_sub), (16, n_sub)),  # M=(L//16, r, mi); N=(L%16, nj)
            lane_to_rh_major=((1, 2),),   # lane bucket0 -> M level0 (L//16); bucket1 -> N level0 (L%16)
            lane_to_rh_minor=((0, 0),),
            register_to_rh_major=(1, 2, 1),   # mi -> M, nj -> N, r -> M
            register_to_rh_minor=(2, 1, 1),   # mi -> M level2, nj -> N level1, r -> M level1
        ),
    )
    return a_desc, b_desc, c_desc


def _wave_descs(m_sub: int = 1, n_sub: int = 1, k_sub: int = 1):
    """Our OWN wave-tile RCC layouts, authored via `make_tile_desc`.

    Reproduce the canonical MFMA operand distributions with the atom grid folded in via
    `block_repeat` (= the atom counts). The result is SUBTILE-CONTIGUOUS -- atoms are contiguous
    register blocks in M-major (A / C) / N-major (B) / K-minor order -- which is exactly how the
    `TileMma` driver slices each atom by register offset. A and B are structurally identical
    (free-dim x K, K the 4-wide vector, K the major lane the atom wants); C is the (M,N) accumulator.
    """
    a_desc = make_tile_desc(
        shape=[m_sub * _ATOM, k_sub * _ATOM],
        thread_tile=[1, 4],
        thread_dist=[16, 4],
        thread_order=[1, 0],
        thread_broadcast=1,
        block_repeat=[m_sub, k_sub],
        wave_dist=[1, 1],
        wave_order=None,
        wave_broadcast=1,
        wave_size=64,
    )
    b_desc = make_tile_desc(   # B as (N, K): same structure as A -> same MFMA operand fragment
        shape=[n_sub * _ATOM, k_sub * _ATOM],
        thread_tile=[1, 4],
        thread_dist=[16, 4],
        thread_order=[1, 0],
        thread_broadcast=1,
        block_repeat=[n_sub, k_sub],
        wave_dist=[1, 1],
        wave_order=None,
        wave_broadcast=1,
        wave_size=64,
    )
    c_desc = make_tile_desc(
        shape=[m_sub * _ATOM, n_sub * _ATOM],
        thread_tile=[4, 1],
        thread_dist=[4, 16],
        thread_order=[0, 1],
        thread_broadcast=1,
        block_repeat=[m_sub, n_sub],
        wave_dist=[1, 1],
        wave_order=None,
        wave_broadcast=1,
        wave_size=64,
    )
    return a_desc, b_desc, c_desc


def _macro_coop_descs(tile_m: int, tile_n: int, tile_k: int, n_waves: int):
    """Block-wide COOPERATIVE load, returned as a (LOAD, STORE) pair per operand.

    LOAD: K-contiguous (KPT contiguous per lane -> coalesced global load, K-stride-1 row-major A/B),
    with the free-dim also owning DPT>=2 elements per lane. STORE: the SAME lane ownership re-ordered
    FREE-DIM-contiguous (M/N inner) -- a pure register `reorder` (transpose = axis-significance swap,
    the free "col-major == transpose of row-major"). The kernel loads with LOAD (fast), re-orders to
    STORE, and writes free-dim-contiguous LDS so the interleaved wave read is stride-1 (conflict-free).
    One interleaved layout end to end; `wave_dist` splits the free dim across the block's waves.
    """
    coop_m, coop_n = tile_m // n_waves, tile_n // n_waves

    def _split(free: int) -> tuple[int, int, int, int]:
        # per-lane elements over the wave's band; split into DPT (free, >=2) x KPT (K, wide-ish).
        per_lane = (free * tile_k) // 64
        kpt = min(4, tile_k)
        while kpt > 1 and (per_lane % kpt or per_lane // kpt < 2 or tile_k % kpt):
            kpt //= 2
        dpt = per_lane // kpt
        m_lanes = free // dpt
        k_lanes = tile_k // kpt
        return m_lanes, k_lanes, dpt, kpt

    def _pair(free: int, tile_free: int):
        m_lanes, k_lanes, dpt, kpt = _split(free)
        load = make_tile_desc(
            shape=[tile_free, tile_k],
            thread_tile=[dpt, kpt],              # K = kpt CONTIGUOUS per lane (coalesced); DPT>=2 free
            thread_dist=[m_lanes, k_lanes],
            thread_order=[0, 1],
            thread_broadcast=1,
            block_repeat=[1, 1],
            wave_dist=[n_waves, 1],
            wave_order=None,
            wave_broadcast=1,
            wave_size=64,
        )
        # STORE = LOAD with the two register axes swapped in significance -> free-dim-contiguous, same
        # elements/lanes (a `reorder`). This is the free transpose (col-major == row-major^T).
        el = load.layout
        store = TileDesc(
            shape=load.shape,
            layout=WarpDistributionEncoding(
                replication_lengths=el.replication_lengths,
                hierarchical_lengths=el.hierarchical_lengths,
                lane_to_rh_major=el.lane_to_rh_major,
                lane_to_rh_minor=el.lane_to_rh_minor,
                register_to_rh_major=el.register_to_rh_major[::-1],
                register_to_rh_minor=el.register_to_rh_minor[::-1],
            ),
        )
        return load, store

    a_load, a_store = _pair(coop_m, tile_m)
    b_load, b_store = _pair(coop_n, tile_n)
    return a_load, a_store, b_load, b_store


def _transpose_desc(td):
    """Swap the two X-dims (M<->K) of a desc -- a free relabel (col-major == row-major^T), register-
    identity (same values, coords swapped). Used to index the M-innermost LDS memref in (K,M) order so
    the free-dim (M) store/read is the memref innermost -> WIDE ds_write/ds_read (never scalar)."""
    e = td.layout
    swap = lambda majs: tuple(tuple((3 - m) if m in (1, 2) else m for m in row) for row in majs)
    return TileDesc(
        shape=(td.shape[1], td.shape[0]),
        layout=WarpDistributionEncoding(
            replication_lengths=e.replication_lengths,
            hierarchical_lengths=(e.hierarchical_lengths[1], e.hierarchical_lengths[0]),
            lane_to_rh_major=swap(e.lane_to_rh_major),
            lane_to_rh_minor=e.lane_to_rh_minor,
            register_to_rh_major=tuple((3 - m) if m in (1, 2) else m for m in e.register_to_rh_major),
            register_to_rh_minor=e.register_to_rh_minor,
        ),
    )


def build_interleaved_gemm(
    M_LEN: int, N_LEN: int, K_LEN: int, *, arch: str = "gfx90a",
    tile_m: int = 64, tile_n: int = 64, tile_k: int = 16,
    waves_m: int = 1, waves_n: int = 1,
    lda: int | None = None, ldb: int | None = None, ldc: int | None = None,
    prefetch: str = "interleaved", mma_layout: str = "interleaved", single_buffer: bool = False,
    reg_prefetch: bool = False, mac_prio: int = 0, lds_pad: int = 0, lds_swizzle=False,
):
    """RCC f16->f32(->f16) GEMM, authored entirely from OUR `make_tile_desc` layouts; `TileMma` drives
    the atom grid (validating only that A/B share the K-distribution).

    Cooperative multi-wave: a ``waves_m x waves_n`` grid of waves per CTA together load the
    ``tile_m x tile_n`` MACRO tile into LDS via the INTERLEAVED (free-dim-contiguous) layout -- each
    wave loads ``tile_m / n_waves`` rows -- then each wave reads its own WARP tile
    (``tile_m/waves_m x tile_n/waves_n``) from LDS in the CANONICAL MMA form and drives it. The macro
    tile is loaded ONCE per K-step and reused across the wave grid (the LDS win). ``waves_m=waves_n=1``
    is the single-wave case (warp tile == macro tile).

    interleaved->canonical is a cross-lane relayout (make_tile_desc's register order is fixed), so LDS
    is the bridge: interleaved global load -> LDS -> canonical read -> driver. Returns ``(kernel, mma)``.
    """
    from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType

    if prefetch not in ("interleaved", "canonical") or mma_layout not in ("interleaved", "canonical"):
        raise ValueError(f"prefetch/mma must be 'interleaved' or 'canonical' -- {prefetch=}, {mma_layout=}")
    n_waves = waves_m * waves_n
    if tile_m % waves_m or tile_n % waves_n:
        raise ValueError(f"macro tile must divide by waves -- {tile_m}x{tile_n} / {waves_m}x{waves_n}")
    warp_m, warp_n = tile_m // waves_m, tile_n // waves_n
    if warp_m % _ATOM or warp_n % _ATOM or tile_k % _ATOM:
        raise ValueError(f"warp tile must be multiples of {_ATOM} -- {warp_m}x{warp_n}x{tile_k}")
    if tile_m % n_waves or tile_n % n_waves:
        raise ValueError(f"macro tile must divide by n_waves={n_waves} for the cooperative load")
    coop_m, coop_n = tile_m // n_waves, tile_n // n_waves
    if coop_m % _ATOM or coop_n % _ATOM:
        raise ValueError(f"cooperative chunk must be multiples of {_ATOM} -- {coop_m}/{coop_n}")
    m_sub, n_sub, k_sub = warp_m // _ATOM, warp_n // _ATOM, tile_k // _ATOM

    # The driver is configured with the WARP tile -- it owns the per-wave m/n/k atom grid + the op.
    mma = TileMma(
        (warp_m, warp_n, tile_k), a="f16", b="f16", c="f32", target=arch,
        tiling=Tiling(atom_shape=(_ATOM, _ATOM, _ATOM)),
    )

    b = IRBuilder(
        f"tiling_gemm_interleaved_{M_LEN}x{N_LEN}x{K_LEN}_{tile_m}x{tile_n}x{tile_k}"
        f"_w{waves_m}x{waves_n}_{arch}"
    )
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size * n_waves

    a_ptr = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    b_ptr = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    c_ptr = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    b.param("M", I32)
    b.param("N", I32)
    b.param("K", I32)

    # Raw thread id is all the verbs need (they derive lane/wave from the encoding). wm/wn = this
    # wave's grid cell, used only for the per-wave LDS window offsets + the C store position.
    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(64))
    wm = b.div(wave, b.const_i32(waves_n))
    wn = b.mod(wave, b.const_i32(waves_n))

    m_macro = b.mul(b.block_id_y(), b.const_i32(tile_m))
    n_macro = b.mul(b.block_id_x(), b.const_i32(tile_n))

    lda = lda if lda is not None else K_LEN   # A (M,K) row-major
    ldb = ldb if ldb is not None else K_LEN   # B (N,K): K contiguous

    a_td = make_tensor_desc((M_LEN, K_LEN), (lda, 1), F16)
    b_td = make_tensor_desc((N_LEN, K_LEN), (ldb, 1), F16)
    # C memory layout follows the MMA knob (the accumulator the MMA produces): interleaved MMA -> native-C
    # -> C-shuffle -> RCR (M,N) N-contiguous (wide, universal-GEMM layout); canonical MMA -> RCC (N,M).
    if mma_layout == "interleaved":
        ldc = ldc if ldc is not None else N_LEN   # C (M,N): N contiguous
        c_td = make_tensor_desc((M_LEN, N_LEN), (ldc, 1), F16)
    else:
        ldc = ldc if ldc is not None else M_LEN   # C (N,M): M contiguous
        c_td = make_tensor_desc((N_LEN, M_LEN), (ldc, 1), F16)

    # TWO orthogonal layout knobs for apples-to-apples sweeps:
    #   prefetch -> coop-load staging into LDS + the layout the wave READ produces (a_pf/b_pf);
    #   mma      -> the operand layout TileMma consumes (a_desc/b_desc) + the accumulator/C-desc.
    # `_read_buf` bridges read->MMA with ONE transform_fragment (register `reorder`) iff prefetch != mma_layout.
    _pf_descs = _wave_descs_interleaved if prefetch == "interleaved" else _wave_descs
    _mma_descs = _wave_descs_interleaved if mma_layout == "interleaved" else _wave_descs
    a_pf, b_pf, _ = _pf_descs(m_sub, n_sub, k_sub)
    a_desc, b_desc, c_desc = _mma_descs(m_sub, n_sub, k_sub)
    a_coop_ld, a_coop_st, b_coop_ld, b_coop_st = _macro_coop_descs(tile_m, tile_n, tile_k, n_waves)

    # Double-buffered LDS as ONE 2x slab; the prefetched NEXT tile goes to the OTHER buffer (disjoint ->
    # the store never waits on the read's lgkmcnt, one barrier/trip). single_buffer halves it so 2 WGs
    # fit per CU (dual occupancy). The PHYSICAL LDS layout is the prefetch knob:
    #   interleaved -> M-innermost ([K, 2*free], M stride-1): free-dim store+read are the memref innermost
    #                  -> WIDE; store = K-contiguous load -> free-dim `reorder` -> re-tag (K,M).
    #   canonical   -> K-innermost ([2*free, K], K stride-1): the K-contiguous coop load stores DIRECTLY
    #                  (identity, no reorder) and the canonical wave read is K-contiguous -> WIDE, no transpose.
    _bufs = 1 if single_buffer else 2
    zero = b.const_i32(0)
    warp_m_c, warp_n_c = b.const_i32(warp_m), b.const_i32(warp_n)
    tile_m_c, tile_n_c = b.const_i32(tile_m), b.const_i32(tile_n)

    if prefetch == "interleaved":
        # lds_pad pads the M-INNERMOST width: the K-row stride becomes (_bufs*tile + lds_pad), so if that
        # is NOT a multiple of 32 dwords (64 f16) the K rows stop aliasing the same banks. The free-dim
        # (M) store/read still address the real M range contiguously (pad is trailing per K-row), so the
        # WIDE access is preserved (keep lds_pad a multiple of 8 f16 = 16 B to hold b128 alignment).
        _pa, _pb = _bufs * tile_m + lds_pad, _bufs * tile_n + lds_pad
        lds_a = b.smem_alloc(F16, [tile_k, _pa], name_hint="lds_a")
        lds_b = b.smem_alloc(F16, [tile_k, _pb], name_hint="lds_b")
        lds_a_td = make_tensor_desc((tile_k, _bufs * tile_m), (_pa, 1), F16)
        lds_b_td = make_tensor_desc((tile_k, _bufs * tile_n), (_pb, 1), F16)
        a_st_t, b_st_t = _transpose_desc(a_coop_st), _transpose_desc(b_coop_st)
        a_rd_t, b_rd_t = _transpose_desc(a_pf), _transpose_desc(b_pf)

        def _store_buf(frags, row_a, row_b):
            fa = make_fragment(a_st_t, F16, transform_fragment(b, frags[0], a_coop_st).value)
            fb = make_fragment(b_st_t, F16, transform_fragment(b, frags[1], b_coop_st).value)
            store_fragment(b, lds_a, make_window(lds_a_td, (zero, row_a)), fa, tid, lds_swizzle=lds_swizzle)
            store_fragment(b, lds_b, make_window(lds_b_td, (zero, row_b)), fb, tid, lds_swizzle=lds_swizzle)

        def _read_lds(row_a, row_b):
            fa = load_fragment(b, lds_a, make_window(lds_a_td, (zero, b.add(row_a, b.mul(wm, warp_m_c)))), a_rd_t, tid, lds_swizzle=lds_swizzle)
            fb = load_fragment(b, lds_b, make_window(lds_b_td, (zero, b.add(row_b, b.mul(wn, warp_n_c)))), b_rd_t, tid, lds_swizzle=lds_swizzle)
            return make_fragment(a_pf, F16, fa.value), make_fragment(b_pf, F16, fb.value)
    else:
        # lds_pad pads the K (innermost) dim: physical row stride = tile_k+lds_pad, so consecutive M rows
        # shift bank mapping (the causation-test knob for canonical bank conflicts). The td LOGICAL shape
        # stays (rows, tile_k); the pad is trailing dead space per row. NOTE: an odd-dword pad misaligns
        # the wide read -- interpret the ds_read width alongside the conflict counter.
        _pk = tile_k + lds_pad
        lds_a = b.smem_alloc(F16, [_bufs * tile_m, _pk], name_hint="lds_a")
        lds_b = b.smem_alloc(F16, [_bufs * tile_n, _pk], name_hint="lds_b")
        lds_a_td = make_tensor_desc((_bufs * tile_m, tile_k), (_pk, 1), F16)
        lds_b_td = make_tensor_desc((_bufs * tile_n, tile_k), (_pk, 1), F16)

        def _store_buf(frags, row_a, row_b):
            store_fragment(b, lds_a, make_window(lds_a_td, (row_a, zero)), frags[0], tid)
            store_fragment(b, lds_b, make_window(lds_b_td, (row_b, zero)), frags[1], tid)

        def _read_lds(row_a, row_b):
            fa = load_fragment(b, lds_a, make_window(lds_a_td, (b.add(row_a, b.mul(wm, warp_m_c)), zero)), a_pf, tid)
            fb = load_fragment(b, lds_b, make_window(lds_b_td, (b.add(row_b, b.mul(wn, warp_n_c)), zero)), b_pf, tid)
            return make_fragment(a_pf, F16, fa.value), make_fragment(b_pf, F16, fb.value)

    _need_xf = prefetch != mma_layout
    if _need_xf and (m_sub > 1 or n_sub > 1 or k_sub > 1):
        # The interleaved<->canonical delta is an intra-lane `reorder` only for a SINGLE 16x16 atom;
        # across a multi-atom wave tile the two full-wave encodings differ in LANE ownership -> cross_lane
        # (deferred). Matched layouts (both interleaved / both canonical) emit no transform and work.
        raise NotImplementedError(
            "crossed prefetch/mma_layout needs a cross-lane transform for multi-atom wave tiles -- "
            f"prefetch={prefetch!r}, mma_layout={mma_layout!r}, subtiles=({m_sub},{n_sub},{k_sub}); "
            "use matched layouts (both 'interleaved' or both 'canonical') -- cross_lane is deferred"
        )

    def _prefetch_load(kb):
        """Whole thread-block cooperatively loads the macro slab at K=kb -> registers (K-contiguous,
        coalesced) via the LOAD descs."""
        return (
            load_fragment(b, a_ptr, make_window(a_td, (m_macro, kb)), a_coop_ld, tid),
            load_fragment(b, b_ptr, make_window(b_td, (n_macro, kb)), b_coop_ld, tid),
        )

    def _read_buf(row_a, row_b):
        """Read the wave's warp tile from LDS (in the PREFETCH layout, wide), then bridge to the MMA
        operand layout with one `transform_fragment` (`reorder`) iff prefetch != mma_layout."""
        fa, fb = _read_lds(row_a, row_b)
        if _need_xf:
            fa = transform_fragment(b, fa, a_desc)
            fb = transform_fragment(b, fb, b_desc)
        return fa, fb

    # Software pipeline (prefetch consumed WITHIN the trip -- no load-value carried across the edge):
    #   prologue: tile 0 -> buffer 0
    #   trip ki (0..N-2): prefetch tile ki+1 (top) -> read tile ki from LDS -> MFMA -> store tile ki+1
    #                     into the OTHER buffer (bottom) -> barrier -> swap.
    #   after the loop: the last prefetched tile (N-1) is in LDS but not yet MFMA'd -- process it.
    # The load (top) and its store (bottom) sit in the SAME trip with the MFMA between, so the vmcnt
    # wait lands AFTER the MFMA (load's shadow); nothing is an iter_arg but `acc`, so the back-edge has
    # no load residency to drain. Double LDS buffer -> store targets the OTHER buffer than the read
    # (disjoint LDS, no WAR) -> one barrier/trip. TileMma drives the inner atom grid.
    n_tiles = K_LEN // tile_k
    tk_c = b.const_i32(tile_k)

    # sched_group_barrier cadence counts (per trip, per wave; VEC=8 f16 per wide mem op).
    _VEC = 8
    n_mfma = m_sub * n_sub * k_sub
    n_dsread = (warp_m * tile_k + warp_n * tile_k) // mma.wave_size // _VEC
    n_vmem = (tile_m * tile_k + tile_n * tile_k) // (mma.wave_size * n_waves) // _VEC
    n_dswrite = n_vmem

    def _pin():
        b.sched_group_barrier(_SGB_DS_READ, n_dsread, 0)
        b.sched_group_barrier(_SGB_VMEM_READ, n_vmem, 0)
        b.sched_group_barrier(_SGB_MFMA, n_mfma, 0)
        b.sched_group_barrier(_SGB_DS_WRITE, n_dswrite, 0)

    def _mma_prio(a_fr, b_fr, acc_v):
        """Wave-balancing (CU load balance): raise wave priority for the MAC block, drop it for the
        memory phases -- so on a SIMD with 2 co-resident waves the MFMA-phase wave wins the issue
        port while the other does its loads (the Aligned "prioritize MAC" stagger). Only meaningful at
        MULTIPLE occupancy (>=2 waves/SIMD); `mac_prio=0` disables it."""
        if mac_prio:
            b.s_setprio(mac_prio)
        out = mma(b, a_fr, b_fr, make_fragment(c_desc, F32, acc_v))
        if mac_prio:
            b.s_setprio(0)
        return out

    accumulator = make_fragment(c_desc, F32)
    fill_fragment(b, accumulator, 0)
    last_k = b.const_i32(K_LEN - tile_k)

    if reg_prefetch and not single_buffer:
        # 2-deep LDS + REGISTER double-buffer: tile ki+1's A/B are read one trip AHEAD (into trip ki's
        # MFMA shadow) and carried as loop-register iter_args, so the ds_read completion latency is off
        # the critical path (the exposed ~36 ns top-of-trip read the plain double-buffer pays). Two
        # tiles stay resident in LDS (buf0/buf1); the global prefetch runs TWO tiles ahead. Costs one
        # extra A/B register set -- affordable only with the macro128/2x2 headroom. Needs >= 2 K-tiles.
        _store_buf(_prefetch_load(zero), zero, zero)          # tile 0 -> buf 0
        _store_buf(_prefetch_load(tk_c), tile_m_c, tile_n_c)  # tile 1 -> buf 1
        b.sync_lds_only()
        a0, b0 = _read_buf(zero, zero)                         # tile 0 registers -> reg_cur
        outer = b.scf_for_iter(
            b.const_i32(0), b.const_i32(K_LEN - tile_k), tk_c,
            [("acc", accumulator.value), ("acur", a0.value), ("bcur", b0.value)], iv_name="k",
        )
        with outer as (kiv, (acc_val, acur, bcur)):
            ki = b.div(kiv, tk_c)
            cur = b.mod(ki, b.const_i32(2))                   # buf parity of tile ki (== ki+2)
            nxt = b.sub(b.const_i32(1), cur)                  # buf parity of tile ki+1
            cur_ra, cur_rb = b.mul(cur, tile_m_c), b.mul(cur, tile_n_c)
            nxt_ra, nxt_rb = b.mul(nxt, tile_m_c), b.mul(nxt, tile_n_c)
            a_nx, b_nx = _read_buf(nxt_ra, nxt_rb)            # read tile ki+1 (in LDS) -> reg_next
            # MFMA on reg_cur (tile ki) -- forces the lgkmcnt wait on reg_cur's read, which drains the
            # tile-ki ds_read from buf[cur] BEFORE the store below overwrites it (WAR-safe).
            acc = _mma_prio(make_fragment(a_desc, F16, acur), make_fragment(b_desc, F16, bcur), acc_val)
            pf = _prefetch_load(b.smin(b.add(kiv, b.mul(b.const_i32(2), tk_c)), last_k))  # tile ki+2
            _store_buf(pf, cur_ra, cur_rb)                    # store ki+2 -> buf[ki%2] (ki vacated)
            _pin()
            b.sync_lds_only()                                 # ki+2 visible (read 2 trips later)
            b.scf_yield(acc.value, a_nx.value, b_nx.value)
        acc_val, a_last, b_last = outer.results
        accumulator = mma(b, make_fragment(a_desc, F16, a_last), make_fragment(b_desc, F16, b_last),
                          make_fragment(c_desc, F32, acc_val))  # last tile (N-1), carried in registers
    elif single_buffer:
        # ONE buffer -> 2 workgroups/CU (dual occupancy). Per trip: read tile ki, prefetch ki+1 (in
        # flight), MFMA, WAR-barrier (reads done), store ki+1 into the SAME buffer, barrier (visible).
        # Two barriers/trip + no prefetch overlap, but 2x resident waves to hide the latency.
        _store_buf(_prefetch_load(zero), zero, zero)          # prologue: tile 0 -> buffer 0
        b.sync_lds_only()
        outer = b.scf_for_iter(
            b.const_i32(0), b.const_i32(K_LEN), tk_c, [("acc", accumulator.value)], iv_name="k",
        )
        with outer as (kiv, (acc_val,)):
            a_frag, b_frag = _read_buf(zero, zero)           # tile ki (already in the buffer)
            pf = _prefetch_load(b.smin(b.add(kiv, tk_c), last_k))  # prefetch ki+1 (clamped), overlaps MFMA
            acc = _mma_prio(a_frag, b_frag, acc_val)
            b.sync_lds_only()                                # WAR: all reads of tile ki done
            _store_buf(pf, zero, zero)                        # overwrite buffer with tile ki+1
            _pin()
            b.sync_lds_only()                                # visible for next trip's read
            b.scf_yield(acc.value)
        accumulator = make_fragment(c_desc, F32, outer.results[0])
    else:
        _store_buf(_prefetch_load(zero), zero, zero)          # prologue: tile 0 -> buffer 0
        b.sync_lds_only()
        outer = b.scf_for_iter(
            b.const_i32(0), b.const_i32(K_LEN - tile_k), tk_c,
            [("acc", accumulator.value)], iv_name="k",
        )
        with outer as (kiv, (acc_val,)):
            ki = b.div(kiv, tk_c)
            cur = b.mod(ki, b.const_i32(2))                   # 0/1: buffer holding THIS tile
            oth = b.sub(b.const_i32(1), cur)                  # the other buffer (prefetch store target)
            cur_ra, cur_rb = b.mul(cur, tile_m_c), b.mul(cur, tile_n_c)
            oth_ra, oth_rb = b.mul(oth, tile_m_c), b.mul(oth, tile_n_c)
            pf = _prefetch_load(b.add(kiv, tk_c))            # prefetch tile ki+1 (always in range)
            a_frag, b_frag = _read_buf(cur_ra, cur_rb)       # read tile ki
            acc = _mma_prio(a_frag, b_frag, acc_val)
            _store_buf(pf, oth_ra, oth_rb)                    # store tile ki+1 -> OTHER buffer
            _pin()                                           # loads in flight, MFMA block, then store
            b.sync_lds_only()                                # cross-trip visibility of the other buffer
            b.scf_yield(acc.value)
        acc_val = outer.results[0]
        last_cur = (n_tiles - 1) % 2                          # process the last prefetched tile (N-1)
        last_ra, last_rb = b.const_i32(last_cur * tile_m), b.const_i32(last_cur * tile_n)
        a_frag, b_frag = _read_buf(last_ra, last_rb)
        accumulator = mma(b, a_frag, b_frag, make_fragment(c_desc, F32, acc_val))

    if mma_layout == "interleaved":
        # C-shuffle (Step4): reorder the native accumulator to the final-write layout (make_tile_desc,
        # store-friendly: lane owns a contiguous 4*m_sub x n_sub (M x N) block, N inner). native->store
        # is a pure `reorder` (same lane ownership), so this is a compile-time register shuffle; the
        # subsequent store is coalesced/wide in the (M,N) N-contiguous direction.
        c_store = make_tile_desc(
            shape=[m_sub * _ATOM, n_sub * _ATOM],
            thread_tile=[4 * m_sub, n_sub],
            thread_dist=[4, 16],
            thread_order=[0, 1],
            thread_broadcast=1,
            block_repeat=[1, 1],
            wave_dist=[1, 1],
            wave_order=None,
            wave_broadcast=1,
            wave_size=64,
        )
        accumulator = transform_fragment(b, accumulator, c_store)

    c_view = c_td if mma_layout == "interleaved" else c_td.permute([1, 0])   # both -> (M, N)
    c_win = make_window(
        c_view,
        (b.add(m_macro, b.mul(wm, b.const_i32(warp_m))),
         b.add(n_macro, b.mul(wn, b.const_i32(warp_n)))),
    )
    store_fragment(b, c_ptr, c_win, accumulator, tid)
    b.ret()
    return b.kernel, mma


def _compile_launcher(kernel, arch: str):
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.launcher import KernelLauncher

    artifact = compile_kernel(kernel, arch=arch)
    signature = (
        SignatureBuilder()
        .ptr("A", "f16").ptr("B", "f16").ptr("C", "f16")
        .scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
        .build()
    )
    return KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=signature
    )


def run_and_verify_interleaved(
    M_LEN: int = 256, N_LEN: int = 256, K_LEN: int = 256, *,
    arch: str = "gfx90a", tile_m: int = 64, tile_n: int = 64, tile_k: int = 16,
    waves_m: int = 1, waves_n: int = 1, prefetch: str = "interleaved", mma_layout: str = "interleaved", single_buffer: bool = False,
    reg_prefetch: bool = False, mac_prio: int = 0, lds_pad: int = 0, lds_swizzle=False,
) -> dict:
    """Compile, launch, and verify `build_interleaved_gemm` is bit-exact vs a numpy golden
    (integer inputs). torch-free: numpy host arrays + `DeviceMem`."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem, LaunchConfig, synchronize_and_release

    kernel, mma = build_interleaved_gemm(
        M_LEN, N_LEN, K_LEN, arch=arch, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        waves_m=waves_m, waves_n=waves_n, prefetch=prefetch, mma_layout=mma_layout, single_buffer=single_buffer,
        reg_prefetch=reg_prefetch, mac_prio=mac_prio, lds_pad=lds_pad, lds_swizzle=lds_swizzle,
    )
    launcher = _compile_launcher(kernel, arch)
    block = (mma.wave_size * waves_m * waves_n, 1, 1)

    rng = np.random.default_rng(0)
    a_buf = rng.integers(-3, 4, size=(M_LEN, K_LEN)).astype(np.float16)
    b_buf = rng.integers(-3, 4, size=(N_LEN, K_LEN)).astype(np.float16)
    # interleaved path stores C as (M,N) N-contiguous (RCR); canonical as (N,M) M-contiguous (RCC).
    c_buf = np.zeros((M_LEN, N_LEN) if mma_layout == "interleaved" else (N_LEN, M_LEN), dtype=np.float16)
    grid = (-(-N_LEN // tile_n), -(-M_LEN // tile_m), 1)

    rt = Runtime()
    a_dev, b_dev, c_dev = (DeviceMem(x.nbytes) for x in (a_buf, b_buf, c_buf))
    rt.memcpy_h2d(a_dev.ptr(), as_u8_buffer(a_buf), a_buf.nbytes)
    rt.memcpy_h2d(b_dev.ptr(), as_u8_buffer(b_buf), b_buf.nbytes)
    rt.memcpy_h2d(c_dev.ptr(), as_u8_buffer(c_buf), c_buf.nbytes)
    launcher(
        {"A": a_dev, "B": b_dev, "C": c_dev, "M": M_LEN, "N": N_LEN, "K": K_LEN},
        config=LaunchConfig(grid=grid, block=block),
    )
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(c_buf), c_dev.ptr(), c_buf.nbytes)
    reference = a_buf.astype(np.float32) @ b_buf.astype(np.float32).T   # (M,N)
    # interleaved buffer is already (M,N); canonical is (N,M) -> transpose to (M,N)
    result = c_buf.astype(np.float32) if mma_layout == "interleaved" else c_buf.astype(np.float32).T
    max_abs_diff = float(np.abs(result - reference).max())
    return {
        "shape": (M_LEN, N_LEN, K_LEN),
        "tile": (tile_m, tile_n, tile_k),
        "waves": (waves_m, waves_n),
        "op_id": mma.op_id,
        "max_abs_diff": max_abs_diff,
        "bit_exact": max_abs_diff == 0.0,
    }


def benchmark_interleaved(
    M_LEN: int = 2048, N_LEN: int = 2048, K_LEN: int = 2048, *,
    arch: str = "gfx90a", tile_m: int = 64, tile_n: int = 64, tile_k: int = 16,
    waves_m: int = 1, waves_n: int = 1, prefetch: str = "interleaved", mma_layout: str = "interleaved", single_buffer: bool = False,
    reg_prefetch: bool = False, mac_prio: int = 0, lds_pad: int = 0, lds_swizzle=False, cold_niters: int = 5, nrepeat: int = 20,
) -> dict:
    """Time `build_interleaved_gemm` (hipEvent-timed, CK-style) and report TFLOPS.

    TFLOPS = 2*M*N*K / time, matching perf_hgemm. Inputs are zero-filled (timing only)."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem, StreamConfig, launch_kernel, make_kernel, wait_stream_and_release,
    )

    kernel, mma = build_interleaved_gemm(
        M_LEN, N_LEN, K_LEN, arch=arch, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        waves_m=waves_m, waves_n=waves_n, prefetch=prefetch, mma_layout=mma_layout, single_buffer=single_buffer,
        reg_prefetch=reg_prefetch, mac_prio=mac_prio, lds_pad=lds_pad, lds_swizzle=lds_swizzle,
    )
    launcher = _compile_launcher(kernel, arch)

    a_buf = np.zeros((M_LEN, K_LEN), dtype=np.float16)
    b_buf = np.zeros((N_LEN, K_LEN), dtype=np.float16)
    c_buf = np.zeros((N_LEN, M_LEN), dtype=np.float16)
    grid = (-(-N_LEN // tile_n), -(-M_LEN // tile_m), 1)

    rt = Runtime()
    a_dev, b_dev, c_dev = (DeviceMem(x.nbytes) for x in (a_buf, b_buf, c_buf))
    for dev, host in ((a_dev, a_buf), (b_dev, b_buf), (c_dev, c_buf)):
        rt.memcpy_h2d(dev.ptr(), as_u8_buffer(host), host.nbytes)

    closure = make_kernel(
        launcher,
        {"A": a_dev, "B": b_dev, "C": c_dev, "M": M_LEN, "N": N_LEN, "K": K_LEN},
        grid, (mma.wave_size * waves_m * waves_n, 1, 1),
    )
    ms = launch_kernel(
        StreamConfig(time_kernel=True, cold_niters=cold_niters, nrepeat=nrepeat, is_gpu_timer=True),
        closure,
    )
    wait_stream_and_release()
    tflops = (2.0 * M_LEN * N_LEN * K_LEN) / (ms * 1e-3) / 1e12
    return {
        "shape": (M_LEN, N_LEN, K_LEN),
        "tile": (tile_m, tile_n, tile_k),
        "ms": ms,
        "tflops": tflops,
    }


def build_lds_staged_gemm(
    M_LEN: int, N_LEN: int, K_LEN: int, *, arch: str = "gfx90a",
    lda: int | None = None, ldb: int | None = None, ldc: int | None = None,
):
    """P2 validation: single wave, single 16x16 atom, but A/B are staged global -> LDS -> register
    before the MMA. Same layout on store + load (identity round-trip) -- proves the LDS
    store/load helpers + `make_tile_desc` LDS layout are bit-exact before multi-wave reuse."""
    from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType

    mma = TileMma(target=arch, atom_override="mfma_f32_16x16x16f16")
    op = mma.emit_op()

    b = IRBuilder(f"tiling_gemm_lds_{M_LEN}x{N_LEN}x{K_LEN}_{arch}")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size

    a_ptr = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    b_ptr = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    c_ptr = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    b.param("M", I32)
    b.param("N", I32)
    b.param("K", I32)

    lane = b.thread_id_x()
    m_base = b.mul(b.block_id_y(), b.const_i32(_ATOM))
    n_base = b.mul(b.block_id_x(), b.const_i32(_ATOM))

    lda = lda if lda is not None else K_LEN
    ldb = ldb if ldb is not None else K_LEN
    ldc = ldc if ldc is not None else M_LEN

    a_td = make_tensor_desc((M_LEN, K_LEN), (lda, 1), F16)
    b_td = make_tensor_desc((N_LEN, K_LEN), (ldb, 1), F16)
    c_td = make_tensor_desc((N_LEN, M_LEN), (ldc, 1), F16)
    a_desc, b_desc, c_desc = _wave_descs()   # single-atom (1,1,1)

    # LDS buffers for the A (M,K) and B (N,K) tiles, addressed as ordinary tiles: a tensor_desc for
    # the LDS shape + a window at origin, then the SAME load/store_fragment verbs (they dispatch to
    # smem ops because the source is an `smem<...>` buffer, not a global ptr).
    lds_a = b.smem_alloc(F16, [_ATOM, _ATOM], name_hint="lds_a")
    lds_b = b.smem_alloc(F16, [_ATOM, _ATOM], name_hint="lds_b")
    lds_a_win = make_window(make_tensor_desc((_ATOM, _ATOM), (_ATOM, 1), F16), (0, 0))
    lds_b_win = make_window(make_tensor_desc((_ATOM, _ATOM), (_ATOM, 1), F16), (0, 0))

    accumulator = make_fragment(c_desc, F32)
    fill_fragment(b, accumulator, 0)
    for tile_k_base in range(0, K_LEN, _ATOM):
        k_base = b.const_i32(tile_k_base)
        a_g = load_fragment(b, a_ptr, make_window(a_td, (m_base, k_base)), a_desc, lane)
        b_g = load_fragment(b, b_ptr, make_window(b_td, (n_base, k_base)), b_desc, lane)
        store_fragment(b, lds_a, lds_a_win, a_g, lane)   # -> smem_store (dispatched by source)
        store_fragment(b, lds_b, lds_b_win, b_g, lane)
        b.sync()
        a_frag = load_fragment(b, lds_a, lds_a_win, a_desc, lane)   # -> smem_load
        b_frag = load_fragment(b, lds_b, lds_b_win, b_desc, lane)
        accumulator.value = b.mma(op, a_frag.value, b_frag.value, accumulator.value)
        b.sync()

    c_win = make_window(c_td.permute([1, 0]), (m_base, n_base))
    store_fragment(b, c_ptr, c_win, accumulator, lane)
    b.ret()
    return b.kernel, mma


def run_and_verify_lds(
    M_LEN: int = 256, N_LEN: int = 256, K_LEN: int = 256, *, arch: str = "gfx90a"
) -> dict:
    """Bit-exact check for the LDS-staged (single-wave, single-atom) GEMM."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem, LaunchConfig, synchronize_and_release

    kernel, mma = build_lds_staged_gemm(M_LEN, N_LEN, K_LEN, arch=arch)
    launcher = _compile_launcher(kernel, arch)
    rng = np.random.default_rng(0)
    a_buf = rng.integers(-3, 4, size=(M_LEN, K_LEN)).astype(np.float16)
    b_buf = rng.integers(-3, 4, size=(N_LEN, K_LEN)).astype(np.float16)
    c_buf = np.zeros((N_LEN, M_LEN), dtype=np.float16)
    grid = (-(-N_LEN // _ATOM), -(-M_LEN // _ATOM), 1)
    rt = Runtime()
    a_dev, b_dev, c_dev = (DeviceMem(x.nbytes) for x in (a_buf, b_buf, c_buf))
    rt.memcpy_h2d(a_dev.ptr(), as_u8_buffer(a_buf), a_buf.nbytes)
    rt.memcpy_h2d(b_dev.ptr(), as_u8_buffer(b_buf), b_buf.nbytes)
    rt.memcpy_h2d(c_dev.ptr(), as_u8_buffer(c_buf), c_buf.nbytes)
    launcher(
        {"A": a_dev, "B": b_dev, "C": c_dev, "M": M_LEN, "N": N_LEN, "K": K_LEN},
        config=LaunchConfig(grid=grid, block=(mma.wave_size, 1, 1)),
    )
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(c_buf), c_dev.ptr(), c_buf.nbytes)
    reference = a_buf.astype(np.float32) @ b_buf.astype(np.float32).T
    result = c_buf.astype(np.float32).T
    max_abs_diff = float(np.abs(result - reference).max())
    return {"shape": (M_LEN, N_LEN, K_LEN), "max_abs_diff": max_abs_diff,
            "bit_exact": max_abs_diff == 0.0}


if __name__ == "__main__":
    print(run_and_verify_interleaved())
    print(run_and_verify_lds())
    print(benchmark_interleaved())
