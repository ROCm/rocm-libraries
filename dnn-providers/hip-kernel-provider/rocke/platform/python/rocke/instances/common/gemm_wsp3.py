# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Warp-specialized producer/consumer 3-stage GEMM pipeline (``wsp3``).

A warp-specialized producer/consumer GEMM for gfx950 / MI355X (CDNA4), built to
push square fp16 GEMM past the ceiling that the standard
``mem``/``compv3``/``compv4`` pipelines (and CK-Tile's own comp_v4 example)
top out at. Full design + phased plan in
``rocke/_wsp3/BUILD_SPEC.md``.

Architecture:
  * 12 warps / 768 threads, ``launch_bounds(768, 2)``.
      - warps 0..3   = PRODUCERS: async global->LDS (``async_buffer_load_lds``).
      - warps 4..11  = CONSUMERS: LDS->reg (``ds_read``) + ``mfma_f32_16x16x32_f16``.
    Role is wave-uniform (``warp_id``-based), so the two complementary
    ``scf.if`` role-loops carry a MATCHED per-iteration ``s_barrier_bare``
    count and rendezvous correctly (no named/split barrier — those ICE
    gfx950; no scf.if-results needed since cross-role state lives only in LDS).
  * 3-stage LDS ring (As[3][2][2], Bs[3][4][2]), ~147 KB, XOR swizzle.
  * Per-iteration rendezvous: ``s_waitcnt(vmcnt=0); sched_barrier(0);
    s_barrier_bare()`` — drains producer async writes WITHOUT serializing the
    next iteration's in-flight loads (why ``sync()`` is wrong here).

"""

from __future__ import annotations

from ...core.ir import I32, IRBuilder, KernelDef, PtrType
from ...helpers.tensor_view import (
    TensorDescriptor,
    TensorView,
    make_global_view,
    make_tile_window,
)
from .gemm_universal import (
    UniversalGemmSpec,
    _atom_frag_lengths,
    _emit_epilogue_default,
    _emit_mma,
    _emit_smem_load,
    _emit_zero_acc_op,
    _resolve_mma_op,
    _storage_dtype,
)

# Producer warps (do the global->LDS load); the remaining warps are consumers
# (LDS->reg + MFMA). 2 measured best for compute-bound square (fewer wasted
# warps -> higher occupancy); override with CK_WSP3_PROD.
NUM_PRODUCER_WARPS = 2


def _pick_load_vec(elems_a: int, elems_b: int, threads: int) -> int:
    """Widest f16 vector width (<=8 halves) that evenly distributes both the
    A and B tile element counts across ``threads`` producer lanes."""
    for v in (8, 4, 2, 1):
        if elems_a % (v * threads) == 0 and elems_b % (v * threads) == 0:
            return v
    return 1


def build_wsp3_gemm(spec: UniversalGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Warp-specialized producer/consumer GEMM — Phase 1: depth-1 skeleton.

    Structure (correctness-first, fully serialized; perf comes in later phases):
      * ``NUM_PRODUCER_WARPS`` producer warps load one K-tile global->LDS.
      * The remaining ``warp_m * warp_n`` consumer warps read LDS and MFMA.
      * Single LDS buffer; two full ``sync()`` barriers per K-tile order the
        producer write (barrier 1) and the consumer read-complete (barrier 2),
        so the producer never overwrites LDS the consumers are still reading.

    Reuses the proven address math: the MFMA per-lane fragment loads
    (:func:`_emit_smem_load` / :func:`_emit_mma`) and the accumulator scatter
    (:func:`_emit_epilogue_default`) from ``gemm_universal``; only the
    coalesced load and the per-warp MFMA loop are re-emitted here under the
    warp-role split. The two roles live in complementary ``scf.if`` blocks
    (role is wave-uniform), each running its own K-loop with a MATCHED count
    of two barriers per iteration, so all warps rendezvous correctly.
    """
    if arch != "gfx950":
        raise ValueError(f"wsp3 pipeline is gfx950-only for now (got {arch!r})")
    if spec.data.dtype_a != "fp16" or spec.data.dtype_b != "fp16":
        raise ValueError("wsp3 Phase 1 supports fp16 inputs only")

    t = spec.tile
    op = _resolve_mma_op(spec, arch)
    if op is None:
        raise ValueError(f"no MMA atom for wsp3 spec on {arch}")
    a_per_lane, b_per_lane, c_per_lane = _atom_frag_lengths(op)
    storage_dtype = _storage_dtype(spec)

    block_m, block_n, block_k = t.tile_m, t.tile_n, t.tile_k
    warp_m, warp_n = t.warp_m, t.warp_n  # consumer warp grid
    wtm, wtn, wtk = t.warp_tile_m, t.warp_tile_n, t.warp_tile_k
    mfmas_m = (block_m // warp_m) // wtm
    mfmas_n = (block_n // warp_n) // wtn
    k_atoms = block_k // wtk

    import os as _os0

    n_prod_warps = int(_os0.environ.get("CK_WSP3_PROD", str(NUM_PRODUCER_WARPS)))
    n_cons_warps = warp_m * warp_n
    total_warps = n_prod_warps + n_cons_warps
    wave = spec.wave_size
    block_size = total_warps * wave
    prod_threads = n_prod_warps * wave

    a_total = block_m * block_k
    b_total = block_n * block_k
    load_vec = _pick_load_vec(a_total, b_total, prod_threads)
    a_vecs = a_total // load_vec // prod_threads
    b_vecs = b_total // load_vec // prod_threads
    kdv = block_k // load_vec  # vec-columns per LDS row

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = block_size
    if spec.trait.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu

    A = b.param(
        "A", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    Bp = b.param(
        "B", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    C = b.param(
        "C", PtrType(storage_dtype, "global"), noalias=True, writeonly=True, align=16
    )
    M = b.param("M", I32)
    N = b.param("N", I32)
    K = b.param("K", I32)

    c0 = b.const_i32(0)
    c_wave = b.const_i32(wave)
    c_block_k = b.const_i32(block_k)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)
    c_prod = b.const_i32(prod_threads)
    c_lv = b.const_i32(load_vec)
    c_kdv = b.const_i32(kdv)
    c_nprodw = b.const_i32(n_prod_warps)
    c_warpn = b.const_i32(warp_n)

    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    is_producer = b.cmp_lt(warp_id, c_nprodw)
    is_consumer = b.cmp_ge(warp_id, c_nprodw)
    cwarp = b.sub(
        warp_id, c_nprodw
    )  # 0..n_cons_warps-1 (garbage for producers, unused)
    cwarp_m = b.div(cwarp, c_warpn)
    cwarp_n = b.mod(cwarp, c_warpn)

    block_m_off = b.mul(b.block_id_y(), b.const_i32(block_m))
    block_n_off = b.mul(b.block_id_x(), b.const_i32(block_n))

    # Ring depth: 1 = serialized (Phase 1); >=2 = ping-pong overlap (Phase 2+).
    import os as _os

    depth = int(_os.environ.get("CK_WSP3_DEPTH", "2"))
    cD = b.const_i32(depth)

    # LDS ring: ``depth`` stacked buffers per operand in one packed alloc;
    # the per-buffer row origin is ``buf * block_m`` / ``buf * block_n``.
    A_smem = b.smem_alloc(storage_dtype, [depth * block_m, block_k], name_hint="A_smem")
    B_smem = b.smem_alloc(storage_dtype, [depth * block_n, block_k], name_hint="B_smem")
    a_lds_view = TensorView(
        base=A_smem,
        desc=TensorDescriptor.packed((depth * block_m, block_k), storage_dtype),
        addr_space="lds",
    )
    b_lds_view = TensorView(
        base=B_smem,
        desc=TensorDescriptor.packed((depth * block_n, block_k), storage_dtype),
        addr_space="lds",
    )
    a_view = make_global_view(
        A, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    b_view = make_global_view(
        Bp, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )

    def _rc(vec_idx):
        row = b.div(vec_idx, c_kdv)
        col = b.mul(b.mod(vec_idx, c_kdv), c_lv)
        return row, col

    def _producer_load(k_off, a_buf_row, b_buf_row):
        """4 producer warps coalesced-load A+B tile at k_off into LDS buffer
        whose row origin is ``a_buf_row`` (A) / ``b_buf_row`` (B)."""
        a_gt = make_tile_window(
            a_view, lengths=(1, block_m, block_k), origin=(c0, block_m_off, k_off)
        )
        b_gt = make_tile_window(
            b_view, lengths=(1, block_n, block_k), origin=(c0, block_n_off, k_off)
        )
        a_lt = make_tile_window(
            a_lds_view, lengths=(depth * block_m, block_k), origin=(c0, c0)
        )
        b_lt = make_tile_window(
            b_lds_view, lengths=(depth * block_n, block_k), origin=(c0, c0)
        )
        for e in range(a_vecs):
            vi = b.add(
                b.mul(b.const_i32(e), c_prod), tid
            )  # producer tid == tid (warps 0..3)
            row, col = _rc(vi)
            val = a_gt.load_vec(b, c0, row, col, n=load_vec)
            a_lt.store_vec(b, b.add(a_buf_row, row), col, value=val, n=load_vec)
        for e in range(b_vecs):
            vi = b.add(b.mul(b.const_i32(e), c_prod), tid)
            row, col = _rc(vi)
            val = b_gt.load_vec(b, c0, row, col, n=load_vec)
            b_lt.store_vec(b, b.add(b_buf_row, row), col, value=val, n=load_vec)

    # ---- async direct-to-LDS producer plumbing (Phase 3) ----
    # 4 producer warps stream the global tile straight into LDS via
    # ``async_buffer_load_lds_addr`` (no VGPR round-trip), so multiple K-tiles'
    # loads stay in flight across an ``s_barrier_bare`` (which does NOT drain
    # vmcnt). The flat per-lane chunk layout is byte-identical to the row-major
    # [block_m, block_k] the consumer ``_emit_smem_load`` reads.
    from ...core.ir import I64 as _I64

    DTL_DWORDS = 4
    DTL_HALVES = DTL_DWORDS * 2  # 8 f16/lane chunk
    DTL_BPL = DTL_DWORDS * 4  # 16 bytes/lane
    async_ok = (block_k % DTL_HALVES) == 0
    a_chunks = (block_m * block_k) // DTL_HALVES
    b_chunks = (block_n * block_k) // DTL_HALVES
    a_passes = (a_chunks + prod_threads - 1) // prod_threads
    b_passes = (b_chunks + prod_threads - 1) // prod_threads
    loads_per_tile = a_passes + b_passes
    a_buf_bytes = block_m * block_k * 2  # one A buffer's byte size
    b_buf_bytes = block_n * block_k * 2
    if async_ok:
        c_chunks_per_row = b.const_i32(block_k // DTL_HALVES)
        c_halves_per_chunk = b.const_i32(DTL_HALVES)
        c2 = b.const_i32(2)
        big = b.const_i32(0x7FFF0000)
        a_rsrc = b.buffer_rsrc(A, big)
        b_rsrc = b.buffer_rsrc(Bp, big)
        a_lds0 = b.smem_addr_of(A_smem)
        b_lds0 = b.smem_addr_of(B_smem)
        zsoff = b.const_i32(0)
        wave_bytes = wave * DTL_BPL
        warp_off = b.zext(
            b.mul(warp_id, b.const_i32(wave_bytes)), _I64
        )  # per-wave LDS slice

    def _producer_load_async(k_off, a_buf, b_buf):
        """Issue async global->LDS for one K-tile into ring buffer ``a_buf``
        (A) / ``b_buf`` (B), where the args are i32 buffer-byte offsets."""
        a_base = b.smem_ptr_add(b.smem_ptr_add(a_lds0, b.zext(a_buf, _I64)), warp_off)
        b_base = b.smem_ptr_add(b.smem_ptr_add(b_lds0, b.zext(b_buf, _I64)), warp_off)
        for p in range(a_passes):
            lds = (
                a_base
                if p == 0
                else b.smem_ptr_add(
                    a_base, b.zext(b.const_i32(p * prod_threads * DTL_BPL), _I64)
                )
            )
            cidx = b.add(tid, b.const_i32(p * prod_threads))
            row = b.div(cidx, c_chunks_per_row)
            col = b.mul(b.mod(cidx, c_chunks_per_row), c_halves_per_chunk)
            off = b.add(b.mul(b.add(block_m_off, row), K), b.add(k_off, col))
            b.async_buffer_load_lds_addr(
                a_rsrc, lds, b.mul(off, c2), zsoff, DTL_DWORDS, coherency=2
            )
        for p in range(b_passes):
            lds = (
                b_base
                if p == 0
                else b.smem_ptr_add(
                    b_base, b.zext(b.const_i32(p * prod_threads * DTL_BPL), _I64)
                )
            )
            cidx = b.add(tid, b.const_i32(p * prod_threads))
            row = b.div(cidx, c_chunks_per_row)
            col = b.mul(b.mod(cidx, c_chunks_per_row), c_halves_per_chunk)
            off = b.add(b.mul(b.add(block_n_off, row), K), b.add(k_off, col))
            b.async_buffer_load_lds_addr(
                b_rsrc, lds, b.mul(off, c2), zsoff, DTL_DWORDS, coherency=2
            )

    m_in_atom = b.mod(lane, b.const_i32(wtm))
    k_blk = b.div(lane, b.const_i32(wtm))
    n_in_atom = b.mod(lane, b.const_i32(wtn))
    warp_m_off = b.mul(cwarp_m, b.const_i32(mfmas_m * wtm))
    warp_n_off = b.mul(cwarp_n, b.const_i32(mfmas_n * wtn))
    k_blk_kbase = b.mul(k_blk, b.const_i32(a_per_lane))

    sched = _os.environ.get("CK_WSP3_SCHED", "1") == "1"

    def _consumer_mfma(accs, a_buf_row, b_buf_row):
        """One K-tile of MFMAs for this consumer warp reading LDS buffer at
        row origin ``a_buf_row`` (A) / ``b_buf_row`` (B); returns new accs."""
        new = list(accs)
        a_base = b.add(a_buf_row, warp_m_off)
        b_base = b.add(b_buf_row, warp_n_off)

        def aload(mi, col_base):
            return _emit_smem_load(
                b,
                A_smem,
                b.add(a_base, b.add(b.const_i32(mi * wtm), m_in_atom)),
                col_base,
                a_per_lane,
                storage_dtype,
            )

        def bload(ni, col_base):
            return _emit_smem_load(
                b,
                B_smem,
                b.add(b_base, b.add(b.const_i32(ni * wtn), n_in_atom)),
                col_base,
                b_per_lane,
                storage_dtype,
            )

        for kk in range(k_atoms):
            col_base = b.add(k_blk_kbase, b.const_i32(kk * wtk))
            if sched:
                # Fine schedule: read A-rows ahead, read each B-col just before
                # its first MFMA, wrap MFMAs with s_setprio + sched_barrier(0)
                # so the post-RA scheduler keeps the MFMA pipe fed and can't
                # hoist all ds_reads above the MFMAs.
                a_rows = [None] * mfmas_m
                b_cols = [None] * mfmas_n
                a_rows[0] = aload(0, col_base)
                b.s_setprio(1)
                for mi in range(mfmas_m):
                    if mi + 1 < mfmas_m:
                        a_rows[mi + 1] = aload(mi + 1, col_base)
                    for ni in range(mfmas_n):
                        if mi == 0:
                            b_cols[ni] = bload(ni, col_base)
                        flat = mi * mfmas_n + ni
                        new[flat] = _emit_mma(b, op, a_rows[mi], b_cols[ni], new[flat])
                        b.sched_barrier(0)
                b.s_setprio(0)
            else:
                a_rows = [aload(mi, col_base) for mi in range(mfmas_m)]
                b_cols = [bload(ni, col_base) for ni in range(mfmas_n)]
                flat = 0
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        new[flat] = _emit_mma(b, op, a_rows[mi], b_cols[ni], new[flat])
                        flat += 1
        return new

    if depth == 1:
        # Phase-1 serialized path: 2 full barriers per iter (no overlap).
        with b.scf_if(is_producer):
            pfor = b.scf_for(c0, K, c_block_k, iv_name="kp")
            with pfor as kp:
                _producer_load(kp, c0, c0)
                b.sync()  # barrier 1: producer writes visible to consumers
                b.sync()  # barrier 2: wait until consumers done reading
        with b.scf_if(is_consumer):
            accs0 = [_emit_zero_acc_op(b, op) for _ in range(mfmas_m * mfmas_n)]
            cfor = b.scf_for_iter(
                c0,
                K,
                c_block_k,
                [(f"acc{i}", a) for i, a in enumerate(accs0)],
                iv_name="kc",
            )
            with cfor as (kc, accs):
                b.sync()
                new = _consumer_mfma(accs, c0, c0)
                b.sync()
                b.scf_yield(*new)
            final_accs = cfor.results
            _emit_epilogue_default(
                b,
                spec,
                op,
                final_accs,
                cwarp_m,
                cwarp_n,
                lane,
                block_m_off,
                block_n_off,
                M,
                N,
                C,
                c_per_lane,
            )
        b.ret()
        return b.kernel

    # ---- depth>=2: ping-pong ring, ONE barrier per iter, producer loads ahead ----
    # Two producer variants (consumer side is identical):
    #   async (Phase 3, default): direct-to-LDS loads stay in flight across an
    #     ``s_barrier_bare`` (no vmcnt drain); the producer issues an explicit
    #     ``s_waitcnt(vmcnt=(depth-2)*loads_per_tile)`` so exactly the tile the
    #     consumer is about to read is complete while depth-2 tiles keep
    #     streaming. depth>2 thus actually hides more HBM latency.
    #   sync (Phase 2 fallback, CK_WSP3_ASYNC=0): global->VGPR->LDS + full
    #     ``sync()`` per iter (drains vmcnt every barrier; depth>2 gives no gain).
    # Default to the SYNC producer (global->VGPR->LDS + full sync()/iter): it is
    # the reliable best (0.44x). The async direct-to-LDS path (CK_WSP3_ASYNC=1)
    # is kept for experimentation but (a) is slower here and (b) cannot do
    # partial-vmcnt deep prefetch correctly — buffer_load_lds completes
    # OUT OF ORDER, so s_waitcnt(vmcnt=N>0) can't isolate "tile T done" across
    # the producer/consumer wave split (verified: depth-3 partial-drain = wrong,
    # full-drain = correct). The reference design likewise drains vmcnt(0) every
    # iter, so depth>2 yields no extra producer overlap; higher throughput comes
    # from the CONSUMER side (PLR + fine MFMA scheduling), the next phases.
    async_mode = (_os.environ.get("CK_WSP3_ASYNC", "0") == "1") and async_ok
    prologue_tiles = depth - 1
    keep_vmcnt = (depth - 2) * loads_per_tile  # tiles left in flight after the wait
    if _os.environ.get("CK_WSP3_DRAIN") == "1":
        keep_vmcnt = 0  # force full drain each iter (out-of-order vmem test)

    with b.scf_if(is_producer):
        if async_mode:
            for e in range(prologue_tiles):
                _producer_load_async(
                    b.const_i32(e * block_k),
                    b.const_i32(e * a_buf_bytes),
                    b.const_i32(e * b_buf_bytes),
                )
        else:
            for e in range(prologue_tiles):
                _producer_load(
                    b.const_i32(e * block_k),
                    b.const_i32(e * block_m),
                    b.const_i32(e * block_n),
                )
        pfor = b.scf_for(c0, K, c_block_k, iv_name="kp")
        with pfor as kp:
            knext = b.add(kp, b.const_i32(prologue_tiles * block_k))
            if async_mode:
                b.s_waitcnt(
                    vmcnt=keep_vmcnt, lgkmcnt=-1
                )  # tile T done; depth-2 still streaming
                b.s_barrier_bare()  # rendezvous T (does NOT drain in-flight loads)
                with b.scf_if(b.cmp_lt(knext, K)):
                    tnext = b.div(knext, c_block_k)
                    wbuf = b.mod(tnext, cD)
                    _producer_load_async(
                        knext,
                        b.mul(wbuf, b.const_i32(a_buf_bytes)),
                        b.mul(wbuf, b.const_i32(b_buf_bytes)),
                    )
            else:
                b.sync()  # rendezvous T (full drain)
                with b.scf_if(b.cmp_lt(knext, K)):
                    tnext = b.div(knext, c_block_k)
                    wbuf = b.mod(tnext, cD)
                    _producer_load(
                        knext, b.mul(wbuf, c_block_m), b.mul(wbuf, c_block_n)
                    )

    with b.scf_if(is_consumer):
        accs0 = [_emit_zero_acc_op(b, op) for _ in range(mfmas_m * mfmas_n)]
        cfor = b.scf_for_iter(
            c0,
            K,
            c_block_k,
            [(f"acc{i}", a) for i, a in enumerate(accs0)],
            iv_name="kc",
        )
        with cfor as (kc, accs):
            if async_mode:
                b.s_barrier_bare()  # rendezvous T (producer guaranteed tile T landed)
            else:
                b.sync()
            tcur = b.div(kc, c_block_k)
            rbuf = b.mod(tcur, cD)
            new = _consumer_mfma(accs, b.mul(rbuf, c_block_m), b.mul(rbuf, c_block_n))
            b.scf_yield(*new)
        final_accs = cfor.results
        _emit_epilogue_default(
            b,
            spec,
            op,
            final_accs,
            cwarp_m,
            cwarp_n,
            lane,
            block_m_off,
            block_n_off,
            M,
            N,
            C,
            c_per_lane,
        )

    b.ret()
    return b.kernel
