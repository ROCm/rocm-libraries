# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Ragged MoE grouped GEMM for gfx950 (CDNA4), bf16 -- variable tokens per
expert with fused gather (A) and scatter (C).

This is the ragged-MoE pivot of the dense ``grouped_gemm_hip.py`` kernel. The
dense kernel is left untouched; this file reuses the same proven inner GEMM
schedule (DTL cooperative direct-to-LDS load, st_16x32 swizzle, inline-asm
b128 ds_read, deep operand pipeline) and only changes three things:

1. Tile scheduling -- instead of ``block_id_z = expert`` over a fixed per-expert
   row count, the grid is ``(ceil(N/TN), num_m_tiles, 1)`` and each m-tile's
   expert comes from ``expert_ids[pid_m]`` (a block-aligned sorted token
   layout produced by a host ``moe_align`` step).
2. Fused gather -- the A (activation) rows are gathered per token via
   ``sorted_token_ids`` inside the DTL load; there is one ``[num_tokens, K]``
   activation buffer (no per-expert copy).
3. Weighted scatter -- each output row is written to the *expanded* output at
   ``C[sorted_token_ids[row]]`` (row = token*top_k + k), multiplied by its
   routing weight; padded rows are clamped to a sink row and discarded. A host
   segment-sum over top_k combines the expanded rows into the final output.

Inputs (``moe_align`` precomputed on host, passed in):
* ``sorted_token_ids[num_m_tiles*TM]`` -- expanded token indices grouped by
  expert, each expert's run padded to a multiple of TM (sentinel = num_expanded).
* ``expert_ids[num_m_tiles]`` -- expert id per m-tile.
* ``routing_weights[num_expanded]`` -- per expanded-row routing weight.
* ``num_expanded`` (scalar) -- = num_tokens*top_k; also the padding sentinel and
  the sink-row index (C is allocated ``[num_expanded+1, N]``).

Epilogue/gather levers (env-gated, all default on):
* ``HOIST=1``   precompute the per-pass A-gather token offset once per tile (it
                is loop-invariant across K-tiles) instead of re-reading the staged
                token + recomputing ``token//TOPK`` every K-tile. Removes the
                per-K-tile narrow ``ds_read_b32`` token re-reads (256 -> 4).
* ``RWLDS=1``   stage the per-row routing weights into LDS alongside the tokens so
                the scatter reads them from LDS, not a scattered global load
                (``global_load_dword`` 33 -> 2). Also *required* by EPIFUSE: a
                fused scatter runs inside the MFMA pipeline and must read only
                LDS-resident metadata (a mid-pipeline scattered global VMEM load
                races the inline-asm vmem/lgkm waits and corrupts operands).
* ``EPIFUSE=1`` emit each m-row's weighted scatter right after that row's final
                MFMA (last K-tile) so the store-issue + address VALU overlap the
                remaining m-rows' MFMAs instead of running as an exposed serial
                tail (implies ``RWLDS``). This is the default epilogue.
* ``CSHUF=1``  (off by default; RCR + square tile ``TN==4*TK`` only) reuse the
                dead-after-K-loop A/B LDS pool as a ``[TM,TN]`` C-stage: scatter the
                weighted accumulators into LDS (the C-lane transpose) then emit WIDE
                b128 coalesced per-row global stores (``global_store_short`` 128 -> 0,
                ``global_store_dwordx4`` 16), zero extra LDS. MEASURED NOT A WIN: the
                two extra workgroup barriers + serial LDS round-trip outweigh the
                store-width gain because the per-element scatter was already
                wave-coalesced -- EPIFUSE (overlap the narrow stores with MFMAs)
                beats it (~+30 TF at store-heavy N=4096/K=256; CSHUF is ~even at
                large K and slightly *below* baseline when store-bound). Kept as a
                lever for experimentation.

Run:
    PYTHONPATH=Python python3 \
        Python/rocke/examples/gfx950/grouped_gemm/ragged_moe_hip.py
"""
from __future__ import annotations

import math
import os

from rocke.core.ir import BF16, F32, I32, I64, IRBuilder, PtrType, VectorType
from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes, mfma_atom_for_dtype
from rocke.helpers.spec import SignatureBuilder


def build_ragged_moe(
    N,
    K,
    E,
    *,
    TOPK=2,
    TM=256,
    TN=256,
    TK=64,
    WM=2,
    WN=4,
    name="ragged_moe",
):
    """Build the ragged-MoE grouped bf16 GEMM ``KernelDef``.

    Returns ``(kernel, block_size, TM, TN)``. N/K/E/TOPK are compile-time; the
    number of m-tiles is a runtime grid dimension (depends on the token
    distribution), so it never appears in the kernel body -- each block computes
    exactly one (m_tile, n_tile).
    """
    atom = mfma_atom_for_dtype("bf16", 16, 16)  # 16x16x32 bf16
    AM, AN, AK = atom.m, atom.n, atom.k
    apl, bpl = atom.a_per_lane, atom.b_per_lane  # 8, 8
    BS = WM * WN * 64
    WTM, WTN = TM // WM, TN // WN
    mm, nn = WTM // AM, WTN // AN
    kchunks = TK // AK
    n_ktiles = K // TK
    assert TK % AK == 0 and TM % (WM * AM) == 0 and TN % (WN * AN) == 0
    assert (TM * TK) % BS == 0 and (TN * TK) % BS == 0

    swz = os.environ.get("SWZ", "1") == "1"
    asm_reads = os.environ.get("ASM", "1") == "1"
    deeppipe = os.environ.get("DEEPPIPE", "1") == "1"
    burstprio = os.environ.get("BURSTPRIO", "0") == "1"
    prio = os.environ.get("PRIO", "1") == "1"
    # CHIP: chiplet/XCD-aware super-tile grid remap (L2 locality across XCDs);
    # +~40 TF in the dense kernel. PIN: hoist wave-uniform indices into SGPRs.
    chiplet = os.environ.get("CHIP", "1") == "1"
    pin = os.environ.get("PIN", "1") == "1"
    chiplet_wgm = int(os.environ.get("WGM", "8"))
    chiplet_xcds = int(os.environ.get("XCDS", "8"))
    n_pid_n = (N + TN - 1) // TN
    # BRRR: native (un-preshuffled) NN weights B=[E,K,N] (N-contiguous), staged
    # into a plain [TK,TN] LDS tile and read via the CK transpose LDS read
    # (ds_read_b64_tr_b16), delivering the same MFMA b-operand as the RCR path.
    b_rrr = os.environ.get("BRRR", "0") == "1"
    # Epilogue/gather optimisation levers (all default on):
    # HOIST: precompute the per-pass A-gather token offset once per tile (the
    #   token index is loop-invariant across K-tiles) instead of re-reading the
    #   staged token + recomputing token//TOPK every K-tile.
    # RWLDS: stage the per-row routing weights into LDS alongside the tokens so
    #   the scatter reads them from LDS instead of a scattered global load.
    # EPIFUSE: emit each m-row's weighted scatter right after that row's final
    #   MFMA (last K-tile) so the store-issue + address VALU overlap the
    #   remaining MFMAs instead of running as an exposed serial tail.
    hoist = os.environ.get("HOIST", "1") == "1"
    rwlds = os.environ.get("RWLDS", "1") == "1"
    epifuse = os.environ.get("EPIFUSE", "1") == "1"
    # CSHUF: reuse the (dead-after-K-loop) A/B LDS as a [TM,TN] C-stage. Each lane
    # scatters its weighted accumulators into the C-stage at their tile-local
    # (row,col) (a cheap LDS ds_write = the "C-lane transpose"), then the workgroup
    # reads the C-stage back in contiguous 8-wide runs and issues WIDE (b128)
    # coalesced global stores -- each output row still redirects to its token, but
    # the N dimension is contiguous so the store widens 16x vs the per-element
    # scatter. Reuses the A/B pool (no extra LDS) only when the pool size
    # 2*nbuf*TM*TK == the C-tile TM*TN, i.e. TM==TN and TN==4*TK (nbuf=2). RCR
    # only (the RRR transpose tile is [TK,TN], incompatible with the flat reuse).
    nbuf = 2
    cshuf = os.environ.get("CSHUF", "0") == "1"
    if cshuf and (b_rrr or TM != TN or TN != 2 * nbuf * TK):
        cshuf = False
    # CKSCHED: emit CK-HotLoop-style sched_group_barrier groups (1 MFMA, then a
    # proportional slice of the next chunk's in-flight ds_reads) so the backend
    # spreads the operand LDS reads evenly across the MFMAs instead of clustering
    # them -- hides the ds_read latency behind the matrix ops. Most impactful for
    # the NN transpose-read path (2x ds_read_tr16_b64 per B fragment).
    # PERMMASCHED: raise s_setprio around EACH MFMA + a sched_barrier fence after,
    # so the backend cannot hoist the next chunk's ds_reads across a matrix op.
    cksched = os.environ.get("CKSCHED", "0") == "1"
    permmasched = os.environ.get("PERMMASCHED", "0") == "1"
    SGB_MFMA, SGB_DSREAD = 0x008, 0x100
    # TR128: deliver the full 8-wide B operand (16x16x32) with ONE
    # ds_read_tr16_b128 (16 B/lane) instead of two ds_read_tr16_b64 + vec_concat.
    # Halves the NN B-operand LDS-read ops. Requires the b128 transpose intrinsic
    # to link on the target ROCm (only the b64 variant is guaranteed).
    tr128 = os.environ.get("TR128", "0") == "1"
    # OPSW: operand switch. Swap the MFMA a/b operands so the 16x16 output
    # fragment is transposed -- each lane then holds c_per_lane CONSECUTIVE N
    # values for a SINGLE token (row = (lane//16)*4 + i is the N-within-atom,
    # col = lane%16 is the M/token). The scatter can then pack those 4 bf16 into
    # one aligned b64 store (C[token, n..n+3]) instead of 4 narrow scattered
    # short stores -- 4x fewer, 8-byte-aligned, same-token contiguous, no LDS
    # restage. Directly attacks the store-BW-bound C-scatter tail.
    opsw = os.environ.get("OPSW", "0") == "1"
    # MEASURED NOT A WIN: the default fragment layout already coalesces each
    # token's N across 16 adjacent lanes (one 32 B transaction); OPSW instead maps
    # adjacent lanes to different token rows -> scattered 8 B stores (slower).
    # RCR (NT) OPSW additionally mismatches the operand lane decode (produces NaN),
    # so OPSW is gated to the NN transpose path and kept only for experimentation.
    if opsw and not b_rrr:
        opsw = False
    if opsw:
        cshuf = False  # OPSW has its own packed-store epilogue
    # A fused scatter runs inside the MFMA software pipeline (interleaved with the
    # inline-asm operand ds_reads + s_setprio), so it must read only LDS-resident
    # metadata: issuing a scattered global routing-weight load mid-pipeline races
    # the manually-managed vmem/lgkm waits and corrupts the operands. EPIFUSE
    # therefore requires the weights be staged in LDS (RWLDS). CSHUF is a separate
    # coalesced epilogue (mutually exclusive with the fused per-row scatter).
    if cshuf:
        epifuse = False
    if epifuse or cshuf:
        rwlds = True

    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    STOK = b.param(
        "sorted_token_ids", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    EIDS = b.param(
        "expert_ids", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    RW = b.param(
        "routing_weights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    NEXP = b.param("num_expanded", I32)
    NMT = b.param("num_m_tiles", I32)  # runtime grid.y extent (for chiplet remap)
    NPB = b.param("num_persist_blocks", I32)  # persistent grid.x extent (stride)

    tid = b.thread_id_x()
    warp = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    warp_row = b.div(warp, b.const_i32(WN))
    warp_col = b.mod(warp, b.const_i32(WN))
    ld = decode_mfma_lanes(b, atom, lane)

    def U(v):
        return b.to_sgpr_u32(v) if pin else v

    persist = os.environ.get("PERSIST", "0") == "1"
    cN = b.const_i32(N)
    cTOPK = b.const_i32(TOPK)
    c_npn = b.const_i32(n_pid_n)
    # Per-tile coords, (re)assigned by set_tile(). The DTL-load / epilogue
    # closures read these as free variables, so a persistent grid-stride loop can
    # re-target each iteration without redefining the closures.
    m_base = n_base = expert = b_eoff = None
    # Per-pass A-gather offsets, precomputed once per tile by precompute_a_gather()
    # (HOIST lever); read as a free variable by dtl_load_a_gather().
    a_gather_base = None

    def set_tile(wgid):
        # Decode a flat workgroup id -> (m_tile, n_tile). The chiplet/super-tile
        # remap keeps adjacent m-tiles on the same XCD for L2 reuse (dynamic
        # num_pid_m = num_m_tiles).
        nonlocal m_base, n_base, expert, b_eoff
        if chiplet:
            from rocke.helpers.grid import chiplet_aware_super_tile_dynamic

            sw = chiplet_aware_super_tile_dynamic(
                b,
                wgid,
                num_pid_m=NMT,
                num_pid_n=c_npn,
                wgm=chiplet_wgm,
                num_xcds=chiplet_xcds,
                chunk_size=64,
            )
            mt, nt = sw.row, sw.col
        else:
            mt, nt = b.div(wgid, c_npn), b.mod(wgid, c_npn)
        m_base = U(b.mul(mt, b.const_i32(TM)))  # offset into sorted_token_ids
        n_base = U(b.mul(nt, b.const_i32(TN)))
        expert = U(b.global_load(EIDS, mt, I32))
        b_eoff = U(b.mul(expert, b.const_i32(N * K)))

    # A/B LDS buffers are carried as (buffer, row_base) tuples so CSHUF can slice
    # them out of one flat pool (and reuse that pool as the C-stage). Non-CSHUF
    # keeps distinct allocations with row_base=0.
    if cshuf:
        # One flat [2*nbuf*TM, TK] pool: As0..As{nbuf-1} then Bs0..Bs{nbuf-1} as
        # row bands; the epilogue reuses the whole pool as the [TM,TN] C-stage
        # (TM*TN == 2*nbuf*TM*TK by the CSHUF guard above).
        AB = b.smem_alloc(BF16, [2 * nbuf * TM, TK], name_hint="AB")
        As = [(AB, i * TM) for i in range(nbuf)]
        Bs = [(AB, (nbuf + i) * TM) for i in range(nbuf)]
    else:
        AB = None
        As = [
            (b.smem_alloc(BF16, [TM, TK], name_hint=f"As{i}"), 0) for i in range(nbuf)
        ]
        # RRR stages B as [TK,TN] (K rows, N cols) for the transpose-read; RCR uses
        # [TN,TK] (N rows, K cols) for the direct ds_read.
        b_shape = [TK, TN] if b_rrr else [TN, TK]
        Bs = [(b.smem_alloc(BF16, b_shape, name_hint=f"Bs{i}"), 0) for i in range(nbuf)]

    def _row(buf, r):  # absolute row within buf's smem handle (adds the band base)
        return b.add(r, b.const_i32(buf[1])) if buf[1] else r

    def _addr(buf):  # i64 LDS base address of this buf's band
        a = b.smem_addr_of(buf[0])
        return (
            b.smem_ptr_add(a, b.zext(b.const_i32(buf[1] * TK * 2), I64))
            if buf[1]
            else a
        )

    # Per-tile token indices staged once into LDS (they are constant across all
    # K-tiles). This replaces the redundant per-element global sorted_token_ids
    # loads in the A-gather and the scatter epilogue with cheap LDS reads.
    STOKL = b.smem_alloc(I32, [TM, 1], name_hint="stok")
    # RWLDS: per-row routing weights staged into LDS (one f32 per tile row) so the
    # scatter reads them from LDS instead of a scattered per-output-row global load.
    RWL = b.smem_alloc(F32, [TM, 1], name_hint="rw") if rwlds else None
    c_zero = b.const_i32(0)
    accs = [atom.zero_acc(b) for _ in range(mm * nn)]

    warp_m_off = b.mul(warp_row, b.const_i32(WTM))
    warp_n_off = b.mul(warp_col, b.const_i32(WTN))

    def stage_tokens():
        # Cooperatively load sorted_token_ids[m_base : m_base+TM] into LDS.
        # Threads past TM clamp to the last slot (redundant same-value write).
        # RWLDS: co-stage each row's routing weight (RW[token], token clamped for
        # padded rows) so the scatter reads it from LDS.
        for p in range((TM + BS - 1) // BS):
            idx = b.add(tid, b.const_i32(p * BS))
            sidx = b.select(b.cmp_lt(idx, b.const_i32(TM)), idx, b.const_i32(TM - 1))
            val = b.global_load(STOK, b.add(m_base, sidx), I32)
            b.smem_store_vN(STOKL, [sidx, c_zero], val, 1)
            if rwlds:
                wv = b.global_load(
                    RW, b.select(b.cmp_lt(val, NEXP), val, b.const_i32(0)), F32
                )
                b.smem_store_vN(RWL, [sidx, c_zero], wv, 1)

    def stok_lds(row):
        return b.vec_extract(b.smem_load_vN(STOKL, row, c_zero, dtype=I32, n=1), 0)

    def rw_lds(row):
        return b.vec_extract(b.smem_load_vN(RWL, row, c_zero, dtype=F32, n=1), 0)

    def swz_col(col, row):
        # st_16x32 XOR swizzle (bank-conflict-free ds_reads): col ^ (((row>>3)&1)<<4).
        if not swz:
            return col
        return b.xor(
            col,
            b.shl(b.mod(b.lshr(row, b.const_i32(3)), b.const_i32(2)), b.const_i32(4)),
        )

    # ---- operand-read setup (RCR layout, mirrors the dense kernel) ----
    a_frag_ty = VectorType(BF16, apl)
    b_frag_ty = VectorType(BF16, bpl)
    raw_ty = VectorType(I32, apl // 2)
    a_base_row = b.add(b.mul(warp_row, b.const_i32(WTM)), ld.m_in_atom)
    b_base_row = b.add(b.mul(warp_col, b.const_i32(WTN)), ld.n_in_atom)
    k_lane = b.mul(ld.k_blk, b.const_i32(apl))
    a_rows = [
        b.add(
            b.add(b.mul(warp_row, b.const_i32(WTM)), b.const_i32(mi * AM)), ld.m_in_atom
        )
        for mi in range(mm)
    ]
    b_rows = [
        b.add(
            b.add(b.mul(warp_col, b.const_i32(WTN)), b.const_i32(nj * AN)), ld.n_in_atom
        )
        for nj in range(nn)
    ]

    # RRR transpose-read setup: read a plain [TK,TN] LDS tile and deliver the
    # MFMA b-operand (B[n,k] per lane) via CK's TransposeLdsReader. AK=32 -> two
    # ds_read_tr16_b64 (read 0/1) concatenated into the 8-wide b-fragment.
    if b_rrr:
        from rocke.helpers.layouts import TransposeLdsReader

        tr_reader = TransposeLdsReader(K=AK, M=AN).bind(b, lane)
        b_warp_n = b.mul(warp_col, b.const_i32(WTN))
        b_n_cols = [
            b.add(b.add(b_warp_n, b.const_i32(nj * AN)), tr_reader.col)
            for nj in range(nn)
        ]
        # HOIST: the transpose-read LDS row indices depend only on the k-chunk
        # position within the TK tile (kk), not on the K-tile -- the LDS layout is
        # identical every K-tile. Precompute the (read0, read1) rows for each kk
        # once instead of recomputing them in every _read_b_rrr call.
        tr_rows = [
            (
                tr_reader.row(b, k_offset=kk * AK, read=0),
                tr_reader.row(b, k_offset=kk * AK, read=1),
            )
            for kk in range(kchunks)
        ]

    def _read_b_rrr(bi, kk):
        r0, r1 = tr_rows[kk]
        out = []
        if tr128:
            # One b128 transpose-read delivers the full 8-wide B fragment per nj.
            for nj in range(nn):
                out.append(b.ds_read_tr16_b128(Bs[bi][0], r0, b_n_cols[nj], dtype=BF16))
            return out
        for nj in range(nn):
            f0 = b.ds_read_tr16_b64(Bs[bi][0], r0, b_n_cols[nj], dtype=BF16)
            f1 = b.ds_read_tr16_b64(Bs[bi][0], r1, b_n_cols[nj], dtype=BF16)
            out.append(b.vec_concat(f0, f1))
        return out

    def ds_read_imm(buf, base_row, col_swz, fi, frag_ty):
        base_off = b.mul(
            b.add(b.mul(base_row, b.const_i32(TK)), col_swz), b.const_i32(2)
        )
        addr = b.add(b.trunc(_addr(buf), I32), base_off)
        imm = fi * AM * TK * 2
        ds_w = apl * 16
        raw = b.inline_asm(
            f"ds_read_b{ds_w} $0, $1 offset:{imm}",
            "=v,v",
            [addr],
            result_type=raw_ty,
            sideeffect=True,
        )
        return b.vec_bitcast(raw, frag_ty)

    def read_chunk(ai, bi, kk):
        col = b.add(b.const_i32(kk * AK), k_lane)
        if asm_reads:
            ca = swz_col(col, a_base_row)
            a = [ds_read_imm(As[ai], a_base_row, ca, mi, a_frag_ty) for mi in range(mm)]
        else:
            a = [
                b.smem_load_vN(
                    As[ai][0],
                    _row(As[ai], a_rows[mi]),
                    swz_col(col, a_rows[mi]),
                    dtype=BF16,
                    n=apl,
                )
                for mi in range(mm)
            ]
        if b_rrr:
            return a, _read_b_rrr(bi, kk)
        if asm_reads:
            cb = swz_col(col, b_base_row)
            bb = [
                ds_read_imm(Bs[bi], b_base_row, cb, nj, b_frag_ty) for nj in range(nn)
            ]
            return a, bb
        bb = [
            b.smem_load_vN(
                Bs[bi][0],
                _row(Bs[bi], b_rows[nj]),
                swz_col(col, b_rows[nj]),
                dtype=BF16,
                n=bpl,
            )
            for nj in range(nn)
        ]
        return a, bb

    def _mma_block(a_fr, b_fr, acc, store_row_cb=None, n_ds=0):
        # EPIFUSE: after finishing an m-row's nn MFMAs on the final K-chunk,
        # scatter that row immediately so its stores overlap the later m-rows'
        # MFMAs (store_row_cb reads its operands from LDS, computed transiently).
        # CKSCHED: interleave 1 MFMA + a proportional slice of the next chunk's
        # in-flight ds_reads so the backend spreads reads across the matrix ops.
        nmma = mm * nn
        ds_done = 0
        for mi in range(mm):
            for nj in range(nn):
                idx = mi * nn + nj
                # OPSW: swap operands -> acc[idx] holds C[m-tile mi, n-tile nj]
                # with N as the per-lane (c_per_lane) fast dimension.
                op_a, op_b = (b_fr[nj], a_fr[mi]) if opsw else (a_fr[mi], b_fr[nj])
                if permmasched:
                    b.s_setprio(1)
                    acc[idx] = atom.emit(b, op_a, op_b, acc[idx])
                    b.s_setprio(0)
                    b.sched_barrier(0)
                else:
                    acc[idx] = atom.emit(b, op_a, op_b, acc[idx])
                if cksched and n_ds:
                    b.sched_group_barrier(SGB_MFMA, 1, 0)
                    want = ((idx + 1) * n_ds) // nmma
                    if want > ds_done:
                        b.sched_group_barrier(SGB_DSREAD, want - ds_done, 0)
                        ds_done = want
            if store_row_cb is not None:
                store_row_cb(acc, mi)

    # ds_reads issued per next-chunk prefetch: mm A-frags + the B reads (RRR
    # transpose: nn x2 b64; RCR: nn b128). Used to size the CKSCHED interleave.
    n_ds_prefetch = mm + (nn * 2 if b_rrr else nn)

    def compute(ai, bi, acc, store_on_last=None):
        if prio and not burstprio:
            b.s_setprio(1)
        if deeppipe and asm_reads:
            a_fr, b_fr = read_chunk(ai, bi, 0)
            for kk in range(kchunks):
                if burstprio:
                    b.s_setprio(0)
                b.s_waitcnt(lgkmcnt=0)
                na = nb = None
                has_next = kk + 1 < kchunks
                if has_next:
                    na, nb = read_chunk(ai, bi, kk + 1)
                if burstprio:
                    b.s_setprio(1)
                sr = store_on_last if kk == kchunks - 1 else None
                _mma_block(
                    a_fr,
                    b_fr,
                    acc,
                    store_row_cb=sr,
                    n_ds=n_ds_prefetch if has_next else 0,
                )
                if has_next:
                    a_fr, b_fr = na, nb
            if burstprio:
                b.s_setprio(0)
        else:
            a_fr, b_fr = read_chunk(ai, bi, 0)
            for kk in range(kchunks):
                if asm_reads:
                    b.s_waitcnt(lgkmcnt=0)
                sr = store_on_last if kk == kchunks - 1 else None
                _mma_block(a_fr, b_fr, acc, store_row_cb=sr)
                if kk + 1 < kchunks:
                    a_fr, b_fr = read_chunk(ai, bi, kk + 1)
        if prio and not burstprio:
            b.s_setprio(0)

    # ---- DTL cooperative direct-to-LDS loads ----
    HALVES, DWORDS = 8, 4  # 16 bytes/lane
    cpr = TK // HALVES
    passes_a = (TM * TK // HALVES) // BS
    passes_b = (TN * TK // HALVES) // BS
    assert TK % HALVES == 0
    wave_off = b.zext(b.mul(warp, b.const_i32(64 * 16)), I64)
    zsoff = b.const_i32(0)
    big = b.const_i32(0x7FFF0000)
    a_rsrc = b.buffer_rsrc(A, big)
    b_rsrc = b.buffer_rsrc(Bp, big)
    c_cpr = b.const_i32(cpr)
    c_hv = b.const_i32(HALVES)
    c_cpr_n = b.const_i32(TN // HALVES)  # N-fast groups per row (RRR B load)

    def dtl_load_b_rrr(buf, kt, coh):
        # B=[E,K,N] (N-contiguous): stage into a plain [TK,TN] LDS tile (row=k,
        # col=n, no swizzle -- tr16 reads the plain layout).
        base = b.smem_ptr_add(_addr(buf), wave_off)
        for p in range(passes_b):
            lds = (
                base
                if p == 0
                else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
            )
            ci = b.add(tid, b.const_i32(p * BS))
            row_k = b.div(ci, c_cpr_n)
            col_n = b.mul(b.mod(ci, c_cpr_n), c_hv)
            off = b.add(
                b_eoff,
                b.add(
                    b.mul(b.add(b.const_i32(kt * TK), row_k), b.const_i32(N)),
                    b.add(n_base, col_n),
                ),
            )
            b.async_buffer_load_lds_addr(
                b_rsrc, lds, b.mul(off, b.const_i32(2)), zsoff, DWORDS, coherency=coh
            )

    def dtl_load_b(buf, kt, coh):
        # B = [E,N,K]: stage rows n_base..+TN, K-contiguous, into [TN,TK] LDS.
        base = b.smem_ptr_add(_addr(buf), wave_off)
        for p in range(passes_b):
            lds = (
                base
                if p == 0
                else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
            )
            ci = b.add(tid, b.const_i32(p * BS))
            row = b.div(ci, c_cpr)
            col = b.mul(b.mod(ci, c_cpr), c_hv)
            off = b.add(
                b_eoff,
                b.add(
                    b.mul(b.add(n_base, row), b.const_i32(K)),
                    b.add(b.const_i32(kt * TK), swz_col(col, row)),
                ),
            )
            b.async_buffer_load_lds_addr(
                b_rsrc, lds, b.mul(off, b.const_i32(2)), zsoff, DWORDS, coherency=coh
            )

    def precompute_a_gather():
        # HOIST: the A-gather source offset for each pass is loop-invariant across
        # K-tiles except for the +kt*TK term. Compute the invariant part once per
        # tile (token LDS read + token//TOPK + row*K + swizzle) and hold it in a
        # register; the K-loop then only adds the compile-time kt*TK.
        nonlocal a_gather_base
        vals = []
        for p in range(passes_a):
            ci = b.add(tid, b.const_i32(p * BS))
            row = b.div(ci, c_cpr)
            col = b.mul(b.mod(ci, c_cpr), c_hv)
            tok = stok_lds(row)
            a_row = b.select(b.cmp_lt(tok, NEXP), b.div(tok, cTOPK), b.const_i32(0))
            vals.append(b.add(b.mul(a_row, b.const_i32(K)), swz_col(col, row)))
        a_gather_base = vals

    def dtl_load_a_gather(buf, kt, coh):
        # A = [num_tokens,K]: gather each tile row's token via sorted_token_ids,
        # activation row = token // TOPK. Padded rows (tok >= NEXP) clamp to row 0
        # (their output is discarded to the sink row in the epilogue).
        base = b.smem_ptr_add(_addr(buf), wave_off)
        for p in range(passes_a):
            lds = (
                base
                if p == 0
                else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
            )
            if hoist:
                off = b.add(a_gather_base[p], b.const_i32(kt * TK))
            else:
                ci = b.add(tid, b.const_i32(p * BS))
                row = b.div(ci, c_cpr)
                col = b.mul(b.mod(ci, c_cpr), c_hv)
                tok = stok_lds(row)  # LDS-staged token index (no global reload)
                a_row = b.select(b.cmp_lt(tok, NEXP), b.div(tok, cTOPK), b.const_i32(0))
                off = b.add(
                    b.mul(a_row, b.const_i32(K)),
                    b.add(b.const_i32(kt * TK), swz_col(col, row)),
                )
            b.async_buffer_load_lds_addr(
                a_rsrc, lds, b.mul(off, b.const_i32(2)), zsoff, DWORDS, coherency=coh
            )

    def load_tile(buf, kt):
        dtl_load_a_gather(As[buf], kt, 0)
        if b_rrr:
            dtl_load_b_rrr(Bs[buf], kt, 0)
        else:
            dtl_load_b(Bs[buf], kt, 0)

    def kloop(acc, final_storer=None):
        stage_tokens()  # stage per-tile token indices (+ weights) into LDS once
        b.sync()
        if hoist:
            precompute_a_gather()  # tokens are in LDS now -> hoist gather offsets
        load_tile(0, 0)  # prologue
        for kt in range(n_ktiles):
            cur = kt % 2
            b.sync()  # drain cur tile's DTL writes + workgroup barrier
            if kt + 1 < n_ktiles:
                load_tile(1 - cur, kt + 1)  # next-tile loads overlap MFMAs
            # EPIFUSE: on the last K-tile, fuse the scatter into the MFMAs.
            sol = final_storer if kt == n_ktiles - 1 else None
            compute(cur, cur, acc, store_on_last=sol)

    def store_row(acc, mi):
        if opsw:
            # OPSW layout: for this lane the c_per_lane accumulator values are
            # CONSECUTIVE N (row = (lane//16)*4 + i) for a SINGLE token (col =
            # lane%16). Pack them (scaled by the row's weight) into one aligned
            # b64 store to C[token, n0..n0+c_per_lane-1].
            r0, c0 = atom.lane_to_output(b, lane, 0)  # r0 = n-in-atom base, c0 = m
            lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), c0)  # token row (m)
            otok = stok_lds(lr)
            valid = b.cmp_lt(otok, NEXP)
            orow = b.select(valid, otok, NEXP)  # padding -> sink row
            w = rw_lds(lr)
            row_base = b.add(b.add(b.mul(orow, cN), n_base), b.add(warp_n_off, r0))
            for nj in range(nn):
                idx = mi * nn + nj
                addr = b.add(row_base, b.const_i32(nj * AN))
                comps = [
                    b.cast_f32_to(b.fmul(b.vec_extract(acc[idx], i), w), BF16)
                    for i in range(atom.c_per_lane)
                ]
                b.global_store_vN(
                    C, addr, b.vec_pack(comps, BF16), n=atom.c_per_lane, align=8
                )
            return
        # Scatter one m-row (all nn n-tiles x c_per_lane elements) to
        # C[sorted_token_ids[row]] (expanded index), scaled by the routing weight.
        # The token/weight/row-base depend on (mi, i) only, not nj -> hoist over nj.
        for i in range(atom.c_per_lane):
            row_in, col_in = atom.lane_to_output(b, lane, i)
            lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), row_in)
            otok = stok_lds(lr)  # LDS-staged token index (no global reload)
            valid = b.cmp_lt(otok, NEXP)
            orow = b.select(valid, otok, NEXP)  # padding -> sink row (NEXP)
            if rwlds:
                w = rw_lds(lr)  # LDS-staged routing weight (no global reload)
            else:
                w = b.global_load(RW, b.select(valid, otok, b.const_i32(0)), F32)
            row_col = b.add(b.add(b.mul(orow, cN), n_base), b.add(warp_n_off, col_in))
            for nj in range(nn):
                idx = mi * nn + nj
                addr = b.add(row_col, b.const_i32(nj * AN))
                val = b.fmul(b.vec_extract(acc[idx], i), w)
                b.global_store(C, addr, b.cast_f32_to(val, BF16), align=2)

    def epilogue(acc):
        # Non-fused fallback (EPIFUSE off): scatter every m-row after the K-loop.
        for mi in range(mm):
            store_row(acc, mi)

    def store_c_shuffle(acc):
        # CSHUF: reuse the A/B pool (dead after the K-loop) as a [TM,TN] C-stage.
        # (1) scatter each lane's weighted accumulators into the stage at their
        # tile-local (row,col) via ds_write -- the C-lane transpose; (2)
        # cooperatively read the stage in contiguous 8-wide runs and issue WIDE
        # (b128) per-row global stores, redirecting each row to its token (padding
        # -> sink row NEXP). The N dimension is contiguous in both the stage and
        # the destination C row, so the store widens 16x vs the per-element scatter.
        cTN = b.const_i32(TN)
        cTK = b.const_i32(TK)
        b.sync()  # last compute's operand ds_reads on AB done before reuse
        for mi in range(mm):
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), row_in)
                w = rw_lds(lr)  # per-row routing weight applied on the way into LDS
                for nj in range(nn):
                    idx = mi * nn + nj
                    lc = b.add(b.add(warp_n_off, b.const_i32(nj * AN)), col_in)
                    flat = b.add(b.mul(lr, cTN), lc)
                    val = b.cast_f32_to(b.fmul(b.vec_extract(acc[idx], i), w), BF16)
                    b.smem_store_vN(AB, [b.div(flat, cTK), b.mod(flat, cTK)], val, 1)
        b.sync()
        nblk = (TM * TN) // (BS * 8)
        for blk in range(nblk):
            flat = b.add(b.const_i32(blk * BS * 8), b.mul(tid, b.const_i32(8)))
            vec = b.smem_load_vN(
                AB, b.div(flat, cTK), b.mod(flat, cTK), dtype=BF16, n=8
            )
            r = b.div(flat, cTN)
            c = b.mod(flat, cTN)
            otok = stok_lds(r)
            orow = b.select(b.cmp_lt(otok, NEXP), otok, NEXP)  # padding -> sink row
            addr = b.add(b.add(b.mul(orow, cN), n_base), c)
            b.global_store_vN(C, addr, vec, n=8, align=16)

    def store_sink(acc):
        # DIAGNOSTIC (STORESINK=1): reduce every accumulator into c_per_lane sums
        # (data-dependent on all MFMAs, so none are DCE'd) and store only those few
        # values. Compute+load stay fully live; store traffic ~ 0. (full-run TF minus
        # STORESINK TF) isolates the real C-store-tail cost.
        sums = [None] * atom.c_per_lane
        for idx in range(mm * nn):
            for i in range(atom.c_per_lane):
                e = b.vec_extract(acc[idx], i)
                sums[i] = e if sums[i] is None else b.fadd(sums[i], e)
        base = b.mul(lane, b.const_i32(atom.c_per_lane))
        for i in range(atom.c_per_lane):
            b.global_store(
                C, b.add(base, b.const_i32(i)), b.cast_f32_to(sums[i], BF16), align=2
            )

    storesink = os.environ.get("STORESINK", "0") == "1"

    def do_tile_body():
        if persist:
            b.sync()  # end previous iteration's LDS use before restaging
        for i in range(len(accs)):
            accs[i] = atom.zero_acc(b)
        if storesink:
            kloop(accs)
            store_sink(accs)
        elif cshuf:
            kloop(accs)
            store_c_shuffle(accs)  # coalesced wide-store epilogue via LDS C-stage
        elif epifuse:
            kloop(accs, final_storer=store_row)  # scatter fused into last K-tile
        else:
            kloop(accs)
            epilogue(accs)

    if persist:
        # Persistent grid-stride: grid=(NPB,1,1); each block walks tiles
        # w = bid, bid+NPB, ... < num_m_tiles*n_pid_n. Fixes small-grid tail.
        total = b.mul(NMT, c_npn)
        floop = b.scf_for(b.block_id_x(), total, NPB, iv_name="tile")
        with floop as w:
            set_tile(w)
            do_tile_body()
    else:
        set_tile(b.add(b.mul(b.block_id_y(), c_npn), b.block_id_x()))
        do_tile_body()
    b.ret()
    return b.kernel, BS, TM, TN


def ragged_moe_signature():
    return (
        SignatureBuilder()
        .ptr("A", "bf16")
        .ptr("B", "bf16")
        .ptr("C", "bf16")
        .ptr("sorted_token_ids", "i32")
        .ptr("expert_ids", "i32")
        .ptr("routing_weights", "f32")
        .scalar("num_expanded", "i32")
        .scalar("num_m_tiles", "i32")
        .scalar("num_persist_blocks", "i32")
        .build()
    )


def _moe_align(expert_of, E, TM, num_expanded):
    """Host sort/align: group expanded rows by expert, pad each run to a
    multiple of TM. Returns (sorted_token_ids, expert_ids) as int32 numpy."""
    import numpy as np

    eo = expert_of.astype(np.int64)
    order = np.argsort(eo, kind="stable")  # expanded indices sorted by expert
    eo_sorted = eo[order]
    sorted_ids, expert_ids = [], []
    for e in range(E):
        idxs = order[eo_sorted == e]
        n = len(idxs)
        npad = ((n + TM - 1) // TM) * TM
        padded = np.full(npad, num_expanded, dtype=np.int32)  # sentinel
        padded[:n] = idxs.astype(np.int32)
        sorted_ids.append(padded)
        expert_ids.extend([e] * (npad // TM))
    sorted_ids = (
        np.concatenate(sorted_ids) if sorted_ids else np.zeros(0, dtype=np.int32)
    )
    return sorted_ids.astype(np.int32), np.asarray(expert_ids, dtype=np.int32)


def _main() -> int:
    import numpy as np
    import torch
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    num_tokens = int(os.environ.get("NTOK", 8192))
    N = int(os.environ.get("N", 1024))
    K = int(os.environ.get("K", 512))
    E = int(os.environ.get("E", 64))
    TOPK = int(os.environ.get("TOPK", 2))
    TM = int(os.environ.get("TM", 256))
    TN = int(os.environ.get("TN", 256))
    TK = int(os.environ.get("TK", 64))
    WM = int(os.environ.get("WM", 2))
    WN = int(os.environ.get("WN", 4))
    brrr = os.environ.get("BRRR", "0") == "1"
    num_expanded = num_tokens * TOPK

    torch.manual_seed(0)
    dt = torch.bfloat16
    A = torch.randn(num_tokens, K, dtype=dt, device="cuda")
    # BRRR: native NN weights [E,K,N]; else preshuffled RCR [E,N,K].
    B = (
        torch.randn(E, K, N, dtype=dt, device="cuda") * 0.05
        if brrr
        else torch.randn(E, N, K, dtype=dt, device="cuda") * 0.05
    )
    weights = torch.rand(num_expanded, dtype=torch.float32, device="cuda")
    # Imbalanced routing: skew expert assignment to exercise ragged tile counts
    # and padding (some experts get many tokens, some few/none).
    skew = torch.rand(num_expanded, device="cuda") ** 2
    expert_of = (skew * E).to(torch.int32).clamp_(0, E - 1)

    sorted_ids_np, expert_ids_np = _moe_align(
        expert_of.cpu().numpy(), E, TM, num_expanded
    )
    num_m_tiles = int(expert_ids_np.shape[0])

    sorted_ids = torch.from_numpy(sorted_ids_np).cuda()
    expert_ids = torch.from_numpy(expert_ids_np).cuda()

    kernel, BS, tm, tn = build_ragged_moe(
        N, K, E, TOPK=TOPK, TM=TM, TN=TN, TK=TK, WM=WM, WN=WN
    )
    print(
        f"[rmoe] tokens={num_tokens} N={N} K={K} E={E} topk={TOPK} "
        f"expanded={num_expanded} m_tiles={num_m_tiles} tile={TM}x{TN}x{TK} BS={BS} "
        f"layout={'NN(native)' if brrr else 'NT(preshuffled)'}"
    )
    art = compile_kernel_via_hipcc(kernel, arch="gfx950")
    print(f"[rmoe] hipcc built {kernel.name}: {len(art.hsaco)} B")

    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=ragged_moe_signature(),
        cache_key=(kernel.name,),
    )
    # Expanded output with a sink row (num_expanded) for padded tiles.
    C = torch.zeros(num_expanded + 1, N, dtype=dt, device="cuda")
    persist = os.environ.get("PERSIST", "0") == "1"
    n_nt = math.ceil(N / tn)
    if persist:
        # 1D persistent grid capped at ~#CUs*blocks_per_cu; each block strides.
        num_persist_blocks = min(n_nt * num_m_tiles, int(os.environ.get("NPB", "608")))
        grid = (num_persist_blocks, 1, 1)
    else:
        num_persist_blocks = n_nt  # unused by the kernel in non-persist mode
        grid = (n_nt, num_m_tiles, 1)
    blk = (BS, 1, 1)

    def call():
        launcher(
            {
                "A": A.data_ptr(),
                "B": B.data_ptr(),
                "C": C.data_ptr(),
                "sorted_token_ids": sorted_ids.data_ptr(),
                "expert_ids": expert_ids.data_ptr(),
                "routing_weights": weights.data_ptr(),
                "num_expanded": num_expanded,
                "num_m_tiles": num_m_tiles,
                "num_persist_blocks": num_persist_blocks,
            },
            config=LaunchConfig(stream=0, grid=grid, block=blk),
        )

    C.zero_()
    call()
    torch.cuda.synchronize()

    # Reference: expanded rows (weighted). row e -> token e//TOPK, expert
    # expert_of[e]; out = weight[e] * (A[token] @ B[expert]^T). Computed
    # per-expert (avoids materializing B[expert_of], which is O(num_expanded*N*K)).
    ref = torch.empty(num_expanded, N, dtype=torch.float32, device="cuda")
    Af = A.float()
    Bf = B.float()
    exp = expert_of.long()
    for e in range(E):
        rows = (exp == e).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        toks = (rows // TOPK).long()
        # NN (BRRR): B[e] is [K,N] -> A@B; RCR: B[e] is [N,K] -> A@B^T.
        ref[rows] = Af[toks] @ (Bf[e] if brrr else Bf[e].transpose(0, 1))
    ref = ref * weights[:, None]
    got = C[:num_expanded].float()
    err = (got - ref).abs().max().item()
    denom = ref.abs().max().item() + 1e-6
    print(
        f"[rmoe] grid={grid} max_abs={err:.4f} rel={err / denom:.4f} "
        f"{'PASS' if err / denom < 0.02 else 'FAIL'}"
    )

    # top-k combine (host segment-sum) sanity: final[token] = sum_k expanded.
    final = C[:num_expanded].float().view(num_tokens, TOPK, N).sum(1)
    ref_final = ref.view(num_tokens, TOPK, N).sum(1)
    ferr = (final - ref_final).abs().max().item()
    print(f"[rmoe] combine max_abs={ferr:.4f}")

    if err / denom >= 0.02:
        return 1

    # perf (valid FLOPs only)
    flops = 2 * num_expanded * N * K
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            call()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 50)
    ts.sort()
    print(
        f"[rmoe] med={flops / (ts[2] * 1e9):.1f} TF  peak={flops / (ts[0] * 1e9):.1f} TF"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
