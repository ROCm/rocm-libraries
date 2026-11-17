# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Decode GEMM family ``"skinny"``: a K-sliced stream of X and W through
wave-private LDS rings (see :mod:`rocke.instances.gfx950.decode_gemm`).

A workgroup owns ``block_n`` rows of W (``block_n / 16`` MFMA row tiles) and
``k_waves`` waves, each over ``K / k_waves`` of K. Per stage of ``block_k``
K elements a wave DMAs ``block_m`` rows of X and its ``block_n`` rows of W
(1 KB per DMA, dense rows of ``2 block_k`` bytes) into slot
``stage % stages`` of its own LDS region, so the K loop needs no barrier: a
wave waits for its own DMAs of the stage (``vmcnt``), reads the operands
with ``ds_read_b128``, refills the slot with the stage ``stages`` ahead and
runs the stage's MFMAs. One X operand read feeds every row tile. The rows
of a stage are swizzled: the 16 B chunk ``c`` of row ``r`` sits at chunk
``c ^ ((r >> 1) & 7)`` (128 B rows) or ``c ^ (r & 15)`` (256 B rows); the DMA
writes LDS linearly and fetches the permuted chunk from global memory. The
k-waves' partial tiles meet in LDS after one barrier, and each thread stores
four consecutive bf16 outputs. X rows past M load zeros and output rows past
M are dropped, both by the buffer bounds. ``xcd=1`` gives each of the
``xccs`` XCDs (consecutive workgroups go round-robin to them) one contiguous
range of tiles: workgroup ``b`` runs tile
``(b % xccs) * (grid / xccs) + b / xccs``.

With ``m_waves x n_waves > 1`` (``k_waves = 1``) the waves share one ring of
``block_m x block_n`` tiles instead: each wave DMAs its share of a stage's
rows, so a stage is complete only after every wave's ``vmcnt`` wait and one
barrier, which also retires the previous stage's reads; the ring then runs
``stages - 1`` stages ahead and the refill goes right after the barrier. Wave
``(wave // n_waves, wave % n_waves)`` computes its ``block_m / m_waves`` by
``block_n / n_waves`` part of the tile and stores it, and the grid walks the
M tiles, each over every column tile. ``refill=1`` issues the refill after
the step's operand reads instead of right after the barrier.

``split_k > 1`` (shared ring) splits the ``K / block_k`` steps of a tile
over ``split_k`` workgroups (workgroup ``b``: split ``b / tiles``, tile
``b % tiles``); split ``s`` runs steps ``[s * steps / split_k, (s + 1) *
steps / split_k)``. When the split is uneven every workgroup runs the longer
count and the short splits load their last stage out of bounds (zeros).
Each split stores its fp32 partial tile into its own workspace slot, takes
the tile's ticket (``sk_sync``: "fence" orders the partials with agent-scope
release / acquire fences, "relaxed" with device-coherent partial stores and
loads), and the workgroup that takes the last ticket sums the slots, stores
bf16 and resets the ticket. ``split_k=-1`` takes the most splits whose grid
fits one workgroup per CU (``tiles x split_k <= cus``), at least 2 steps each.

An LDS DMA needs one wait state after the SALU write of M0 that addresses
it. ``m0fill=1`` lets up to four consecutive DMAs into one operand share an
M0 write: each later one adds its LDS step as the instruction's immediate
offset (which the hardware applies to the global address too, so the step
is taken back out of its voffset). ``m0fill=2`` issues the refill among the
step's MFMAs, one MFMA ahead of each DMA (shared ring: with ``refill=1``);
``m0fill=3`` does both.
"""

from __future__ import annotations

from ...core.ir import BF16, F32, I32, I64, VectorType

SKINNY_BLOCK_M = (8, 16, 24, 32)
SHARED_BLOCK_M = (16, 32, 48, 64, 128)
SHARED_BLOCK_N = (16, 32, 48, 64, 128, 144)
SPLIT_FINISH = ("last", "epoch", "shared")
SKINNY_BLOCK_K = (64, 128)
LDS_DMA_BYTES = 1024
OOB = 0x7FFF0000  # a buffer offset past every tensor
M0_SHARE_BYTES = 3072  # largest LDS step a DMA's immediate offset (12 bits) covers in 1 KB steps
SGB_MFMA, SGB_VMEM_READ = 0x008, 0x020  # sched_group_barrier classes


def skinny_shared(spec) -> bool:
    """Whether the workgroup's waves share one ring (``m_waves x n_waves > 1``)."""
    return spec.m_waves * spec.n_waves > 1


def skinny_block_m(spec) -> int:
    """X rows a stage stages: ``block_m``, or the smallest legal size >= M."""
    if spec.block_m:
        return spec.block_m
    return next((bm for bm in SKINNY_BLOCK_M if bm >= spec.M), 0)


def skinny_auto_split(spec) -> int:
    """The split count of ``split_k=-1``: the most splits with ``tiles x
    split_k <= cus`` that leave every split at least 2 steps (1 when the
    tiling is not a shared ring or already fills the CUs)."""
    bm, bn, bk = skinny_block_m(spec), spec.block_n, spec.block_k
    if not skinny_shared(spec) or bk not in SKINNY_BLOCK_K or bn < 16 or spec.N % bn or not bm:
        return 1
    tiles = -(-spec.M // bm) * (spec.N // bn)
    return max(1, min(spec.cus // tiles, spec.K // bk // 2))


def skinny_plan(spec) -> dict:
    """The concrete tiling of a skinny spec (no IR)."""
    bm, bn, bk = skinny_block_m(spec), spec.block_n, spec.block_k
    kwv, mw, nw = spec.k_waves, spec.m_waves, spec.n_waves
    waves = kwv * mw * nw
    row_bytes = 2 * bk
    rows_dma = LDS_DMA_BYTES // row_bytes
    kw_len = spec.K // kwv
    sk = skinny_auto_split(spec) if spec.split_k == -1 and skinny_shared(spec) else spec.split_k
    k_steps = kw_len // bk
    steps = -(-k_steps // sk) if sk > 0 else 0  # per split (the longer count)
    stages = min(spec.stages, steps)
    stage_bytes = (bm + bn) * row_bytes  # per wave, or per workgroup when shared
    RG, MG = bn // (16 * nw), -(-bm // (16 * mw))
    dmas = (bm + bn) // rows_dma
    m_tiles = -(-spec.M // bm) if bm else 0
    return {
        "block_m": bm,
        "RG": RG,
        "MG": MG,
        "row_bytes": row_bytes,
        "rows_dma": rows_dma,
        "x_dmas": bm // rows_dma,
        "w_dmas": bn // rows_dma,
        "wave_dmas": dmas // waves if skinny_shared(spec) else dmas,
        "kw_len": kw_len,
        "k_steps": k_steps,
        "steps": steps,
        "stages": stages,
        "stage_bytes": stage_bytes,
        "lds": kwv * stages * stage_bytes,
        "waves": waves,
        "threads": 64 * waves,
        "m_tiles": m_tiles,
        "tiles": m_tiles * (spec.N // bn),
        "grid": m_tiles * (spec.N // bn) * sk,
    }


def valid_skinny(spec, lds_limit: int) -> tuple[bool, str]:
    if spec.m_waves < 1 or spec.n_waves < 1:
        return False, "skinny: m_waves, n_waves >= 1"
    if spec.xcd not in (0, 1):
        return False, "skinny: xcd in (0, 1)"
    if spec.refill not in (0, 1) or (spec.refill and not skinny_shared(spec)):
        return False, "skinny: refill in (0, 1), 1 only with the shared ring"
    if spec.m0fill not in (0, 1, 2, 3) or (spec.m0fill & 2 and skinny_shared(spec) and not spec.refill):
        return False, "skinny: m0fill in 0..3; 2 and 3 with the shared ring need refill=1"
    if spec.split_k < 1 or (spec.split_k > 1 and not skinny_shared(spec)):
        return False, "skinny: split_k >= 1, > 1 only with the shared ring"
    if spec.split_k > 1 and spec.sk_sync not in ("fence", "relaxed"):
        return False, "skinny: split-K syncs with sk_sync 'fence' or 'relaxed'"
    sk = spec.split_k
    if spec.sk_finish not in SPLIT_FINISH or (
        spec.sk_finish != "last" and (sk < 2 or sk & (sk - 1) or (spec.sk_finish == "shared" and spec.sk_sync != "relaxed"))
    ):
        return False, (
            f"skinny: sk_finish in {SPLIT_FINISH}; 'epoch' and 'shared' count the ticket across calls (a power-of-two "
            "split_k > 1), 'shared' also needs sk_sync='relaxed'"
        )
    ok, why = (_valid_shared if skinny_shared(spec) else _valid_private)(spec, lds_limit)
    if ok and spec.sk_finish == "shared" and skinny_plan(spec)["grid"] > spec.cus:
        return False, "skinny: sk_finish='shared' spins on its partners: a grid of at most cus workgroups"
    if ok and spec.xcd and skinny_plan(spec)["grid"] % spec.xccs:
        return False, "skinny: xcd=1 needs the grid a multiple of xccs"
    return ok, why


def _valid_private(spec, lds_limit: int) -> tuple[bool, str]:
    bm = skinny_block_m(spec)
    if bm == 0 or spec.block_m not in (0, *SKINNY_BLOCK_M) or spec.M > bm:
        return False, f"skinny: block_m in {SKINNY_BLOCK_M} (0: the smallest >= M), M <= block_m"
    if spec.block_n not in (16, 32, 48, 64) or spec.N % spec.block_n:
        return False, "skinny: block_n in (16, 32, 48, 64) dividing N"
    if spec.block_k not in SKINNY_BLOCK_K:
        return False, f"skinny: block_k in {SKINNY_BLOCK_K}"
    if not 1 <= spec.k_waves <= 16 or spec.K % (spec.k_waves * spec.block_k):
        return False, "skinny: k_waves in 1..16 and K a multiple of k_waves * block_k"
    if spec.stages < 1:
        return False, "skinny: stages >= 1"
    if spec.w_cache not in (0, 1, 2, 3):
        return False, "skinny: w_cache in 0..3 (LDS DMA cache bits)"
    p = skinny_plan(spec)
    if p["lds"] > lds_limit:
        return False, "skinny: LDS exceeds the workgroup limit"
    if p["RG"] * p["MG"] * 1024 > p["stage_bytes"]:
        return False, "skinny: a wave's partial tiles must fit its region of one ring slot"
    return True, "ok"


def _valid_shared(spec, lds_limit: int) -> tuple[bool, str]:
    mw, nw = spec.m_waves, spec.n_waves
    if spec.k_waves != 1 or mw * nw > 16:
        return False, "skinny: shared ring needs k_waves = 1 and m_waves * n_waves <= 16"
    if spec.block_m not in SHARED_BLOCK_M or spec.block_m % (16 * mw):
        return False, f"skinny: shared ring needs block_m in {SHARED_BLOCK_M}, a multiple of 16 * m_waves"
    if spec.block_n not in SHARED_BLOCK_N or spec.N % spec.block_n or spec.block_n % (16 * nw):
        return False, f"skinny: shared ring needs block_n in {SHARED_BLOCK_N} dividing N, a multiple of 16 * n_waves"
    if spec.block_k not in SKINNY_BLOCK_K or spec.K % spec.block_k:
        return False, f"skinny: block_k in {SKINNY_BLOCK_K} dividing K"
    if spec.w_cache not in (0, 1, 2, 3):
        return False, "skinny: w_cache in 0..3 (LDS DMA cache bits)"
    p = skinny_plan(spec)
    if p["stages"] < 2:
        return False, "skinny: shared ring needs stages >= 2 and K >= 2 block_k"
    if spec.split_k > p["k_steps"] // 2:
        return False, "skinny: split_k leaves every split at least 2 steps of block_k"
    if p["x_dmas"] % p["waves"] or p["w_dmas"] % p["waves"]:
        return False, "skinny: shared ring needs the X and W DMAs of a stage to split evenly over the waves"
    if p["lds"] > lds_limit:
        return False, "skinny: LDS exceeds the workgroup limit"
    return True, "ok"


def m0_groups(nd: int, stride: int, x_dmas: int) -> tuple[list[int], list[int]]:
    """m0fill=1: for each of a wave's ``nd`` DMAs (stage DMA ``j * stride``)
    the DMA whose M0 it uses and its immediate offset. A group stays within
    one operand (X DMAs are the first ``x_dmas`` stage DMAs) and within the
    immediate's reach."""
    head, imm = list(range(nd)), [0] * nd
    for j in range(1, nd):
        h = head[j - 1]
        gap = (j - h) * stride * LDS_DMA_BYTES
        if (j * stride < x_dmas) == (h * stride < x_dmas) and gap <= M0_SHARE_BYTES:
            head[j], imm[j] = h, gap
    return head, imm


def _xor_term(b, row, row_bytes):
    i = b.const_i32
    if row_bytes == 128:
        return b.land(b.lshr(row, i(1)), i(7))
    return b.land(row, i(15))


def build_skinny(b, spec, ptrs):
    p = skinny_plan(spec)
    i = b.const_i32
    M, N, K = spec.M, spec.N, spec.K
    bm, bn, bk = p["block_m"], spec.block_n, spec.block_k
    RG, MG, KWV = p["RG"], p["MG"], spec.k_waves
    rb, R = p["row_bytes"], p["rows_dma"]
    xd = p["x_dmas"]
    S, ST, sb = p["steps"], p["stages"], p["stage_bytes"]
    shared, WV = skinny_shared(spec), p["waves"]
    nd = p["wave_dmas"]
    ahead = ST - 1 if shared else ST  # stages in flight past the one computed
    b.kernel.attrs["max_workgroup_size"] = p["threads"]
    b.kernel.attrs["lds_alias_scopes"] = True  # LLVM then waits only for DMAs into the slot read

    SK = spec.split_k
    tile = b.block_id_x()
    X = spec.xccs
    if spec.xcd and X & (X - 1) == 0:
        tile = b.add(b.mul(b.land(tile, i(X - 1)), i(p["grid"] // X)), b.lshr(tile, i(X.bit_length() - 1)))
    elif spec.xcd:
        tile = b.add(b.mul(b.mod(tile, i(X)), i(p["grid"] // X)), b.div(tile, i(X)))
    split = None
    if SK > 1:
        ks = b.div(tile, i(p["tiles"]))
        tile = b.sub(tile, b.mul(ks, i(p["tiles"])))
        start = b.div(b.mul(ks, i(p["k_steps"])), i(SK))
        count = b.sub(b.div(b.mul(b.add(ks, i(1)), i(p["k_steps"])), i(SK)), start)
        split = {"ks": ks, "tile": tile, "k0": b.mul(start, i(bk * 2)), "count": count}
    if p["m_tiles"] > 1:
        mt = b.div(tile, i(N // bn))
        tile = b.sub(tile, b.mul(mt, i(N // bn)))
        rows_left = b.sub(i(M), b.mul(mt, i(bm)))
        x_ptr = b.global_ptr_add(ptrs["X"], b.mul(mt, i(bm * K * 2)))
        y_ptr = b.global_ptr_add(ptrs["Y"], b.mul(mt, i(bm * N * 2)))
        x_bytes, y_bytes = b.mul(rows_left, i(K * 2)), b.mul(rows_left, i(N * 2))
    else:
        x_ptr, y_ptr, x_bytes, y_bytes = ptrs["X"], ptrs["Y"], None, None
    if split is not None:
        split["m_base"] = b.mul(mt, i(bm)) if p["m_tiles"] > 1 else i(0)
        split["n_tile"] = tile
    w_tile = b.global_ptr_add(ptrs["W"], b.mul(tile, i(bn * K * 2)))
    rw = b.buffer_rsrc(w_tile, i(bn * K * 2))
    rx = b.buffer_rsrc(x_ptr, i(M * K * 2) if x_bytes is None else x_bytes)
    ry = b.buffer_rsrc(y_ptr, i(M * N * 2) if y_bytes is None else y_bytes)
    tid = b.thread_id_x()
    lane = b.land(tid, i(63))
    wave = b.readfirstlane(b.lshr(tid, i(6))) if WV > 1 else i(0)
    kbase2 = i(0) if shared else b.mul(b.mul(wave, i(p["kw_len"])), i(2))  # bytes into a row
    chunks = rb // 16
    d_row = b.lshr(lane, i(chunks.bit_length() - 1))
    d_chunk = b.land(lane, i(chunks - 1))

    slots = [
        b.smem_alloc(I32, [KWV * sb // 4], name_hint=f"dg_s{k}") for k in range(ST)
    ]
    bases = [b.smem_addr_of(s) for s in slots]
    # wave_off: the wave's region (reads); dma_off: where its first DMA lands
    wave_off = i(0) if shared else b.mul(wave, i(sb))
    dma_off = b.mul(wave, i(LDS_DMA_BYTES)) if shared else wave_off
    stride = WV if shared else 1  # stage DMA of the wave's DMA j: j * stride (+ wave)

    voff = []  # per DMA of the wave: X rows first, then W rows
    for j in range(nd):
        d = j * stride
        row = b.add(i(d * R), d_row)  # row of the stage
        if shared:
            row = b.add(row, b.mul(wave, i(R)))
        col = b.mul(b.xor(d_chunk, _xor_term(b, row, rb)), i(16))
        src = b.sub(row, i(bm)) if d >= xd else row
        voff.append(b.add(b.mul(src, i(K * 2)), b.add(kbase2, col)))
    head, imm = m0_groups(nd, stride, xd) if spec.m0fill & 1 else (list(range(nd)), [0] * nd)
    for j in range(nd):
        if imm[j]:  # the immediate also moves the global address: take it back out of the voffset
            voff[j] = b.sub(voff[j], i(imm[j]))
    voff_last = voff
    if split is not None and p["k_steps"] % SK:
        # the short splits load their last stage out of bounds (zeros)
        full = b.cmp_eq(split["count"], i(S))
        voff_last = [b.select(full, v, i(OOB)) for v in voff]

    def issue(t):
        lds = None
        for j in range(nd):
            d = j * stride
            if head[j] == j:
                lds = b.smem_ptr_add(bases[t % ST], b.sext(b.add(dma_off, i(d * LDS_DMA_BYTES)), I64))
            b.async_buffer_load_lds_addr(
                rx if d < xd else rw,
                lds,
                voff_last[j] if t == S - 1 else voff[j],
                i(t * bk * 2) if split is None else b.add(split["k0"], i(t * bk * 2)),
                dwords=4,
                coherency=0 if d < xd else spec.w_cache,
                offset=imm[j],
            )

    r16 = b.land(lane, i(15))
    g4 = b.lshr(lane, i(4))

    def operand_index(row0, ks, base=None):
        row = b.add(i(row0), r16)
        if base is not None:
            row = b.add(row, base)
        chunk = b.xor(b.add(g4, i(4 * ks)), _xor_term(b, row, rb))
        byte = b.add(wave_off, b.add(b.mul(row, i(rb)), b.mul(chunk, i(16))))
        return b.lshr(byte, i(2))

    if shared:
        wm = b.div(wave, i(spec.n_waves))
        wn = b.sub(wave, b.mul(wm, i(spec.n_waves)))
        m0, n0 = b.mul(wm, i(16 * MG)), b.mul(wn, i(16 * RG))
    else:
        m0 = n0 = None
    ksubs = bk // 32
    xi = [[operand_index(16 * mg, ks, m0) for mg in range(MG)] for ks in range(ksubs)]
    wi = [[operand_index(bm + 16 * rg, ks, n0) for rg in range(RG)] for ks in range(ksubs)]

    for t in range(ahead):
        issue(t)
    if spec.pin:
        b.sched_barrier(0)
    zero = b.vec_pack([b.const_f32(0.0)] * 4, F32)
    acc = [[zero] * MG for _ in range(RG)]
    for t in range(S):
        b.s_waitcnt(vmcnt=min(63, (min(t + ahead, S) - t - 1) * nd))  # stage t has landed
        nxt = t + (ahead if shared else ST)
        late = spec.m0fill & 2 and nxt < S  # the refill goes among the MFMAs
        if shared:
            # every wave's DMAs of stage t have landed and stage t - 1's reads are done
            b.sync_lds_only(fence="compiler")
            if t + ahead < S and not spec.refill:
                issue(t + ahead)
        slot = slots[t % ST]
        ops = []
        for ks in range(ksubs):
            x = [b.smem_load_vN(slot, xi[ks][mg], dtype=I32, n=4) for mg in range(MG)]
            w = [b.smem_load_vN(slot, wi[ks][rg], dtype=I32, n=4) for rg in range(RG)]
            ops.append((x, w))
        if spec.refill and t + ahead < S and not late:
            if spec.pin:
                b.sched_barrier(0)
            issue(t + ahead)
        if not shared and t + ST < S:
            b.s_waitcnt(lgkmcnt=0)  # every read of the slot is done before its refill
            if not late:
                issue(t + ST)
        if spec.pin:
            b.sched_barrier(0)
        if late:
            issue(nxt)
        if spec.prio == 1:
            b.s_setprio(1)
        for x, w in ops:
            for mg in range(MG):
                xb = b.bitcast(x[mg], VectorType(BF16, 8))
                for rg in range(RG):
                    acc[rg][mg] = b.mfma_f32_16x16x32_bf16(
                        b.bitcast(w[rg], VectorType(BF16, 8)), xb, acc[rg][mg]
                    )
        if spec.prio == 1:
            b.s_setprio(0)
        if late:
            # one MFMA ahead of each DMA fills the wait state after its M0 write
            mfmas = ksubs * MG * RG
            for _ in range(min(nd, mfmas)):
                b.sched_group_barrier(SGB_MFMA, 1)
                b.sched_group_barrier(SGB_VMEM_READ, 1)
            if nd > mfmas:
                b.sched_group_barrier(SGB_VMEM_READ, nd - mfmas)
            elif mfmas > nd:
                b.sched_group_barrier(SGB_MFMA, mfmas - nd)
        if spec.pin:
            b.sched_barrier(0)
    if split is not None:
        _split_epilogue(b, spec, p, ptrs, acc, split, tid, lane, m0, n0, ry)
        return
    if shared:
        for rg in range(RG):
            for mg in range(MG):
                _store_tile4(b, spec, ry, tile, rg, mg, lane, acc[rg][mg], m0, n0)
        return
    _skinny_epilogue(b, spec, p, acc, tile, tid, lane, wave, ry, slots[0])


def _store_tile4(b, spec, ry, tile, rg, mg, lane, vals, m0=None, n0=None):
    """Four consecutive outputs of lane ``lane`` of tile (rg, mg) as bf16
    (``m0`` / ``n0``: the wave's offset into the workgroup tile)."""
    i = b.const_i32
    m = b.add(i(16 * mg), b.land(lane, i(15)))
    n = b.add(b.mul(tile, i(spec.block_n)), b.add(i(16 * rg), b.mul(b.lshr(lane, i(4)), i(4))))
    if m0 is not None:
        m, n = b.add(m, m0), b.add(n, n0)
    voff = b.mul(b.add(b.mul(m, i(spec.N)), n), i(2))
    b.buffer_store_vN_bf16(ry, voff, i(0), b.vec_trunc_f32_to_bf16(vals), dwords=2)


def _split_epilogue(b, spec, p, ptrs, acc, split, tid, lane, m0, n0, ry):
    """split_k > 1: the wave's partial tiles into slot ``ks`` of the
    workspace (``[split][M][N]`` fp32, 16 B per lane and tile), the tile's
    ticket, and the last workgroup sums the slots, stores bf16 and resets
    the ticket."""
    i = b.const_i32
    M, N, SK = spec.M, spec.N, spec.split_k
    bm, bn, threads = p["block_m"], spec.block_n, p["threads"]
    relaxed = spec.sk_sync == "relaxed"
    aux = 16 if relaxed else 0  # sc1: device-coherent partials
    rws = b.buffer_rsrc(ptrs["WS"], i(SK * M * N * 4))
    m_base, n_base = split["m_base"], b.mul(split["n_tile"], i(bn))
    slot0 = b.mul(split["ks"], i(M * N))
    for rg in range(p["RG"]):
        for mg in range(p["MG"]):
            m = b.add(m_base, b.add(m0, b.add(i(16 * mg), b.land(lane, i(15)))))
            n = b.add(n_base, b.add(n0, b.add(i(16 * rg), b.mul(b.lshr(lane, i(4)), i(4)))))
            flat = b.add(slot0, b.add(b.mul(m, i(N)), n))
            voff = b.select(b.cmp_lt(m, i(M)), b.mul(flat, i(4)), i(OOB))
            b.buffer_store_cp(b.bitcast(acc[rg][mg], VectorType(I32, 4)), rws, voff, i(0), aux=aux)
    # every wave's partial writes are performed before the ticket is taken
    if relaxed:
        b.s_waitcnt(vmcnt=0)
    else:
        b.fence(scope="agent", ordering="release")
    per_row = bn // 4
    total = bm * per_row
    if spec.sk_finish == "shared":
        _split_shared_finish(b, spec, p, ptrs, split, tid, rws, ry, m_base, n_base, per_row, total)
        return
    ticket = b.smem_alloc(I32, [1], name_hint="dg_ticket")
    b.sync_lds_only(fence="compiler")
    with b.scf_if(b.cmp_eq(tid, i(0))):
        old = b.global_atomic_add(ptrs["CNT"], split["tile"], i(1), ordering="monotonic")
        b.smem_store_vN(ticket, [i(0)], old, 1)
    b.sync_lds_only(fence="compiler")
    last = b.readfirstlane(b.vec_extract(b.smem_load_vN(ticket, i(0), dtype=I32, n=1), 0))
    epoch = spec.sk_finish == "epoch"
    if epoch:  # the ticket counts up across calls: this call's last partner holds SK - 1 mod SK
        last = b.land(last, i(SK - 1))
    with b.scf_if(b.cmp_eq(last, i(SK - 1))):
        if not relaxed:
            b.fence(scope="agent", ordering="acquire")
        f4 = VectorType(F32, 4)
        for q in range(-(-total // threads)):
            e = b.add(tid, i(q * threads))
            ml = b.div(e, i(per_row))
            n = b.add(n_base, b.mul(b.sub(e, b.mul(ml, i(per_row))), i(4)))
            ok = b.cmp_lt(b.add(m_base, ml), i(M))
            if total % threads:
                ok = b.land(ok, b.cmp_lt(e, i(total)))
            flat = b.add(b.mul(b.add(m_base, ml), i(N)), n)
            voff4 = b.select(ok, b.mul(flat, i(4)), i(OOB))
            got = b.bitcast(b.buffer_load_cp(rws, voff4, i(0), dwords=4, aux=aux), f4)
            for s in range(1, SK):
                raw = b.buffer_load_cp(rws, voff4, i(s * M * N * 4), dwords=4, aux=aux)
                got = b.fadd(got, b.bitcast(raw, f4))
            voff2 = b.select(ok, b.mul(b.add(b.mul(ml, i(N)), n), i(2)), i(OOB))
            b.buffer_store_vN_bf16(ry, voff2, i(0), b.vec_trunc_f32_to_bf16(got), dwords=2)
        if not epoch:
            with b.scf_if(b.cmp_eq(tid, i(0))):
                b.global_store(ptrs["CNT"], split["tile"], i(0), align=4)


def _split_shared_finish(b, spec, p, ptrs, split, tid, rws, ry, m_base, n_base, per_row, total):
    """sk_finish="shared": take the tile's ticket (word 1 of its pair; it counts
    up across calls, so this call's partners bring it to the next multiple of
    split_k), spin until it gets there, then sum this partner's 1 / split_k
    of the tile over every slot and store it as bf16."""
    i = b.const_i32
    M, N, SK, threads = spec.M, spec.N, spec.split_k, p["threads"]
    with b.scf_if(b.cmp_eq(tid, i(0))):
        cnt = b.readfirstlane(b.global_ptr_to_i64(ptrs["CNT"]))
        voff = b.add(b.mul(split["tile"], i(8)), i(4))
        old = b.global_atomic_add_scoped(cnt, voff, i(1), scope="agent", returns=True)
        target = b.add(b.land(old, i(-SK)), i(SK))
        rcnt = b.buffer_rsrc(ptrs["CNT"], i(p["tiles"] * 8))
        b.tagged_poll(target, [(rcnt, b.mul(split["tile"], i(8)), 2)], aux=16, pause="s_sleep 1", reload="launder")
    b.sync_lds_only(fence="compiler")
    share = -(-total // SK)  # 4-column vectors of the tile this partner sums
    f4 = VectorType(F32, 4)
    for q in range(-(-share // threads)):
        e_loc = b.add(tid, i(q * threads))
        e = b.add(b.mul(split["ks"], i(share)), e_loc)
        ml = b.div(e, i(per_row))
        n = b.add(n_base, b.mul(b.sub(e, b.mul(ml, i(per_row))), i(4)))
        ok = b.land(b.cmp_lt(b.add(m_base, ml), i(M)), b.cmp_lt(e, i(total)))
        if share % threads:
            ok = b.land(ok, b.cmp_lt(e_loc, i(share)))
        flat = b.add(b.mul(b.add(m_base, ml), i(N)), n)
        voff4 = b.select(ok, b.mul(flat, i(4)), i(OOB))
        got = b.bitcast(b.buffer_load_cp(rws, voff4, i(0), dwords=4, aux=16), f4)
        for s in range(1, SK):
            raw = b.buffer_load_cp(rws, voff4, i(s * M * N * 4), dwords=4, aux=16)
            got = b.fadd(got, b.bitcast(raw, f4))
        voff2 = b.select(ok, b.mul(b.add(b.mul(ml, i(N)), n), i(2)), i(OOB))
        b.buffer_store_vN_bf16(ry, voff2, i(0), b.vec_trunc_f32_to_bf16(got), dwords=2)


def _skinny_epilogue(b, spec, p, acc, tile, tid, lane, wave, ry, slot):
    """k_waves == 1: stores from the accumulators. Otherwise each wave's
    partial tiles go as ``[rg][mg][lane]`` 16 B entries into its own region
    of ring slot 0 (its last DMA into it has landed and no other wave
    touches it, so no barrier comes first), one barrier, and thread ``e``
    sums entry ``e`` over the waves and stores it."""
    i = b.const_i32
    RG, MG, KWV = p["RG"], p["MG"], spec.k_waves
    if KWV == 1:
        for rg in range(RG):
            for mg in range(MG):
                _store_tile4(b, spec, ry, tile, rg, mg, lane, acc[rg][mg])
        return
    E = RG * MG * 64  # entries per wave
    wave_words = p["stage_bytes"] // 4
    for rg in range(RG):
        for mg in range(MG):
            idx = b.add(b.mul(wave, i(wave_words)), b.mul(b.add(i((rg * MG + mg) * 64), lane), i(4)))
            b.smem_store_vN(slot, [idx], b.bitcast(acc[rg][mg], VectorType(I32, 4)), 4)
    b.sync_lds_only(fence="compiler")
    threads = p["threads"]
    f4 = VectorType(F32, 4)
    for q in range(-(-E // threads)):
        e0 = q * threads

        def one(e0=e0):
            e = b.add(tid, i(e0))
            e4 = b.mul(e, i(4))
            got = b.bitcast(b.smem_load_vN(slot, e4, dtype=I32, n=4), f4)
            for k in range(1, KWV):
                raw = b.smem_load_vN(slot, b.add(e4, i(k * wave_words)), dtype=I32, n=4)
                got = b.fadd(got, b.bitcast(raw, f4))
            tm = b.lshr(e, i(6))  # rg * MG + mg
            rg = b.div(tm, i(MG)) if MG > 1 else tm
            mg = b.sub(tm, b.mul(rg, i(MG))) if MG > 1 else i(0)
            ln = b.land(e, i(63))
            m = b.add(b.mul(mg, i(16)), b.land(ln, i(15)))
            n = b.add(
                b.mul(tile, i(spec.block_n)),
                b.add(b.mul(rg, i(16)), b.mul(b.lshr(ln, i(4)), i(4))),
            )
            voff = b.mul(b.add(b.mul(m, i(spec.N)), n), i(2))
            b.buffer_store_vN_bf16(ry, voff, i(0), b.vec_trunc_f32_to_bf16(got), dwords=2)

        if e0 + threads > E:
            with b.scf_if(b.cmp_lt(b.add(tid, i(e0)), i(E))):
                one()
        else:
            one()
