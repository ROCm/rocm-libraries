"""Dense flash-attention prefill kernel for gfx950 (MI355X).

Productized from the ``flash_dense_dualwave_swp`` experiment
(``kernels/gfx950/experiments/flash_dense_dualwave_swp/``). This is the shippable
step-1 pipeline with every WINNING lever baked in as always-on (no env gates):

  * **CK-1 transposed PV** — P feeds the PV MFMA in its native QK-output layout via a
    half-local V load (``pv32_v_load_paired``); the cross-half P-relayout shuffle is
    gone (~96 ``ds_bpermute`` removed). +35% over the pre-CK-1 winner.
  * **LDS bank-conflict padding on K** (``[NBUF, BN, D+8]``) — kills the 8-way conflict
    on the QK K-reads. The dominant base win (+80% over the naive baseline).
  * **native exp2_fast** (``v_exp_f32``, no overflow guard — the softmax argument is
    always <= 0) — +11.5%.
  * **full-population ``sched_group_barrier`` template** naming DS_READ/MFMA/VALU/TRANS
    per PV step.
  * **diagonal-only causal masking** — a mask-free body loop over below-diagonal KV
    tiles (~94% at Sq=8192) plus a masked diagonal tail.
  * **depth-1 cluster split** fusing exp2 into the PV MFMA loop for MFMA/VALU co-exec.
  * **vectorized O store**.

Measured on MI355X: **521 TFLOPS @ Sq=8192, bf16, D=128, causal, 0 spill**, error
~1.46e-3 vs SDPA. Shape (batch/seqlen/heads/head_dim) is baked at build time (dense,
compile-time-sized ABI); only the KV tile and occupancy hint are tunable knobs.

Experimental/negative levers from the sweep (step-2 8-cluster, K-staging, per-nsub
staging, score truncation, s_setprio, PV V-prefetch, lazy rescale) are intentionally
NOT carried over — see the experiment's ``plan.md`` for their measured results.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from rocke.core.ir import IRBuilder, KernelDef, PtrType, BF16, F16, F32, I64
from rocke.helpers.attention import mfma_32x32x16_for_dtype, pv32_v_load_paired
from rocke.helpers.schedule import MFMA, VALU, TRANS, DS_READ
from rocke.helpers.spec import kernel_name_join
from kernels.gfx950.attention_tiled_2d import _mfma_32x32_c_row, _mfma_32x32_c_col

LOG2E = 1.4426950408889634
_DTYPE_IR = {"bf16": BF16, "fp16": F16}

# Baked pipeline constants (NOT tunable knobs — these are load-bearing):
#   _BLOCK_M: query rows per CTA. The causal mask + P relayout assume 256; the
#             kernel FAULTS at other values until those hardcodes are lifted.
#   num_waves = _BLOCK_M // 32 = 8 (block = 512 threads).
#   _NBUF=2 double-buffer (NBUF=3 is a measured dead end: 256 VGPR + 58 spills).
#   _LDS_PAD=8 bf16 elements of K-row padding (the +80% bank-conflict fix).
_BLOCK_M = 256
_NBUF = 2
_LDS_PAD = 8


@dataclass(frozen=True)
class AttentionDenseSpec:
    """Compile-time spec for the dense flash-attention prefill kernel.

    Functional fields (batch / seqlen / heads / head_size / causal / dtype) are baked
    into the kernel as constants — this is a dense, statically-sized ABI. ``block_n``
    and ``waves_per_eu`` are the only performance knobs; every algorithmic lever is
    always-on (see the module docstring).
    """

    # --- functional (compile-time shape) ---
    batch: int
    seqlen_q: int
    seqlen_kv: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    causal: bool = True
    dtype: str = "bf16"

    # --- validated performance knobs ---
    # block_n: KV tile length. 64 (66 KB LDS, WPE-tunable) and 128 (135 KB LDS, pins
    #   the 256-VGPR cap) both match ~peak; 64 is strictly more resource-efficient.
    block_n: int = 64
    # waves_per_eu: occupancy hint. 2 is a free win (tighter allocation, still 2
    #   waves/SIMD); 3 is a measured trap (VGPR<=170 forces spills -> -20%).
    waves_per_eu: int = 2
    # persistent: emit the grid-stride PERSISTENT variant instead of one CTA per
    #   (query-block, head, batch). A 1-D grid of ``num_persistent`` long-lived CTAs
    #   grid-strides over the W = (seqlen_q//256)*Hq*B work items, so the per-CTA
    #   launch/dispatch + scalar setup + K/V-prime cold-start (~4.5 tile-equivalents,
    #   plan.md "CAUSAL GAP = FIXED-COST AMORTIZATION") is paid once per CU instead of
    #   once per query-block. Inner compute is byte-identical to the default path.
    #   Measured MI355X Sq=8192 causal: 512 -> 853 TFLOPS (+70%), 0 spill, err 1.46e-3.
    persistent: bool = False
    # num_persistent: number of long-lived CTAs when ``persistent``. 256 = exactly one
    #   8-wave block per CU on MI355X (256 CUs) at 2 waves/SIMD; larger oversubscribes
    #   the CUs -> a serialized 2nd block -> tail loss (304 measured -20%).
    num_persistent: int = 256
    # interleave: boustrophedon query-block ordering that reverses qb on alternating
    #   (hq,bt) planes to spread the triangular causal load across CTAs. A large-Sq
    #   lever (helps Sq>=16384) that slightly hurts small Sq; only used when persistent.
    interleave: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_IR:
            raise ValueError(
                f"dtype must be one of {sorted(_DTYPE_IR)}, got {self.dtype}"
            )
        if self.head_size % 32 != 0:
            raise ValueError(
                f"head_size must be a multiple of 32, got {self.head_size}"
            )
        if self.block_n % 32 != 0:
            raise ValueError(f"block_n must be a multiple of 32, got {self.block_n}")
        if self.seqlen_q % _BLOCK_M != 0:
            raise ValueError(
                f"seqlen_q must be a multiple of {_BLOCK_M}, got {self.seqlen_q}"
            )
        if self.seqlen_kv % self.block_n != 0:
            raise ValueError(
                f"seqlen_kv must be a multiple of block_n={self.block_n}, got {self.seqlen_kv}"
            )
        if self.num_kv_heads == 0 or self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be a positive multiple "
                f"of num_kv_heads ({self.num_kv_heads})"
            )
        if self.block_n % 32 != 0 or self.block_n <= 0:
            raise ValueError(
                f"block_n must be a positive multiple of 32, got {self.block_n}"
            )
        if self.persistent and self.num_persistent <= 0:
            raise ValueError(
                f"num_persistent must be positive, got {self.num_persistent}"
            )

    @property
    def num_waves(self) -> int:
        return _BLOCK_M // 32

    @property
    def dtype_ir(self):
        return _DTYPE_IR[self.dtype]

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    def kernel_name(self) -> str:
        parts = [
            "rocke_attention_dense",
            f"d{self.head_size}",
            f"hq{self.num_query_heads}",
            f"kv{self.num_kv_heads}",
            f"bn{self.block_n}",
            self.dtype,
            f"sq{self.seqlen_q}",
            f"sk{self.seqlen_kv}",
            "causal" if self.causal else "full",
        ]
        if self.persistent:
            parts.append(f"persist{self.num_persistent}")
            if self.interleave:
                parts.append("intl")
        return kernel_name_join(*parts)


def supports_attention_dense(
    spec: AttentionDenseSpec, *, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return (ok, reason). The kernel is gfx950-only and dense (no paging/bias)."""
    if arch != "gfx950":
        return False, f"attention_dense is gfx950-only (got {arch})"
    try:
        AttentionDenseSpec(**{f.name: getattr(spec, f.name) for f in spec.__dataclass_fields__.values()})  # type: ignore[attr-defined]
    except ValueError as e:
        return False, str(e)
    return True, ""


def build_attention_dense(
    spec: AttentionDenseSpec, *, arch: str = "gfx950"
) -> KernelDef:
    """Emit the dense flash-attention prefill kernel described by ``spec``."""
    if arch != "gfx950":
        raise NotImplementedError(f"attention_dense is gfx950-only (got {arch})")

    if spec.persistent:
        return _build_attention_dense_persistent(spec)

    B = spec.batch
    Sq = spec.seqlen_q
    Skv = spec.seqlen_kv
    Hq = spec.num_query_heads
    Hkv = spec.num_kv_heads
    D = spec.head_size
    causal = spec.causal
    dtype = spec.dtype_ir

    BLOCK_M = _BLOCK_M
    WAVES = spec.num_waves
    BN = spec.block_n
    NBUF = _NBUF
    PAD = _LDS_PAD

    K_STEPS = D // 16
    D_TILES = D // 32
    N_SUB = BN // 32
    KK_STEPS = BN // 16
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)

    q = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k = b.param(
        "k_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    v = b.param(
        "v_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    o = b.param(
        "o_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    scale = b.param("scale", F32)
    qk_scale = b.fmul(scale, b.const_f32(LOG2E))

    _exp2 = b.exp2_fast  # native v_exp_f32 (softmax arg always <= 0)

    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    lane_h = b.div(lane, b.const_i32(32))
    d_base = b.mul(lane_h, b.const_i32(8))
    neg_inf = b.const_f32(-1e30)

    qb = b.block_id_x()
    hq = b.block_id_y()
    bt = b.block_id_z()
    hkv = b.div(hq, b.const_i32(gqa))
    q_tok0 = b.add(b.mul(qb, b.const_i32(BLOCK_M)), b.mul(wave, b.const_i32(32)))
    q_base = b.add(
        b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
        b.mul(hq, b.const_i32(D)),
    )
    k_base = b.add(
        b.mul(b.mul(bt, b.const_i32(Skv)), b.const_i32(stride_k_tok)),
        b.mul(hkv, b.const_i32(D)),
    )

    # --- LDS allocation: PAD on K (bank-conflict fix), unpadded V ---
    LDROW = D + PAD
    K_lds = b.smem_alloc(dtype, [NBUF, BN, LDROW], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [NBUF, BN, D], name_hint="Vlds")

    # Q packs (B operand), scaled once by qk_scale = softmax_scale * log2(e).
    q_tok = b.add(q_tok0, lane_m)
    q_packs = []
    for ks in range(K_STEPS):
        col = b.add(b.const_i32(ks * 16), d_base)
        addr = b.add(b.add(q_base, b.mul(q_tok, b.const_i32(stride_q_tok))), col)
        raw = b.global_load_vN(q, addr, dtype, 8, align=16)
        elems = [
            b.cast_f32_to(b.fmul(b.cast_to_f32(b.vec_extract(raw, j)), qk_scale), dtype)
            for j in range(8)
        ]
        q_packs.append(b.vec_pack(elems, dtype))

    n_ktiles = Skv // BN
    n_per = BLOCK_M // BN

    K_BYTES_PER_BUF = BN * LDROW * 2
    K_LDROW_BYTES = LDROW * 2
    V_BYTES_PER_BUF = BN * D * 2
    ROWS_PER_WAVE = BN // WAVES
    WAVE_BYTES = 64 * 16
    V_DMA_PASSES = (BN * D) // (WAVES * 64 * 8)
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    k_rsrc = b.buffer_rsrc(k, b.const_i32(B * Skv * Hkv * D * 2))
    v_rsrc = b.buffer_rsrc(v, b.const_i32(B * Skv * Hkv * D * 2))
    v_wave_off_i64 = b.zext(b.to_sgpr_u32(b.mul(wave, b.const_i32(WAVE_BYTES))), I64)

    def async_load_k(lds_base, buf_val, tile_key0):
        """Row-by-row async DMA into padded [BN, LDROW] K_lds layout."""
        buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(K_BYTES_PER_BUF))
        for r in range(ROWS_PER_WAVE):
            row = b.add(b.mul(wave, b.const_i32(ROWS_PER_WAVE)), b.const_i32(r))
            row_lds_off = b.add(
                buf_off, b.zext(b.mul(row, b.const_i32(K_LDROW_BYTES)), I64)
            )
            row_base = b.smem_ptr_add(lds_base, row_lds_off)
            gkey = b.add(tile_key0, row)
            gcol = b.mul(lane, b.const_i32(2))
            voff = b.add(b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), gcol)
            b.async_buffer_load_lds_addr(
                k_rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
            )

    def async_load_v(lds_base, buf_val, tile_key0):
        """Contiguous async DMA into unpadded [BN, D] V_lds layout."""
        buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(V_BYTES_PER_BUF))
        base_off = b.add(v_wave_off_i64, buf_off)
        for c in range(V_DMA_PASSES):
            wave_base = b.smem_ptr_add(
                lds_base, b.add(base_off, b.const_i64(c * WAVES * WAVE_BYTES))
            )
            flat = b.mul(
                b.add(b.mul(b.const_i32(c), b.const_i32(WAVES * 64)), tid),
                b.const_i32(8),
            )
            krow = b.div(flat, b.const_i32(D))
            kcol = b.mod(flat, b.const_i32(D))
            gkey = b.add(tile_key0, krow)
            voff = b.add(b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), kcol)
            b.async_buffer_load_lds_addr(
                v_rsrc, wave_base, b.mul(voff, b.const_i32(2)), zero_soff, 4
            )

    def load_tile(buf_val, tile_idx):
        tk0 = b.mul(tile_idx, b.const_i32(BN))
        async_load_k(K_lds_addr, buf_val, tk0)
        async_load_v(V_lds_addr, buf_val, tk0)

    # ---- per-tile compute closures ----

    def do_qk(kbuf):
        """QK MFMA: S^T = K@Q^T. mfma(a=K, bv=Q) => key on the 16 per-lane accumulator
        regs (+lane^32), query on lane%32 -- the layout that keeps softmax a cheap
        in-lane reduce + one lane^32 exchange, and lets CK-1's transposed PV consume P
        with no relayout shuffle."""
        s_reg = []
        for nsub in range(N_SUB):
            acc = b.zero_vec_f32(16)
            krow = b.add(b.const_i32(nsub * 32), lane_m)
            for ks in range(K_STEPS):
                col = b.add(b.const_i32(ks * 16), d_base)
                k_pack = b.smem_load_vN(K_lds, kbuf, krow, col, dtype=dtype, n=8)
                acc = mfma_32x32x16_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
            s_reg.append([b.vec_extract(acc, i) for i in range(16)])
        return s_reg

    def do_mask(s_reg, tile_idx):
        if not causal:
            return
        tile_key0 = b.mul(tile_idx, b.const_i32(BN))
        query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        for nsub in range(N_SUB):
            sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
            for i in range(16):
                ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                s_reg[nsub][i] = b.select(
                    b.cmp_le(ktok, query_tok), s_reg[nsub][i], neg_inf
                )

    def softmax_max(s_reg, m_i):
        local_max = neg_inf
        for nsub in range(N_SUB):
            for i in range(16):
                local_max = b.fmax(local_max, s_reg[nsub][i])
        tile_max = b.fmax(local_max, b.warp_shuffle_xor(local_max, 32))
        m_new = b.fmax(m_i, tile_max)
        alpha = _exp2(b.fsub(m_i, m_new))
        return m_new, alpha

    def relayout_p(p):
        """CK-1 half-local P feed: assemble the PV B-operand from lane-local P regs
        only (a bf16 cast + pack, NO cross-half warp_shuffle_xor/select). Pairs with
        the half-local V load in ``read_v`` so the K-axis stays aligned."""
        packs = []
        for kk_step in range(KK_STEPS):
            elems = []
            for kk in range(8):
                local_in_group = kk % 4
                band = kk // 4
                key_idx = kk_step * 16 + band * 8 + local_in_group
                p_tile = key_idx // 32
                row_static = key_idx % 32
                preg = (row_static // 8) * 4 + (row_static % 4)
                elems.append(b.cast_f32_to(p[p_tile][preg], dtype))
            packs.append(b.vec_pack(elems, dtype))
        return packs

    def read_v(dt, kk_step, vbuf):
        """CK-1 half-local transposed V A-operand (matches ``relayout_p``)."""
        return pv32_v_load_paired(
            b,
            V_lds=V_lds,
            v_buf=vbuf,
            n=dt,
            k=kk_step,
            lane_half32=lane_h,
            lane_col32=lane_m,
            dtype=dtype,
        )

    def do_pv(o_acc_in, p_packs, vbuf):
        out = []
        for dt in range(D_TILES):
            acc_o = o_acc_in[dt]
            for kk_step in range(KK_STEPS):
                acc_o = mfma_32x32x16_for_dtype(
                    b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], acc_o
                )
            out.append(acc_o)
        return out

    def rescale_o(o_acc, alpha):
        return [
            b.vec_pack(
                [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)], F32
            )
            for dt in range(D_TILES)
        ]

    def pv_fused_exp(o_acc_in, p_packs, vbuf, s_reg, m_new):
        """Depth-1 cluster: interleave exp2(s - m_new) into the PV MFMA loop so the
        softmax VALU/TRANS co-executes in the MFMA shadow. The full per-step
        instruction population (DS_READ/MFMA/VALU/TRANS) is named to sched_group_barrier
        so the IGLP grouping matches the real stream."""
        exp_per = -(-(N_SUB * 16) // (D_TILES * KK_STEPS))
        slots = [(nsub, i) for nsub in range(N_SUB) for i in range(16)]
        p_vals = [[None] * 16 for _ in range(N_SUB)]
        it = iter(slots)
        out = []
        for dt in range(D_TILES):
            acc_o = o_acc_in[dt]
            for kk_step in range(KK_STEPS):
                acc_o = mfma_32x32x16_for_dtype(
                    b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], acc_o
                )
                n_emit = 0
                for _ in range(exp_per):
                    slot = next(it, None)
                    if slot is None:
                        break
                    nsub, i = slot
                    p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
                    n_emit += 1
                b.sched_group_barrier(DS_READ, 2, 0)
                b.sched_group_barrier(MFMA, 1, 0)
                b.sched_group_barrier(VALU, max(1, n_emit), 0)
                b.sched_group_barrier(TRANS, max(1, n_emit), 0)
            out.append(acc_o)
        for slot in it:
            nsub, i = slot
            p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
        l_local = b.const_f32(0.0)
        for nsub in range(N_SUB):
            for i in range(16):
                l_local = b.fadd(l_local, p_vals[nsub][i])
        l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
        return out, p_vals, l_tile

    if causal:
        n_upper = b.add(b.mul(qb, b.const_i32(n_per)), b.const_i32(n_per))
        n_upper = b.select(
            b.cmp_lt(n_upper, b.const_i32(n_ktiles)), n_upper, b.const_i32(n_ktiles)
        )
    else:
        n_upper = b.const_i32(n_ktiles)

    # Prologue: prime the K/V double buffer and compute tile 0.
    load_tile(b.const_i32(0), b.const_i32(0))
    load_tile(b.const_i32(1), b.const_i32(1))
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare()
    s0 = do_qk(b.const_i32(0))
    do_mask(s0, b.const_i32(0))
    m0, _alpha0 = softmax_max(s0, neg_inf)
    # tile-0 softmax exp + relayout only; PV lags by one tile (fused into the loop).
    p0_vals = [
        [_exp2(b.fsub(s0[nsub][i], m0)) for i in range(16)] for nsub in range(N_SUB)
    ]
    l0_local = b.const_f32(0.0)
    for nsub in range(N_SUB):
        for i in range(16):
            l0_local = b.fadd(l0_local, p0_vals[nsub][i])
    l0 = b.fadd(l0_local, b.warp_shuffle_xor(l0_local, 32))
    o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
    pk0 = relayout_p(p0_vals)

    iter_args = (
        [("m", m0), ("l", l0)]
        + [(f"o{dt}", o0[dt]) for dt in range(D_TILES)]
        + [(f"pk{kk}", pk0[kk]) for kk in range(KK_STEPS)]
    )

    def emit_loop_body(j, carry, masked):
        m_i = carry[0]
        l_i = carry[1]
        o_acc = list(carry[2 : 2 + D_TILES])
        p_prev = list(carry[2 + D_TILES : 2 + D_TILES + KK_STEPS])
        kbuf = b.mod(j, b.const_i32(NBUF))
        vbuf_prev = b.mod(b.add(j, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
        pbuf = b.mod(b.add(j, b.const_i32(1)), b.const_i32(NBUF))

        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        s = do_qk(kbuf)
        if masked:
            do_mask(s, j)
        m_new, alpha = softmax_max(s, m_i)
        b.sched_barrier(0)  # depth-1 fence: m_new region-live-in
        o_acc, p_vals, l_tile = pv_fused_exp(o_acc, p_prev, vbuf_prev, s, m_new)
        l_new = b.fadd(b.fmul(l_i, alpha), l_tile)
        o_acc = rescale_o(o_acc, alpha)
        p_packs = relayout_p(p_vals)
        b.s_barrier_bare()
        load_tile(pbuf, b.add(j, b.const_i32(1)))
        b.scf_yield(m_new, l_new, *o_acc, *p_packs)

    if causal:
        # Diagonal-only masking: below-diagonal tiles need no mask (~94% at Sq=8192).
        diag_start = b.mul(qb, b.const_i32(n_per))
        body_upper = b.select(b.cmp_lt(diag_start, n_upper), diag_start, n_upper)
        body = b.scf_for_iter(
            b.const_i32(1), body_upper, b.const_i32(1), iter_args, iv_name="nb"
        )
        with body as (j, carry):
            emit_loop_body(j, carry, masked=False)
        tail_args = [
            (name + "_t", val) for (name, _), val in zip(iter_args, body.results)
        ]
        tail_lo = b.select(
            b.cmp_lt(diag_start, b.const_i32(1)), b.const_i32(1), diag_start
        )
        loop = b.scf_for_iter(tail_lo, n_upper, b.const_i32(1), tail_args, iv_name="nt")
        with loop as (j, carry):
            emit_loop_body(j, carry, masked=True)
    else:
        loop = b.scf_for_iter(
            b.const_i32(1), n_upper, b.const_i32(1), iter_args, iv_name="nkt"
        )
        with loop as (j, carry):
            emit_loop_body(j, carry, masked=False)

    res = loop.results
    l_i = res[1]
    o_acc = list(res[2 : 2 + D_TILES])
    p_prev = list(res[2 + D_TILES : 2 + D_TILES + KK_STEPS])

    last_vbuf = b.mod(b.add(n_upper, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
    o_acc = do_pv(o_acc, p_prev, last_vbuf)

    # Epilogue: O = (P@V) / l, vectorized bf16 store.
    rcp_l = b.rcp(l_i)
    o_base = b.add(
        b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
        b.mul(hq, b.const_i32(D)),
    )
    qtok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
    q_row_byte = b.add(o_base, b.mul(qtok, b.const_i32(stride_q_tok)))
    d_half = b.mul(lane_h, b.const_i32(4))
    for dt in range(D_TILES):
        for g in range(4):
            d0 = b.add(b.const_i32(dt * 32 + g * 8), d_half)
            addr = b.add(q_row_byte, d0)
            vals = [
                b.cast_f32_to(
                    b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l), dtype
                )
                for kk in range(4)
            ]
            b.global_store_vN(o, addr, b.vec_pack(vals, dtype), 4, align=8)
    b.ret()
    return b.kernel


def _build_attention_dense_persistent(spec: AttentionDenseSpec) -> KernelDef:
    """Persistent (grid-stride) variant of the dense flash-attention kernel.

    Launches a 1-D grid of ``spec.num_persistent`` long-lived CTAs; each CTA
    grid-strides over the flattened work-item space ``W = (Sq//BLOCK_M)*Hq*B`` and
    runs the byte-identical inner step-1 CK-1 pipeline per work item, so the per-CTA
    launch/dispatch + scalar setup + K/V-prime cold-start is amortized once per CU
    instead of once per query-block (see the ``persistent`` spec field). Every
    algorithmic lever is the same always-on set as the default build; the only
    differences are the outer work loop, the qb-major work decode (load-balances the
    causal triangle), the per-work-item state reset, and ``exp_per=1`` (keeps the
    extra loop-carried index math within 256 VGPR at 0 spill; numerically identical
    to the default's ``exp_per=2`` — pure emission ordering)."""
    B = spec.batch
    Sq = spec.seqlen_q
    Skv = spec.seqlen_kv
    Hq = spec.num_query_heads
    Hkv = spec.num_kv_heads
    D = spec.head_size
    causal = spec.causal
    dtype = spec.dtype_ir

    BLOCK_M = _BLOCK_M
    WAVES = spec.num_waves
    BN = spec.block_n
    NBUF = _NBUF
    PAD = _LDS_PAD
    NP = spec.num_persistent
    INTERLEAVE = spec.interleave

    K_STEPS = D // 16
    D_TILES = D // 32
    N_SUB = BN // 32
    KK_STEPS = BN // 16
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D
    n_ktiles = Skv // BN
    n_per = BLOCK_M // BN
    NQB = Sq // BLOCK_M
    W = NQB * Hq * B  # total work items

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)

    q = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k = b.param(
        "k_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    v = b.param(
        "v_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    o = b.param(
        "o_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    scale = b.param("scale", F32)
    qk_scale = b.fmul(scale, b.const_f32(LOG2E))
    _exp2 = b.exp2_fast

    # ----- CTA-invariant scalar setup (paid ONCE per persistent CTA) -----
    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    lane_h = b.div(lane, b.const_i32(32))
    d_base = b.mul(lane_h, b.const_i32(8))
    neg_inf = b.const_f32(-1e30)

    LDROW = D + PAD
    K_lds = b.smem_alloc(dtype, [NBUF, BN, LDROW], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [NBUF, BN, D], name_hint="Vlds")

    K_BYTES_PER_BUF = BN * LDROW * 2
    K_LDROW_BYTES = LDROW * 2
    V_BYTES_PER_BUF = BN * D * 2
    ROWS_PER_WAVE = BN // WAVES
    WAVE_BYTES = 64 * 16
    V_DMA_PASSES = (BN * D) // (WAVES * 64 * 8)
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    k_rsrc = b.buffer_rsrc(k, b.const_i32(B * Skv * Hkv * D * 2))
    v_rsrc = b.buffer_rsrc(v, b.const_i32(B * Skv * Hkv * D * 2))
    v_wave_off_i64 = b.zext(b.to_sgpr_u32(b.mul(wave, b.const_i32(WAVE_BYTES))), I64)

    # ----- persistent grid-stride loop over the flattened work-item space -----
    cta_id = b.block_id_x()
    outer = b.scf_for(cta_id, b.const_i32(W), b.const_i32(NP), iv_name="wi")
    with outer as wi:
        # Cross-work-item LDS reuse safety: drain the previous item's trailing DMA
        # and barrier so all waves finished the previous epilogue reads before we
        # reissue into the shared K/V buffers.
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()

        # qb-MAJOR decode: wi = qb*(Hq*B) + hq*B + bt. Putting qb (the triangular
        # causal cost index) in the MSB spreads cheap+expensive query blocks across
        # each CTA under grid-stride; a qb-fast decode would alias qb to a constant
        # per CTA when NP is a multiple of NQB (32x imbalance).
        bt = b.mod(wi, b.const_i32(B))
        rem = b.div(wi, b.const_i32(B))
        hq = b.mod(rem, b.const_i32(Hq))
        qb0 = b.div(rem, b.const_i32(Hq))
        if INTERLEAVE and causal and NQB > 1:
            odd = b.cmp_eq(b.mod(rem, b.const_i32(2)), b.const_i32(1))
            qb = b.select(odd, b.sub(b.const_i32(NQB - 1), qb0), qb0)
        else:
            qb = qb0
        hkv = b.div(hq, b.const_i32(gqa))

        q_tok0 = b.add(b.mul(qb, b.const_i32(BLOCK_M)), b.mul(wave, b.const_i32(32)))
        q_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq, b.const_i32(D)),
        )
        k_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Skv)), b.const_i32(stride_k_tok)),
            b.mul(hkv, b.const_i32(D)),
        )

        q_tok = b.add(q_tok0, lane_m)
        q_packs = []
        for ks in range(K_STEPS):
            col = b.add(b.const_i32(ks * 16), d_base)
            addr = b.add(b.add(q_base, b.mul(q_tok, b.const_i32(stride_q_tok))), col)
            raw = b.global_load_vN(q, addr, dtype, 8, align=16)
            elems = [
                b.cast_f32_to(
                    b.fmul(b.cast_to_f32(b.vec_extract(raw, j)), qk_scale), dtype
                )
                for j in range(8)
            ]
            q_packs.append(b.vec_pack(elems, dtype))

        def async_load_k(lds_base, buf_val, tile_key0):
            buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(K_BYTES_PER_BUF))
            row0 = b.mul(wave, b.const_i32(ROWS_PER_WAVE))
            row_lds_off = b.add(
                buf_off, b.zext(b.mul(row0, b.const_i32(K_LDROW_BYTES)), I64)
            )
            gcol = b.mul(lane, b.const_i32(2))
            voff = b.add(
                b.add(
                    k_base,
                    b.mul(b.add(tile_key0, row0), b.const_i32(stride_k_tok)),
                ),
                gcol,
            )
            for r in range(ROWS_PER_WAVE):
                row_base = b.smem_ptr_add(lds_base, row_lds_off)
                b.async_buffer_load_lds_addr(
                    k_rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                )
                if r + 1 < ROWS_PER_WAVE:
                    row_lds_off = b.add(row_lds_off, b.const_i64(K_LDROW_BYTES))
                    voff = b.add(voff, b.const_i32(stride_k_tok))

        def async_load_v(lds_base, buf_val, tile_key0):
            buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(V_BYTES_PER_BUF))
            base_off = b.add(v_wave_off_i64, buf_off)
            for c in range(V_DMA_PASSES):
                wave_base = b.smem_ptr_add(
                    lds_base, b.add(base_off, b.const_i64(c * WAVES * WAVE_BYTES))
                )
                flat = b.mul(
                    b.add(b.mul(b.const_i32(c), b.const_i32(WAVES * 64)), tid),
                    b.const_i32(8),
                )
                krow = b.div(flat, b.const_i32(D))
                kcol = b.mod(flat, b.const_i32(D))
                gkey = b.add(tile_key0, krow)
                voff = b.add(
                    b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), kcol
                )
                b.async_buffer_load_lds_addr(
                    v_rsrc, wave_base, b.mul(voff, b.const_i32(2)), zero_soff, 4
                )

        def load_tile(buf_val, tile_idx):
            tk0 = b.mul(tile_idx, b.const_i32(BN))
            async_load_k(K_lds_addr, buf_val, tk0)
            async_load_v(V_lds_addr, buf_val, tk0)

        def do_qk(kbuf):
            s_reg = []
            for nsub in range(N_SUB):
                acc = b.zero_vec_f32(16)
                krow = b.add(b.const_i32(nsub * 32), lane_m)
                for ks in range(K_STEPS):
                    col = b.add(b.const_i32(ks * 16), d_base)
                    k_pack = b.smem_load_vN(K_lds, kbuf, krow, col, dtype=dtype, n=8)
                    acc = mfma_32x32x16_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                s_reg.append([b.vec_extract(acc, i) for i in range(16)])
            return s_reg

        def do_mask(s_reg, tile_idx):
            if not causal:
                return
            tile_key0 = b.mul(tile_idx, b.const_i32(BN))
            query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
            for nsub in range(N_SUB):
                sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
                for i in range(16):
                    ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                    s_reg[nsub][i] = b.select(
                        b.cmp_le(ktok, query_tok), s_reg[nsub][i], neg_inf
                    )

        def softmax_max(s_reg, m_i):
            local_max = neg_inf
            for nsub in range(N_SUB):
                for i in range(16):
                    local_max = b.fmax(local_max, s_reg[nsub][i])
            tile_max = b.fmax(local_max, b.warp_shuffle_xor(local_max, 32))
            m_new = b.fmax(m_i, tile_max)
            alpha = _exp2(b.fsub(m_i, m_new))
            return m_new, alpha

        def softmax_stats(s_reg, m_i):
            m_new, alpha = softmax_max(s_reg, m_i)
            p = [
                [_exp2(b.fsub(s_reg[nsub][i], m_new)) for i in range(16)]
                for nsub in range(N_SUB)
            ]
            l_local = b.const_f32(0.0)
            for nsub in range(N_SUB):
                for i in range(16):
                    l_local = b.fadd(l_local, p[nsub][i])
            l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            return m_new, alpha, p, l_tile

        def relayout_p(p):
            packs = []
            for kk_step in range(KK_STEPS):
                elems = []
                for kk in range(8):
                    local_in_group = kk % 4
                    band = kk // 4
                    key_idx = kk_step * 16 + band * 8 + local_in_group
                    p_tile = key_idx // 32
                    row_static = key_idx % 32
                    preg = (row_static // 8) * 4 + (row_static % 4)
                    elems.append(b.cast_f32_to(p[p_tile][preg], dtype))
                packs.append(b.vec_pack(elems, dtype))
            return packs

        def read_v(dt, kk_step, vbuf):
            return pv32_v_load_paired(
                b,
                V_lds=V_lds,
                v_buf=vbuf,
                n=dt,
                k=kk_step,
                lane_half32=lane_h,
                lane_col32=lane_m,
                dtype=dtype,
            )

        def do_pv(o_acc_in, p_packs, vbuf):
            out = []
            for dt in range(D_TILES):
                acc_o = o_acc_in[dt]
                for kk_step in range(KK_STEPS):
                    acc_o = mfma_32x32x16_for_dtype(
                        b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], acc_o
                    )
                out.append(acc_o)
            return out

        def rescale_o(o_acc, alpha):
            return [
                b.vec_pack(
                    [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)],
                    F32,
                )
                for dt in range(D_TILES)
            ]

        def pv_fused_exp(o_acc_in, p_packs, vbuf, s_reg, m_new):
            exp_per = (
                1  # one exp2 per PV-MFMA step -> 256 VGPR / 0 spill (see docstring)
            )
            slots = [(nsub, i) for nsub in range(N_SUB) for i in range(16)]
            p_vals = [[None] * 16 for _ in range(N_SUB)]
            it = iter(slots)
            out = []
            for dt in range(D_TILES):
                acc_o = o_acc_in[dt]
                for kk_step in range(KK_STEPS):
                    acc_o = mfma_32x32x16_for_dtype(
                        b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], acc_o
                    )
                    n_emit = 0
                    for _ in range(exp_per):
                        slot = next(it, None)
                        if slot is None:
                            break
                        nsub, i = slot
                        p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
                        n_emit += 1
                    b.sched_group_barrier(DS_READ, 2, 0)
                    b.sched_group_barrier(MFMA, 1, 0)
                    b.sched_group_barrier(VALU, max(1, n_emit), 0)
                    b.sched_group_barrier(TRANS, max(1, n_emit), 0)
                out.append(acc_o)
            for slot in it:
                nsub, i = slot
                p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
            l_local = b.const_f32(0.0)
            for nsub in range(N_SUB):
                for i in range(16):
                    l_local = b.fadd(l_local, p_vals[nsub][i])
            l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            return out, p_vals, l_tile

        def emit_loop_body(j, carry, masked):
            m_i = carry[0]
            l_i = carry[1]
            o_acc = list(carry[2 : 2 + D_TILES])
            p_prev = list(carry[2 + D_TILES : 2 + D_TILES + KK_STEPS])
            pbuf = b.mod(b.add(j, b.const_i32(1)), b.const_i32(NBUF))
            kbuf = b.mod(j, b.const_i32(NBUF))
            vbuf_prev = b.mod(b.add(j, b.const_i32(NBUF - 1)), b.const_i32(NBUF))

            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare()
            s = do_qk(kbuf)
            if masked:
                do_mask(s, j)
            m_new, alpha = softmax_max(s, m_i)
            b.sched_barrier(0)
            o_acc, p_vals, l_tile = pv_fused_exp(o_acc, p_prev, vbuf_prev, s, m_new)
            l_new = b.fadd(b.fmul(l_i, alpha), l_tile)
            o_acc = rescale_o(o_acc, alpha)
            p_packs = relayout_p(p_vals)
            b.s_barrier_bare()
            load_tile(pbuf, b.add(j, b.const_i32(1)))
            b.scf_yield(m_new, l_new, *o_acc, *p_packs)

        if causal:
            n_upper = b.add(b.mul(qb, b.const_i32(n_per)), b.const_i32(n_per))
            n_upper = b.select(
                b.cmp_lt(n_upper, b.const_i32(n_ktiles)),
                n_upper,
                b.const_i32(n_ktiles),
            )
        else:
            n_upper = b.const_i32(n_ktiles)

        load_tile(b.const_i32(0), b.const_i32(0))
        load_tile(b.const_i32(1), b.const_i32(1))
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        s0 = do_qk(b.const_i32(0))
        do_mask(s0, b.const_i32(0))
        m0, _alpha0, p0, l0 = softmax_stats(s0, neg_inf)
        o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
        pk0 = relayout_p(p0)

        iter_args = (
            [("m", m0), ("l", l0)]
            + [(f"o{dt}", o0[dt]) for dt in range(D_TILES)]
            + [(f"pk{kk}", pk0[kk]) for kk in range(KK_STEPS)]
        )

        if causal:
            diag_start = b.mul(qb, b.const_i32(n_per))
            body_upper = b.select(b.cmp_lt(diag_start, n_upper), diag_start, n_upper)
            body = b.scf_for_iter(
                b.const_i32(1), body_upper, b.const_i32(1), iter_args, iv_name="nb"
            )
            with body as (j, carry):
                emit_loop_body(j, carry, masked=False)
            tail_args = [
                (name + "_t", val) for (name, _), val in zip(iter_args, body.results)
            ]
            tail_lo = b.select(
                b.cmp_lt(diag_start, b.const_i32(1)), b.const_i32(1), diag_start
            )
            loop = b.scf_for_iter(
                tail_lo, n_upper, b.const_i32(1), tail_args, iv_name="nt"
            )
            with loop as (j, carry):
                emit_loop_body(j, carry, masked=True)
        else:
            loop = b.scf_for_iter(
                b.const_i32(1), n_upper, b.const_i32(1), iter_args, iv_name="nkt"
            )
            with loop as (j, carry):
                emit_loop_body(j, carry, masked=False)

        res = loop.results
        l_i = res[1]
        o_acc = list(res[2 : 2 + D_TILES])
        p_prev = list(res[2 + D_TILES : 2 + D_TILES + KK_STEPS])

        last_vbuf = b.mod(b.add(n_upper, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
        o_acc = do_pv(o_acc, p_prev, last_vbuf)

        # Epilogue: recompute (bt, hq) from the live loop IV so they need not cross
        # the KV loop (keeps the loop-carried live set minimal -> 0 spill).
        rcp_l = b.rcp(l_i)
        bt_e = b.mod(wi, b.const_i32(B))
        hq_e = b.mod(b.div(wi, b.const_i32(B)), b.const_i32(Hq))
        o_base = b.add(
            b.mul(b.mul(bt_e, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq_e, b.const_i32(D)),
        )
        qtok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        q_row_byte = b.add(o_base, b.mul(qtok, b.const_i32(stride_q_tok)))
        d_half = b.mul(lane_h, b.const_i32(4))
        for dt in range(D_TILES):
            for g in range(4):
                d0 = b.add(b.const_i32(dt * 32 + g * 8), d_half)
                addr = b.add(q_row_byte, d0)
                vals = [
                    b.cast_f32_to(
                        b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l), dtype
                    )
                    for kk in range(4)
                ]
                b.global_store_vN(o, addr, b.vec_pack(vals, dtype), 4, align=8)

    b.ret()
    return b.kernel
