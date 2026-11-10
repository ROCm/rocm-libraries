# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Ported from: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   path:     aiter/ops/flydsl/kernels/gdr_decode.py (create_vk_gdr_decode_kernel)
#   compiler: flydsl 0.3.4 (pip wheel)
#
# Modifications:
#   1. Written against the flydsl 0.3.4 API: global memory goes through
#      kernels.common.buffer_ops buffer resources built from the operands' raw
#      device pointers (every operand is a pointer kernel argument), not
#      upstream's gdr_common copy-atom views.
#   2. Token-packed with cu_seqlens (hipDNN gdn_attributes.fbs) instead of a
#      dense [B, seq_length] batch: one workgroup runs one (sequence, output head,
#      V slice) and walks the sequence's tokens [cu[n], cu[n+1]) in a runtime
#      scf.for that carries the state slice in registers. seq_length is no longer
#      compile-time, so the same object serves decode (one token per sequence)
#      and prefill (a recurrent prefill, not upstream's chunked one).
#   3. Every stride is a runtime argument (upstream bakes them), and each output
#      head h reads q head h / q_group, k head h / k_group and v head h / v_group
#      (hipDNN's HO = max(H, HV) head replication, which covers upstream's
#      hv // (HV / HK)).
#   4. The state is hipDNN's [N or P, HO, V, K] f32 (upstream's "vk" layout),
#      read through its own strides from initial_state and written through
#      final_state's, optionally addressed through state_indices. Absent
#      initial_state reads as zero and absent final_state is not written: both
#      by giving the buffer resource zero records, so the hardware's range check
#      suppresses the access. A K-contiguous state moves in 128-bit vectors; a
#      K-strided one (STATE_K_CONTIGUOUS False) element by element.
#   5. The gate and write strength follow hipDNN's flags at runtime: alpha is
#      exp(g) (LOG), max(g, 1e-10) (LINEAR) or exp(-exp(a_log) softplus(g +
#      dt_bias)) (safe_gate, upstream's only form); beta is beta, sigmoid(beta) or
#      2 sigmoid(beta). Absent a_log / dt_bias read as zero (unit amplitude, zero
#      bias). g, beta, a_log and dt_bias are each f32, f16 or bf16, chosen per
#      operand at runtime.
#   6. use_qk_l2norm and the scale are runtime arguments; with the norm, q and k
#      become x * rsqrt(sum(x^2) + 1e-6) and the scale multiplies the normalized
#      q, as upstream.
#   7. No padding-slot sentinel (state index < 0) and no MTP / speculative
#      variants; upstream's create_vk_gdr_mtp_kernel is not ported.
#
# Upstream-first: prefer landing these changes in AITER; this copy exists so the
# provider is not blocked on that.

"""Gated DeltaNet forward as a token-recurrent kernel.

Per sequence n, output head h and token t, with the f32 state S [V, K]:
    S_t = alpha_t S_{t-1} + beta_t (v_t - alpha_t S_{t-1} k_t) k_t^T
    o_t = scale S_t q_t

Thread mapping (upstream's): a 64-lane wave is WARP_THREADS_V rows of V by
WARP_THREADS_K lanes along K, each lane holding VALUES_PER_THREAD_K consecutive
K values of WARP_TILE_K_ITERS K tiles. The workgroup's NUM_WARPS waves stack
along V; a workgroup covers TILE_V = V / V_SPLIT rows of V, so V_SPLIT
workgroups share one (sequence, head). K reductions are butterfly shuffles over
the WARP_THREADS_K lanes of a row, which leave every lane the same sum.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from kernels.common import buffer_ops

KERNEL_NAME = "gdn_recurrent"

LOG2E = 1.4426950408889634
L2_NORM_EPS = 1e-6
LINEAR_GATE_FLOOR = 1e-10
SOFTPLUS_THRESHOLD = 20.0

# Bits of the `flags` argument.
FLAG_L2NORM = 1 << 0
FLAG_BETA_SIGMOID = 1 << 1
FLAG_BETA_DOUBLE = 1 << 2  # allow_neg_eigval: 2 * sigmoid(beta)
FLAG_GATE_LINEAR = 1 << 3
FLAG_GATE_SAFE = 1 << 4
FLAG_STATE_INDICES = 1 << 5
FLAG_INITIAL_STATE = 1 << 6
FLAG_FINAL_STATE = 1 << 7
FLAG_A_LOG = 1 << 8
FLAG_DT_BIAS = 1 << 9

# Two-bit dtype codes packed into the `aux_dtypes` argument, one per operand.
AUX_F32 = 0
AUX_F16 = 1
AUX_BF16 = 2
AUX_SHIFT_G = 0
AUX_SHIFT_BETA = 2
AUX_SHIFT_A_LOG = 4
AUX_SHIFT_DT_BIAS = 6

NUM_WARPS = 4
WARP_SIZE = 64
WARP_THREADS_K = 8
WARP_THREADS_V = WARP_SIZE // WARP_THREADS_K
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
VALUES_PER_THREAD_K = 4  # one 128-bit f32 vector of the state

# Records of a present operand's buffer: the largest byte offset a 32-bit buffer
# offset reaches. The matcher bounds every operand under it.
_PRESENT_RECORDS = 0x7FFFFFFF

# The kernel's parameter list after the twelve operand addresses, in order.
INT_PARAMS = (
    "num_seqs",
    "num_heads_out",
    "q_group",
    "k_group",
    "v_group",
    "stride_q_t",
    "stride_q_h",
    "stride_k_t",
    "stride_k_h",
    "stride_v_t",
    "stride_v_h",
    "stride_o_t",
    "stride_o_h",
    "stride_g_t",
    "stride_g_h",
    "stride_beta_t",
    "stride_beta_h",
    "stride_a_log",
    "stride_dt_bias",
    "stride_h0_n",
    "stride_h0_h",
    "stride_h0_v",
    "stride_h0_k",
    "stride_ht_n",
    "stride_ht_h",
    "stride_ht_v",
    "stride_ht_k",
    "flags",
    "aux_dtypes",
)
POINTER_PARAMS = (
    "Q",
    "K",
    "V",
    "G",
    "Beta",
    "CuSeqlens",
    "ALog",
    "DtBias",
    "StateIndices",
    "InitialState",
    "FinalState",
    "O",
)


def _io_numeric(dtype: str):
    return {"bf16": fx.BFloat16, "f16": fx.Float16}[dtype]


def build_gdn_recurrent_module(
    dtype: str,
    head_k_dim: int,
    head_v_dim: int,
    v_split: int,
    state_k_contiguous: bool = True,
):
    """The launcher for one (dtype, K, V, V split, state layout) object.

    Grid: (num_seqs * v_split, num_heads_out, 1); block: (BLOCK_THREADS, 1, 1).
    """
    io_num = _io_numeric(dtype)

    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    if head_k_dim % WARP_TILE_K != 0:
        raise ValueError(f"head_k_dim {head_k_dim} is not a multiple of {WARP_TILE_K}")
    K_ITERS = head_k_dim // WARP_TILE_K

    WARP_GROUP_TILE_V = NUM_WARPS * WARP_THREADS_V
    if head_v_dim % v_split != 0:
        raise ValueError(f"head_v_dim {head_v_dim} does not split {v_split} ways")
    TILE_V = head_v_dim // v_split
    if TILE_V % WARP_GROUP_TILE_V != 0:
        raise ValueError(
            f"a V slice of {TILE_V} rows is not a multiple of {WARP_GROUP_TILE_V}"
        )
    V_ITERS = TILE_V // WARP_GROUP_TILE_V
    NUM_VECS = V_ITERS * K_ITERS

    K_SHUFFLE_OFFSETS = []
    offset_ = WARP_THREADS_K // 2
    while offset_ >= 1:
        K_SHUFFLE_OFFSETS.append(offset_)
        offset_ //= 2

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def gdn_recurrent_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        G: fx.Pointer,
        Beta: fx.Pointer,
        CuSeqlens: fx.Pointer,
        ALog: fx.Pointer,
        DtBias: fx.Pointer,
        StateIndices: fx.Pointer,
        InitialState: fx.Pointer,
        FinalState: fx.Pointer,
        O: fx.Pointer,
        num_seqs: fx.Int32,
        num_heads_out: fx.Int32,
        q_group: fx.Int32,
        k_group: fx.Int32,
        v_group: fx.Int32,
        stride_q_t: fx.Int32,
        stride_q_h: fx.Int32,
        stride_k_t: fx.Int32,
        stride_k_h: fx.Int32,
        stride_v_t: fx.Int32,
        stride_v_h: fx.Int32,
        stride_o_t: fx.Int32,
        stride_o_h: fx.Int32,
        stride_g_t: fx.Int32,
        stride_g_h: fx.Int32,
        stride_beta_t: fx.Int32,
        stride_beta_h: fx.Int32,
        stride_a_log: fx.Int32,
        stride_dt_bias: fx.Int32,
        stride_h0_n: fx.Int32,
        stride_h0_h: fx.Int32,
        stride_h0_v: fx.Int32,
        stride_h0_k: fx.Int32,
        stride_ht_n: fx.Int32,
        stride_ht_h: fx.Int32,
        stride_ht_v: fx.Int32,
        stride_ht_k: fx.Int32,
        flags: fx.Int32,
        aux_dtypes: fx.Int32,
        scale: fx.Float32,
    ):
        c0_i32 = fx.Int32(0)
        f32_0 = fx.Float32(0.0)
        f32_1 = fx.Float32(1.0)

        tid = fx.Int32(fx.thread_idx.x)
        bx = fx.Int32(fx.block_idx.x)
        h_i = fx.Int32(fx.block_idx.y)
        w_tid = tid % WARP_SIZE
        wid = tid // WARP_SIZE
        k_lane = w_tid % WARP_THREADS_K
        v_lane = w_tid // WARP_THREADS_K

        n_i = bx // v_split
        tile_v_start = (bx % v_split) * TILE_V
        q_h = h_i // q_group
        k_h = h_i // k_group
        v_h = h_i // v_group

        def flag(bit):
            return (flags & fx.Int32(bit)) != c0_i32

        def records(present):
            return present.select(fx.Int64(_PRESENT_RECORDS), fx.Int64(0))

        def rsrc_at(address, byte_offset=None, num_records=None):
            addr = fx.Int64(fx.ptrtoint(address))
            if byte_offset is not None:
                addr = addr + byte_offset
            if num_records is None:
                num_records = fx.Int64(_PRESENT_RECORDS)
            return buffer_ops.create_buffer_resource_from_addr(
                as_mlir_value(addr), num_records_bytes=num_records
            )

        def load_aux(rsrc, index, shift):
            """One f32, f16 or bf16 element as f32; the unused width's load is
            pushed out of range, so nothing past the element is touched."""
            code = (aux_dtypes >> fx.Int32(shift)) & fx.Int32(3)
            is_f32 = code == fx.Int32(AUX_F32)
            wide = fx.Float32(
                buffer_ops.buffer_load(
                    rsrc,
                    as_mlir_value(index),
                    vec_width=1,
                    dtype=fx.Float32,
                    mask=as_mlir_value(is_f32),
                )
            )
            narrow_bits = fx.Int16(
                buffer_ops.buffer_load(
                    rsrc,
                    as_mlir_value(index),
                    vec_width=1,
                    dtype=fx.Int16,
                    mask=as_mlir_value(~is_f32),
                )
            )
            as_bf16 = narrow_bits.bitcast(fx.BFloat16).to(fx.Float32)
            as_f16 = narrow_bits.bitcast(fx.Float16).to(fx.Float32)
            narrow = (code == fx.Int32(AUX_BF16)).select(as_bf16, as_f16)
            return is_f32.select(wide, narrow)

        def fast_exp(x):
            return fx.Float32(fx.math.exp2(fx.Float32(x) * fx.Float32(LOG2E)))

        def k_reduce(x):
            for offset in K_SHUFFLE_OFFSETS:
                x = x + x.shuffle_xor(fx.Int32(offset), fx.Int32(WARP_SIZE))
            return x

        def vec_sum(vec):
            return fx.Float32(fx.Vector(vec).reduce(fx.ReductionOp.ADD))

        def filled(value):
            return fx.Vector.filled(VALUES_PER_THREAD_K, fx.Float32(value), fx.Float32)

        # --- Sequence bounds and state row ------------------------------------
        cu_rsrc = rsrc_at(CuSeqlens)
        bos = fx.Int32(
            buffer_ops.buffer_load(
                cu_rsrc, as_mlir_value(n_i), vec_width=1, dtype=fx.Int32
            )
        )
        eos = fx.Int32(
            buffer_ops.buffer_load(
                cu_rsrc, as_mlir_value(n_i + 1), vec_width=1, dtype=fx.Int32
            )
        )
        has_indices = flag(FLAG_STATE_INDICES)
        idx_rsrc = rsrc_at(StateIndices, num_records=records(has_indices))
        indexed_row = fx.Int32(
            buffer_ops.buffer_load(
                idx_rsrc, as_mlir_value(n_i), vec_width=1, dtype=fx.Int32
            )
        )
        state_row = has_indices.select(indexed_row, n_i)

        # Each (row, head) state tile gets its own 64-bit base, so the pool may
        # exceed what a 32-bit buffer offset reaches.
        h0_rsrc = rsrc_at(
            InitialState,
            (
                fx.Int64(state_row) * fx.Int64(stride_h0_n)
                + fx.Int64(h_i) * fx.Int64(stride_h0_h)
            )
            * fx.Int64(4),
            records(flag(FLAG_INITIAL_STATE)),
        )
        ht_rsrc = rsrc_at(
            FinalState,
            (
                fx.Int64(state_row) * fx.Int64(stride_ht_n)
                + fx.Int64(h_i) * fx.Int64(stride_ht_h)
            )
            * fx.Int64(4),
            records(flag(FLAG_FINAL_STATE)),
        )

        q_rsrc = rsrc_at(Q)
        k_rsrc = rsrc_at(K)
        v_rsrc = rsrc_at(V)
        o_rsrc = rsrc_at(O)
        g_rsrc = rsrc_at(G)
        beta_rsrc = rsrc_at(Beta)
        a_log_rsrc = rsrc_at(ALog, num_records=records(flag(FLAG_A_LOG)))
        dt_bias_rsrc = rsrc_at(DtBias, num_records=records(flag(FLAG_DT_BIAS)))

        # --- Per-head constants -----------------------------------------------
        amplitude = fast_exp(load_aux(a_log_rsrc, h_i * stride_a_log, AUX_SHIFT_A_LOG))
        dt_bias = load_aux(dt_bias_rsrc, h_i * stride_dt_bias, AUX_SHIFT_DT_BIAS)
        use_l2norm = flag(FLAG_L2NORM)
        beta_sigmoid = flag(FLAG_BETA_SIGMOID)
        beta_factor = flag(FLAG_BETA_DOUBLE).select(fx.Float32(2.0), f32_1)
        gate_linear = flag(FLAG_GATE_LINEAR)
        gate_safe = flag(FLAG_GATE_SAFE)

        def k_index(ki):
            return k_lane * VALUES_PER_THREAD_K + ki * WARP_TILE_K

        def v_index(vi):
            return tile_v_start + wid * WARP_THREADS_V + v_lane + vi * WARP_GROUP_TILE_V

        def state_offset(stride_v, stride_k, vi, ki, element):
            k = k_index(ki) + element
            return v_index(vi) * stride_v + k * stride_k

        # --- Incoming state -----------------------------------------------------
        state = []
        for vi in range_constexpr(V_ITERS):
            for ki in range_constexpr(K_ITERS):
                if const_expr(state_k_contiguous):
                    vec = buffer_ops.buffer_load(
                        h0_rsrc,
                        as_mlir_value(
                            state_offset(stride_h0_v, fx.Int32(1), vi, ki, 0)
                        ),
                        vec_width=VALUES_PER_THREAD_K,
                        dtype=fx.Float32,
                    )
                    state.append(fx.Vector(vec))
                else:
                    elements = [
                        fx.Float32(
                            buffer_ops.buffer_load(
                                h0_rsrc,
                                as_mlir_value(
                                    state_offset(stride_h0_v, stride_h0_k, vi, ki, e)
                                ),
                                vec_width=1,
                                dtype=fx.Float32,
                            )
                        )
                        for e in range(VALUES_PER_THREAD_K)
                    ]
                    state.append(fx.Vector.from_elements(elements, fx.Float32))

        # --- Token recurrence ---------------------------------------------------
        final = state
        for t_idx, carry in range(
            fx.Index(bos), fx.Index(eos), fx.Index(1), init=state
        ):
            t = fx.Int32(t_idx)
            cur = [fx.Vector(c) for c in carry]

            q_vecs = []
            k_vecs = []
            for ki in range_constexpr(K_ITERS):
                q_raw = buffer_ops.buffer_load(
                    q_rsrc,
                    as_mlir_value(t * stride_q_t + q_h * stride_q_h + k_index(ki)),
                    vec_width=VALUES_PER_THREAD_K,
                    dtype=io_num,
                )
                k_raw = buffer_ops.buffer_load(
                    k_rsrc,
                    as_mlir_value(t * stride_k_t + k_h * stride_k_h + k_index(ki)),
                    vec_width=VALUES_PER_THREAD_K,
                    dtype=io_num,
                )
                q_vecs.append(fx.Vector(q_raw).to(fx.Float32))
                k_vecs.append(fx.Vector(k_raw).to(fx.Float32))

            sum_qq = filled(f32_0)
            sum_kk = filled(f32_0)
            for ki in range_constexpr(K_ITERS):
                sum_qq = fx.math.fma(q_vecs[ki], q_vecs[ki], sum_qq)
                sum_kk = fx.math.fma(k_vecs[ki], k_vecs[ki], sum_kk)
            norm_q = k_reduce(vec_sum(sum_qq))
            norm_k = k_reduce(vec_sum(sum_kk))
            eps = fx.Float32(L2_NORM_EPS)
            inv_q = use_l2norm.select(fx.Float32(fx.math.rsqrt(norm_q + eps)), f32_1)
            inv_k = use_l2norm.select(fx.Float32(fx.math.rsqrt(norm_k + eps)), f32_1)
            q_mul = filled(inv_q * scale)
            k_mul = filled(inv_k)
            for ki in range_constexpr(K_ITERS):
                q_vecs[ki] = q_vecs[ki] * q_mul
                k_vecs[ki] = k_vecs[ki] * k_mul

            # alpha: exp(g) (LOG), max(g, floor) (LINEAR), or the safe gate.
            g_raw = load_aux(g_rsrc, t * stride_g_t + h_i * stride_g_h, AUX_SHIFT_G)
            x = g_raw + dt_bias
            softplus = (x <= fx.Float32(SOFTPLUS_THRESHOLD)).select(
                fx.Float32(fx.math.log1p(fast_exp(x))), x
            )
            alpha_safe = fast_exp(-(amplitude * softplus))
            alpha_linear = g_raw.maximumf(fx.Float32(LINEAR_GATE_FLOOR))
            alpha_log = fast_exp(g_raw)
            alpha = gate_safe.select(
                alpha_safe, gate_linear.select(alpha_linear, alpha_log)
            )

            b_raw = load_aux(
                beta_rsrc, t * stride_beta_t + h_i * stride_beta_h, AUX_SHIFT_BETA
            )
            b_sig = (f32_1 / (f32_1 + fast_exp(-b_raw))) * beta_factor
            beta = beta_sigmoid.select(b_sig, b_raw)

            dot_kq = filled(f32_0)
            for ki in range_constexpr(K_ITERS):
                dot_kq = fx.math.fma(k_vecs[ki], q_vecs[ki], dot_kq)
            kq = k_reduce(vec_sum(dot_kq))

            alpha_vec = filled(alpha)
            nxt = [None] * NUM_VECS
            for vi in range_constexpr(V_ITERS):
                v_row = v_index(vi)
                v_val = fx.Float32(
                    fx.Numeric.from_ir_type(io_num.ir_type)(
                        buffer_ops.buffer_load(
                            v_rsrc,
                            as_mlir_value(t * stride_v_t + v_h * stride_v_h + v_row),
                            vec_width=1,
                            dtype=io_num,
                        )
                    ).to(fx.Float32)
                )
                sum_hk = filled(f32_0)
                sum_hq = filled(f32_0)
                decayed = []
                for ki in range_constexpr(K_ITERS):
                    h_cur = cur[vi * K_ITERS + ki] * alpha_vec
                    decayed.append(h_cur)
                    sum_hk = fx.math.fma(h_cur, k_vecs[ki], sum_hk)
                    sum_hq = fx.math.fma(h_cur, q_vecs[ki], sum_hq)
                hk = k_reduce(vec_sum(sum_hk))
                hq = k_reduce(vec_sum(sum_hq))

                v_new = (v_val - hk) * beta
                v_new_vec = filled(v_new)
                for ki in range_constexpr(K_ITERS):
                    nxt[vi * K_ITERS + ki] = fx.math.fma(
                        k_vecs[ki], v_new_vec, decayed[ki]
                    )

                out = (hq + v_new * kq).to(io_num)
                if k_lane == c0_i32:
                    buffer_ops.buffer_store(
                        as_mlir_value(out),
                        o_rsrc,
                        as_mlir_value(t * stride_o_t + h_i * stride_o_h + v_row),
                    )
            final = yield [as_mlir_value(v) for v in nxt]

        # --- Outgoing state -----------------------------------------------------
        for vi in range_constexpr(V_ITERS):
            for ki in range_constexpr(K_ITERS):
                vec = fx.Vector(final[vi * K_ITERS + ki])
                if const_expr(state_k_contiguous):
                    buffer_ops.buffer_store(
                        as_mlir_value(vec),
                        ht_rsrc,
                        as_mlir_value(
                            state_offset(stride_ht_v, fx.Int32(1), vi, ki, 0)
                        ),
                    )
                else:
                    for e in range_constexpr(VALUES_PER_THREAD_K):
                        buffer_ops.buffer_store(
                            as_mlir_value(fx.Float32(vec[e])),
                            ht_rsrc,
                            as_mlir_value(
                                state_offset(stride_ht_v, stride_ht_k, vi, ki, e)
                            ),
                        )

    @flyc.jit
    def launch_gdn_recurrent(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        G: fx.Pointer,
        Beta: fx.Pointer,
        CuSeqlens: fx.Pointer,
        ALog: fx.Pointer,
        DtBias: fx.Pointer,
        StateIndices: fx.Pointer,
        InitialState: fx.Pointer,
        FinalState: fx.Pointer,
        O: fx.Pointer,
        num_seqs: fx.Int32,
        num_heads_out: fx.Int32,
        q_group: fx.Int32,
        k_group: fx.Int32,
        v_group: fx.Int32,
        stride_q_t: fx.Int32,
        stride_q_h: fx.Int32,
        stride_k_t: fx.Int32,
        stride_k_h: fx.Int32,
        stride_v_t: fx.Int32,
        stride_v_h: fx.Int32,
        stride_o_t: fx.Int32,
        stride_o_h: fx.Int32,
        stride_g_t: fx.Int32,
        stride_g_h: fx.Int32,
        stride_beta_t: fx.Int32,
        stride_beta_h: fx.Int32,
        stride_a_log: fx.Int32,
        stride_dt_bias: fx.Int32,
        stride_h0_n: fx.Int32,
        stride_h0_h: fx.Int32,
        stride_h0_v: fx.Int32,
        stride_h0_k: fx.Int32,
        stride_ht_n: fx.Int32,
        stride_ht_h: fx.Int32,
        stride_ht_v: fx.Int32,
        stride_ht_k: fx.Int32,
        flags: fx.Int32,
        aux_dtypes: fx.Int32,
        scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gdn_recurrent_kernel(
            Q,
            K,
            V,
            G,
            Beta,
            CuSeqlens,
            ALog,
            DtBias,
            StateIndices,
            InitialState,
            FinalState,
            O,
            num_seqs,
            num_heads_out,
            q_group,
            k_group,
            v_group,
            stride_q_t,
            stride_q_h,
            stride_k_t,
            stride_k_h,
            stride_v_t,
            stride_v_h,
            stride_o_t,
            stride_o_h,
            stride_g_t,
            stride_g_h,
            stride_beta_t,
            stride_beta_h,
            stride_a_log,
            stride_dt_bias,
            stride_h0_n,
            stride_h0_h,
            stride_h0_v,
            stride_h0_k,
            stride_ht_n,
            stride_ht_h,
            stride_ht_v,
            stride_ht_k,
            flags,
            aux_dtypes,
            scale,
        ).launch(
            grid=(num_seqs * v_split, num_heads_out, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_recurrent
