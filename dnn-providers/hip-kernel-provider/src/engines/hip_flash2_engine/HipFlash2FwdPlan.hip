// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Exported: flash2_v7_hipdnn_d128, flash2_v7_hipdnn_d64
// Build: hipcc --offload-arch=gfx942 -O3 --cuda-device-only -x hip
//        -I/opt/rocm/include HipFlash2FwdPlan.hip -o kernels/hip_flash2_fwd_gfx942.co
//
// ---------------------------------------------------------------------------
// v13 -- FULL MFMA for BOTH GEMMs via the S-transpose formulation.
//
// MEASURED rocWMMA 16x16x16 f16 register layouts on gfx942 (MI300X), where
// lane = l (0..63) and e = fragment element index (0..3):
//
//   matrix_a     : A[row = l%16        ][k   = (l/16)*4 + e]
//   matrix_b     : B[k   = (l/16)*4 + e][col = l%16        ]
//   accumulator  : C[row = (l/16)*4 + e][col = l%16        ]
//
// KEY OBSERVATION: the accumulator mapping is the TRANSPOSE of the matrix_a
// mapping, but is IDENTICAL to the matrix_b mapping. Therefore an accumulator
// holding S = Q*K^T can NEVER be reinterpreted in-register as a matrix_a
// fragment holding P -- doing so silently yields P^T. Every previous attempt
// to MFMA the P*V product failed for this reason, forcing a scalar LDS
// fallback (~4096 scalar MADs/lane).
//
// THE FIX: compute the TRANSPOSED score S^T = K*Q^T instead of S = Q*K^T.
//   S^T accumulator: C[row = key][col = query]
//        -> lane l, elem e holds key = (l/16)*4+e, query = l%16
//   This is EXACTLY the matrix_b register mapping for P^T[key][query].
//   So P^T is produced in-register by an element-wise exp() on the S^T
//   accumulator -- no shuffles, no LDS round-trip, no transpose.
//
// Then accumulate the transposed output:
//   O^T = V^T * P^T        (equivalently O = P*V)
//   O^T accumulator: C[row = d][col = query]
//   matrix_a = V^T : A[row = d  ][k = key]  <- sV[d][key], row_major, ld=64
//   matrix_b = P^T : B[k = key  ][col = q]  <- in-register from S^T accumulator
//
// CONSEQUENCES vs the scalar-LDS version:
//   * P*V is now 4*D16 MFMA ops per KV tile instead of 4096 scalar MADs/lane.
//   * The sP LDS staging buffer and its two __syncthreads() per kg are gone.
//   * The so[64][D] output staging buffer is gone -- each lane writes its own
//     4 contiguous halves straight to global. LDS drops 64KB -> 32KB (D=128),
//     which lets two CTAs share a CU again.
//
// SOFTMAX AXIS: each lane owns query column l%16 and holds keys along e and
// across lane groups l/16. The row-wise (per-query) max/sum reduction is
// therefore over e, over kg, and ACROSS LANE GROUPS -- i.e. __shfl_xor with
// offsets 16 and 32. (Offsets 1,2,4,8 would reduce across queries: wrong axis.)
//
// LDS: sK[D][64] + sV[D][64] = 32KB (D=128), 16KB (D=64).
// ---------------------------------------------------------------------------

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

__device__ __forceinline__ float flash2_expf(float x)
{
#if defined(__gfx942__)
    return __builtin_amdgcn_exp2f(x * 1.4426950408f);
#else
    return exp2f(x * 1.4426950408f);
#endif
}

template <int D, int D16>
__device__ void flash2_fwd_body(const __half* __restrict__ ptr_q,
                                const __half* __restrict__ ptr_k,
                                const __half* __restrict__ ptr_v,
                                __half* __restrict__ ptr_o,
                                int batch, int num_heads_q, int num_heads_k,
                                int seq_len_q, int seq_len_kv, int head_dim,
                                float scale, int causal,
                                int q_stride_batch, int q_stride_head, int q_stride_seq,
                                int k_stride_batch, int k_stride_head, int k_stride_seq,
                                int v_stride_batch, int v_stride_head, int v_stride_seq,
                                int o_stride_batch, int o_stride_head, int o_stride_seq)
{
    using namespace rocwmma;

    // K as matrix_a: A[row=key][k=d] read from sK[d][key] => col_major, ld=64.
    using fk_a = fragment<matrix_a, 16, 16, 16, __half, col_major>;
    // Q^T as matrix_b: B[k=d][col=query] read from Q[query][d] => col_major, ld=q_stride_seq.
    using fq_b = fragment<matrix_b, 16, 16, 16, __half, col_major>;
    // V^T as matrix_a: A[row=d][k=key] read from sV[d][key] => row_major, ld=64.
    using fv_a = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    // P^T as matrix_b: filled in-register from the S^T accumulator (layout tag unused).
    using fp_b = fragment<matrix_b, 16, 16, 16, __half, col_major>;
    using facc = fragment<accumulator, 16, 16, 16, float>;

    const int qi_blk = blockIdx.x, b = blockIdx.y, hq = blockIdx.z;
    const int lane   = threadIdx.x;
    const int hk     = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);
    const int qi_base = qi_blk * 64;
    const int qi_max  = qi_base + 63;

    // Register-layout roles for this lane.
    const int my_qcol   = lane % 16;        // query column owned by this lane
    const int my_e_base = (lane / 16) * 4;  // base index along the (l/16)*4+e axis

    __shared__ __half sK[D][64];
    __shared__ __half sV[D][64];

    // ---- Load Q^T fragments once (matrix_b, straight from global) ----------
    fq_b q_frags[4][D16];
    const __half* Qp = ptr_q + (size_t)b * q_stride_batch + (size_t)hq * q_stride_head
                       + (size_t)qi_base * q_stride_seq;
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
        {
            if(qi_base + qg * 16 + 15 < seq_len_q)
                load_matrix_sync(q_frags[qg][d16],
                                 Qp + (size_t)qg * 16 * q_stride_seq + d16 * 16,
                                 q_stride_seq);
            else
            {
                // Partial tile: zero-fill, then fill valid columns element-wise.
                fill_fragment(q_frags[qg][d16], __half(0));
                const int qrow = qi_base + qg * 16 + my_qcol;
                if(qrow < seq_len_q)
                    for(int e = 0; e < 4; e++)
                        q_frags[qg][d16].x[e] =
                            Qp[(size_t)(qg * 16 + my_qcol) * q_stride_seq + d16 * 16
                               + my_e_base + e];
            }
        }

    // ---- O^T accumulators: C[row=d][col=query] -----------------------------
    facc o_acc[4][D16];
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++) fill_fragment(o_acc[qg][d16], 0.f);

    // Per-lane online softmax state: one query column (my_qcol) per qg.
    float row_m[4], row_d[4];
    for(int qg = 0; qg < 4; qg++) { row_m[qg] = -1e38f; row_d[qg] = 0.f; }

    for(int kb = 0; kb < (seq_len_kv + 63) / 64; kb++)
    {
        const int  ki0       = kb * 64;
        if(causal && ki0 > qi_max) break;
        const bool full_tile = (!causal) || (ki0 + 63 <= qi_base);

        // ---- Stage K and V tiles (transposed: [d][key]) --------------------
        const int ki_g = ki0 + lane;
        if(ki_g < seq_len_kv)
        {
            const __half* kp = ptr_k + (size_t)b * k_stride_batch + (size_t)hk * k_stride_head
                               + (size_t)ki_g * k_stride_seq;
            const __half* vp = ptr_v + (size_t)b * v_stride_batch + (size_t)hk * v_stride_head
                               + (size_t)ki_g * v_stride_seq;
            for(int d = 0; d < D; d++) { sK[d][lane] = kp[d]; sV[d][lane] = vp[d]; }
        }
        else
        {
            for(int d = 0; d < D; d++) { sK[d][lane] = __half(0); sV[d][lane] = __half(0); }
        }
        __syncthreads();

        // ---- GEMM 1: S^T = K * Q^T  (MFMA) --------------------------------
        // st[kg][qg].x[e] = S^T[key = ki0+kg*16+my_e_base+e][query = qi_base+qg*16+my_qcol]
        facc st[4][4];
        for(int kg = 0; kg < 4; kg++)
            for(int qg = 0; qg < 4; qg++) fill_fragment(st[kg][qg], 0.f);
        for(int kg = 0; kg < 4; kg++)
            for(int d16 = 0; d16 < D16; d16++)
            {
                fk_a kf;
                load_matrix_sync(kf, &sK[d16 * 16][kg * 16], 64);
                for(int qg = 0; qg < 4; qg++)
                    mma_sync(st[kg][qg], kf, q_frags[qg][d16], st[kg][qg]);
            }

        // ---- Scale + mask in place; per-lane tile max over keys ------------
        float tile_m[4];
        for(int qg = 0; qg < 4; qg++) tile_m[qg] = -1e38f;
        for(int kg = 0; kg < 4; kg++)
            for(int qg = 0; qg < 4; qg++)
                for(int e = 0; e < 4; e++)
                {
                    const int key   = ki0 + kg * 16 + my_e_base + e;
                    const int query = qi_base + qg * 16 + my_qcol;
                    float     s     = st[kg][qg].x[e] * scale;
                    if(key >= seq_len_kv)          s = -1e38f;
                    if(!full_tile && key > query)  s = -1e38f;
                    st[kg][qg].x[e] = s;
                    tile_m[qg]      = fmaxf(tile_m[qg], s);
                }

        // ---- Reduce the max ACROSS LANE GROUPS (keys), not across queries --
        for(int qg = 0; qg < 4; qg++)
        {
            float gm = tile_m[qg];
            gm = fmaxf(gm, __shfl_xor(gm, 16));
            gm = fmaxf(gm, __shfl_xor(gm, 32));
            tile_m[qg] = gm;
        }

        // ---- Rescale running state to the new max --------------------------
        for(int qg = 0; qg < 4; qg++)
        {
            const float new_m = fmaxf(row_m[qg], tile_m[qg]);
            const float br    = (row_m[qg] <= -1e30f) ? 0.f : flash2_expf(row_m[qg] - new_m);
            row_m[qg] = new_m;
            row_d[qg] *= br;
            for(int d16 = 0; d16 < D16; d16++)
                for(int e = 0; e < 4; e++) o_acc[qg][d16].x[e] *= br;
        }

        // ---- P^T in-register: exp() on the S^T accumulator -----------------
        // The accumulator register mapping IS the matrix_b mapping, so this is
        // a pure element-wise conversion: no shuffle, no LDS, no transpose.
        fp_b pf[4][4];
        for(int kg = 0; kg < 4; kg++)
            for(int qg = 0; qg < 4; qg++)
                for(int e = 0; e < 4; e++)
                {
                    const float s  = st[kg][qg].x[e];
                    const float pv = (s > -1e30f) ? flash2_expf(s - row_m[qg]) : 0.f;
                    pf[kg][qg].x[e] = (_Float16)pv;
                    row_d[qg]      += pv;
                }

        // ---- GEMM 2: O^T += V^T * P^T  (MFMA) ------------------------------
        for(int kg = 0; kg < 4; kg++)
            for(int d16 = 0; d16 < D16; d16++)
            {
                fv_a vf;
                load_matrix_sync(vf, &sV[d16 * 16][kg * 16], 64);
                for(int qg = 0; qg < 4; qg++)
                    mma_sync(o_acc[qg][d16], vf, pf[kg][qg], o_acc[qg][d16]);
            }
        __syncthreads();
    }

    // ---- Finalize: sum the normalizer across lane groups -------------------
    for(int qg = 0; qg < 4; qg++)
    {
        row_d[qg] += __shfl_xor(row_d[qg], 16);
        row_d[qg] += __shfl_xor(row_d[qg], 32);
    }

    // ---- Normalize and write O straight to global --------------------------
    // Lane l owns query column my_qcol and head-dim indices my_e_base + e,
    // so each lane stores 4 contiguous __half per (qg, d16).
    for(int qg = 0; qg < 4; qg++)
    {
        const int oq = qi_base + qg * 16 + my_qcol;
        if(oq >= seq_len_q) continue;
        const float inv = (row_d[qg] > 0.f) ? 1.f / row_d[qg] : 0.f;
        __half* op = ptr_o + (size_t)b * o_stride_batch + (size_t)hq * o_stride_head
                     + (size_t)oq * o_stride_seq;
        for(int d16 = 0; d16 < D16; d16++)
            for(int e = 0; e < 4; e++)
                op[d16 * 16 + my_e_base + e] = __float2half(o_acc[qg][d16].x[e] * inv);
    }
}

extern "C" __global__
    __launch_bounds__(64, 2) void flash2_v7_hipdnn_d128(
        const __half* __restrict__ ptr_q, const __half* __restrict__ ptr_k,
        const __half* __restrict__ ptr_v, __half* __restrict__ ptr_o,
        int batch, int num_heads_q, int num_heads_k,
        int seq_len_q, int seq_len_kv, int head_dim, float scale, int causal,
        int q_stride_batch, int q_stride_head, int q_stride_seq,
        int k_stride_batch, int k_stride_head, int k_stride_seq,
        int v_stride_batch, int v_stride_head, int v_stride_seq,
        int o_stride_batch, int o_stride_head, int o_stride_seq)
{
    flash2_fwd_body<128, 8>(ptr_q, ptr_k, ptr_v, ptr_o,
        batch, num_heads_q, num_heads_k, seq_len_q, seq_len_kv, head_dim, scale, causal,
        q_stride_batch, q_stride_head, q_stride_seq,
        k_stride_batch, k_stride_head, k_stride_seq,
        v_stride_batch, v_stride_head, v_stride_seq,
        o_stride_batch, o_stride_head, o_stride_seq);
}

extern "C" __global__
    __launch_bounds__(64, 2) void flash2_v7_hipdnn_d64(
        const __half* __restrict__ ptr_q, const __half* __restrict__ ptr_k,
        const __half* __restrict__ ptr_v, __half* __restrict__ ptr_o,
        int batch, int num_heads_q, int num_heads_k,
        int seq_len_q, int seq_len_kv, int head_dim, float scale, int causal,
        int q_stride_batch, int q_stride_head, int q_stride_seq,
        int k_stride_batch, int k_stride_head, int k_stride_seq,
        int v_stride_batch, int v_stride_head, int v_stride_seq,
        int o_stride_batch, int o_stride_head, int o_stride_seq)
{
    flash2_fwd_body<64, 4>(ptr_q, ptr_k, ptr_v, ptr_o,
        batch, num_heads_q, num_heads_k, seq_len_q, seq_len_kv, head_dim, scale, causal,
        q_stride_batch, q_stride_head, q_stride_seq,
        k_stride_batch, k_stride_head, k_stride_seq,
        v_stride_batch, v_stride_head, v_stride_seq,
        o_stride_batch, o_stride_head, o_stride_seq);
}
