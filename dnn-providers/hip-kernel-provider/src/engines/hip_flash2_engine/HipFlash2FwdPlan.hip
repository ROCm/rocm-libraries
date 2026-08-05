// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Exported: flash2_v7_hipdnn_d128, flash2_v7_hipdnn_d64
// Build: hipcc --offload-arch=gfx942 -O3 --cuda-device-only -x hip
//        -I/opt/rocm/include HipFlash2FwdPlan.hip -o kernels/hip_flash2_fwd_gfx942.co
//
// FRAGMENT LAYOUT (measured by Samuel Reeder on gfx942, rocWMMA 16x16x16):
//   accumulator: lane l, elem e -> C[row=(l/16)*4+e][col=l%16]
//   matrix_b:    lane l, elem e -> B[row=(l%16)*4+e???] -- see note
//
// For S = Q*K^T (accumulator):
//   sf[qg][kg].x[e] = S[ qg*16+(l/16)*4+e ][ kg*16+l%16 ]
//
// KEY INSIGHT for O accumulation:
//   After softmax, we need O[q_row][d] += sum_k P[q_row][k] * V[k][d]
//   Using the accumulator layout: o_frags[qg][d16].x[e] holds O[qg*16+(l/16)*4+e][d16*16+l%16]
//   Each lane l owns output columns multiples of l%16 across d16 blocks.
//   Each lane l owns exactly the key column l%16 from each key group.
//   So: o[qg][d16].x[e] += P[qg*16+(l/16)*4+e][kg*16+l%16] * V[kg*16+l%16][d16*16+?]
//   V[kg*16+l%16][d] is in shared memory sV[d][kg*16+l%16].
//   For output col d16*16+l%16: sV[d16*16+l%16... NO -- sV[d][lane] stores V[ki_g][d]
//   where ki_g = ki0+lane. So sV[d16*16+c][kg*16+l%16] = V[ki0+kg*16+l%16][d16*16+c].
//   We want V[ki0+kg*16+l%16][d16*16+l%16] for output col d16*16+l%16.
//   But lane l contributes to output col l%16 + k*16 -- it only fills certain columns.
//
// CORRECT O ACCUMULATION (scalar, no pf fragment):
//   For each kg, d16: lane l accumulates:
//     o_frags[qg][d16].x[e] += P_val[qg][kg][e] * sV[d16*16+l%16][kg*16+l%16]
//   This is: O[q_row][out_col=l%16+d16*16] += P[q_row][key_col=kg*16+l%16] * V[key_col][out_col]
//   Each lane handles output column out_col=l%16 (mod 16), hitting all d16 blocks.
//   V[key_col][out_col] = sV[out_col][key_col] = sV[d16*16+l%16][kg*16+l%16]

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
    using fq = fragment<matrix_a,    16, 16, 16, __half, row_major>;
    using fk = fragment<matrix_b,    16, 16, 16, __half, col_major>;
    using fs = fragment<accumulator, 16, 16, 16, float>;
    using fo = fragment<accumulator, 16, 16, 16, float>;

    const int qi_blk   = blockIdx.x, b = blockIdx.y, hq = blockIdx.z;
    const int lane      = threadIdx.x;
    const int hk        = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);
    const int qi_base   = qi_blk * 64;
    const int qi_max    = qi_base + 63;
    // Accumulator layout: lane l, elem e -> row=(l/16)*4+e, col=l%16
    const int my_e_base = (lane / 16) * 4;   // first query-row-offset for this lane's elems
    const int my_kcol   = lane % 16;           // key column this lane owns in each kg tile

    __shared__ __half sK[D][64], sV[D][64];

    // Load Q fragments
    fq q_frags[4][D16];
    const __half* Qp = ptr_q + (size_t)b * q_stride_batch + (size_t)hq * q_stride_head
                       + (size_t)qi_base * q_stride_seq;
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
        {
            if(qi_base + qg * 16 < seq_len_q)
                load_matrix_sync(q_frags[qg][d16],
                                 Qp + (size_t)qg * 16 * q_stride_seq + d16 * 16, q_stride_seq);
            else
                fill_fragment(q_frags[qg][d16], __half(0));
        }

    // Output accumulators: each lane owns row (qg*16+my_e_base+e), col (d16*16+my_kcol)
    // We accumulate O SCALAR: o_acc[qg][d16][e] = sum over kg of P[q_row][kg*16+my_kcol] * V[kg*16+my_kcol][d16*16+my_kcol]
    // Store in fo fragment (accumulator layout) so store_matrix_sync works at the end
    fo o_frags[4][D16];
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
            fill_fragment(o_frags[qg][d16], 0.f);

    // Per-lane softmax state [qg][e]: local (not reduced across lanes in tile loop)
    float row_m[4][4], row_d[4][4];
    for(int qg = 0; qg < 4; qg++)
        for(int e = 0; e < 4; e++)
        {
            row_m[qg][e] = -1e38f;
            row_d[qg][e] = 0.f;
        }

    for(int kb = 0; kb < (seq_len_kv + 63) / 64; kb++)
    {
        const int ki0        = kb * 64;
        if(causal && ki0 > qi_max)
            break;
        const bool full_tile = (!causal) || (ki0 + 63 <= qi_base);

        // Load K, V into shared memory
        const int ki_g = ki0 + lane;
        if(ki_g < seq_len_kv)
        {
            const __half* kp = ptr_k + (size_t)b * k_stride_batch + (size_t)hk * k_stride_head
                               + (size_t)ki_g * k_stride_seq;
            const __half* vp = ptr_v + (size_t)b * v_stride_batch + (size_t)hk * v_stride_head
                               + (size_t)ki_g * v_stride_seq;
            for(int d = 0; d < D; d++)
            {
                sK[d][lane] = kp[d];
                sV[d][lane] = vp[d];
            }
        }
        else
        {
            for(int d = 0; d < D; d++)
            {
                sK[d][lane] = __half(0);
                sV[d][lane] = __half(0);
            }
        }
        __syncthreads();

        // S = Q * K^T  (accumulator layout)
        fs sf[4][4];
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                fill_fragment(sf[qg][kg], 0.f);
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int d16 = 0; d16 < D16; d16++)
                {
                    fk kf;
                    load_matrix_sync(kf, &sK[d16 * 16][kg * 16], 64);
                    mma_sync(sf[qg][kg], q_frags[qg][d16], kf, sf[qg][kg]);
                }

        // Per-lane tile max: sf[qg][kg].x[e] = S[qg*16+my_e_base+e][kg*16+my_kcol]
        float tile_m[4][4];
        float pval[4][4][4];  // softmax weights (unnormalized) after rescaling
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
                tile_m[qg][e] = -1e38f;

        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int e = 0; e < 4; e++)
                {
                    float s          = sf[qg][kg].x[e] * scale;
                    const int key_col   = ki0 + kg * 16 + my_kcol;
                    const int query_row = qi_base + qg * 16 + my_e_base + e;
                    if(!full_tile && key_col > query_row)
                        s = -1e38f;
                    if(key_col >= seq_len_kv)
                        s = -1e38f;
                    pval[qg][kg][e] = s;
                    tile_m[qg][e]   = fmaxf(tile_m[qg][e], s);
                }

        // Update running max and rescale previous o_frags and row_d
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                const float new_m = fmaxf(row_m[qg][e], tile_m[qg][e]);
                const float br    = flash2_expf(row_m[qg][e] - new_m);
                row_m[qg][e]      = new_m;
                for(int d16 = 0; d16 < D16; d16++)
                    o_frags[qg][d16].x[e] *= br;
                row_d[qg][e] *= br;
            }

        // Compute softmax weights and accumulate O SCALAR
        // Lane l owns output column my_kcol (and my_kcol+16, +32, ... via d16)
        // Key column this lane covers: kg*16 + my_kcol for each kg
        // V value for that key col and output col: sV[out_col][kg*16+my_kcol]
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int e = 0; e < 4; e++)
                {
                    float pv      = flash2_expf(pval[qg][kg][e] - row_m[qg][e]);
                    row_d[qg][e] += pv;
                    // Accumulate O: O[q_row][out_col] += pv * V[kg*16+my_kcol][out_col]
                    // out_col = d16*16 + my_kcol for each d16 block
                    for(int d16 = 0; d16 < D16; d16++)
                    {
                        // sV[d][col]: V[ki0+col][d]. Here key = ki0+kg*16+my_kcol.
                        // out_col = d16*16 + my_kcol
                        float v_val = __half2float(sV[d16 * 16 + my_kcol][kg * 16 + my_kcol]);
                        o_frags[qg][d16].x[e] += pv * v_val;
                    }
                }
        __syncthreads();
    }

    // After tile loop: reduce row_m and row_d, then normalize o_frags
    // Lanes sharing query row (qg,e) have same (lane/16) → shfl_xor 1,2,4,8
    for(int qg = 0; qg < 4; qg++)
        for(int e = 0; e < 4; e++)
        {
            // Global max
            float gm = row_m[qg][e];
            gm = fmaxf(gm, __shfl_xor(gm, 1));
            gm = fmaxf(gm, __shfl_xor(gm, 2));
            gm = fmaxf(gm, __shfl_xor(gm, 4));
            gm = fmaxf(gm, __shfl_xor(gm, 8));

            // Rescale this lane's contributions to global max
            const float br = flash2_expf(row_m[qg][e] - gm);
            for(int d16 = 0; d16 < D16; d16++)
                o_frags[qg][d16].x[e] *= br;
            row_d[qg][e] *= br;

            // Sum row_d across lanes (each lane sums its own partial normalizer)
            row_d[qg][e] += __shfl_xor(row_d[qg][e], 1);
            row_d[qg][e] += __shfl_xor(row_d[qg][e], 2);
            row_d[qg][e] += __shfl_xor(row_d[qg][e], 4);
            row_d[qg][e] += __shfl_xor(row_d[qg][e], 8);
        }

    // Normalize and write output
    {
        __shared__ float so[64][D];
        for(int qg = 0; qg < 4; qg++)
        {
            for(int d16 = 0; d16 < D16; d16++)
            {
                for(int e = 0; e < 4; e++)
                {
                    const float inv        = (row_d[qg][e] > 0.f) ? 1.f / row_d[qg][e] : 0.f;
                    o_frags[qg][d16].x[e] *= inv;
                }
                store_matrix_sync(&so[qg * 16][d16 * 16], o_frags[qg][d16], D, mem_row_major);
            }
        }
        __syncthreads();

        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                const int oq = qi_base + qg * 16 + my_e_base + e;
                if(oq < seq_len_q)
                {
                    __half* op = ptr_o + (size_t)b * o_stride_batch
                                 + (size_t)hq * o_stride_head
                                 + (size_t)oq * o_stride_seq;
                    for(int k = 0; k < D16; k++)
                        op[my_kcol + k * 16] =
                            __float2half(so[qg * 16 + my_e_base + e][my_kcol + k * 16]);
                }
            }
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
    flash2_fwd_body<128, 8>(
        ptr_q, ptr_k, ptr_v, ptr_o, batch, num_heads_q, num_heads_k,
        seq_len_q, seq_len_kv, head_dim, scale, causal,
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
    flash2_fwd_body<64, 4>(
        ptr_q, ptr_k, ptr_v, ptr_o, batch, num_heads_q, num_heads_k,
        seq_len_q, seq_len_kv, head_dim, scale, causal,
        q_stride_batch, q_stride_head, q_stride_seq,
        k_stride_batch, k_stride_head, k_stride_seq,
        v_stride_batch, v_stride_head, v_stride_seq,
        o_stride_batch, o_stride_head, o_stride_seq);
}
