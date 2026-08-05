// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Exported: flash2_v7_hipdnn_d128, flash2_v7_hipdnn_d64
// Build: hipcc --offload-arch=gfx942 -O3 --cuda-device-only -x hip
//        -I/opt/rocm/include HipFlash2FwdPlan.hip -o kernels/hip_flash2_fwd_gfx942.co
//
// FRAGMENT LAYOUT (measured by Samuel Reeder on gfx942, rocWMMA 16x16x16):
//   accumulator: lane l, elem e -> C[row=(l/16)*4+e][col=l%16]
//
// For S = Q*K^T (accumulator):
//   sf[qg][kg].x[e] = S[ qg*16+(l/16)*4+e ][ kg*16+l%16 ]
//
// APPROACH: each lane handles one key column (my_kcol=l%16) per kg group.
//   Softmax weights pv[qg][kg] are stored to shared memory (sP[qg][kg][my_kcol]).
//   Each lane then reads ALL key columns to accumulate its output columns:
//     o_acc[qg][d16][e] += sum_{kg,kc} sP[qg][kg][kc] * sV[d16*16+my_kcol][kg*16+kc]
//   After tile loop: reduce row_m/row_d across lanes (shfl_xor 1,2,4,8),
//   rescale o_acc per-lane, write via shared memory.
//
// sP layout: sP[qg=0..3][kg=0..3][kc=0..15] = P weight for query-group qg,
//   key-group kg, key-column kc. Indexed as sP[qg*64 + kg*16 + kc].

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

    const int qi_blk   = blockIdx.x, b = blockIdx.y, hq = blockIdx.z;
    const int lane      = threadIdx.x;
    const int hk        = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);
    const int qi_base   = qi_blk * 64;
    const int qi_max    = qi_base + 63;
    const int my_e_base = (lane / 16) * 4;   // query row offset for this lane
    const int my_kcol   = lane % 16;           // key column this lane owns per kg

    // Shared memory layout:
    // sK[D][64], sV[D][64] -- K and V tiles
    // sP[4][4][16] -- softmax weights: sP[qg][kg][kc]
    // so[64][D] -- output accumulator (written at end)
    // Use union to reuse memory: sKV is used during tile loop, sP during P*V
    // We interleave: after computing sf, write sP, syncthreads, accumulate O, syncthreads
    __shared__ __half sK[D][64];
    __shared__ __half sV[D][64];
    __shared__ float  sP[4][4][16];  // [qg][kg][kc]

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

    // Output accumulator: o_acc[qg][d16][e] for lane's output col (d16*16+my_kcol)
    float o_acc[4][D16][4];
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
            for(int e = 0; e < 4; e++)
                o_acc[qg][d16][e] = 0.f;

    // Per-lane local softmax state
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

        // Apply scale, causal mask, update row_m per lane
        float tile_m[4][4];
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
                tile_m[qg][e] = -1e38f;

        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int e = 0; e < 4; e++)
                {
                    float s = sf[qg][kg].x[e] * scale;
                    if(!full_tile && (ki0 + kg * 16 + my_kcol) > (qi_base + qg * 16 + my_e_base + e))
                        s = -1e38f;
                    if(ki0 + kg * 16 + my_kcol >= seq_len_kv)
                        s = -1e38f;
                    sf[qg][kg].x[e] = s;  // store masked score back
                    tile_m[qg][e]   = fmaxf(tile_m[qg][e], s);
                }

        // Rescale previous output and normalizer
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                const float new_m = fmaxf(row_m[qg][e], tile_m[qg][e]);
                const float br    = flash2_expf(row_m[qg][e] - new_m);
                row_m[qg][e]      = new_m;
                for(int d16 = 0; d16 < D16; d16++)
                    o_acc[qg][d16][e] *= br;
                row_d[qg][e] *= br;
            }

        // Compute softmax weights and write to shared memory
        // Each lane l computes pv for its own key column my_kcol across all qg,kg,e
        // sP[qg][kg][my_kcol] = softmax weight for (qg, kg*16+my_kcol)
        // Note: e indexes query row, but sP needs to be per-e too
        // Use a 2D layout: sP[qg*4+e][kg*16+kc]
        __shared__ float sP2[16][64];  // [qg*4+e][kg*16+kc]
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int e = 0; e < 4; e++)
                {
                    float pv = flash2_expf(sf[qg][kg].x[e] - row_m[qg][e]);
                    row_d[qg][e]              += pv;
                    sP2[qg * 4 + e][kg * 16 + my_kcol] = pv;
                }
        __syncthreads();

        // Each lane accumulates O for its output column my_kcol across all query rows (qg,e)
        // O[qg*16+my_e_base+e][d16*16+my_kcol] += sum_{kg,kc} sP2[qg*4+e][kg*16+kc] * sV[d16*16+my_kcol][kg*16+kc]
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
                for(int d16 = 0; d16 < D16; d16++)
                {
                    float acc = 0.f;
                    for(int kg = 0; kg < 4; kg++)
                        for(int kc = 0; kc < 16; kc++)
                            acc += sP2[qg * 4 + e][kg * 16 + kc]
                                   * __half2float(sV[d16 * 16 + my_kcol][kg * 16 + kc]);
                    o_acc[qg][d16][e] += acc;
                }
        __syncthreads();
    }

    // After tile loop: reduce row_m and row_d across 16 lanes sharing each query row (qg,e)
    // Lanes with same (l/16) share the same (qg,e) → different my_kcol (l%16)
    // shfl_xor offsets 1,2,4,8 reduce across l%16 within same (l/16) group
    for(int qg = 0; qg < 4; qg++)
        for(int e = 0; e < 4; e++)
        {
            float gm = row_m[qg][e];
            gm = fmaxf(gm, __shfl_xor(gm, 1));
            gm = fmaxf(gm, __shfl_xor(gm, 2));
            gm = fmaxf(gm, __shfl_xor(gm, 4));
            gm = fmaxf(gm, __shfl_xor(gm, 8));

            const float br = flash2_expf(row_m[qg][e] - gm);
            for(int d16 = 0; d16 < D16; d16++)
                o_acc[qg][d16][e] *= br;
            row_d[qg][e] *= br;

            row_d[qg][e] += __shfl_xor(row_d[qg][e], 1);
            row_d[qg][e] += __shfl_xor(row_d[qg][e], 2);
            row_d[qg][e] += __shfl_xor(row_d[qg][e], 4);
            row_d[qg][e] += __shfl_xor(row_d[qg][e], 8);
        }

    // Normalize and write to global memory via shared mem
    {
        __shared__ float so[64][D];
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                const int row_in_so  = qg * 16 + my_e_base + e;
                const float inv      = (row_d[qg][e] > 0.f) ? 1.f / row_d[qg][e] : 0.f;
                for(int d16 = 0; d16 < D16; d16++)
                    so[row_in_so][d16 * 16 + my_kcol] = o_acc[qg][d16][e] * inv;
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
