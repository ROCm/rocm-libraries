// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Exported: flash2_v7_hipdnn_d128, flash2_v7_hipdnn_d64
// Build: hipcc --offload-arch=gfx942 -O3 --cuda-device-only -x hip
//        -I/opt/rocm/include HipFlash2FwdPlan.hip -o kernels/hip_flash2_fwd_gfx942.co
//
// FRAGMENT LAYOUT (measured by Samuel Reeder on gfx942, rocWMMA 16×16×16):
//   accumulator : lane l, elem e  →  C[ row = (l/16)*4 + e ][ col = l%16 ]
//   matrix_a    : lane l, elem e  →  A[ row = l%16 ][ col = (l/16)*4 + e ]
//
// For S = Q·Kᵀ in accumulator layout:
//   sf[qg][kg].x[e]  holds  S[ qg*16 + (lane/16)*4 + e ][ kg*16 + lane%16 ]
//   → e indexes QUERY row within tile; lane%16 indexes KEY column
//
// Row-max reduction: lanes sharing the same query row (qg,e) have the same
// (lane/16) value but differ in lane%16 → shfl_xor offsets 1,2,4,8.
//
// KEY INVARIANT: row_m[qg][e] and row_d[qg][e] are PER-LANE local accumulators.
// DO NOT reduce them inside the tile loop — doing so would make all 16 lanes
// equal after tile 1 and exponentially inflate row_d on subsequent tiles.
// Reduce ONCE after the tile loop when computing the final output.
//
// Param layout matches Flash2KernelArgs exactly:
//   ptr_q,ptr_k,ptr_v,ptr_o, batch,num_heads_q,num_heads_k,seq_len_q,seq_len_kv,
//   head_dim, scale, causal, q/k/v/o strides (batch, head, seq)

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
    using fp = fragment<matrix_a,    16, 16, 16, __half, row_major>;
    using fv = fragment<matrix_b,    16, 16, 16, __half, row_major>;
    using fo = fragment<accumulator, 16, 16, 16, float>;

    const int qi_blk = blockIdx.x, b = blockIdx.y, hq = blockIdx.z;
    const int lane   = threadIdx.x;
    const int hk     = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);
    const int qi_base  = qi_blk * 64;
    const int qi_max   = qi_base + 63;
    // lane l: e indexes query row as qg*16+(l/16)*4+e; key col is kg*16+l%16
    const int my_e_base = (lane / 16) * 4;
    const int my_kcol   = lane % 16;

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
                                 Qp + (size_t)qg * 16 * q_stride_seq + d16 * 16,
                                 q_stride_seq);
            else
                fill_fragment(q_frags[qg][d16], __half(0));
        }

    // Output accumulators
    fo o_frags[4][D16];
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
            fill_fragment(o_frags[qg][d16], 0.f);

    // Per-lane softmax state: [qg][e] — NOT reduced across lanes inside tile loop.
    // Each lane tracks its own local row max and normalizer for each (qg,e) query row.
    // After the tile loop we reduce once across the 16 lanes sharing each query row.
    float row_m[4][4], row_d[4][4];
    for(int qg = 0; qg < 4; qg++)
        for(int e = 0; e < 4; e++)
        {
            row_m[qg][e] = -1e38f;
            row_d[qg][e] = 0.f;
        }

    for(int kb = 0; kb < (seq_len_kv + 63) / 64; kb++)
    {
        const int ki0 = kb * 64;
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
            for(int d = 0; d < D; d++) { sK[d][lane] = kp[d]; sV[d][lane] = vp[d]; }
        }
        else
        {
            for(int d = 0; d < D; d++) { sK[d][lane] = __half(0); sV[d][lane] = __half(0); }
        }
        __syncthreads();

        // S = Q·Kᵀ
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

        // -------------------------------------------------------------------
        // Per-lane local tile max (no cross-lane reduction here)
        // For lane l, sf[qg][kg].x[e] = S[qg*16+my_e_base+e][kg*16+my_kcol]
        // This lane is responsible for key column my_kcol in each key group.
        // row_m[qg][e] accumulates the running max for THIS LANE'S key columns only.
        // -------------------------------------------------------------------
        float tile_m[4][4];
        float ms[4][4][4];
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
                tile_m[qg][e] = row_m[qg][e];

        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int e = 0; e < 4; e++)
                {
                    float s = sf[qg][kg].x[e] * scale;
                    const int key_col   = ki0 + kg * 16 + my_kcol;
                    const int query_row = qi_base + qg * 16 + my_e_base + e;
                    if(!full_tile && key_col > query_row) s = -1e38f;
                    if(key_col >= seq_len_kv) s = -1e38f;
                    ms[qg][kg][e] = s;
                    tile_m[qg][e] = fmaxf(tile_m[qg][e], s);
                }

        // Rescale previous output and normalizer with new local max
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                const float br = flash2_expf(row_m[qg][e] - tile_m[qg][e]);
                row_m[qg][e] = tile_m[qg][e];
                for(int d16 = 0; d16 < D16; d16++)
                    o_frags[qg][d16].x[e] *= br;
                row_d[qg][e] *= br;
            }

        // Softmax weights and local row_d accumulation (no reduction)
        fp pf[4][4];
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
            {
                fill_fragment(pf[qg][kg], __half(0));
                for(int e = 0; e < 4; e++)
                {
                    float pv = flash2_expf(ms[qg][kg][e] - row_m[qg][e]);
                    pf[qg][kg].x[e] = (_Float16)pv;
                    row_d[qg][e] += pv;  // local accumulation only
                }
            }

        // O += P·V
        for(int kg = 0; kg < 4; kg++)
            for(int d16 = 0; d16 < D16; d16++)
            {
                fv vf;
                load_matrix_sync(vf, &sV[d16 * 16][kg * 16], 64);
                for(int qg = 0; qg < 4; qg++)
                    mma_sync(o_frags[qg][d16], pf[qg][kg], vf, o_frags[qg][d16]);
            }
        __syncthreads();
    }

    // -------------------------------------------------------------------
    // After tile loop: reduce row_m and row_d across the 16 lanes sharing
    // each query row. Lanes sharing query row (qg,e) have the same (lane/16)
    // but differ in lane%16 → shfl_xor offsets 1,2,4,8.
    // Also rescale o_frags: each lane's local row_m may differ from the
    // global max, so we need a two-step: find global max, rescale, then sum.
    // -------------------------------------------------------------------
    {
        // Step 1: find per-query-row global max across 16 lanes
        float global_m[4][4];
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                global_m[qg][e] = row_m[qg][e];
                global_m[qg][e] = fmaxf(global_m[qg][e], __shfl_xor(global_m[qg][e], 1));
                global_m[qg][e] = fmaxf(global_m[qg][e], __shfl_xor(global_m[qg][e], 2));
                global_m[qg][e] = fmaxf(global_m[qg][e], __shfl_xor(global_m[qg][e], 4));
                global_m[qg][e] = fmaxf(global_m[qg][e], __shfl_xor(global_m[qg][e], 8));
            }

        // Step 2: rescale this lane's o_frags and row_d to the global max
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                const float br = flash2_expf(row_m[qg][e] - global_m[qg][e]);
                for(int d16 = 0; d16 < D16; d16++)
                    o_frags[qg][d16].x[e] *= br;
                row_d[qg][e] *= br;
            }

        // Step 3: sum row_d and o_frags across the 16 lanes
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                row_d[qg][e] += __shfl_xor(row_d[qg][e], 1);
                row_d[qg][e] += __shfl_xor(row_d[qg][e], 2);
                row_d[qg][e] += __shfl_xor(row_d[qg][e], 4);
                row_d[qg][e] += __shfl_xor(row_d[qg][e], 8);
            }
        for(int qg = 0; qg < 4; qg++)
            for(int d16 = 0; d16 < D16; d16++)
                for(int e = 0; e < 4; e++)
                {
                    o_frags[qg][d16].x[e] += __shfl_xor(o_frags[qg][d16].x[e], 1);
                    o_frags[qg][d16].x[e] += __shfl_xor(o_frags[qg][d16].x[e], 2);
                    o_frags[qg][d16].x[e] += __shfl_xor(o_frags[qg][d16].x[e], 4);
                    o_frags[qg][d16].x[e] += __shfl_xor(o_frags[qg][d16].x[e], 8);
                }
    }

    // -------------------------------------------------------------------
    // Write output: normalize and store
    // o_frags[qg][d16].x[e] holds (after reduction) the full sum for
    // query row qg*16+my_e_base+e, output col range covered by this warp.
    // After normalize, store via shared mem and write to global.
    // -------------------------------------------------------------------
    {
        __shared__ float so[64][D];
        for(int qg = 0; qg < 4; qg++)
        {
            for(int d16 = 0; d16 < D16; d16++)
            {
                for(int e = 0; e < 4; e++)
                {
                    const float inv = (row_d[qg][e] > 0.f) ? 1.f / row_d[qg][e] : 0.f;
                    o_frags[qg][d16].x[e] *= inv;
                }
                store_matrix_sync(&so[qg * 16][d16 * 16], o_frags[qg][d16], D, mem_row_major);
            }
        }
        __syncthreads();

        // Write: each lane writes column my_kcol for all its query rows
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
