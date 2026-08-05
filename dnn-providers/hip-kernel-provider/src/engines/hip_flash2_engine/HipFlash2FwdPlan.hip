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
// Consequences used throughout:
//   - softmax state is per (qg, e): row_m[4][4], row_d[4][4]  (16 states/lane)
//   - causal mask:  key_col = kg*16 + lane%16
//                   query_row = qg*16 + (lane/16)*4 + e
//   - row-max reduction over keys ↔ reduction over lane%16 → shfl_xor offsets 1,2,4,8
//   - P as matrix_a: matrix_a[row=lane%16][col=(lane/16)*4+e]
//     so pf[qg][kg].x[e] must hold P[ kg*16+lane%16 ][ qg*16+(lane/16)*4+e ]
//     which equals the value at sf[qg][kg].x[e] after softmax (same slot, re-interpreted)
//
// Param layout matches Flash2KernelArgs exactly (raw buffer via HIP_LAUNCH_PARAM_BUFFER_POINTER):
//   ptr_q,ptr_k,ptr_v,ptr_o, batch,num_heads_q,num_heads_k,seq_len_q,seq_len_kv,head_dim,
//   scale,causal, q_stride_{batch,head,seq}, k_stride_{batch,head,seq},
//   v_stride_{batch,head,seq}, o_stride_{batch,head,seq}

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

// Generic kernel body templated on D and D16 = D/16
template <int D, int D16>
__device__ void flash2_fwd_body(const __half* __restrict__ ptr_q,
                                const __half* __restrict__ ptr_k,
                                const __half* __restrict__ ptr_v,
                                __half* __restrict__ ptr_o,
                                int batch,
                                int num_heads_q,
                                int num_heads_k,
                                int seq_len_q,
                                int seq_len_kv,
                                int head_dim,
                                float scale,
                                int causal,
                                int q_stride_batch,
                                int q_stride_head,
                                int q_stride_seq,
                                int k_stride_batch,
                                int k_stride_head,
                                int k_stride_seq,
                                int v_stride_batch,
                                int v_stride_head,
                                int v_stride_seq,
                                int o_stride_batch,
                                int o_stride_head,
                                int o_stride_seq)
{
    using namespace rocwmma;
    using fq = fragment<matrix_a,    16, 16, 16, __half, row_major>;
    using fk = fragment<matrix_b,    16, 16, 16, __half, col_major>;
    using fs = fragment<accumulator, 16, 16, 16, float>;
    // P as matrix_a: lane l, elem e → P[row=l%16][col=(l/16)*4+e]
    // The accumulator sf.x[e] holds S[(l/16)*4+e][l%16].
    // Re-interpreting that slot as matrix_a gives P[l%16][(l/16)*4+e] = S[(l/16)*4+e][l%16]
    // which is exactly the transpose — so direct assignment pf.x[e] = softmax(sf.x[e]) is correct.
    using fp = fragment<matrix_a,    16, 16, 16, __half, row_major>;
    using fv = fragment<matrix_b,    16, 16, 16, __half, row_major>;
    using fo = fragment<accumulator, 16, 16, 16, float>;

    const int qi_blk = blockIdx.x;
    const int b      = blockIdx.y;
    const int hq     = blockIdx.z;
    const int lane   = threadIdx.x;
    const int hk     = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);

    // In the accumulator layout, lane l covers query rows (l/16)*4 .. (l/16)*4+3
    // across all 4 groups (qg=0..3), and key column l%16 within each key group.
    const int qi_base = qi_blk * 64;
    const int qi_max  = qi_base + 63;  // last query index in this block
    const int my_e_base = (lane / 16) * 4;  // first query-row-within-group for this lane
    const int my_kcol   = lane % 16;         // key column index within a 16-wide key tile

    __shared__ __half sK[D][64], sV[D][64];

    // Load Q fragments (4 groups × D16 tiles)
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

    // Output accumulators and softmax state
    fo o_frags[4][D16];
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
            fill_fragment(o_frags[qg][d16], 0.f);

    // Softmax state: per (qg, e) — 4 groups × 4 elems = 16 states per lane
    // row_m[qg][e] = running max for query row qg*16 + my_e_base + e
    // row_d[qg][e] = running normalizer
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

        // true when the entire key tile is strictly to the left of this query tile
        // (no masking needed within the tile)
        const bool full_tile = (!causal) || (ki0 + 63 <= qi_base);

        // Load K, V into shared memory (lane loads column ki0+lane)
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

        // S = Q · Kᵀ  (accumulator layout)
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
        // Softmax over sf, using the measured accumulator layout:
        //   sf[qg][kg].x[e]  →  S[ qg*16 + (lane/16)*4 + e ][ kg*16 + lane%16 ]
        //
        // Row index:  query_row = qg*16 + my_e_base + e   (per lane, per elem)
        // Col index:  key_col   = kg*16 + my_kcol         (same for all e in this lane)
        //
        // Row-max reduction: all lanes sharing the same query row must exchange.
        // Lanes sharing query row r = qg*16 + e_base + e are those with the same
        // qg, same e, and any my_kcol (lane%16).  They differ only in (lane/16),
        // i.e. offsets 16, 32, 48 from the base.  So shfl_xor offsets 1,2,4,8
        // exchange across lane%16 (different key columns, same query row within the
        // same (qg,e)) which is NOT what we want.
        //
        // Wait — let's be precise.  my_e_base = (lane/16)*4.
        // Two lanes l1 and l2 share the same query row for the same (qg,e) when:
        //   qg*16 + (l1/16)*4 + e  ==  qg*16 + (l2/16)*4 + e
        //   ↔  l1/16 == l2/16
        // So lanes sharing the same query row (qg,e) have the same (lane/16) group
        // but differ only in lane%16 (0..15).
        // Reduction across them uses shfl_xor offsets 1,2,4,8.
        // -------------------------------------------------------------------

        // Step 1: compute per-lane per-elem max over this key tile
        float nm[4][4];
        float ms[4][4][4];  // scaled scores after masking
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
                nm[qg][e] = row_m[qg][e];

        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int e = 0; e < 4; e++)
                {
                    float s = sf[qg][kg].x[e] * scale;

                    // Causal mask: mask out positions where key_col > query_row
                    const int key_col   = ki0 + kg * 16 + my_kcol;
                    const int query_row = qi_base + qg * 16 + my_e_base + e;
                    if(!full_tile && key_col > query_row)
                        s = -1e38f;
                    // Out-of-bounds key
                    if(key_col >= seq_len_kv)
                        s = -1e38f;

                    ms[qg][kg][e] = s;
                    nm[qg][e] = fmaxf(nm[qg][e], s);
                }

        // Step 2: reduce nm[qg][e] across lanes sharing the same query row (same qg, e, lane/16)
        // Those lanes differ only in lane%16 → shfl_xor offsets 1,2,4,8
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                nm[qg][e] = fmaxf(nm[qg][e], __shfl_xor(nm[qg][e], 1));
                nm[qg][e] = fmaxf(nm[qg][e], __shfl_xor(nm[qg][e], 2));
                nm[qg][e] = fmaxf(nm[qg][e], __shfl_xor(nm[qg][e], 4));
                nm[qg][e] = fmaxf(nm[qg][e], __shfl_xor(nm[qg][e], 8));
            }

        // Step 3: rescale existing output and normalizer, update row_m
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                const float br = flash2_expf(row_m[qg][e] - nm[qg][e]);
                row_m[qg][e] = nm[qg][e];
                // rescale o_frags: each output elem belongs to one (qg,e) query row
                // o_frags[qg][d16].x[e] holds O[ qg*16+(lane/16)*4+e ][ d16*16+lane%16 ]
                for(int d16 = 0; d16 < D16; d16++)
                    o_frags[qg][d16].x[e] *= br;
                row_d[qg][e] *= br;
            }

        // Step 4: compute P = softmax weights, accumulate into pf (matrix_a layout)
        // and reduce row_d
        fp pf[4][4];
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
            {
                fill_fragment(pf[qg][kg], __half(0));
                for(int e = 0; e < 4; e++)
                {
                    float pv = flash2_expf(ms[qg][kg][e] - row_m[qg][e]);
                    // pf[qg][kg].x[e] will be consumed as matrix_a:
                    // matrix_a lane l elem e → A[row=l%16][col=(l/16)*4+e]
                    // sf (accumulator) lane l elem e → C[row=(l/16)*4+e][col=l%16]
                    // So pf.x[e] = sf.x[e] gives P[ key_col=l%16 ][ query_row_e=(l/16)*4+e ]
                    // which is P^T in the sense needed for the second GEMM.
                    pf[qg][kg].x[e] = (_Float16)pv;
                    row_d[qg][e] += pv;
                }
            }

        // Step 5: reduce row_d (same axis as nm: shfl_xor 1,2,4,8)
        for(int qg = 0; qg < 4; qg++)
            for(int e = 0; e < 4; e++)
            {
                row_d[qg][e] += __shfl_xor(row_d[qg][e], 1);
                row_d[qg][e] += __shfl_xor(row_d[qg][e], 2);
                row_d[qg][e] += __shfl_xor(row_d[qg][e], 4);
                row_d[qg][e] += __shfl_xor(row_d[qg][e], 8);
            }

        // Step 6: O += P · V
        // pf is matrix_a with layout [row=lane%16][col=(lane/16)*4+e]
        // vf is matrix_b: V loaded row-major from sV
        // o_frags is accumulator [row=(lane/16)*4+e][col=lane%16]
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
    // Write output
    // o_frags[qg][d16].x[e] holds O[ qg*16 + (lane/16)*4 + e ][ d16*16 + lane%16 ]
    // Store to shared memory then write to global memory row by row.
    // -------------------------------------------------------------------
    {
        __shared__ float so[64][D];

        for(int qg = 0; qg < 4; qg++)
        {
            for(int d16 = 0; d16 < D16; d16++)
            {
                // Normalize: divide each (qg,e) output element by its normalizer
                for(int e = 0; e < 4; e++)
                {
                    const float inv = (row_d[qg][e] > 0.f) ? 1.f / row_d[qg][e] : 0.f;
                    o_frags[qg][d16].x[e] *= inv;
                }
                // store_matrix_sync writes the accumulator to shared in row-major order:
                // so[ qg*16 + (lane/16)*4 + e ][ d16*16 + lane%16 ] = o_frags[qg][d16].x[e]
                store_matrix_sync(&so[qg * 16][d16 * 16], o_frags[qg][d16], D, mem_row_major);
            }
        }
        __syncthreads();

        // Each lane writes its portion of the output
        // lane covers column positions: my_kcol, my_kcol+16, my_kcol+32, ... (via so layout)
        // Simpler: let each lane write the rows it owns in so (my_e_base+e, for each qg)
        for(int qg = 0; qg < 4; qg++)
        {
            for(int e = 0; e < 4; e++)
            {
                const int oq = qi_base + qg * 16 + my_e_base + e;
                if(oq < seq_len_q)
                {
                    // This lane is responsible for column my_kcol of row oq
                    // But we need all D columns. Use the so array written by store_matrix_sync.
                    // Only one lane per row/column cell owns it; each row oq has all 64 columns
                    // written by 16 lanes (the 16 with same (lane/16)*4+e = my_e_base+e, same qg).
                    // Actually store_matrix_sync distributes so each lane handles its own column
                    // subset. For a complete row write, iterate all D using the shared array.
                    // The row qg*16+my_e_base+e in so[] is fully populated after syncthreads.
                    // Divide the D-wide row among the 16 lanes in the same (lane/16) group:
                    // lane%16 ∈ [0,15], so each handles column my_kcol + k*16 for k=0..D16-1.
                    __half* op = ptr_o + (size_t)b * o_stride_batch + (size_t)hq * o_stride_head
                                 + (size_t)oq * o_stride_seq;
                    for(int k = 0; k < D16; k++)
                        op[my_kcol + k * 16] = __float2half(so[qg * 16 + my_e_base + e][my_kcol + k * 16]);
                }
            }
        }
    }
}

extern "C" __global__
    __launch_bounds__(64, 2) void flash2_v7_hipdnn_d128(const __half* __restrict__ ptr_q,
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
    flash2_fwd_body<128, 8>(ptr_q, ptr_k, ptr_v, ptr_o, batch, num_heads_q, num_heads_k,
                            seq_len_q, seq_len_kv, head_dim, scale, causal,
                            q_stride_batch, q_stride_head, q_stride_seq,
                            k_stride_batch, k_stride_head, k_stride_seq,
                            v_stride_batch, v_stride_head, v_stride_seq,
                            o_stride_batch, o_stride_head, o_stride_seq);
}

extern "C" __global__
    __launch_bounds__(64, 2) void flash2_v7_hipdnn_d64(const __half* __restrict__ ptr_q,
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
    flash2_fwd_body<64, 4>(ptr_q, ptr_k, ptr_v, ptr_o, batch, num_heads_q, num_heads_k,
                           seq_len_q, seq_len_kv, head_dim, scale, causal,
                           q_stride_batch, q_stride_head, q_stride_seq,
                           k_stride_batch, k_stride_head, k_stride_seq,
                           v_stride_batch, v_stride_head, v_stride_seq,
                           o_stride_batch, o_stride_head, o_stride_seq);
}
