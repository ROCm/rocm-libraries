// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Exported: flash2_v7_hipdnn_d128, flash2_v7_hipdnn_d64
// Build: hipcc --offload-arch=gfx{942,950} -O3 --cuda-device-only -x hip
//        -I/opt/rocm/include HipFlash2FwdPlan.hip -o kernels/hip_flash2_fwd_gfx{942,950}.co
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

extern "C" __global__
    __launch_bounds__(64, 2) void flash2_v7_hipdnn_d128(const __half* __restrict__ ptr_q,
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
    constexpr int D = 128, D16 = 8;
    using fq = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    using fk = fragment<matrix_b, 16, 16, 16, __half, col_major>;
    using fs = fragment<accumulator, 16, 16, 16, float>;
    using fp = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    using fv = fragment<matrix_b, 16, 16, 16, __half, row_major>;
    using fo = fragment<accumulator, 16, 16, 16, float>;
    const int qi_blk = blockIdx.x, b = blockIdx.y, hq = blockIdx.z, lane = threadIdx.x;
    const int hk = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);
    const int qi_base = qi_blk * 64, my_qrow = lane % 16, my_cg = lane / 16, qi_max = qi_base + 63;
    const bool my_valid = (qi_base + my_cg * 16 + my_qrow < seq_len_q);
    __shared__ __half sK[D][64], sV[D][64];
    fq q_frags[4][D16];
    const __half* Qp = ptr_q + (size_t)b * q_stride_batch + (size_t)hq * q_stride_head
                       + (size_t)qi_base * q_stride_seq;
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
        {
            if(qi_base + qg * 16 < seq_len_q)
                load_matrix_sync(
                    q_frags[qg][d16], Qp + (size_t)qg * 16 * q_stride_seq + d16 * 16, q_stride_seq);
            else
                fill_fragment(q_frags[qg][d16], __half(0));
        }
    fo o_frags[4][D16];
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
            fill_fragment(o_frags[qg][d16], 0.f);
    // Per-group softmax state: each of the 4 query groups has its own max+normalizer
    float row_m[4] = {-1e38f, -1e38f, -1e38f, -1e38f}, row_d[4] = {0.f, 0.f, 0.f, 0.f};
    for(int kb = 0; kb < (seq_len_kv + 63) / 64; kb++)
    {
        const int ki0 = kb * 64;
        if(causal && ki0 > qi_max)
            break;
        const bool ft = (!causal) || (ki0 + 63 <= qi_base);
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
                sK[d][lane] = __float2half(0.f);
                sV[d][lane] = __float2half(0.f);
            }
        }
        __syncthreads();
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
        // Defect A fix: per-group softmax state (row_m[4], row_d[4])
        // Each group qg tracks its own running max and normalizer independently.
        float nm[4], ms[4][4][4];
        for(int qg = 0; qg < 4; qg++)
            nm[qg] = row_m[qg];
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int e = 0; e < 4; e++)
                {
                    float s = sf[qg][kg].x[e] * scale;
                    if(!ft && (ki0 + kg * 16 + my_cg * 4 + e) > (qi_base + qg * 16 + my_qrow))
                        s = -1e38f;
                    if(ki0 + kg * 16 + my_cg * 4 + e >= seq_len_kv)
                        s = -1e38f;
                    ms[qg][kg][e] = s;
                    // Only update group qg's max from lanes that own group qg rows
                    if(qg == my_cg)
                        nm[qg] = fmaxf(nm[qg], s);
                }
        // Reduce nm[qg] within the 16-lane group (offsets 1,2,4,8 stay in-group)
        for(int qg = 0; qg < 4; qg++)
            for(int off = 1; off < 16; off <<= 1)
                nm[qg] = fmaxf(nm[qg], __shfl_xor(nm[qg], off));
        // Per-group rescale of output accumulators and denominator
        for(int qg = 0; qg < 4; qg++)
        {
            const float br = flash2_expf(row_m[qg] - nm[qg]);
            row_m[qg] = nm[qg];
            for(int d16 = 0; d16 < D16; d16++)
                for(int e = 0; e < 4; e++)
                    o_frags[qg][d16].x[e] *= br;
            row_d[qg] *= br;
        }
        fp pf[4][4];
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
            {
                fill_fragment(pf[qg][kg], __half(0));
                for(int e = 0; e < 4; e++)
                {
                    float pv = flash2_expf(ms[qg][kg][e] - row_m[qg]);
                    pf[qg][kg].x[e] = (_Float16)pv;
                    if(qg == my_cg)
                        row_d[qg] += pv;
                }
            }
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
    if(my_valid)
    {
        __shared__ float so[64][D];
        // Each group uses its own normalizer
        for(int qg = 0; qg < 4; qg++)
        {
            const float inv = (row_d[qg] > 0.f) ? 1.f / row_d[qg] : 0.f;
            for(int d16 = 0; d16 < D16; d16++)
            {
                for(int e = 0; e < 4; e++)
                    o_frags[qg][d16].x[e] *= inv;
                store_matrix_sync(&so[qg * 16][d16 * 16], o_frags[qg][d16], D, mem_row_major);
            }
        }
        __syncthreads();
        for(int qg = 0; qg < 4; qg++)
        {
            const int oq = qi_base + qg * 16 + my_qrow;
            if(oq < seq_len_q)
            {
                __half* op = ptr_o + (size_t)b * o_stride_batch + (size_t)hq * o_stride_head
                             + (size_t)oq * o_stride_seq;
                for(int d = 0; d < D; d++)
                    op[d] = __float2half(so[qg * 16 + my_qrow][d]);
            }
        }
    }
}

extern "C" __global__
    __launch_bounds__(64, 2) void flash2_v7_hipdnn_d64(const __half* __restrict__ ptr_q,
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
    constexpr int D = 64, D16 = 4;
    using fq = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    using fk = fragment<matrix_b, 16, 16, 16, __half, col_major>;
    using fs = fragment<accumulator, 16, 16, 16, float>;
    using fp = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    using fv = fragment<matrix_b, 16, 16, 16, __half, row_major>;
    using fo = fragment<accumulator, 16, 16, 16, float>;
    const int qi_blk = blockIdx.x, b = blockIdx.y, hq = blockIdx.z, lane = threadIdx.x;
    const int hk = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);
    const int qi_base = qi_blk * 64, my_qrow = lane % 16, my_cg = lane / 16, qi_max = qi_base + 63;
    const bool my_valid = (qi_base + my_cg * 16 + my_qrow < seq_len_q);
    __shared__ __half sK[D][64], sV[D][64];
    fq q_frags[4][D16];
    const __half* Qp = ptr_q + (size_t)b * q_stride_batch + (size_t)hq * q_stride_head
                       + (size_t)qi_base * q_stride_seq;
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
        {
            if(qi_base + qg * 16 < seq_len_q)
                load_matrix_sync(
                    q_frags[qg][d16], Qp + (size_t)qg * 16 * q_stride_seq + d16 * 16, q_stride_seq);
            else
                fill_fragment(q_frags[qg][d16], __half(0));
        }
    fo o_frags[4][D16];
    for(int qg = 0; qg < 4; qg++)
        for(int d16 = 0; d16 < D16; d16++)
            fill_fragment(o_frags[qg][d16], 0.f);
    // Per-group softmax state: each of the 4 query groups has its own max+normalizer
    float row_m[4] = {-1e38f, -1e38f, -1e38f, -1e38f}, row_d[4] = {0.f, 0.f, 0.f, 0.f};
    for(int kb = 0; kb < (seq_len_kv + 63) / 64; kb++)
    {
        const int ki0 = kb * 64;
        if(causal && ki0 > qi_max)
            break;
        const bool ft = (!causal) || (ki0 + 63 <= qi_base);
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
                sK[d][lane] = __float2half(0.f);
                sV[d][lane] = __float2half(0.f);
            }
        }
        __syncthreads();
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
        // Defect A fix: per-group softmax state (row_m[4], row_d[4])
        // Each group qg tracks its own running max and normalizer independently.
        float nm[4], ms[4][4][4];
        for(int qg = 0; qg < 4; qg++)
            nm[qg] = row_m[qg];
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
                for(int e = 0; e < 4; e++)
                {
                    float s = sf[qg][kg].x[e] * scale;
                    if(!ft && (ki0 + kg * 16 + my_cg * 4 + e) > (qi_base + qg * 16 + my_qrow))
                        s = -1e38f;
                    if(ki0 + kg * 16 + my_cg * 4 + e >= seq_len_kv)
                        s = -1e38f;
                    ms[qg][kg][e] = s;
                    // Only update group qg's max from lanes that own group qg rows
                    if(qg == my_cg)
                        nm[qg] = fmaxf(nm[qg], s);
                }
        // Reduce nm[qg] within the 16-lane group (offsets 1,2,4,8 stay in-group)
        for(int qg = 0; qg < 4; qg++)
            for(int off = 1; off < 16; off <<= 1)
                nm[qg] = fmaxf(nm[qg], __shfl_xor(nm[qg], off));
        // Per-group rescale of output accumulators and denominator
        for(int qg = 0; qg < 4; qg++)
        {
            const float br = flash2_expf(row_m[qg] - nm[qg]);
            row_m[qg] = nm[qg];
            for(int d16 = 0; d16 < D16; d16++)
                for(int e = 0; e < 4; e++)
                    o_frags[qg][d16].x[e] *= br;
            row_d[qg] *= br;
        }
        fp pf[4][4];
        for(int qg = 0; qg < 4; qg++)
            for(int kg = 0; kg < 4; kg++)
            {
                fill_fragment(pf[qg][kg], __half(0));
                for(int e = 0; e < 4; e++)
                {
                    float pv = flash2_expf(ms[qg][kg][e] - row_m[qg]);
                    pf[qg][kg].x[e] = (_Float16)pv;
                    if(qg == my_cg)
                        row_d[qg] += pv;
                }
            }
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
    if(my_valid)
    {
        __shared__ float so[64][D];
        // Each group uses its own normalizer
        for(int qg = 0; qg < 4; qg++)
        {
            const float inv = (row_d[qg] > 0.f) ? 1.f / row_d[qg] : 0.f;
            for(int d16 = 0; d16 < D16; d16++)
            {
                for(int e = 0; e < 4; e++)
                    o_frags[qg][d16].x[e] *= inv;
                store_matrix_sync(&so[qg * 16][d16 * 16], o_frags[qg][d16], D, mem_row_major);
            }
        }
        __syncthreads();
        for(int qg = 0; qg < 4; qg++)
        {
            const int oq = qi_base + qg * 16 + my_qrow;
            if(oq < seq_len_q)
            {
                __half* op = ptr_o + (size_t)b * o_stride_batch + (size_t)hq * o_stride_head
                             + (size_t)oq * o_stride_seq;
                for(int d = 0; d < D; d++)
                    op[d] = __float2half(so[qg * 16 + my_qrow][d]);
            }
        }
    }
}
