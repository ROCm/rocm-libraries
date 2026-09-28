// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// v23 = fully parameterized S-transpose Flash-Attention 2 for joint tuning.
//
// ---------------------------------------------------------------------------
// WHY THIS EXISTS
// Earlier versions were tuned one knob at a time, which mis-attributes results
// whenever knobs interact. Two known cases:
//   * QG=2 was measured on the OLD (pre-v18) LDS layout and looked bad. Its
//     register saving may only pay off now that staging/fragment loads are wide.
//   * Pipelining was measured only at QG=4, where the live set is already
//     ~288 VGPR + 256 AGPR. It doubled s_waitcnt and lost 30%. At QG=2 there is
//     real headroom, so the same transform may win.
// So every knob below is a compile-time parameter and the whole grid gets
// benchmarked together.
//
// KNOBS
//   F2_WAVES  waves per CTA (4 or 8)         -> blockDim = 64*WAVES
//   F2_QG     query groups per wave (1,2,3)  -> queries/CTA = WAVES*QG*16
//   F2_KG     key groups per tile  (2,4)     -> BK = KG*16 (32 or 64)
//
// The five variants the dispatcher ships, and their launch geometry:
//   w4q1k4  WAVES=4 QG=1 KG=4  ->  256 threads,  64 queries/CTA
//   w8q1k4  WAVES=8 QG=1 KG=4  ->  512 threads, 128 queries/CTA
//   w8q2k4  WAVES=8 QG=2 KG=4  ->  512 threads, 256 queries/CTA
//   w8q3k2  WAVES=8 QG=3 KG=2  ->  512 threads, 384 queries/CTA
//   w8q3k4  WAVES=8 QG=3 KG=4  ->  512 threads, 384 queries/CTA
// The defaults below are the tuning-sweep defaults, not a shipped variant.
//   F2_PIPE   0/1 software-pipelined prefetch of the next KV tile
//   F2_PAD    pad on the D axis of sK   (LDS bank skew)
//   F2_PADK   pad on the key axis of sVT (LDS bank skew; 0 is a known trap)
//   F2_VTMAJOR 1 = d-major V (wide V fragment reads), 0 = key-major (v18)
//
// REGISTER MODEL (per wave, D=128 => D16=8), which is what QG trades against:
//   q_frags[QG][8] matrix_b  2 VGPR each -> 16*QG
//   o_acc  [QG][8] accum     4 VGPR each -> 32*QG   (AGPRs)
//   st  [KG][QG]   accum     4 VGPR each ->  4*KG*QG
//   pf  [KG][QG]   matrix_b  2 VGPR each ->  2*KG*QG
//   + pipelining: 2*NCH half8 = 8*NCH VGPR
//
// THE MATH IS FIXED (validated, maxErr <= 1e-3 vs CPU FP32 and vs torch SDPA).
// Measured gfx942 layouts, lane l elem e:
//   matrix_a    : A[row=l%16][k=(l/16)*4+e]
//   matrix_b    : B[k=(l/16)*4+e][col=l%16]
//   accumulator : C[row=(l/16)*4+e][col=l%16]  == matrix_b == transpose(a)
// so S^T = K*Q^T leaves the score accumulator already in matrix_b layout and
// P^T needs no conversion; then O^T = V^T*P^T.
// Softmax reduces across lane groups: __shfl_xor offsets 16 and 32.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#ifndef F2_WAVES
#define F2_WAVES 4
#endif
#ifndef F2_QG
#define F2_QG 4
#endif
#ifndef F2_KG
#define F2_KG 4
#endif
#ifndef F2_PIPE
#define F2_PIPE 0
#endif
#ifndef F2_SKIPRESCALE
#define F2_SKIPRESCALE 1
#endif
#ifndef F2_PAD
#define F2_PAD 8
#endif
#ifndef F2_PADK
#define F2_PADK 8
#endif
#ifndef F2_VTMAJOR
#define F2_VTMAJOR 1
#endif

#define F2_BLOCK (F2_WAVES * 64)
#define F2_BK (F2_KG * 16)
#define F2_QPER (F2_WAVES * F2_QG * 16)

__device__ __forceinline__ float flash2_expf(float x)
{
#if defined(__gfx942__)
    return __builtin_amdgcn_exp2f(x * 1.4426950408f);
#else
    return exp2f(x * 1.4426950408f);
#endif
}

struct __align__(16) half8
{
    __half h[8];
};

template <int D, int D16>
__device__ void flash2_body_v23(const __half* __restrict__ ptr_q,
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
    constexpr int LD = D + F2_PAD; // sK  row stride (row = key)
    constexpr int LDK = F2_BK + F2_PADK; // sVT row stride (row = d)

    using fq_b = fragment<matrix_b, 16, 16, 16, __half, col_major>;
    using fk_a = fragment<matrix_a, 16, 16, 16, __half, row_major>;
#if F2_VTMAJOR
    using fv_a = fragment<matrix_a, 16, 16, 16, __half, row_major>;
#else
    using fv_a = fragment<matrix_a, 16, 16, 16, __half, col_major>;
#endif
    using fp_b = fragment<matrix_b, 16, 16, 16, __half, col_major>;
    using facc = fragment<accumulator, 16, 16, 16, float>;

    const int b = blockIdx.y, hq = blockIdx.z;
    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int hk = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);

    const int cta_q0 = blockIdx.x * F2_QPER;
    const int qi_base = cta_q0 + wave * (F2_QG * 16);
    const int qi_max = qi_base + F2_QG * 16 - 1;
    const int cta_qmax = cta_q0 + F2_QPER - 1;

    const int my_qcol = lane % 16;
    const int my_e_base = (lane / 16) * 4;

    __shared__ __half sK[F2_BK * LD];
#if F2_VTMAJOR
    __shared__ __half sVT[D * LDK];
#else
    __shared__ __half sVT[F2_BK * LD];
#endif

    fq_b q_frags[F2_QG][D16];
    const __half* Qp = ptr_q + (size_t)b * q_stride_batch + (size_t)hq * q_stride_head
                       + (size_t)qi_base * q_stride_seq;
    for(int qg = 0; qg < F2_QG; qg++)
        for(int d16 = 0; d16 < D16; d16++)
        {
            if(qi_base + qg * 16 + 15 < seq_len_q)
                load_matrix_sync(
                    q_frags[qg][d16], Qp + (size_t)qg * 16 * q_stride_seq + d16 * 16, q_stride_seq);
            else
            {
                fill_fragment(q_frags[qg][d16], __half(0));
                const int qrow = qi_base + qg * 16 + my_qcol;
                if(qrow < seq_len_q)
                    for(int e = 0; e < 4; e++)
                        q_frags[qg][d16].x[e] = Qp[(size_t)(qg * 16 + my_qcol) * q_stride_seq
                                                   + d16 * 16 + my_e_base + e];
            }
        }

    facc o_acc[F2_QG][D16];
    for(int qg = 0; qg < F2_QG; qg++)
        for(int d16 = 0; d16 < D16; d16++)
            fill_fragment(o_acc[qg][d16], 0.f);

    float row_m[F2_QG], row_d[F2_QG];
    for(int qg = 0; qg < F2_QG; qg++)
    {
        row_m[qg] = -1e38f;
        row_d[qg] = 0.f;
    }

    constexpr int CH = D / 8; // 16B chunks per key row
    constexpr int NTOT = F2_BK * CH; // chunks per tile
    constexpr int NCH = (NTOT + F2_BLOCK - 1) / F2_BLOCK; // chunks per thread

    const bool vec_ok = ((k_stride_seq % 8) == 0) && ((v_stride_seq % 8) == 0) && ((D % 8) == 0)
                        && ((((uintptr_t)ptr_k) & 15) == 0) && ((((uintptr_t)ptr_v) & 15) == 0);

    const __half* Kb = ptr_k + (size_t)b * k_stride_batch + (size_t)hk * k_stride_head;
    const __half* Vb = ptr_v + (size_t)b * v_stride_batch + (size_t)hk * v_stride_head;

    const int ntiles = (seq_len_kv + F2_BK - 1) / F2_BK;
    const int kb_stop
        = (causal && (cta_qmax / F2_BK + 1) < ntiles) ? (cta_qmax / F2_BK + 1) : ntiles;

    // ---- load one KV tile from global into registers -----------------------
    auto fetch = [&](int ki0, half8* pk, half8* pv) {
        for(int j = 0; j < NCH; j++)
        {
            const int idx = tid + j * F2_BLOCK;
            if(NTOT % F2_BLOCK != 0 && idx >= NTOT)
                break; // dead when NTOT divides evenly
            const int key = idx / CH, c = (idx % CH) * 8;
            const int kgl = ki0 + key;
            if(kgl < seq_len_kv)
            {
                pk[j] = *reinterpret_cast<const half8*>(Kb + (size_t)kgl * k_stride_seq + c);
                pv[j] = *reinterpret_cast<const half8*>(Vb + (size_t)kgl * v_stride_seq + c);
            }
            else
                for(int i = 0; i < 8; i++)
                {
                    pk[j].h[i] = __half(0);
                    pv[j].h[i] = __half(0);
                }
        }
    };
    // ---- commit registers into LDS ----------------------------------------
    auto commit = [&](const half8* pk, const half8* pv) {
        for(int j = 0; j < NCH; j++)
        {
            const int idx = tid + j * F2_BLOCK;
            if(NTOT % F2_BLOCK != 0 && idx >= NTOT)
                break; // dead when NTOT divides evenly
            const int key = idx / CH, c = (idx % CH) * 8;
            *reinterpret_cast<half8*>(&sK[key * LD + c]) = pk[j];
#if F2_VTMAJOR
            for(int i = 0; i < 8; i++)
                sVT[(c + i) * LDK + key] = pv[j].h[i];
#else
            *reinterpret_cast<half8*>(&sVT[key * LD + c]) = pv[j];
#endif
        }
    };

#if F2_PIPE
    half8 pk[NCH], pv[NCH];
    if(vec_ok && kb_stop > 0)
        fetch(0, pk, pv);
#endif

    for(int kb = 0; kb < kb_stop; kb++)
    {
        const int ki0 = kb * F2_BK;

        __syncthreads();
        if(vec_ok)
        {
#if F2_PIPE
            commit(pk, pv);
            if(kb + 1 < kb_stop)
                fetch((kb + 1) * F2_BK, pk, pv); // overlaps the MFMA below
#else
            half8 tk[NCH], tv[NCH];
            fetch(ki0, tk, tv);
            commit(tk, tv);
#endif
        }
        else
        {
            for(int idx = tid; idx < F2_BK * D; idx += F2_BLOCK)
            {
                const int key = idx / D, d = idx % D;
                const int kgl = ki0 + key;
                const bool ok = (kgl < seq_len_kv);
                sK[key * LD + d] = ok ? Kb[(size_t)kgl * k_stride_seq + d] : __half(0);
                const __half vv = ok ? Vb[(size_t)kgl * v_stride_seq + d] : __half(0);
#if F2_VTMAJOR
                sVT[d * LDK + key] = vv;
#else
                sVT[key * LD + d] = vv;
#endif
            }
        }
        __syncthreads();

        if(causal && ki0 > qi_max)
            continue;
        const bool full_tile = (!causal) || (ki0 + F2_BK - 1 <= qi_base);

        // ---- GEMM 1: S^T = K * Q^T ----------------------------------------
        facc st[F2_KG][F2_QG];
        for(int kg = 0; kg < F2_KG; kg++)
            for(int qg = 0; qg < F2_QG; qg++)
                fill_fragment(st[kg][qg], 0.f);
        for(int kg = 0; kg < F2_KG; kg++)
            for(int d16 = 0; d16 < D16; d16++)
            {
                fk_a kf;
                load_matrix_sync(kf, &sK[(kg * 16) * LD + d16 * 16], LD);
                for(int qg = 0; qg < F2_QG; qg++)
                    mma_sync(st[kg][qg], kf, q_frags[qg][d16], st[kg][qg]);
            }

        float tile_m[F2_QG];
        for(int qg = 0; qg < F2_QG; qg++)
            tile_m[qg] = -1e38f;
        for(int kg = 0; kg < F2_KG; kg++)
            for(int qg = 0; qg < F2_QG; qg++)
                for(int e = 0; e < 4; e++)
                {
                    const int key = ki0 + kg * 16 + my_e_base + e;
                    const int query = qi_base + qg * 16 + my_qcol;
                    float s = st[kg][qg].x[e] * scale;
                    if(key >= seq_len_kv)
                        s = -1e38f;
                    if(!full_tile && key > query)
                        s = -1e38f;
                    st[kg][qg].x[e] = s;
                    tile_m[qg] = fmaxf(tile_m[qg], s);
                }

        for(int qg = 0; qg < F2_QG; qg++)
        {
            float gm = tile_m[qg];
            gm = fmaxf(gm, __shfl_xor(gm, 16));
            gm = fmaxf(gm, __shfl_xor(gm, 32));
            tile_m[qg] = gm;
        }

        // Rescaling the O accumulators is the most expensive VALU step in the
        // loop: it touches QG*D16*4 elements that live in AGPRs, so each
        // `*= br` becomes accvgpr_read + v_mul + accvgpr_write. The measured
        // mix was 192 accvgpr_write + 160 accvgpr_read vs only 128 v_mfma --
        // 2.75 register moves per useful MFMA.
        //
        // br == 1.0 exactly whenever this tile did not raise the running max,
        // which is the common case once the max stabilizes. Skip the whole
        // read-modify-write then. tile_m was broadcast by the shfl_xor
        // reduction above, so br is wave-uniform and the branch is coherent;
        // the ballot makes that explicit to the compiler.
        for(int qg = 0; qg < F2_QG; qg++)
        {
            const float new_m = fmaxf(row_m[qg], tile_m[qg]);
            const float br = (row_m[qg] <= -1e30f) ? 0.f : flash2_expf(row_m[qg] - new_m);
            row_m[qg] = new_m;
            row_d[qg] *= br;
#if F2_SKIPRESCALE
            if(__builtin_amdgcn_ballot_w64(br != 1.0f) != 0ull)
#endif
            {
                for(int d16 = 0; d16 < D16; d16++)
                    for(int e = 0; e < 4; e++)
                        o_acc[qg][d16].x[e] *= br;
            }
        }

        fp_b pf[F2_KG][F2_QG];
        for(int kg = 0; kg < F2_KG; kg++)
            for(int qg = 0; qg < F2_QG; qg++)
                for(int e = 0; e < 4; e++)
                {
                    const float s = st[kg][qg].x[e];
                    const float pvv = (s > -1e30f) ? flash2_expf(s - row_m[qg]) : 0.f;
                    pf[kg][qg].x[e] = (_Float16)pvv;
                    row_d[qg] += pvv;
                }

        // ---- GEMM 2: O^T += V^T * P^T -------------------------------------
        for(int kg = 0; kg < F2_KG; kg++)
            for(int d16 = 0; d16 < D16; d16++)
            {
                fv_a vf;
#if F2_VTMAJOR
                load_matrix_sync(vf, &sVT[(d16 * 16) * LDK + kg * 16], LDK);
#else
                load_matrix_sync(vf, &sVT[(kg * 16) * LD + d16 * 16], LD);
#endif
                for(int qg = 0; qg < F2_QG; qg++)
                    mma_sync(o_acc[qg][d16], vf, pf[kg][qg], o_acc[qg][d16]);
            }
    }

    for(int qg = 0; qg < F2_QG; qg++)
    {
        row_d[qg] += __shfl_xor(row_d[qg], 16);
        row_d[qg] += __shfl_xor(row_d[qg], 32);
    }

    for(int qg = 0; qg < F2_QG; qg++)
    {
        const int oq = qi_base + qg * 16 + my_qcol;
        if(oq >= seq_len_q)
            continue;
        const float inv = (row_d[qg] > 0.f) ? 1.f / row_d[qg] : 0.f;
        __half* op = ptr_o + (size_t)b * o_stride_batch + (size_t)hq * o_stride_head
                     + (size_t)oq * o_stride_seq;
        for(int d16 = 0; d16 < D16; d16++)
            for(int e = 0; e < 4; e++)
                op[d16 * 16 + my_e_base + e] = __float2half(o_acc[qg][d16].x[e] * inv);
    }
}

extern "C" __global__
    __launch_bounds__(F2_BLOCK, 1) void flash2_v7_hipdnn_d128(const __half* __restrict__ ptr_q,
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
    flash2_body_v23<128, 8>(ptr_q,
                            ptr_k,
                            ptr_v,
                            ptr_o,
                            batch,
                            num_heads_q,
                            num_heads_k,
                            seq_len_q,
                            seq_len_kv,
                            head_dim,
                            scale,
                            causal,
                            q_stride_batch,
                            q_stride_head,
                            q_stride_seq,
                            k_stride_batch,
                            k_stride_head,
                            k_stride_seq,
                            v_stride_batch,
                            v_stride_head,
                            v_stride_seq,
                            o_stride_batch,
                            o_stride_head,
                            o_stride_seq);
}

extern "C" __global__
    __launch_bounds__(F2_BLOCK, 1) void flash2_v7_hipdnn_d64(const __half* __restrict__ ptr_q,
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
    flash2_body_v23<64, 4>(ptr_q,
                           ptr_k,
                           ptr_v,
                           ptr_o,
                           batch,
                           num_heads_q,
                           num_heads_k,
                           seq_len_q,
                           seq_len_kv,
                           head_dim,
                           scale,
                           causal,
                           q_stride_batch,
                           q_stride_head,
                           q_stride_seq,
                           k_stride_batch,
                           k_stride_head,
                           k_stride_seq,
                           v_stride_batch,
                           v_stride_head,
                           v_stride_seq,
                           o_stride_batch,
                           o_stride_head,
                           o_stride_seq);
}
