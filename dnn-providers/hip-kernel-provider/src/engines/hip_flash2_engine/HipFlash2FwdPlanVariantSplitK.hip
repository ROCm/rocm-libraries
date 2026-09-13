// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// v30 = SPLIT-K (flash-decoding style) on top of the v26 S-transpose kernel.
//
// ---------------------------------------------------------------------------
// WHY
// Measured deficits vs CK/AITER are concentrated in long-S / low-head shapes:
//   B2H8  S8192 D128 c  0.79x     B1H16 S6144 D128 c  0.80x
//   B1H32 S8192 D128 n  0.85x     B1H32 S4096 D128 n  0.86x
// These have few CTAs but each CTA scans a very long KV axis, so the GPU runs
// a short, deep grid: latency is exposed and there is nothing else to overlap.
//
// SPLIT-K fixes the shape of the grid rather than the inner loop. Partition the
// KV axis into NSPLIT chunks and give each (query-tile, chunk) pair its own CTA,
// multiplying grid parallelism by NSPLIT. Each CTA runs the ordinary online
// softmax over ITS chunk only, then writes a PARTIAL result plus that chunk's
// softmax statistics. A cheap second kernel merges the partials.
//
// MERGE MATH (standard, and exact -- not an approximation):
//   chunk j produces  m_j = max logit in chunk j
//                     l_j = sum of exp(logit - m_j)
//                     O_j = (1/l_j) * sum exp(logit - m_j) V     [already normalized]
//   global m = max_j m_j
//   global l = sum_j l_j * exp(m_j - m)
//   O = sum_j O_j * (l_j * exp(m_j - m)) / l
// so each chunk's contribution is reweighted by its own mass. Storing O_j
// pre-normalized keeps the partial buffer in the same fp32 range regardless of
// chunk length, which matters because chunks can differ wildly under causal
// masking.
//
// PARTIAL BUFFER (fp32, allocated by the caller):
//   partial_o : [B, H, NSPLIT, Sq, D]   float
//   partial_m : [B, H, NSPLIT, Sq]      float
//   partial_l : [B, H, NSPLIT, Sq]      float
// Extra HBM traffic is NSPLIT * Sq * D * 4 bytes per (b,h); the win has to beat
// that, which is why split-K only pays when the grid is starved.
//
// CAUSAL + SPLIT-K: chunk j covers keys [j*chunk, (j+1)*chunk). A query tile
// only needs chunks that start at or below its last query index; CTAs for
// higher chunks exit immediately after writing an empty partial (m=-inf, l=0),
// which the merge skips.
//
// Everything inside the KV loop is unchanged from v26 (S^T = K*Q^T so the score
// accumulator is already in matrix_b layout for P^T; O^T = V^T*P^T).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#ifndef F2_WAVES
#define F2_WAVES 8
#endif
#ifndef F2_QG
#define F2_QG 2
#endif
#ifndef F2_KG
#define F2_KG 4
#endif
#ifndef F2_PAD
#define F2_PAD 8
#endif
#ifndef F2_PADK
#define F2_PADK 8
#endif

#define F2_BLOCK (F2_WAVES * 64)
#define F2_BK    (F2_KG * 16)
#define F2_QPER  (F2_WAVES * F2_QG * 16)

__device__ __forceinline__ float flash2_expf(float x)
{
#if defined(__gfx942__)
    return __builtin_amdgcn_exp2f(x * 1.4426950408f);
#else
    return exp2f(x * 1.4426950408f);
#endif
}

struct __align__(16) half8 { __half h[8]; };

// ---------------------------------------------------------------------------
// Split-K partial pass. grid = (qtiles, B*H, NSPLIT), block = F2_BLOCK.
// ---------------------------------------------------------------------------
template <int D, int D16>
__device__ void flash2_split_body(const __half* __restrict__ ptr_q,
                                  const __half* __restrict__ ptr_k,
                                  const __half* __restrict__ ptr_v,
                                  float* __restrict__ part_o,
                                  float* __restrict__ part_m,
                                  float* __restrict__ part_l,
                                  int batch, int num_heads_q, int num_heads_k,
                                  int seq_len_q, int seq_len_kv, int head_dim,
                                  float scale, int causal, int nsplit,
                                  int q_stride_batch, int q_stride_head, int q_stride_seq,
                                  int k_stride_batch, int k_stride_head, int k_stride_seq,
                                  int v_stride_batch, int v_stride_head, int v_stride_seq)
{
    using namespace rocwmma;
    constexpr int LD  = D + F2_PAD;
    constexpr int LDK = F2_BK + F2_PADK;

    using fq_b = fragment<matrix_b,    16, 16, 16, __half, col_major>;
    using fk_a = fragment<matrix_a,    16, 16, 16, __half, row_major>;
    using fv_a = fragment<matrix_a,    16, 16, 16, __half, row_major>;
    using fp_b = fragment<matrix_b,    16, 16, 16, __half, col_major>;
    using facc = fragment<accumulator, 16, 16, 16, float>;

    const int bh    = blockIdx.y;
    const int b     = bh / num_heads_q;
    const int hq    = bh % num_heads_q;
    const int split = blockIdx.z;
    const int tid   = threadIdx.x;
    const int wave  = tid >> 6;
    const int lane  = tid & 63;
    const int hk    = (num_heads_k == num_heads_q) ? hq : hq / (num_heads_q / num_heads_k);

    const int cta_q0  = blockIdx.x * F2_QPER;
    const int qi_base = cta_q0 + wave * (F2_QG * 16);
    const int qi_max  = qi_base + F2_QG * 16 - 1;

    const int my_qcol   = lane % 16;
    const int my_e_base = (lane / 16) * 4;

    // This CTA's slice of the KV axis, rounded to whole BK tiles.
    const int tiles_all = (seq_len_kv + F2_BK - 1) / F2_BK;
    const int per_split = (tiles_all + nsplit - 1) / nsplit;
    const int t_begin   = split * per_split;
    int       t_end     = t_begin + per_split;
    if(t_end > tiles_all) t_end = tiles_all;
    // Causal cap MUST be CTA-wide, not per-wave. qi_max is the last query of
    // THIS wave, so capping on it makes different waves run different trip
    // counts -- and the loop below contains __syncthreads(). Divergent barriers
    // are undefined behaviour, which showed up as causal-only corruption
    // (maxErr 0.7-1.2) while noncausal stayed exact at 1e-4.
    // Use the CTA's last query instead; per-wave work is still skipped inside
    // the loop by the `causal && ki0 > qi_max` continue, which is safe because
    // it sits after both barriers.
    const int cta_qmax = cta_q0 + F2_QPER - 1;
    if(causal)
    {
        const int t_cap = cta_qmax / F2_BK + 1;
        if(t_end > t_cap) t_end = t_cap;
    }

    __shared__ __half sK[F2_BK * LD];
    __shared__ __half sVT[D * LDK];

    fq_b q_frags[F2_QG][D16];
    const __half* Qp = ptr_q + (size_t)b * q_stride_batch + (size_t)hq * q_stride_head
                       + (size_t)qi_base * q_stride_seq;
    for(int qg = 0; qg < F2_QG; qg++)
        for(int d16 = 0; d16 < D16; d16++)
        {
            if(qi_base + qg * 16 + 15 < seq_len_q)
                load_matrix_sync(q_frags[qg][d16],
                                 Qp + (size_t)qg * 16 * q_stride_seq + d16 * 16, q_stride_seq);
            else
            {
                fill_fragment(q_frags[qg][d16], __half(0));
                const int qrow = qi_base + qg * 16 + my_qcol;
                if(qrow < seq_len_q)
                    for(int e = 0; e < 4; e++)
                        q_frags[qg][d16].x[e] =
                            Qp[(size_t)(qg * 16 + my_qcol) * q_stride_seq + d16 * 16
                               + my_e_base + e];
            }
        }

    facc o_acc[F2_QG][D16];
    for(int qg = 0; qg < F2_QG; qg++)
        for(int d16 = 0; d16 < D16; d16++) fill_fragment(o_acc[qg][d16], 0.f);
    float row_m[F2_QG], row_d[F2_QG];
    for(int qg = 0; qg < F2_QG; qg++) { row_m[qg] = -1e38f; row_d[qg] = 0.f; }

    constexpr int CH   = D / 8;
    constexpr int NTOT = F2_BK * CH;
    constexpr int NCH  = (NTOT + F2_BLOCK - 1) / F2_BLOCK;

    const __half* Kb = ptr_k + (size_t)b * k_stride_batch + (size_t)hk * k_stride_head;
    const __half* Vb = ptr_v + (size_t)b * v_stride_batch + (size_t)hk * v_stride_head;

    auto fetch = [&](int ki0, half8* pk, half8* pv) {
        for(int j = 0; j < NCH; j++)
        {
            const int idx = tid + j * F2_BLOCK;
            if(NTOT % F2_BLOCK != 0 && idx >= NTOT) break;
            const int key = idx / CH, cc = (idx % CH) * 8;
            const int kgl = ki0 + key;
            if(kgl < seq_len_kv)
            {
                pk[j] = *reinterpret_cast<const half8*>(Kb + (size_t)kgl * k_stride_seq + cc);
                pv[j] = *reinterpret_cast<const half8*>(Vb + (size_t)kgl * v_stride_seq + cc);
            }
            else
                for(int i = 0; i < 8; i++) { pk[j].h[i] = __half(0); pv[j].h[i] = __half(0); }
        }
    };

    half8 pk[NCH], pv[NCH];
    if(t_begin < t_end) fetch(t_begin * F2_BK, pk, pv);

    for(int t = t_begin; t < t_end; t++)
    {
        const int ki0 = t * F2_BK;
        __syncthreads();
        for(int j = 0; j < NCH; j++)
        {
            const int idx = tid + j * F2_BLOCK;
            if(NTOT % F2_BLOCK != 0 && idx >= NTOT) break;
            const int key = idx / CH, cc = (idx % CH) * 8;
            *reinterpret_cast<half8*>(&sK[key * LD + cc]) = pk[j];
            for(int i = 0; i < 8; i++) sVT[(cc + i) * LDK + key] = pv[j].h[i];
        }
        if(t + 1 < t_end) fetch((t + 1) * F2_BK, pk, pv);
        __syncthreads();

        if(causal && ki0 > qi_max) continue;
        const bool full_tile = (!causal) || (ki0 + F2_BK - 1 <= qi_base);

        facc st[F2_KG][F2_QG];
        for(int kg = 0; kg < F2_KG; kg++)
            for(int qg = 0; qg < F2_QG; qg++) fill_fragment(st[kg][qg], 0.f);
        for(int kg = 0; kg < F2_KG; kg++)
            for(int d16 = 0; d16 < D16; d16++)
            {
                fk_a kf;
                load_matrix_sync(kf, &sK[(kg * 16) * LD + d16 * 16], LD);
                for(int qg = 0; qg < F2_QG; qg++)
                    mma_sync(st[kg][qg], kf, q_frags[qg][d16], st[kg][qg]);
            }

        float tile_m[F2_QG];
        for(int qg = 0; qg < F2_QG; qg++) tile_m[qg] = -1e38f;
        for(int kg = 0; kg < F2_KG; kg++)
            for(int qg = 0; qg < F2_QG; qg++)
                for(int e = 0; e < 4; e++)
                {
                    const int key   = ki0 + kg * 16 + my_e_base + e;
                    const int query = qi_base + qg * 16 + my_qcol;
                    float     s     = st[kg][qg].x[e] * scale;
                    if(key >= seq_len_kv)         s = -1e38f;
                    if(!full_tile && key > query) s = -1e38f;
                    st[kg][qg].x[e] = s;
                    tile_m[qg]      = fmaxf(tile_m[qg], s);
                }
        for(int qg = 0; qg < F2_QG; qg++)
        {
            float gm = tile_m[qg];
            gm = fmaxf(gm, __shfl_xor(gm, 16));
            gm = fmaxf(gm, __shfl_xor(gm, 32));
            tile_m[qg] = gm;
        }
        for(int qg = 0; qg < F2_QG; qg++)
        {
            const float new_m = fmaxf(row_m[qg], tile_m[qg]);
            const float br    = (row_m[qg] <= -1e30f) ? 0.f : flash2_expf(row_m[qg] - new_m);
            row_m[qg] = new_m;
            row_d[qg] *= br;
            if(__builtin_amdgcn_ballot_w64(br != 1.0f) != 0ull)
                for(int d16 = 0; d16 < D16; d16++)
                    for(int e = 0; e < 4; e++) o_acc[qg][d16].x[e] *= br;
        }
        fp_b pf[F2_KG][F2_QG];
        for(int kg = 0; kg < F2_KG; kg++)
            for(int qg = 0; qg < F2_QG; qg++)
                for(int e = 0; e < 4; e++)
                {
                    const float s   = st[kg][qg].x[e];
                    const float pvv = (s > -1e30f) ? flash2_expf(s - row_m[qg]) : 0.f;
                    pf[kg][qg].x[e] = (_Float16)pvv;
                    row_d[qg]      += pvv;
                }
        for(int kg = 0; kg < F2_KG; kg++)
            for(int d16 = 0; d16 < D16; d16++)
            {
                fv_a vf;
                load_matrix_sync(vf, &sVT[(d16 * 16) * LDK + kg * 16], LDK);
                for(int qg = 0; qg < F2_QG; qg++)
                    mma_sync(o_acc[qg][d16], vf, pf[kg][qg], o_acc[qg][d16]);
            }
    }

    for(int qg = 0; qg < F2_QG; qg++)
    {
        row_d[qg] += __shfl_xor(row_d[qg], 16);
        row_d[qg] += __shfl_xor(row_d[qg], 32);
    }

    // Write this chunk's partial: O normalized by its own l, plus (m, l).
    const size_t base_ml = ((size_t)bh * nsplit + split) * (size_t)seq_len_q;
    const size_t base_o  = base_ml * (size_t)D;
    for(int qg = 0; qg < F2_QG; qg++)
    {
        const int oq = qi_base + qg * 16 + my_qcol;
        if(oq >= seq_len_q) continue;
        const bool empty = (t_begin >= t_end) || (row_d[qg] <= 0.f);
        const float inv  = empty ? 0.f : 1.f / row_d[qg];
        for(int d16 = 0; d16 < D16; d16++)
            for(int e = 0; e < 4; e++)
                part_o[base_o + (size_t)oq * D + d16 * 16 + my_e_base + e] =
                    o_acc[qg][d16].x[e] * inv;
        if(my_e_base == 0 && (lane / 16) == 0)
        {
            part_m[base_ml + oq] = empty ? -1e38f : row_m[qg];
            part_l[base_ml + oq] = empty ? 0.f    : row_d[qg];
        }
    }
}

extern "C" __global__ __launch_bounds__(F2_BLOCK, 1)
void flash2_split_d128(const __half* __restrict__ q, const __half* __restrict__ k,
                       const __half* __restrict__ v, float* __restrict__ po,
                       float* __restrict__ pm, float* __restrict__ pl,
                       int batch, int hq, int hk, int sq, int skv, int d,
                       float scale, int causal, int nsplit,
                       int qsb, int qsh, int qss, int ksb, int ksh, int kss,
                       int vsb, int vsh, int vss)
{
    flash2_split_body<128, 8>(q, k, v, po, pm, pl, batch, hq, hk, sq, skv, d,
                              scale, causal, nsplit, qsb, qsh, qss, ksb, ksh, kss,
                              vsb, vsh, vss);
}

extern "C" __global__ __launch_bounds__(F2_BLOCK, 1)
void flash2_split_d64(const __half* __restrict__ q, const __half* __restrict__ k,
                      const __half* __restrict__ v, float* __restrict__ po,
                      float* __restrict__ pm, float* __restrict__ pl,
                      int batch, int hq, int hk, int sq, int skv, int d,
                      float scale, int causal, int nsplit,
                      int qsb, int qsh, int qss, int ksb, int ksh, int kss,
                      int vsb, int vsh, int vss)
{
    flash2_split_body<64, 4>(q, k, v, po, pm, pl, batch, hq, hk, sq, skv, d,
                             scale, causal, nsplit, qsb, qsh, qss, ksb, ksh, kss,
                             vsb, vsh, vss);
}

// ---------------------------------------------------------------------------
// Merge pass: combine NSPLIT partials into the final fp16 output.
// grid = (ceil(Sq/16), B*H), block = 256 (one 64-lane wave per 4 rows, D lanes).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 1)
void flash2_merge(const float* __restrict__ po, const float* __restrict__ pm,
                  const float* __restrict__ pl, __half* __restrict__ out,
                  int batch, int hq, int sq, int d, int nsplit,
                  int osb, int osh, int oss)
{
    const int bh  = blockIdx.y;
    const int b   = bh / hq;
    const int h   = bh % hq;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);   // 4 query rows per CTA
    if(row >= sq) return;
    const int lane = threadIdx.x & 63;

    const size_t stride_ml = (size_t)sq;
    const size_t stride_o  = (size_t)sq * d;
    const size_t base      = (size_t)bh * nsplit;

    float m = -1e38f;
    for(int j = 0; j < nsplit; j++)
        m = fmaxf(m, pm[(base + j) * stride_ml + row]);

    float ltot = 0.f;
    for(int j = 0; j < nsplit; j++)
    {
        const float lj = pl[(base + j) * stride_ml + row];
        if(lj > 0.f) ltot += lj * flash2_expf(pm[(base + j) * stride_ml + row] - m);
    }
    const float inv = (ltot > 0.f) ? 1.f / ltot : 0.f;

    for(int c = lane; c < d; c += 64)
    {
        float acc = 0.f;
        for(int j = 0; j < nsplit; j++)
        {
            const float lj = pl[(base + j) * stride_ml + row];
            if(lj <= 0.f) continue;
            const float w = lj * flash2_expf(pm[(base + j) * stride_ml + row] - m);
            acc += po[(base + j) * stride_o + (size_t)row * d + c] * w;
        }
        out[(size_t)b * osb + (size_t)h * osh + (size_t)row * oss + c] =
            __float2half(acc * inv);
    }
}
