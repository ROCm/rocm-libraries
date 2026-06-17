// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2FwdPlan: execute() implementation
// This is the actual HIP kernel that runs when hipDNN dispatches to our engine.
//
// The Flash-Attention 2 kernel here is our V7:
//   - Full rocWMMA MFMA on both QK^T and P@V (5.83-6.46x over scalar on MI300X/MI325X)
//   - Causal tile skip: break when ki0 > qi_max (1.39-1.50x additional gain at seq>=4096)
//   - Fast exp2 approximation (numerically validated)
//   - GQA native support (hk = hq / gqa_ratio, no K/V replication)
//
// Performance (MI325X gfx942, ROCm 7.2, seq=4096 causal):
//   V7b: 78.98 TFLOPS MHA, 78.16 TFLOPS GQA4 (Alola job 366655)
//
// This file is compiled as a .hip file by hipcc for gfx942 and gfx950 targets.
// The hipDNN hip-kernel-provider includes it as a HIPRTC source or pre-compiled .co.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifdef __has_include
#if __has_include(<rocwmma/rocwmma.hpp>)
#include <rocwmma/rocwmma.hpp>
#define HAVE_ROCWMMA 1
#endif
#endif

#include "HipFlash2FwdPlanBuilder.hpp"

namespace hip_flash2_engine {

// ── Portable fast-exp helper ───────────────────────────────────────────────────
// On gfx942 (CDNA3): __builtin_amdgcn_exp2f is a single-cycle v_exp_f32 instruction.
// On gfx950 (CDNA4) and other targets: fall back to standard exp2f.
//
// Brian / gfx950 validation note: The gfx942 intrinsic was causing validation
// failures on gfx950 because the instruction's rounding behaviour differs
// between CDNA3 and CDNA4.  The portable path uses exp2f which is IEEE-754
// correct on both architectures.  Performance impact on gfx942: < 1%.
__device__ __forceinline__ float flash2_exp2f(float x) {
#if defined(__gfx942__)
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}

// exp(x) via exp2: exp(x) = 2^(x * log2(e))
__device__ __forceinline__ float flash2_expf(float x) {
    return flash2_exp2f(x * 1.4426950408f);  // log2(e) = 1/ln(2)
}

// ── Flash-Attention 2 V7 kernel (full MFMA + causal skip) ─────────────────────
// Grid:  (ceil(Lq/64), B, Hq)  — 64 query rows per CTA, 1 wavefront
// Block: (64,) — one AMD wavefront = 64 lanes
//
// Key design (from profiler analysis, OCI jobs 35347, 35379):
//   - rocWMMA 16x16x16 MFMA for both QK^T and P@V (5.8x over scalar baseline)
//   - Transposed LDS sK[D][64]: zero read conflicts, 5.88 store conflicts (hidden by MFMA)
//   - Causal skip: break when ki0 > qi_base+63 (+1.39-1.50x for seq>=4096)
//   - Fast exp2 via __builtin_amdgcn_exp2f (+0% at current TFLOPS, safe for future)
//   - GQA: hk = hq / (Hq/Hk) — no K/V memory replication needed

#ifdef HAVE_ROCWMMA

template <int D>
__global__ __launch_bounds__(64, 2) void flash2_v7_hipdnn(
    const __half* __restrict__ Q,  // [B, Lq, Hq, D]
    const __half* __restrict__ K,  // [B, Lk, Hk, D]
    const __half* __restrict__ V,  // [B, Lk, Hk, D]
    __half* __restrict__ O,        // [B, Lq, Hq, D]
    int qo_len, int kv_len, int Hq, int Hk, float sm_scale, int causal,
    // Strides in elements (for non-standard layouts from hipDNN)
    int64_t q_stride_batch, int64_t q_stride_seq, int64_t q_stride_head, int64_t k_stride_batch,
    int64_t k_stride_seq, int64_t k_stride_head, int64_t v_stride_batch, int64_t v_stride_seq,
    int64_t v_stride_head, int64_t o_stride_batch, int64_t o_stride_seq, int64_t o_stride_head) {
    using namespace rocwmma;
    using fq = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    using fk = fragment<matrix_b, 16, 16, 16, __half, col_major>;
    using fs = fragment<accumulator, 16, 16, 16, float>;
    using fp = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    using fv = fragment<matrix_b, 16, 16, 16, __half, row_major>;
    using fo = fragment<accumulator, 16, 16, 16, float>;
    static constexpr int D16 = D / 16;

    const int qi_blk = blockIdx.x;
    const int b = blockIdx.y;
    const int hq = blockIdx.z;
    const int lane = threadIdx.x;
    const int hk = hq / (Hq / Hk);
    const int qi_base = qi_blk * 64;
    const int my_qrow = lane % 16;
    const int my_cg = lane / 16;
    const int my_qi = qi_base + my_cg * 16 + my_qrow;
    const bool my_valid = (my_qi < qo_len);
    const int qi_max = qi_base + 63;

    // Transposed LDS: sK[D][64] — lane ki writes to column ki for row d
    // Zero read conflicts; store conflicts (5.88/access) hidden by MFMA latency
    __shared__ __half sK[D][64];
    __shared__ __half sV[D][64];

    // Load Q fragments for all 4 groups of 16 query rows
    fq q_frags[4][D16];
    const __half* Qp = Q + (size_t)b * q_stride_batch + (size_t)qi_base * q_stride_seq +
                       (size_t)hq * q_stride_head;
    for (int qg = 0; qg < 4; qg++)
        for (int d16 = 0; d16 < D16; d16++) {
            if (qi_base + qg * 16 < qo_len)
                load_matrix_sync(q_frags[qg][d16], Qp + qg * 16 * q_stride_seq + d16 * 16,
                                 q_stride_seq);
            else
                fill_fragment(q_frags[qg][d16], __half(0));
        }

    // Output accumulators + online softmax state
    fo o_frags[4][D16];
    for (int qg = 0; qg < 4; qg++)
        for (int d16 = 0; d16 < D16; d16++) fill_fragment(o_frags[qg][d16], 0.f);
    float row_m = -1e38f, row_d = 0.f;

    for (int kb = 0; kb < (kv_len + 63) / 64; kb++) {
        int ki0 = kb * 64;

        // ── Causal tile skip (AMDable KB: flash2_causal_tile_skip) ────────────
        // Break when the entire tile is fully masked — all remaining tiles also masked
        if (causal && ki0 > qi_max) break;
        const bool full_tile = (!causal) || (ki0 + 63 <= qi_base);

        // Cooperative LDS load: lane l loads KV token ki0+l for all D dims
        int ki_g = ki0 + lane;
        if (ki_g < kv_len) {
            const __half* kp = K + (size_t)b * k_stride_batch + (size_t)ki_g * k_stride_seq +
                               (size_t)hk * k_stride_head;
            const __half* vp = V + (size_t)b * v_stride_batch + (size_t)ki_g * v_stride_seq +
                               (size_t)hk * v_stride_head;
#pragma unroll
            for (int d = 0; d < D; d++) {
                sK[d][lane] = kp[d];
                sV[d][lane] = vp[d];
            }
        } else {
#pragma unroll
            for (int d = 0; d < D; d++) {
                sK[d][lane] = __float2half(0.f);
                sV[d][lane] = __float2half(0.f);
            }
        }
        __syncthreads();

        // ── QK^T via MFMA: 4 query groups × 4 KV groups × D16 calls ─────────
        fs score_frags[4][4];
        for (int qg = 0; qg < 4; qg++)
            for (int kg = 0; kg < 4; kg++) fill_fragment(score_frags[qg][kg], 0.f);

        for (int qg = 0; qg < 4; qg++)
            for (int kg = 0; kg < 4; kg++)
                for (int d16 = 0; d16 < D16; d16++) {
                    fk kf;
                    load_matrix_sync(kf, &sK[d16 * 16][kg * 16], 64);
                    mma_sync(score_frags[qg][kg], q_frags[qg][d16], kf, score_frags[qg][kg]);
                }

        // ── Online softmax ────────────────────────────────────────────────────
        float tile_new_m = row_m;
        float ms[4][4][4];
        for (int qg = 0; qg < 4; qg++)
            for (int kg = 0; kg < 4; kg++)
                for (int e = 0; e < 4; e++) {
                    float s = score_frags[qg][kg].x[e] * sm_scale;
                    if (!full_tile) {
                        int kv = ki0 + kg * 16 + my_cg * 4 + e;
                        int qi = qi_base + qg * 16 + my_qrow;
                        if (kv > qi) s = -1e38f;
                    }
                    if (ki0 + kg * 16 + my_cg * 4 + e >= kv_len) s = -1e38f;
                    ms[qg][kg][e] = s;
                    if (qg == my_cg) tile_new_m = fmaxf(tile_new_m, s);
                }
        for (int g = 1; g < 4; g++) tile_new_m = fmaxf(tile_new_m, __shfl_xor(tile_new_m, g * 16));

        // Fast exp2 for rescale — portable across gfx942/gfx950
        float br = flash2_expf(row_m - tile_new_m);
        row_m = tile_new_m;
        for (int qg = 0; qg < 4; qg++)
            for (int d16 = 0; d16 < D16; d16++)
                for (int e = 0; e < 4; e++) o_frags[qg][d16].x[e] *= br;
        row_d *= br;

        // Convert scores to fp16 probability fragments
        fp p_frags[4][4];
        for (int qg = 0; qg < 4; qg++)
            for (int kg = 0; kg < 4; kg++) {
                fill_fragment(p_frags[qg][kg], __half(0));
                for (int e = 0; e < 4; e++) {
                    float pv = flash2_expf(ms[qg][kg][e] - row_m);
                    p_frags[qg][kg].x[e] = (_Float16)pv;
                    if (qg == my_cg) row_d += pv;
                }
            }

        // ── P@V via MFMA ──────────────────────────────────────────────────────
        for (int kg = 0; kg < 4; kg++)
            for (int d16 = 0; d16 < D16; d16++) {
                fv vf;
                load_matrix_sync(vf, &sV[d16 * 16][kg * 16], 64);
                for (int qg = 0; qg < 4; qg++)
                    mma_sync(o_frags[qg][d16], p_frags[qg][kg], vf, o_frags[qg][d16]);
            }
        __syncthreads();
    }

    // ── Write output ──────────────────────────────────────────────────────────
    if (my_valid) {
        float inv = (row_d > 0.f) ? 1.f / row_d : 0.f;
        __shared__ float smem_out[64][D];

        for (int qg = 0; qg < 4; qg++) {
            for (int d16 = 0; d16 < D16; d16++) {
                for (int e = 0; e < 4; e++) o_frags[qg][d16].x[e] *= inv;
                store_matrix_sync(&smem_out[qg * 16][d16 * 16], o_frags[qg][d16], D, mem_row_major);
            }
        }
        __syncthreads();

        for (int qg = 0; qg < 4; qg++) {
            int out_qi = qi_base + qg * 16 + my_qrow;
            if (out_qi < qo_len) {
                __half* op = O + (size_t)b * o_stride_batch + (size_t)out_qi * o_stride_seq +
                             (size_t)hq * o_stride_head;
                for (int d = 0; d < D; d++) op[d] = __float2half(smem_out[qg * 16 + my_qrow][d]);
            }
        }
    }
}

#endif  // HAVE_ROCWMMA

// ── Host-side launcher: called from hipDNN plan execute() ────────────────────
void launch_flash2_v7(const Flash2FwdParams& p, const __half* Q, const __half* K, const __half* V,
                      __half* O, hipStream_t stream) {
#ifndef HAVE_ROCWMMA
    // rocWMMA not available — should not reach here (isApplicable returns false)
    (void)p;
    (void)Q;
    (void)K;
    (void)V;
    (void)O;
    (void)stream;
    return;
#else
    const float sm_scale = (p.attn_scale != 0.f) ? p.attn_scale : 1.f / sqrtf((float)p.head_dim);

    dim3 grid((p.seq_len_q + 63) / 64, p.batch, p.num_heads_q);
    dim3 block(64);

    if (p.head_dim == 128) {
        hipLaunchKernelGGL(
            (flash2_v7_hipdnn<128>), grid, block, 0, stream,
            Q, K, V, O,
            p.seq_len_q, p.seq_len_kv,
