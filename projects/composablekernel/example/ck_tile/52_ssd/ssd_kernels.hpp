// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Mamba-2 SSD — ck_tile custom HIP kernels for element-wise operations.
// These kernels handle the non-GEMM parts of the SSD algorithm:
//   cumsum, segsum, pre-multiply, state propagation, epilogue.

#pragma once

#include <cmath>

namespace ck_tile {

// ---------------------------------------------------------------------------
// Kernel: cumulative sum of DeltaA along L dimension
// Input:  DeltaA [BEH, C, L]
// Output: Cumsum [BEH, C, L]  where Cumsum[c,l] = sum_{i=0}^{l} DeltaA[c,i]
// Grid:   (BEH, 1, 1)  Block: (C, 1, 1)
// ---------------------------------------------------------------------------
__global__ void ssd_cumsum_kernel(
    const float* __restrict__ delta_a,
    float*       __restrict__ cumsum,
    int C, int L)
{
    const int beh = blockIdx.x;
    const int ci  = threadIdx.x;
    if (ci >= C) return;

    const float* in  = delta_a + static_cast<size_t>(beh) * C * L + static_cast<size_t>(ci) * L;
    float*       out = cumsum  + static_cast<size_t>(beh) * C * L + static_cast<size_t>(ci) * L;

    float acc = 0.0f;
    for (int l = 0; l < L; ++l) {
        acc   += in[l];
        out[l] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel: segsum + pre_intra_bmm2
// Fuses the lower-triangular exponential decay matrix with delta and IntraBMM1.
//   out[i, j] = segsum(i,j) * delta[j] * intra_bmm1[i, j]
// where segsum(i,j) = exp(cumsum[i]-cumsum[j]) if j<i, 1 if j==i, 0 if j>i.
// Grid:   (BEH, C, 1)  Block: (256, 1, 1)
// ---------------------------------------------------------------------------
__global__ void ssd_segsum_pre_intra2_kernel(
    const float* __restrict__ cumsum,
    const float* __restrict__ delta,
    const float* __restrict__ intra_bmm1,
    float*       __restrict__ pre_intra2,
    int C, int L)
{
    const int beh   = blockIdx.x;
    const int ci    = blockIdx.y;
    const int total = L * L;

    const float* cum  = cumsum     + static_cast<size_t>(beh) * C * L + static_cast<size_t>(ci) * L;
    const float* dlt  = delta      + static_cast<size_t>(beh) * C * L + static_cast<size_t>(ci) * L;
    const float* bmm1 = intra_bmm1 + (static_cast<size_t>(beh) * C + ci) * L * L;
    float*       out  = pre_intra2 + (static_cast<size_t>(beh) * C + ci) * L * L;

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int i = idx / L;
        const int j = idx % L;

        float seg;
        if      (j < i)  seg = expf(cum[i] - cum[j]);
        else if (j == i) seg = 1.0f;
        else             seg = 0.0f;

        out[idx] = seg * dlt[j] * bmm1[idx];
    }
}

// ---------------------------------------------------------------------------
// Kernel: pre_inter_bmm1
// Computes:
//   pre_inter1[n, l] = exp(last - cumsum[l]) * delta[l] * B_mat[n, l]
//   cumsum_exp[l]    = exp(cumsum[l])
//   last_vals[ci]    = cumsum[L-1]
// Grid:   (BEH, C, 1)  Block: (256, 1, 1)
// ---------------------------------------------------------------------------
__global__ void ssd_pre_inter1_kernel(
    const float* __restrict__ cumsum,
    const float* __restrict__ delta,
    const float* __restrict__ b_mat,      // [B*G, N, C, L]
    float*       __restrict__ pre_inter1, // [BEH, C, N, L]
    float*       __restrict__ cumsum_exp, // [BEH, C, L]
    float*       __restrict__ last_vals,  // [BEH, C]
    int C, int L, int N, int G, int grp_ratio)
{
    const int beh = blockIdx.x;
    const int ci  = blockIdx.y;
    const int b   = beh / (G * grp_ratio);
    const int eh  = beh % (G * grp_ratio);
    const int g   = eh / grp_ratio;
    const int bg  = b * G + g;

    const float* cum = cumsum + static_cast<size_t>(beh) * C * L + static_cast<size_t>(ci) * L;
    const float* dlt = delta  + static_cast<size_t>(beh) * C * L + static_cast<size_t>(ci) * L;
    float* ce   = cumsum_exp + static_cast<size_t>(beh) * C * L + static_cast<size_t>(ci) * L;
    float* pi1  = pre_inter1 + (static_cast<size_t>(beh) * C + ci) * N * L;

    const float last = cum[L - 1];
    if (threadIdx.x == 0) last_vals[beh * C + ci] = last;

    for (int l = threadIdx.x; l < L; l += blockDim.x)
        ce[l] = expf(cum[l]);

    for (int idx = threadIdx.x; idx < N * L; idx += blockDim.x) {
        const int n = idx / L;
        const int l = idx % L;
        float bv = b_mat[(static_cast<size_t>(bg) * N + n) * C * L +
                          static_cast<size_t>(ci) * L + l];
        pi1[idx] = expf(last - cum[l]) * dlt[l] * bv;
    }
}

// ---------------------------------------------------------------------------
// Kernel: sequential state propagation across chunks
//   state[0] = 0
//   state[ci] = inter_bmm1[ci-1] + exp(last[ci-1]) * state[ci-1]
// Grid:   (BEH, 1, 1)  Block: (256, 1, 1)
// ---------------------------------------------------------------------------
__global__ void ssd_state_propagation_kernel(
    const float* __restrict__ inter_bmm1,
    const float* __restrict__ last_vals,
    float*       __restrict__ state,
    int C, int N, int D)
{
    const int beh    = blockIdx.x;
    const int ND     = N * D;
    const float* bmm = inter_bmm1 + static_cast<size_t>(beh) * C * ND;
    const float* lv  = last_vals  + static_cast<size_t>(beh) * C;
    float*       st  = state      + static_cast<size_t>(beh) * C * ND;

    for (int idx = threadIdx.x; idx < ND; idx += blockDim.x)
        st[idx] = 0.0f;
    __syncthreads();

    for (int ci = 1; ci < C; ++ci) {
        float el = expf(lv[ci - 1]);
        for (int idx = threadIdx.x; idx < ND; idx += blockDim.x)
            st[ci * ND + idx] = bmm[(ci - 1) * ND + idx] + el * st[(ci - 1) * ND + idx];
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernel: epilogue
//   Y[d, ci, l] = cumsum_exp[l] * inter_bmm2[l, d]
//               + intra_bmm2[l, d]
//               + D_param[d] * X[d, ci, l]
// Grid:   (BEH, C, 1)  Block: (256, 1, 1)
// ---------------------------------------------------------------------------
__global__ void ssd_epilogue_kernel(
    const float* __restrict__ inter_bmm2,
    const float* __restrict__ intra_bmm2,
    const float* __restrict__ cumsum_exp,
    const float* __restrict__ x,
    const float* __restrict__ d_param,
    float*       __restrict__ y,
    int C, int L, int D, int EH)
{
    const int beh = blockIdx.x;
    const int ci  = blockIdx.y;
    const int eh  = beh % EH;
    const int total = L * D;

    const float* i2 = inter_bmm2 + (static_cast<size_t>(beh) * C + ci) * L * D;
    const float* a2 = intra_bmm2 + (static_cast<size_t>(beh) * C + ci) * L * D;
    const float* ce = cumsum_exp + static_cast<size_t>(beh) * C * L + static_cast<size_t>(ci) * L;

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int l = idx / D;
        const int d = idx % D;
        float val = ce[l] * i2[l * D + d] + a2[l * D + d];
        float dp  = d_param[eh * D + d];
        float xv  = x[static_cast<size_t>(beh) * D * C * L +
                      static_cast<size_t>(d) * C * L +
                      static_cast<size_t>(ci) * L + l];
        val += dp * xv;
        y[static_cast<size_t>(beh) * D * C * L +
          static_cast<size_t>(d) * C * L +
          static_cast<size_t>(ci) * L + l] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel: final state
//   F[d, n] = inter_bmm1[C-1, n, d] + exp(last[C-1]) * state[C-1, n, d]
// Grid:   (BEH, 1, 1)  Block: (256, 1, 1)
// ---------------------------------------------------------------------------
__global__ void ssd_final_state_kernel(
    const float* __restrict__ inter_bmm1,
    const float* __restrict__ state,
    const float* __restrict__ last_vals,
    float*       __restrict__ fstate,
    int C, int N, int D)
{
    const int beh = blockIdx.x;
    const int ND  = N * D;
    const float* bmm = inter_bmm1 + static_cast<size_t>(beh) * C * ND + (C - 1) * ND;
    const float* st  = state      + static_cast<size_t>(beh) * C * ND + (C - 1) * ND;
    float el = expf(last_vals[beh * C + C - 1]);

    for (int idx = threadIdx.x; idx < ND; idx += blockDim.x) {
        int n = idx / D;
        int d = idx % D;
        fstate[static_cast<size_t>(beh) * D * N + static_cast<size_t>(d) * N + n] =
            bmm[idx] + el * st[idx];
    }
}

// ---------------------------------------------------------------------------
// Batched pack: B_mat/C_mat slices for all (beh, ci) at once.
// src_base: [B*G, N, C, L], packs each [N,L] slice (stride C*L) into
// contiguous [L,N] (transpose=true) or [N,L] (transpose=false).
// Grid: (elem_blocks, BEH*C)  Block: (256,1,1)
// ---------------------------------------------------------------------------
__global__ void pack_bcmat_batched_kernel(
    const float* __restrict__ src_base,
    float*       __restrict__ dst,
    int EH, int G, int grp,
    int N, int C, int L,
    bool do_transpose)
{
    const int batch_idx = blockIdx.y;
    const int beh = batch_idx / C;
    const int ci  = batch_idx % C;
    const int b   = beh / EH;
    const int eh  = beh % EH;
    const int g   = eh / grp;
    const int bg  = b * G + g;

    const int total = N * L;
    const float* src = src_base + static_cast<size_t>(bg) * N * C * L + static_cast<size_t>(ci) * L;
    float* out = dst + static_cast<size_t>(batch_idx) * N * L;

    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        const int r = idx / L;
        const int c = idx % L;
        float val = src[r * (C * L) + c];
        if (do_transpose)
            out[c * N + r] = val;
        else
            out[r * L + c] = val;
    }
}

// ---------------------------------------------------------------------------
// Batched pack: X slices for all (beh, ci) at once.
// src_base: [BEH, D, C, L], packs each [D,L] (stride C*L) -> [D,L] contig.
// Grid: (elem_blocks, BEH*C)  Block: (256,1,1)
// ---------------------------------------------------------------------------
__global__ void pack_x_batched_kernel(
    const float* __restrict__ src_base,
    float*       __restrict__ dst,
    int D, int C, int L)
{
    const int batch_idx = blockIdx.y;
    const int beh = batch_idx / C;
    const int ci  = batch_idx % C;
    const int total = D * L;

    const float* src = src_base + static_cast<size_t>(beh) * D * C * L + static_cast<size_t>(ci) * L;
    float* out = dst + static_cast<size_t>(batch_idx) * D * L;

    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        const int r = idx / L;
        const int c = idx % L;
        out[r * L + c] = src[r * (C * L) + c];
    }
}

// ---------------------------------------------------------------------------
// Batched pack+transpose: State [N,D] -> [D,N] for all batches.
// src: [total_batches, N, D] contiguous  ->  dst: [total_batches, D, N]
// Grid: (elem_blocks, total_batches)  Block: (256,1,1)
// ---------------------------------------------------------------------------
__global__ void pack_state_batched_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int N, int D)
{
    const int batch_idx = blockIdx.y;
    const int total = N * D;
    const float* in  = src + static_cast<size_t>(batch_idx) * N * D;
    float*       out = dst + static_cast<size_t>(batch_idx) * D * N;

    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        const int r = idx / D;
        const int c = idx % D;
        out[c * N + r] = in[r * D + c];
    }
}

} // namespace ck_tile
