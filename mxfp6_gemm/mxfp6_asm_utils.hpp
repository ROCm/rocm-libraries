#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace mxfp6 {

// ---- Vector types ----

using v3i  = __attribute__((__vector_size__(3 * 4))) int;
using v8i  = __attribute__((__vector_size__(8 * 4))) int;
using v16f = __attribute__((__vector_size__(16 * 4))) float;

// ---- Wait helpers ----

__device__ __forceinline__ void wait_vmcnt(int n) {
    if      (n == 0) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    else if (n == 1) asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
    else if (n == 2) asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
    else if (n == 3) asm volatile("s_waitcnt vmcnt(3)" ::: "memory");
    else if (n == 4) asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
}

__device__ __forceinline__ void wait_lgkmcnt(int n) {
    if      (n == 0) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    else if (n == 1) asm volatile("s_waitcnt lgkmcnt(1)" ::: "memory");
    else if (n == 2) asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
    else if (n == 3) asm volatile("s_waitcnt lgkmcnt(3)" ::: "memory");
    else if (n == 4) asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
}

// ---- GLOBAL_LOAD_LDS_DWORDX4: async HBM → LDS (zero VGPR) ----
// LDS destination = M0.  Set M0 via set_m0() before calling.

__device__ __forceinline__ void set_m0(uint32_t val) {
    asm volatile("s_mov_b32 m0, %0" : : "s"(val));
}

__device__ __forceinline__ void async_load_lds_b128(
    void* smem_anchor, const void* gaddr)
{
    asm volatile(
        "global_load_lds_dwordx4 %1, off offset:0"
        : "=r"(smem_anchor)
        : "v"(gaddr)
        : "memory"
    );
}

// ---- DS_READ for FP6: read 24 bytes (32 FP6 values) ----
//
// Split into issue/complete so the caller can hide LDS latency.
//
//   v3i lo, hi;
//   ds_read_fp6x32_issue(addr, lo, hi);
//   // ... overlap other work ...
//   wait_lgkmcnt(0);
//   v8i reg = ds_read_fp6x32_complete(lo, hi);

__device__ __forceinline__ void ds_read_fp6x32_issue(
    uint32_t lds_byte_offset, v3i& lo, v3i& hi)
{
    asm volatile(
        "ds_read_b96 %0, %2 offset:0\n"
        "ds_read_b96 %1, %2 offset:12"
        : "=v"(lo), "=v"(hi)
        : "v"(lds_byte_offset)
        : "memory"
    );
}

__device__ __forceinline__ v8i ds_read_fp6x32_complete(v3i lo, v3i hi) {
    int d0, d1, d2, d3, d4, d5;
    asm volatile(
        "v_mov_b32 %0, %6\n v_mov_b32 %1, %7\n v_mov_b32 %2, %8\n"
        "v_mov_b32 %3, %9\n v_mov_b32 %4, %10\n v_mov_b32 %5, %11"
        : "=v"(d0), "=v"(d1), "=v"(d2), "=v"(d3), "=v"(d4), "=v"(d5)
        : "v"(lo[0]), "v"(lo[1]), "v"(lo[2]),
          "v"(hi[0]), "v"(hi[1]), "v"(hi[2])
    );
    return v8i{d0, d1, d2, d3, d4, d5, 0, 0};
}

// Convenience wrapper (for simple tests only — stalls pipeline)
__device__ __forceinline__ v8i ds_read_fp6x32(uint32_t lds_byte_offset) {
    v3i lo, hi;
    ds_read_fp6x32_issue(lds_byte_offset, lo, hi);
    wait_lgkmcnt(0);
    return ds_read_fp6x32_complete(lo, hi);
}

// ---- MFMA ----
//
// V_MFMA_SCALE_F32_32x32x64_F8F6F4  (FP6 E2M3 × FP6 E2M3)
//
// BYTE_SEL: which byte of the scale VGPR to use (0-3).
// scale_a/b MUST be in VGPR with only the selected byte meaningful.

struct alignas(64) AccTile { v16f vec; };

__device__ __forceinline__ void clear_acc(AccTile& acc) { acc.vec = v16f{}; }

// TransposeC variant: swap src0/src1 so output layout has each lane
// holding 1 M-row × 16 N-columns, enabling vectorized stores.
//
// Calling mfma(B, A, ...) computes D_out[n][m] = C[m][n] = A*B.
// The output is C transposed, but the store maps it back correctly.
//
// For FP6 × FP6: cbsz=2 for src0(=B), blgp=2 for src1(=A).
// scale_a in the builtin → src0 scale = B's scale.
// scale_b in the builtin → src1 scale = A's scale.
template <int BYTE_SEL>
__device__ __forceinline__ void mfma_scale_f32_32x32x64_fp6(
    AccTile& acc, v8i a, v8i b, int scale_a, int scale_b)
{
    static_assert(BYTE_SEL >= 0 && BYTE_SEL <= 3, "byte_sel must be 0-3");
    acc.vec = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        b, a, acc.vec,
        /*cbsz=*/2, /*blgp=*/2,
        /*opsel_a=*/BYTE_SEL, scale_b,
        /*opsel_b=*/BYTE_SEL, scale_a);
}

// ---- Epilogue: store 32×32 AccTile to global ----
//
// With TransposeC (src0=B, src1=A), each lane holds:
//   m = tid % 32 (one M-row)
//   16 acc values spanning N columns
//
// N mapping per acc item:
//   m_half=0 (lanes 0-31):  acc[p] → N = (p%4) + (p/4)*8
//   m_half=1 (lanes 32-63): acc[p] → N = (p%4) + (p/4)*8 + 4
//
// Groups of 4 consecutive acc values map to 4 consecutive N positions
// → enables global_store_dwordx4 (4 stores instead of 16).

__device__ __forceinline__ void store_acc_f32(
    float* __restrict__ D, int D_stride, const AccTile& acc,
    int m_tile_base, int n_tile_base)
{
    int m = m_tile_base + (threadIdx.x % 32);
    int m_half = threadIdx.x / 32;
    float* row = &D[m * D_stride + n_tile_base];

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        int n = g * 8 + m_half * 4;
        *reinterpret_cast<float4*>(&row[n]) =
            make_float4(acc.vec[g*4], acc.vec[g*4+1], acc.vec[g*4+2], acc.vec[g*4+3]);
    }
}

} // namespace mxfp6
