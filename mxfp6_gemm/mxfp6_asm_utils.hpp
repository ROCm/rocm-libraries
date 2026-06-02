#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace mxfp6 {

// ---- Vector types ----

using v3i  = __attribute__((__vector_size__(3 * 4))) int;
using v4i  = __attribute__((__vector_size__(4 * 4))) int;
using v6i  = __attribute__((__vector_size__(6 * 4))) int;
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
        "s_waitcnt lgkmcnt(0)\n"
        "v_mov_b32 %0, %6\n v_mov_b32 %1, %7\n v_mov_b32 %2, %8\n"
        "v_mov_b32 %3, %9\n v_mov_b32 %4, %10\n v_mov_b32 %5, %11"
        : "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3), "=&v"(d4), "=&v"(d5)
        : "v"(lo[0]), "v"(lo[1]), "v"(lo[2]),
          "v"(hi[0]), "v"(hi[1]), "v"(hi[2])
    );
    return v8i{d0, d1, d2, d3, d4, d5, 0, 0};
}

__device__ __forceinline__ v8i ds_read_fp6x32(uint32_t lds_byte_offset) {
    v3i lo, hi;
    ds_read_fp6x32_issue(lds_byte_offset, lo, hi);
    return ds_read_fp6x32_complete(lo, hi);
}

// ---- DS_READ for FP6, COMPILER-MANAGED waitcnt ----
//
// Plain typed LDS load (no inline asm) so the backend's SIInsertWaitcnts pass
// sees real DS_READ machine instructions and inserts *relative* lgkmcnt itself
// — enabling automatic overlap with MFMA. `aligned(4)` keeps the access at
// 4-byte alignment so the compiler picks ds_read_b96 (matching the verified
// zero-bank-conflict layout) instead of assuming 16B-aligned ds_read_b128.
// Returns a v6i (single SSA vector → 6 *contiguous* VGPRs) so the MFMA operand
// constraint is satisfiable without scalar reconstruction. Building a v8i from
// 6 scalars lets the allocator scatter the two b96 reads into non-adjacent regs
// and skip the gather — corrupting the MFMA operand (HANDOFF 问题 #2).
__device__ __forceinline__ v6i ds_read_fp6x32_plain(
    const void* lds, uint32_t lds_byte_offset)
{
    using v6i_a = int __attribute__((__vector_size__(24), __aligned__(4)));
    const char* p = reinterpret_cast<const char*>(lds) + lds_byte_offset;
    v6i_a x = *reinterpret_cast<const v6i_a*>(p);
    return v6i{x[0], x[1], x[2], x[3], x[4], x[5]};
}

// ---- DS_WRITE, COMPILER-MANAGED ----
// Plain typed LDS store so the compiler tracks the lgkmcnt for __syncthreads.
__device__ __forceinline__ void ds_write_b128_plain(
    void* lds, uint32_t lds_byte_offset, v4i data)
{
    using v4i_a = int __attribute__((__vector_size__(16), __aligned__(4)));
    char* p = reinterpret_cast<char*>(lds) + lds_byte_offset;
    *reinterpret_cast<v4i_a*>(p) = v4i_a{data[0], data[1], data[2], data[3]};
}

// ---- DS_WRITE: VGPR → LDS ----

__device__ __forceinline__ void ds_write_b128(uint32_t lds_byte_offset, v4i data) {
    asm volatile(
        "ds_write_b128 %0, %1"
        : : "v"(lds_byte_offset), "v"(data) : "memory"
    );
}

// ---- v8i ↔ v6i conversion ----

__device__ __forceinline__ v6i to_v6i(v8i x) {
    return v6i{x[0], x[1], x[2], x[3], x[4], x[5]};
}

// ---- MFMA ----
//
// V_MFMA_SCALE_F32_32x32x64_F8F6F4  (FP6 E2M3 × FP6 E2M3)
//
// TransposeC: src0=B, src1=A → each lane holds 1 M-row × 16 N-cols.
// cbsz=2 (FP6 for src0=B), blgp=2 (FP6 for src1=A).
//
// Accumulator tile types:
//   AccTileV — accumulator in Arch VGPR  (ACC_CD=0)
//   AccTileA — accumulator in AccVGPR    (ACC_CD=1)
//
// On gfx950, src0/src1 (A/B) can be VGPR or AccVGPR (ACC bit).
// src2/vdst (C/D) can be VGPR or AccVGPR (ACC_CD bit).
// Data loads (global_load, ds_read) write to Arch VGPR.
// Data stores (global_store) can read from either register file.
// → No explicit cross-file moves needed in the data path.

struct alignas(64) AccTileV { v16f vec; };
struct alignas(64) AccTileA { v16f vec; };

using AccTile = AccTileV;

__device__ __forceinline__ void clear_acc(AccTileV& acc) { acc.vec = v16f{}; }
__device__ __forceinline__ void clear_acc(AccTileA& acc) { acc.vec = v16f{}; }

// MFMA: A/B in Arch VGPR, accumulator in Arch VGPR
template <int BYTE_SEL>
__device__ __forceinline__ void mfma_scale_f32_32x32x64_fp6(
    AccTileV& acc, v8i a, v8i b, int scale_a, int scale_b)
{
    static_assert(BYTE_SEL == 0, "only BYTE_SEL=0 supported for now");
    v6i b6 = to_v6i(b), a6 = to_v6i(a);
    asm volatile(
        "v_mfma_scale_f32_32x32x64_f8f6f4 %0, %1, %2, %0, %3, %4 cbsz:2 blgp:2"
        : "+v"(acc.vec)
        : "v"(b6), "v"(a6), "v"(scale_b), "v"(scale_a)
    );
}

// MFMA: A/B in Arch VGPR, accumulator in AccVGPR
template <int BYTE_SEL>
__device__ __forceinline__ void mfma_scale_f32_32x32x64_fp6(
    AccTileA& acc, v8i a, v8i b, int scale_a, int scale_b)
{
    static_assert(BYTE_SEL == 0, "only BYTE_SEL=0 supported for now");
    v6i b6 = to_v6i(b), a6 = to_v6i(a);
    asm volatile(
        "v_mfma_scale_f32_32x32x64_f8f6f4 %0, %1, %2, %0, %3, %4 cbsz:2 blgp:2"
        : "+a"(acc.vec)
        : "v"(b6), "v"(a6), "v"(scale_b), "v"(scale_a)
    );
}

// MFMA: v6i operands (already 6 contiguous VGPRs from plain loads).
// No to_v6i round-trip → no scalar reconstruction → operand stays contiguous.
template <int BYTE_SEL>
__device__ __forceinline__ void mfma_scale_f32_32x32x64_fp6(
    AccTileA& acc, v6i a, v6i b, int scale_a, int scale_b)
{
    static_assert(BYTE_SEL == 0, "only BYTE_SEL=0 supported for now");
    asm volatile(
        "v_mfma_scale_f32_32x32x64_f8f6f4 %0, %1, %2, %0, %3, %4 cbsz:2 blgp:2"
        : "+a"(acc.vec)
        : "v"(b), "v"(a), "v"(scale_b), "v"(scale_a)
    );
}

// MFMA: v6i operands, accumulator in Arch VGPR (ACC_CD=0).
// Mirror of the AccTileA v6i overload — lets a tile mix AGPR + Arch-VGPR
// accumulators so a single wave can hold more than 256 AGPR worth of acc.
template <int BYTE_SEL>
__device__ __forceinline__ void mfma_scale_f32_32x32x64_fp6(
    AccTileV& acc, v6i a, v6i b, int scale_a, int scale_b)
{
    static_assert(BYTE_SEL == 0, "only BYTE_SEL=0 supported for now");
    asm volatile(
        "v_mfma_scale_f32_32x32x64_f8f6f4 %0, %1, %2, %0, %3, %4 cbsz:2 blgp:2"
        : "+v"(acc.vec)
        : "v"(b), "v"(a), "v"(scale_b), "v"(scale_a)
    );
}

// ---- Epilogue: store 32×32 AccTile to global ----
//
// With TransposeC, each lane holds 1 M-row × 16 N-columns.
// N mapping: acc[p] → N = (p%4) + (p/4)*8 + m_half*4
// Groups of 4 consecutive acc → 4 consecutive N → global_store_dwordx4.
//
// On gfx950, global_store can read directly from AccVGPR,
// so both AccTileV and AccTileA use the same store logic.

__device__ __forceinline__ void store_acc_f32(
    float* __restrict__ D, int D_stride, const AccTileV& acc,
    int m_tile_base, int n_tile_base)
{
    int lane = threadIdx.x % 64;
    int m = m_tile_base + (lane % 32);
    int m_half = lane / 32;
    float* row = &D[m * D_stride + n_tile_base];

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        int n = g * 8 + m_half * 4;
        *reinterpret_cast<float4*>(&row[n]) =
            make_float4(acc.vec[g*4], acc.vec[g*4+1], acc.vec[g*4+2], acc.vec[g*4+3]);
    }
}

__device__ __forceinline__ void store_acc_f32(
    float* __restrict__ D, int D_stride, const AccTileA& acc,
    int m_tile_base, int n_tile_base)
{
    int lane = threadIdx.x % 64;
    int m = m_tile_base + (lane % 32);
    int m_half = lane / 32;
    float* row = &D[m * D_stride + n_tile_base];

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        int n = g * 8 + m_half * 4;
        *reinterpret_cast<float4*>(&row[n]) =
            make_float4(acc.vec[g*4], acc.vec[g*4+1], acc.vec[g*4+2], acc.vec[g*4+3]);
    }
}

} // namespace mxfp6
