#pragma once
// Device-side primitives for the MXFP6 GEMM kernel (kernel.hpp): vector types, waitcnt
// helpers, the FP6 LDS read, the scale loads, and the (swapped-operand) scaled MFMA.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace mxfp6 {

// ---- Vector types ----

using v3i = __attribute__((__vector_size__(3 * 4))) int;
using v4i = __attribute__((__vector_size__(4 * 4))) int;
using v6i = __attribute__((__vector_size__(6 * 4))) int;
using v16f = __attribute__((__vector_size__(16 * 4))) float;

// ---- Wait helpers ----
// Only vmcnt(0) is used in steady state (the hard RAW guard before each double-buffer
// barrier); the parameter keeps the helper reusable for asm experiments.
__device__ __forceinline__ void wait_vmcnt(int n) {
    if (n == 0)
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    else if (n == 1)
        asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
    else if (n == 2)
        asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
    else if (n == 3)
        asm volatile("s_waitcnt vmcnt(3)" ::: "memory");
    else if (n == 4)
        asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
}

// Set M0 (the LDS destination address for cooperative buffer_load ... lds).
__device__ __forceinline__ void set_m0(uint32_t val) {
    asm volatile("s_mov_b32 m0, %0" : : "s"(val));
}

// ---- FP6 LDS read (compiler-managed waitcnt) ----
//
// Read 24 bytes (32 FP6 values / lane) of an MFMA operand from LDS. A plain typed load (no
// inline asm) so the backend's SIInsertWaitcnts pass sees real DS_READ instructions and
// inserts *relative* lgkmcnt itself — enabling automatic overlap with MFMA. `aligned(4)`
// keeps the access 4-byte aligned so the compiler picks ds_read_b96 (matching the verified
// zero-bank-conflict layout) instead of assuming 16B-aligned ds_read_b128. Returns a v6i
// (single SSA vector -> 6 *contiguous* VGPRs) so the MFMA operand constraint is satisfiable
// without scalar reconstruction (a scattered build corrupts the operand).
__device__ __forceinline__ v6i ds_read_fp6x32_plain(const void* lds, uint32_t lds_byte_offset) {
    using v6i_a = int __attribute__((__vector_size__(24), __aligned__(4)));
    const char* p = reinterpret_cast<const char*>(lds) + lds_byte_offset;
    v6i_a x = *reinterpret_cast<const v6i_a*>(p);
    return v6i{x[0], x[1], x[2], x[3], x[4], x[5]};
}

// ds_read one 32x32 MFMA operand (32 FP6 / lane = 24B) from an LDS tile staged by
// issue_A_chunks (kernel.hpp). blk = 32-row block in the tile, sub = which 64-K sub-slab,
// k-half + row from lane. Operand layout matches the load: row (blk*32+lane%32) slab at
// row*KT_BYTES, sub-slab at sub*48, k-half at (lane/32)*24.
template <int KT_BYTES>
__device__ __forceinline__ v6i read_op(const char* smem, uint32_t lds_base, int blk, int sub,
                                       int lane) {
    uint32_t off =
        lds_base + (uint32_t)((blk * 32 + (lane & 31)) * KT_BYTES + sub * 48 + (lane >> 5) * 24);
    return ds_read_fp6x32_plain(smem, off);
}

// ---- Scale loads (no-wait inline asm) ----
//
// Inline-asm global load to VGPR with NO waitcnt (caller manages vmcnt). Keeps scale loads
// off the compiler's vmcnt accounting so a typed load wouldn't drain the in-flight prefetch.
// "memory" clobber preserves ordering vs the manual waits.
__device__ __forceinline__ int asm_load_dword_nowait(const void* a) {
    int v;
    asm volatile("global_load_dword %0, %1, off" : "=v"(v) : "v"(a) : "memory");
    return v;
}

// Wide no-wait loads: bring K64_PER_TILE contiguous scale dwords for a whole K-tile in ONE
// instruction (tile-grouped scale layout) instead of one dword per sub. Cuts scale vmem op
// count ~K64_PER_TILE x. out[] gets the dwords. Only NW in {2,3,4} (gfx950 dwordx3).
__device__ __forceinline__ void asm_load_dwordxN_nowait(int* out, const void* a, int nw) {
    if (nw == 2) {
        int2 v;
        asm volatile("global_load_dwordx2 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v.x;
        out[1] = v.y;
    } else if (nw == 3) {
        v3i v;
        asm volatile("global_load_dwordx3 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v[0];
        out[1] = v[1];
        out[2] = v[2];
    } else if (nw == 4) {
        v4i v;
        asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v[0];
        out[1] = v[1];
        out[2] = v[2];
        out[3] = v[3];
    } else {
        out[0] = asm_load_dword_nowait(a);
    }
}

// ---- Scaled MFMA: V_MFMA_SCALE_F32_32x32x64_F8F6F4 (FP6 E2M3 x FP6 E2M3) ----
//
// Accumulator lives in AccVGPR ("+a"); 32x32 output = v16f / lane. cbsz=2/blgp=2 select FP6.
struct alignas(64) AccTile {
    v16f vec;
};

__device__ __forceinline__ void clear_acc(AccTile& acc) {
    acc.vec = v16f{};
}

// SWAPPED operands: src0=A, src1=B (vs the stock src0=B,src1=A "TransposeC"). This makes the
// accumulator come out as C^T in registers -> each lane holds 1 N-COLUMN x 16 M-rows. The
// payoff is in the EPILOGUE: storing row-major D[m][n] with n = base + lane%32 is then
// NATURALLY COALESCED (consecutive lanes -> consecutive N -> consecutive addresses), so no
// LDS-transpose / no barrier / no extra LDS is needed, and it works for any OutT.
// Operands are symmetric (both "lane holds 32 K-values"), so swapping src0/src1 is valid;
// each operand keeps its own scale (scale0=scale_a with src0=a, scale1=scale_b with src1=b).
__device__ __forceinline__ void mfma_scale_f32_32x32x64_fp6_swapC(AccTile& acc, v6i a, v6i b,
                                                                  int scale_a, int scale_b) {
    asm volatile("v_mfma_scale_f32_32x32x64_f8f6f4 %0, %1, %2, %0, %3, %4 cbsz:2 blgp:2"
                 : "+a"(acc.vec)
                 : "v"(a), "v"(b), "v"(scale_a), "v"(scale_b));
}

}  // namespace mxfp6
