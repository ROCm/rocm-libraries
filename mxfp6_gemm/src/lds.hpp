#pragma once
// Device helpers for the LDS-staged MXFP6 GEMM: ds_read of an MFMA operand from LDS +
// no-wait inline-asm scale loads. (Host-side tile_scale layout lives in preprocess.hpp;
// the cooperative A loader + kernel live in lds_hybrid.hpp.)
#include <vector>

#include "asm_utils.hpp"
#include "mxfp6/preprocess.hpp"
namespace mxfp6 {
// ds_read one 32x32 MFMA operand (32 FP6 / lane = 24B) from an LDS tile staged by
// issue_A_chunks (lds_hybrid.hpp). blk = 32-row block in the tile, sub = which 64-K sub-slab,
// kh/m from lane. Operand layout matches load: row (blk*32+lane%32) slab at
// row*KT_BYTES, sub-slab at sub*48, k-half at (lane/32)*24.
template <int KT_BYTES>
__device__ __forceinline__ v6i read_op(const char* smem, uint32_t lds_base, int blk, int sub,
                                       int lane) {
    uint32_t off =
        lds_base + (uint32_t)((blk * 32 + (lane & 31)) * KT_BYTES + sub * 48 + (lane >> 5) * 24);
    return ds_read_fp6x32_plain(smem, off);
}

// Inline-asm global load to VGPR with NO waitcnt (caller manages vmcnt). Keeps scale loads
// off the compiler's vmcnt accounting so a typed load wouldn't drain the in-flight prefetch.
// "memory" clobber preserves ordering vs the manual waits.
__device__ __forceinline__ int asm_load_dword_nowait(const void* a) {
    int v;
    asm volatile("global_load_dword %0, %1, off" : "=v"(v) : "v"(a) : "memory");
    return v;
}

// Wide no-wait loads: bring K64_PER_TILE contiguous scale dwords for a whole K-tile in ONE
// instruction (tile-grouped scale layout) instead of one dword per sub. Cuts scale
// vmem op count ~K64_PER_TILE x. out[] gets the dwords. Only NW in {2,3,4} (gfx950 dwordx3).
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

}  // namespace mxfp6
