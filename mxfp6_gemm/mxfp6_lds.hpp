#pragma once
// LDS deep-K staged MXFP6 GEMM kernel + tile-grouped scale layout.
// Deep K staged in LDS enlarges the MFMA window past the load latency. Uses 32x32x64 MFMA
// (not 16x16x128: that halves FLOPs/inst and doubles operand bandwidth per FLOP).
#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include <vector>
namespace mxfp6 {
// Tile-grouped scale layout: a wave's `group` consecutive 32-blocks AND its K64_PER_TILE
// k64 sub-slabs of one K-tile become contiguous per lane, so the kernel fetches a
// whole K-tile's scales in ONE dwordx{K64_PER_TILE} load (NDA=1). Layout:
//   out[(((g*k_tiles+kt)*64 + lane)*K64_PER_TILE + sub)*group_pad + j]
//        = scale of (block g*group+j, k64 = kt*K64_PER_TILE+sub)
struct TiledScale { std::vector<uint8_t> data; };
static TiledScale tile_scale(const PreprocessedScale& ps, int group, int subs) {
    int group_pad = (group + 3) / 4 * 4;
    int k_tiles = ps.k64_iters / subs;
    int ng = ps.num_tiles / group;
    TiledScale ts;
    ts.data.assign((size_t)ng * k_tiles * 64 * subs * group_pad, 0);
    for (int g = 0; g < ng; g++)
        for (int kt = 0; kt < k_tiles; kt++)
            for (int lane = 0; lane < 64; lane++)
                for (int sub = 0; sub < subs; sub++)
                    for (int j = 0; j < group; j++) {
                        int k64 = kt * subs + sub, tile = g * group + j;
                        ts.data[(((size_t)(g * k_tiles + kt) * 64 + lane) * subs + sub) * group_pad + j] =
                            ps.data[(size_t)(tile * ps.k64_iters + k64) * 64 + lane];
                    }
    return ts;
}

// Cooperative load of a (ROWS x K_TILE) FP6 tile into LDS via buffer_load_dwordx4 (MUBUF,
// LDS dest = M0 + lane*16). Row-major LDS layout:
//   LDS[lds_base + m*KT_BYTES + ck*16] = global[gbase + m*row_stride + kt_byte + ck*16]
// KT_BYTES = K_TILE*6/8, CPR = KT_BYTES/16 chunks/row; each lane fetches its own 16B,
// walking chunks (m-major, ck-minor) to match the layout.
//
// MUBUF (default): the 64-bit base lives in the buffer descriptor (V#), so each load is just
// a 32-bit voffset (~half the per-load address VALU vs FLAT). -DUSE_GLOBAL_LDS = FLAT fallback.
//
// MXFP6_M0_NOP (default 0): explicit s_nop for the "SALU writes M0 -> load(LDS=1)" 1-wait-
// state hazard. Off for KT192 (compiler already puts many instrs between set_m0 and the load,
// so it's redundant + free in this latency-bound kernel). Set =1 for configs packing loads
// back-to-back with no separating instr (KT256 single-buffer raced without it). FLAT handles
// the hazard automatically.
#ifndef MXFP6_M0_NOP
#define MXFP6_M0_NOP 0
#endif
// ds_read one 32x32 MFMA operand (32 FP6 / lane = 24B) from an LDS tile staged by the
// the A cooperative loader. blk = 32-row block within the tile, sub = which 64-K sub-slab,
// kh/m from lane. Operand layout matches load: row (blk*32+lane%32) slab at
// row*KT_BYTES, sub-slab at sub*48, k-half at (lane/32)*24.
template <int KT_BYTES>
__device__ __forceinline__ v6i read_op(const char* smem, uint32_t lds_base, int blk,
                                      int sub, int lane) {
    uint32_t off = lds_base + (uint32_t)((blk * 32 + (lane & 31)) * KT_BYTES + sub * 48 +
                                         (lane >> 5) * 24);
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
        int2 v; asm volatile("global_load_dwordx2 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v.x; out[1] = v.y;
    } else if (nw == 3) {
        v3i v; asm volatile("global_load_dwordx3 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
    } else if (nw == 4) {
        v4i v; asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v[0]; out[1] = v[1]; out[2] = v[2]; out[3] = v[3];
    } else {
        out[0] = asm_load_dword_nowait(a);
    }
}

} // namespace mxfp6
