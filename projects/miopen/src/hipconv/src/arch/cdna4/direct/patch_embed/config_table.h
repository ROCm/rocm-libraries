#pragma once

#include "config.h"

#include <array>

// The compiled set of patch_embed configs.
//
// Intentionally free of device code: the autoshard generator includes this header on the
// host to count the configs it must instantiate.

namespace hipconv::cdna4::patch_embed
{

// The tiles the ladder picks between; kernel.h's is_valid_config chooses.
//
// Narrow, wide, huge, then the narrow tile at half the chunk for the K depths 128 cannot
// pair up. Twenty tiles from M32xN128 to M256xN256 were measured over n 1..256 of the
// 16x16 stem (200 iters, all correct); these three are a ladder that lands within 9.0% of
// the per-shape best everywhere, against 19.3% for the best single tile.
//
// What the ladder is really climbing is traffic, not occupancy. Every one of these runs a
// single workgroup per CU, and the two re-reads trade directly: weight bytes are (M tiles)
// * the whole filter and input bytes are (N tiles) * the whole image, so widening both
// axes is the only move that cuts either without paying the other. At 76800 rows that is
// 1534 MB for M96xN192 against 826 MB for M256xN256, and the measured throughput follows
// it monotonically -- 563 against 670 TFLOP/s. Halving the chunk to buy a second workgroup
// per CU was tried on the wide tile and lost 4 to 11%, so the occupancy is not what is
// short.
//
// The huge tile only pays once there are rows to amortise it over: it trails badly below
// ~14000 rows, where its 900-row-tall grid cannot fill the machine at all.
//
// The last three are the same ladder for the filter rows the granule does not divide --
// 14x14 over 3 channels, so ViT-L/14 and CLIP. They are not a fallback: the explicit-GEMM
// family is off by default, so a shape this family declines has no kernel at all.
//
// The huge one takes a quarter of the chunk the aligned ladder gives it, and that is the
// whole of its lead: at 32 it holds 65536 B and two workgroups per CU where 64 holds
// 131072 B and one, worth 1.7x at 65536 rows (0.295 -> 0.175 ms). The narrow and wide ones
// go the other way and keep 128, where the padded axis is short enough that the phase count
// costs more than the rounding does.
constexpr Config configs[] = {
    {.m_tile16 = 2, .n_tile16 = 8, .k_chunk = 128, .waves_m = 1, .waves_n = 4},
    {.m_tile16 = 4, .n_tile16 = 4, .k_chunk = 64, .waves_m = 2, .waves_n = 2},
    {.m_tile16 = 6, .n_tile16 = 12, .k_chunk = 128, .waves_m = 2, .waves_n = 6},
    {.m_tile16 = 16, .n_tile16 = 16, .k_chunk = 64, .waves_m = 2, .waves_n = 4},
    {.m_tile16 = 2, .n_tile16 = 8, .k_chunk = 64, .waves_m = 1, .waves_n = 4},
    {.m_tile16 = 8, .n_tile16 = 16, .k_chunk = 64, .waves_m = 1, .waves_n = 8},
    {.m_tile16 = 8, .n_tile16 = 8, .k_chunk = 64, .waves_m = 1, .waves_n = 4},
    {.m_tile16 = 2, .n_tile16 = 8, .k_chunk = 128, .waves_m = 1, .waves_n = 4, .row_pad = true},
    {.m_tile16 = 4, .n_tile16 = 8, .k_chunk = 64, .waves_m = 1, .waves_n = 8, .row_pad = true},
    {.m_tile16 = 8, .n_tile16 = 8, .k_chunk = 32, .waves_m = 1, .waves_n = 4, .row_pad = true},
    {.m_tile16 = 16, .n_tile16 = 16, .k_chunk = 32, .waves_m = 1, .waves_n = 8, .row_pad = true},
};

constexpr int num_configs = std::size(configs);

// Whether every entry names a kernel the compute and staging phases can build.
constexpr bool table_is_well_formed()
{
    for(const Config& cfg : configs)
    {
        // The wave grid has to divide the workgroup tile, or a wave would own a
        // fractional MFMA tile.
        if(cfg.m_tile16 % cfg.waves_m != 0 || cfg.n_tile16 % cfg.waves_n != 0)
            return false;
        // A staged chunk has to be a whole number of MFMA steps.
        if(cfg.k_chunk % MFMA_K != 0)
            return false;
        // The LDS swizzle XORs a granule index with a row, so the period must be a
        // power of two that covers every granule in a row.
        const int g = cfg.granules();
        if(g == 0 || (g & (g - 1)) != 0)
            return false;
        // Staging deals whole passes over the workgroup: no thread may be left with a
        // partial piece.
        if(cfg.a_pieces() % cfg.threads() != 0 || cfg.b_pieces() % cfg.threads() != 0)
            return false;
        if(cfg.lds_bytes() > 160 * 1024)
            return false;
        // The output has to reach global memory in some number of whole parts.
        if(cfg.drains() == 0)
            return false;
    }
    return true;
}

static_assert(table_is_well_formed(),
              "a patch_embed config is malformed: the wave grid must divide the workgroup "
              "tile; k_chunk must be a whole number of MFMA K steps and hold a power-of-two "
              "number of 16-byte granules for the XOR swizzle; both tiles must stage in whole "
              "passes over the workgroup; and the two tiles together must fit LDS");

} // namespace hipconv::cdna4::patch_embed
