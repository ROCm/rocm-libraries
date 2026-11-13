#pragma once

#include "config.h"

// The compiled set of direct_l1 configs and the attribute-keyed index lookup.
//
// Intentionally free of device code so the unit test can name a config
// (config_index()) without recompiling the kernel; the geometry helpers live
// elsewhere.

namespace hipconv::cdna4::direct_l1
{
using namespace hipconv;

constexpr Config configs[] = {
    // 3x3
    {
        .wave_k16 = 4,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
    },

    // 3x3 K256 tile (waves_k=4): P(16) x Q(16) x K(256).
    //
    // Selected when its narrower 16-column W-block overcomputes less than the K128
    // tile's 32-column one (see is_valid_config), reusing one input tile across 4x the
    // output channels. Per-wave geometry matches the K128 wave, so register pressure is
    // unchanged.
    {
        .waves_k  = 4,
        .wave_k16 = 4,
        .wave_p   = 8,
        .waves_q  = 1,
        .kh       = 3,
        .kw       = 3,
    },

    // 3x3 K256 arbitrary-K fallback (k_divisible=false).
    //
    // Supports the K256 window for K not a multiple of 256 (padded, straddling stores
    // guarded).
    {
        .waves_k     = 4,
        .wave_k16    = 4,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .k_divisible = false,
    },

    // 3x3 batch-unfold K128 (K32/wave): P(8) x N(4) x W(8) x K(128).
    //
    // Supports low-resolution maps (output width <= 8) by packing 4 batch images along
    // the columns. The 4-way K split halves per-wave weight traffic for more input
    // LDS reads, the right trade when the tiny input tile makes weights dominate.
    // Ranks below the K256 unfold sibling below on K % 256 == 0.
    {
        .waves_k  = 4,
        .wave_k16 = 2,
        .wave_p   = 8,
        .waves_p  = 1,
        .waves_q  = 2,
        .unfold_n = 4,
        .kh       = 3,
        .kw       = 3,
    },

    // 3x3 batch-unfold K256 (K64/wave): the deeper-K sibling of the K128 unfold.
    //
    // Same P(8) x N(4) x W(8) low-res tile, but block_k = 256. Deepening K doubles
    // input-tile reuse at the same LDS traffic; per-wave geometry matches the K256
    // tile (proven spill-free). Supports K % 256 == 0; the rest stay on K128.
    {
        .waves_k  = 4,
        .wave_k16 = 4,
        .wave_p   = 8,
        .waves_p  = 1,
        .waves_q  = 2,
        .unfold_n = 4,
        .kh       = 3,
        .kw       = 3,
    },

    // ======================= tf32 tiles (elem_bytes = 4) =======================
    //
    // The doubled LDS tile picks the whole family's tall-narrow shape:
    //   - waves_k = 4, waves_q = 1 (K128, Q16). LDS only fits the doubled tile at
    //     block_h * block_w <= 319, which a Q(32) block blows past. And a tf32 lane
    //     covers 4 channels, so a wave loads 4 columns and the W axis needs
    //     ceil(block_w / 4) = 5 waves, which only a 4-way K split supplies.
    //   - wave_k16 = 2, so the writer takes its narrow register store: the LDS-staged
    //     wide store is 16-bit only, and skipping it frees the staging pad the doubled
    //     tile needs.
    //   - wave_p is the largest keeping block_h * block_w <= 319, per filter, except
    //     for 3x1, which runs out of registers first (see its tiles).
    //
    // Per (filter, direction) there are four entries: the divisible tile and the three
    // fallback corners (output count not a multiple of 128; a reduction of one C(64)
    // iteration; both). There is no unfold, K256, or K96 sibling, so the standard tile
    // supports every output width and the padded K128 covers those counts.

    // ---- 3x3 tf32: P(14) x Q(16) x K(128), block_h 16 x block_w 18 = 288 ----
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },

    // ---- 2x2 tf32: P(16) x Q(16) x K(128), block_h 17 x block_w 17 = 289 ----
    //
    // The shallowest halo of the family, so it affords the full wave_p = 8.
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 8,
        .waves_q    = 1,
        .kh         = 2,
        .kw         = 2,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 2,
        .kw          = 2,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 8,
        .waves_q    = 1,
        .kh         = 2,
        .kw         = 2,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 2,
        .kw          = 2,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 8,
        .waves_q    = 1,
        .kh         = 2,
        .kw         = 2,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 2,
        .kw          = 2,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 8,
        .waves_q    = 1,
        .kh         = 2,
        .kw         = 2,
        .direction  = hipconv::Direction::Dgrad,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 2,
        .kw          = 2,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },

    // ---- 4x4 tf32: P(12) x Q(16) x K(128), block_h 15 x block_w 19 = 285 ----
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 4,
        .kw         = 4,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 6,
        .waves_q     = 1,
        .kh          = 4,
        .kw          = 4,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 4,
        .kw         = 4,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 6,
        .waves_q     = 1,
        .kh          = 4,
        .kw          = 4,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 4,
        .kw         = 4,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 6,
        .waves_q     = 1,
        .kh          = 4,
        .kw          = 4,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 4,
        .kw         = 4,
        .direction  = hipconv::Direction::Dgrad,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 6,
        .waves_q     = 1,
        .kh          = 4,
        .kw          = 4,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },

    // ---- 5x5 tf32: P(10) x Q(16) x K(128), block_h 14 x block_w 20 = 280 ----
    //
    // The deepest halo, so the shortest tile: wave_p = 6 would need 160.5 KiB.
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 5,
        .waves_q    = 1,
        .kh         = 5,
        .kw         = 5,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 5,
        .waves_q     = 1,
        .kh          = 5,
        .kw          = 5,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 5,
        .waves_q    = 1,
        .kh         = 5,
        .kw         = 5,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 5,
        .waves_q     = 1,
        .kh          = 5,
        .kw          = 5,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 5,
        .waves_q    = 1,
        .kh         = 5,
        .kw         = 5,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 5,
        .waves_q     = 1,
        .kh          = 5,
        .kw          = 5,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 5,
        .waves_q    = 1,
        .kh         = 5,
        .kw         = 5,
        .direction  = hipconv::Direction::Dgrad,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 5,
        .waves_q     = 1,
        .kh          = 5,
        .kw          = 5,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },

    // ---- 3x1 tf32: P(12) x Q(16) x K(128), block_h 14 x block_w 16 = 224 ----
    //
    // The one tile sized by registers, not LDS. With no W halo, LDS would afford
    // wave_p = 8, but kw = 1 leaves only kw * 2 = 2 load steps, so each one stages half
    // the tile's rows in VGPRs before the bf16 split: 2 x 9 rows x 4 = 72 registers at
    // wave_p = 8. wave_p = 7 still spills; wave_p = 6 stages 56 and fits.
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 1,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 6,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 1,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 1,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 6,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 1,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 1,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 6,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 1,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .elem_bytes  = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 1,
        .direction  = hipconv::Direction::Dgrad,
        .single_c   = true,
        .elem_bytes = 4,
    },
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 6,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 1,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },
    // ---- tf32 K(64) divisible tiles (wave_k16 = 1) ----
    //
    // Same tile as the K128 entry of each filter, split 4 ways over K(16) instead of
    // K(32). It supports output == 64, the one count where halving block_k costs nothing:
    // the K128 tile runs a single block there anyway and just pads. Above
    // wide_k128_min_output the shared threshold in relative_throughput_conv2d ranks K128 first.
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 8,
        .waves_q    = 1,
        .kh         = 2,
        .kw         = 2,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 8,
        .waves_q    = 1,
        .kh         = 2,
        .kw         = 2,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 4,
        .kw         = 4,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 4,
        .kw         = 4,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 5,
        .waves_q    = 1,
        .kh         = 5,
        .kw         = 5,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 5,
        .waves_q    = 1,
        .kh         = 5,
        .kw         = 5,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 1,
        .elem_bytes = 4,
    },
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 6,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 1,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },

    // ---- 3x1 tf32 large-tensor tiles (large_tensor): K128 / K64, wave_p = 6 ----
    //
    // The 3x1 tf32 K128 and K64 divisible tiles with the 16-bit 3x1 large_tensor tiles'
    // row-origin fold, for a single image whose span overflows int32 -- which at 4
    // bytes takes half the elements a 16-bit image does. The size gate in
    // is_valid_config makes them claim only shapes the baseline cannot. Like the
    // 16-bit trio they support divisible outputs only.
    {
        .waves_k      = 4,
        .wave_k16     = 2,
        .wave_p       = 6,
        .waves_q      = 1,
        .kh           = 3,
        .kw           = 1,
        .elem_bytes   = 4,
        .large_tensor = true,
    },
    {
        .waves_k      = 4,
        .wave_k16     = 1,
        .wave_p       = 6,
        .waves_q      = 1,
        .kh           = 3,
        .kw           = 1,
        .elem_bytes   = 4,
        .large_tensor = true,
    },
    {
        .waves_k      = 4,
        .wave_k16     = 2,
        .wave_p       = 6,
        .waves_q      = 1,
        .kh           = 3,
        .kw           = 1,
        .direction    = hipconv::Direction::Dgrad,
        .elem_bytes   = 4,
        .large_tensor = true,
    },
    {
        .waves_k      = 4,
        .wave_k16     = 1,
        .wave_p       = 6,
        .waves_q      = 1,
        .kh           = 3,
        .kw           = 1,
        .direction    = hipconv::Direction::Dgrad,
        .elem_bytes   = 4,
        .large_tensor = true,
    },
    // ===================== end of the tf32 tiles =====================

    // 2x2
    {
        .wave_k16 = 4,
        .wave_p   = 8,
        .kh       = 2,
        .kw       = 2,
    },

    // 2x2 K256 tile (waves_k=4): mirrors the 3x3 K256 for the 2x2 filter.
    {
        .waves_k  = 4,
        .wave_k16 = 4,
        .wave_p   = 8,
        .waves_q  = 1,
        .kh       = 2,
        .kw       = 2,
    },
    // 2x2 K256 arbitrary-K fallback (k_divisible=false).
    {
        .waves_k     = 4,
        .wave_k16    = 4,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 2,
        .kw          = 2,
        .k_divisible = false,
    },

    // 2x2 batch-unfold (K32/wave): mirrors the other K32 unfolds for the 2x2 filter.
    {
        .waves_k  = 4,
        .wave_k16 = 2,
        .wave_p   = 8,
        .waves_p  = 1,
        .waves_q  = 2,
        .unfold_n = 4,
        .kh       = 2,
        .kw       = 2,
    },

    // 4x4 K128 tall-narrow tile (waves_k=4, wave_k16=2): P(16) x Q(16) x K(128).
    //
    // The only K128 4x4 fprop tile; replaced the wide P(14) x Q(32) tile it
    // outperforms (H%16==0 fits P(16) exactly; measured ~24% faster on H32xW32).
    {
        .waves_k  = 4,
        .wave_k16 = 2,
        .wave_p   = 8,
        .waves_q  = 1,
        .kh       = 4,
        .kw       = 4,
    },

    // 5x5 K128 tall-narrow tile (waves_k=4, wave_k16=2): P(16) x Q(16) x K(128).
    //
    // The only K128 5x5 fprop tile; replaced the wide P(12) x Q(32) tile it
    // outperforms (narrow Q(16) also shrinks the halo; measured ~16% faster on
    // H32xW32). 5x5 has no K256 tile, so this supports every 5x5 fprop count.
    {
        .waves_k  = 4,
        .wave_k16 = 2,
        .wave_p   = 8,
        .waves_q  = 1,
        .kh       = 5,
        .kw       = 5,
    },

    // 4x4 batch-unfold: the sole 4x4 unfold tile (K32/wave).
    {
        .waves_k  = 4,
        .wave_k16 = 2,
        .wave_p   = 8,
        .waves_p  = 1,
        .waves_q  = 2,
        .unfold_n = 4,
        .kh       = 4,
        .kw       = 4,
    },

    // 5x5 batch-unfold: the sole 5x5 unfold tile (K32/wave): P(8) x N(4) x W(8) x K(128).
    //
    // Split 4 ways along K (per wave K(32)); the split halves per-wave weight traffic
    // for more input LDS reads (measured ~20% win, +39% small N). waves_p=1 and
    // waves_q=2 are both explicit: waves_p=1 halves weight redundancy, waves_q=2
    // keeps the 4-image column block; either at the default 2 breaks the tile.
    {
        .waves_k  = 4,
        .wave_k16 = 2,
        .wave_p   = 8,
        .waves_p  = 1,
        .waves_q  = 2,
        .unfold_n = 4,
        .kh       = 5,
        .kw       = 5,
    },

    // Batch-unfold arbitrary-output fallback (K128 unfold, k_divisible=false).
    //
    // Supports the W<=8 unfold window when the output channel count is not a multiple of
    // 128 (e.g. 96, 160, 224); the divisible unfold tiles above require output %
    // block_k == 0. The writer pads output to 128 and guards the straddling wave.
    // One per filter; 3x3 needs no K256 variant (%256 is a subset of %128, which the
    // K256 divisible unfold takes).
    // 3x3
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 1,
        .waves_q     = 2,
        .unfold_n    = 4,
        .kh          = 3,
        .kw          = 3,
        .k_divisible = false,
    },
    // 2x2
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 1,
        .waves_q     = 2,
        .unfold_n    = 4,
        .kh          = 2,
        .kw          = 2,
        .k_divisible = false,
    },
    // 4x4
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 1,
        .waves_q     = 2,
        .unfold_n    = 4,
        .kh          = 4,
        .kw          = 4,
        .k_divisible = false,
    },
    // 5x5
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 1,
        .waves_q     = 2,
        .unfold_n    = 4,
        .kh          = 5,
        .kw          = 5,
        .k_divisible = false,
    },


    // K(96) divisible tiles (wave_k16=3), for K divisible by 3*2^n.
    //
    // wave_p depends only on filter size and block_p (weights bypass LDS), so each
    // filter reuses the K128 wave_p above.
    // 3x3
    {
        .wave_k16 = 3,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
    },
    // 2x2
    {
        .wave_k16 = 3,
        .wave_p   = 8,
        .kh       = 2,
        .kw       = 2,
    },
    // 4x4
    {
        .wave_k16 = 3,
        .wave_p   = 7,
        .kh       = 4,
        .kw       = 4,
    },
    // 5x5
    {
        .wave_k16 = 3,
        .wave_p   = 6,
        .kh       = 5,
        .kw       = 5,
    },
    // K(64) divisible tiles (wave_k16=2). Same per-filter wave_p as above.
    // 3x3, in 3 steps of C(64)
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .c64_step = true,
    },
    // 2x2, in 2 steps of C(64)
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 2,
        .kw       = 2,
        .c64_step = true,
    },
    // 4x4, in 4 steps of C(64)
    {
        .wave_k16 = 2,
        .wave_p   = 7,
        .kh       = 4,
        .kw       = 4,
        .c64_step = true,
    },
    // 5x5, in 5 steps of C(64)
    {
        .wave_k16 = 2,
        .wave_p   = 6,
        .kh       = 5,
        .kw       = 5,
        .c64_step = true,
    },

    // 3x1 divisible tiles (asymmetric filter): K128/K96/K64, wave_p=8.
    //
    // The one-wide filter has no W halo (block_w == block_q), so its input tile is
    // strictly smaller than 3x3's and wave_p=8 is safe. num_load_steps = kw*2 = 2.
    // The specialist comparisons in relative_throughput_conv2d (unfold, K256) gate on kw == kh, so
    // a 3x1 config falls through to the generic divisible family with no ties. K128
    {
        .wave_k16 = 4,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 1,
    },
    // K96
    {
        .wave_k16 = 3,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 1,
    },
    // K64
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 1,
    },

    // 3x1 large-tensor tiles (large_tensor): K128/K96/K64, wave_p=8.
    //
    // Identical geometry to the 3x1 divisible tiles above, but with the row-origin
    // fold that lets a single image exceed 2 GiB (see kernel.h is_valid_config size
    // gate and input_loader/output_writer). Motivating case: an NFHWC 3D tensor folded
    // to NF(HW)C and convolved with the 3-tap F axis, where one image is > 10 GiB. The
    // size gate makes these valid only on shapes whose per-image span overflows int32, so
    // they never shadow the baseline tiles. The K256 tiles need no large_tensor variant:
    // relative_throughput_conv2d exempts large_tensor from the K256 comparison, so these rank first
    // on
    // the shape even when the K256 overcompute comparison would otherwise prefer K256.
    // K128
    {
        .wave_k16     = 4,
        .wave_p       = 8,
        .kh           = 3,
        .kw           = 1,
        .large_tensor = true,
    },
    // K96
    {
        .wave_k16     = 3,
        .wave_p       = 8,
        .kh           = 3,
        .kw           = 1,
        .large_tensor = true,
    },
    // K64
    {
        .wave_k16     = 2,
        .wave_p       = 8,
        .kh           = 3,
        .kw           = 1,
        .large_tensor = true,
    },

    // 3x1 K256 tile (waves_k=4): P(16) x Q(16) x K(256), mirrors the 3x3.
    //
    // Selected when its 16-column W-block overcomputes less than the K128 tile's
    // 32-column one, reusing one input tile across 4x the output channels. Divisible
    // (K % 256 == 0) variant.
    {
        .waves_k  = 4,
        .wave_k16 = 4,
        .wave_p   = 8,
        .waves_q  = 1,
        .kh       = 3,
        .kw       = 1,
    },
    // 3x1 K256 arbitrary-K fallback (k_divisible=false).
    //
    // When the K256 selection ranks first, a K not a multiple of 256 would have no
    // K256 configuration without this padded tile (the divisible K256 sibling takes only
    // % 256 == 0).
    {
        .waves_k     = 4,
        .wave_k16    = 4,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 1,
        .k_divisible = false,
    },
    // Arbitrary-K K128 fallback (k_divisible=false): accepts any K (padded).
    //
    // Compiles the writer's per-channel K bounds check that the divisible configs
    // omit; matches only K no divisible variant covers. One entry per filter size.
    // 3x3
    {
        .wave_k16    = 4,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 3,
        .k_divisible = false,
    },
    // 2x2
    {
        .wave_k16    = 4,
        .wave_p      = 8,
        .kh          = 2,
        .kw          = 2,
        .k_divisible = false,
    },
    // 4x4 narrow arbitrary-K fallback (waves_k=4, wave_k16=2, k_divisible=false).
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 4,
        .kw          = 4,
        .k_divisible = false,
    },
    // 5x5 fprop narrow arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 5,
        .kw          = 5,
        .k_divisible = false,
    },
    // 3x1
    //
    // The 3x1 K256 arbitrary-K sibling covers only the counts the K256 selection
    // ranks first, which at a wide output map it never does, so K=32 had no tile.
    {
        .wave_k16    = 4,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 1,
        .k_divisible = false,
    },
    // Small-C fallback (K64, single_c): supports C in [2, 64].
    //
    // The single lone C(64) peel iteration over-computes to C_padded=64 against zero
    // weights; for completeness, not speed. One entry per filter size.
    // 3x3
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .single_c = true,
        .c64_step = true,
    },
    // 2x2
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 2,
        .kw       = 2,
        .single_c = true,
        .c64_step = true,
    },
    // 4x4
    {
        .wave_k16 = 2,
        .wave_p   = 7,
        .kh       = 4,
        .kw       = 4,
        .single_c = true,
        .c64_step = true,
    },
    // 5x5
    {
        .wave_k16 = 2,
        .wave_p   = 6,
        .kh       = 5,
        .kw       = 5,
        .single_c = true,
        .c64_step = true,
    },
    // 3x1
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 1,
        .single_c = true,
    },
    // Small-C arbitrary-output fallback (K64, single_c, k_divisible=false).
    //
    // Supports C in [2, 64] combined with an output count not a multiple of 64 (e.g.
    // 96, 160, 224), which the divisible single_c tile above rejects. The writer
    // guards the padded output tail. For completeness, not speed. One per filter.
    // 3x3
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 3,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },
    // 2x2
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 2,
        .kw          = 2,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },
    // 4x4
    {
        .wave_k16    = 2,
        .wave_p      = 7,
        .kh          = 4,
        .kw          = 4,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },
    // 5x5
    {
        .wave_k16    = 2,
        .wave_p      = 6,
        .kh          = 5,
        .kw          = 5,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },
    // 3x1
    //
    // The pointwise fold of a (3,1,1) conv3d lands here whenever C and K are both
    // small: C=32, K=32 fails the divisible tiles on both counts at once.
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 1,
        .k_divisible = false,
        .single_c    = true,
    },
    // Dgrad (backward-data) K128 tiles (wave_k16=4): 128 din/workgroup.
    //
    // The dgrad-formatted weights carry the C<->K swap and tap flip, so the compute
    // kernel is identical to fprop. wave_p matches fprop per filter. Placed after
    // fprop so those indices are unchanged.
    // 3x3
    {
        .wave_k16  = 4,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
    },

    // 3x3 dgrad K256 tile (waves_k=4): mirrors the fprop K256 for dgrad.
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_q   = 1,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
    },

    // 3x3 dgrad K256 arbitrary-din fallback (k_divisible=false).
    {
        .waves_k     = 4,
        .wave_k16    = 4,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },

    // 3x3 dgrad batch-unfold K128 (K32/wave): the 3x3 dgrad unfold tile.
    //
    // Ranks below the K256 unfold sibling below on din % 256 == 0 (same split as fprop).
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 4,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
    },

    // 3x3 dgrad batch-unfold K256 (K64/wave): mirrors the fprop K256 unfold for dgrad.
    //
    // The deeper-K sibling of the K128 unfold above; ranks first on din % 256 == 0.
    // Before it existed the K128 unfold ceded those counts and no tile supported the W<=8
    // dgrad window.
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 4,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
    },

    // 2x2
    {
        .wave_k16  = 4,
        .wave_p    = 8,
        .kh        = 2,
        .kw        = 2,
        .direction = hipconv::Direction::Dgrad,
    },

    // 2x2 dgrad K256 tile (waves_k=4): mirrors the fprop 2x2 K256 for dgrad.
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_q   = 1,
        .kh        = 2,
        .kw        = 2,
        .direction = hipconv::Direction::Dgrad,
    },
    // 2x2 dgrad K256 arbitrary-din fallback (k_divisible=false).
    {
        .waves_k     = 4,
        .wave_k16    = 4,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 2,
        .kw          = 2,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },

    // 2x2 dgrad batch-unfold (K32/wave): mirrors the fprop 2x2 K32 unfold for dgrad.
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 4,
        .kh        = 2,
        .kw        = 2,
        .direction = hipconv::Direction::Dgrad,
    },

    // 4x4 dgrad K128 tall-narrow tile (waves_k=4): mirrors the fprop 4x4 narrow tile.
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_q   = 1,
        .kh        = 4,
        .kw        = 4,
        .direction = hipconv::Direction::Dgrad,
    },
    // 5x5 dgrad K128 tall-narrow tile (waves_k=4): mirrors the fprop 5x5 narrow tile.
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_q   = 1,
        .kh        = 5,
        .kw        = 5,
        .direction = hipconv::Direction::Dgrad,
    },

    // 4x4 dgrad batch-unfold: the sole 4x4 dgrad unfold tile (K32/wave).
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 4,
        .kh        = 4,
        .kw        = 4,
        .direction = hipconv::Direction::Dgrad,
    },
    // 5x5 dgrad batch-unfold: the sole 5x5 dgrad unfold tile (K32/wave).
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 4,
        .kh        = 5,
        .kw        = 5,
        .direction = hipconv::Direction::Dgrad,
    },

    // Dgrad batch-unfold arbitrary-output fallback (K128 unfold, k_divisible=false).
    //
    // Mirrors the fprop arbitrary unfold for backward-data: supports the W<=8 window
    // when the dgrad output (din, = C) is not a multiple of 128. One per filter.
    // 3x3
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 1,
        .waves_q     = 2,
        .unfold_n    = 4,
        .kh          = 3,
        .kw          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },
    // 2x2
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 1,
        .waves_q     = 2,
        .unfold_n    = 4,
        .kh          = 2,
        .kw          = 2,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },
    // 4x4
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 1,
        .waves_q     = 2,
        .unfold_n    = 4,
        .kh          = 4,
        .kw          = 4,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },
    // 5x5
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 1,
        .waves_q     = 2,
        .unfold_n    = 4,
        .kh          = 5,
        .kw          = 5,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },

    // Dgrad K(96) divisible tiles (wave_k16=3): mirror the fprop K96 configs.
    //
    // Supports a dgrad output channel count (din) that is a multiple of 96 with exact
    // work instead of the padded K128 fallback.
    // 3x3
    {
        .wave_k16  = 3,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
    },
    // 2x2
    {
        .wave_k16  = 3,
        .wave_p    = 8,
        .kh        = 2,
        .kw        = 2,
        .direction = hipconv::Direction::Dgrad,
    },
    // 4x4
    {
        .wave_k16  = 3,
        .wave_p    = 7,
        .kh        = 4,
        .kw        = 4,
        .direction = hipconv::Direction::Dgrad,
    },
    // 5x5
    {
        .wave_k16  = 3,
        .wave_p    = 6,
        .kh        = 5,
        .kw        = 5,
        .direction = hipconv::Direction::Dgrad,
    },

    // Dgrad K(64) divisible tiles (wave_k16=2): mirror the fprop K64 configs.
    // 3x3, in 3 steps of C(64)
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
        .c64_step  = true,
    },
    // 2x2, in 2 steps of C(64)
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 2,
        .kw        = 2,
        .direction = hipconv::Direction::Dgrad,
        .c64_step  = true,
    },
    // 4x4, in 4 steps of C(64)
    {
        .wave_k16  = 2,
        .wave_p    = 7,
        .kh        = 4,
        .kw        = 4,
        .direction = hipconv::Direction::Dgrad,
        .c64_step  = true,
    },
    // 5x5, in 5 steps of C(64)
    {
        .wave_k16  = 2,
        .wave_p    = 6,
        .kh        = 5,
        .kw        = 5,
        .direction = hipconv::Direction::Dgrad,
        .c64_step  = true,
    },

    // Dgrad 3x1 divisible tiles (asymmetric filter): mirror the fprop 3x1 trio.
    //
    // The dgrad transpose flips taps per axis (kh over 3, kw over 1), so the compute
    // kernel is unchanged. K128
    {
        .wave_k16  = 4,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },
    // K96
    {
        .wave_k16  = 3,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },
    // K64
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },

    // Dgrad 3x1 large-tensor tiles: mirror the fprop 3x1 large_tensor trio.
    // K128
    {
        .wave_k16     = 4,
        .wave_p       = 8,
        .kh           = 3,
        .kw           = 1,
        .direction    = hipconv::Direction::Dgrad,
        .large_tensor = true,
    },
    // K96
    {
        .wave_k16     = 3,
        .wave_p       = 8,
        .kh           = 3,
        .kw           = 1,
        .direction    = hipconv::Direction::Dgrad,
        .large_tensor = true,
    },
    // K64
    {
        .wave_k16     = 2,
        .wave_p       = 8,
        .kh           = 3,
        .kw           = 1,
        .direction    = hipconv::Direction::Dgrad,
        .large_tensor = true,
    },

    // 3x1 dgrad K256 tile (waves_k=4): mirrors the fprop 3x1 K256 for dgrad.
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_q   = 1,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },
    // 3x1 dgrad K256 arbitrary-din fallback (k_divisible=false).
    {
        .waves_k     = 4,
        .wave_k16    = 4,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 1,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },

    // Dgrad arbitrary-din fallback (k_divisible=false).
    //
    // The dgrad analog of the fprop arbitrary-K fallback: accepts any din (padded),
    // matching only din no divisible dgrad variant covers.
    // 3x3
    {
        .wave_k16    = 4,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },
    // 2x2
    {
        .wave_k16    = 4,
        .wave_p      = 8,
        .kh          = 2,
        .kw          = 2,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },
    // 4x4 dgrad narrow arbitrary-din fallback (waves_k=4, wave_k16=2, k_divisible=false).
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 4,
        .kw          = 4,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },
    // 5x5 dgrad narrow arbitrary-din fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 5,
        .kw          = 5,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },
    // 3x1: mirrors the fprop 3x1 arbitrary-K fallback.
    {
        .wave_k16    = 4,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 1,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },
    // Dgrad small-dout fallback (single_c): supports dout in [2, 64].
    //
    // The dgrad analog of the fprop small-C config (dout is the reduction). dout
    // over-computed to 64 against zero weights; for completeness, not speed.
    // 3x3
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
        .single_c  = true,
        .c64_step  = true,
    },
    // 2x2
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 2,
        .kw        = 2,
        .direction = hipconv::Direction::Dgrad,
        .single_c  = true,
        .c64_step  = true,
    },
    // 4x4
    {
        .wave_k16  = 2,
        .wave_p    = 7,
        .kh        = 4,
        .kw        = 4,
        .direction = hipconv::Direction::Dgrad,
        .single_c  = true,
        .c64_step  = true,
    },
    // 5x5
    {
        .wave_k16  = 2,
        .wave_p    = 6,
        .kh        = 5,
        .kw        = 5,
        .direction = hipconv::Direction::Dgrad,
        .single_c  = true,
        .c64_step  = true,
    },
    // 3x1
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
        .single_c  = true,
    },
    // Dgrad small-dout arbitrary-din fallback (single_c, k_divisible=false).
    //
    // The dgrad analog of the fprop small-C arbitrary-output config: dout (reduction)
    // in [2, 64] with a din (output) count not a multiple of 64. One per filter.
    // 3x3
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },
    // 2x2
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 2,
        .kw          = 2,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },
    // 4x4
    {
        .wave_k16    = 2,
        .wave_p      = 7,
        .kh          = 4,
        .kw          = 4,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },
    // 5x5
    {
        .wave_k16    = 2,
        .wave_p      = 6,
        .kh          = 5,
        .kw          = 5,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },
    // 3x1
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 1,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
    },

    // ---- conv3d (kd > 1) ----
    //
    // Appended, not interleaved: the autoshard assigns config i to shard
    // i % num_shards, so appending leaves every 2D config's shard where it was.
    //
    // relative_throughput_conv3d ranks these separately from the 2D family. Every tile its
    // comparisons favour must exist here, or nothing ranks first on the shape. It has
    // no unfold comparison: an unfold tile packs consecutive images along its columns,
    // which under the depth fold are depth planes needing different tap offsets.

    // 3x3x3 K128: P(16) x Q(32) x K(128). Supports the c384/k384 VAE layers.
    {
        .wave_k16 = 4,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .kd       = 3,
    },

    // 3x3x3 K256 tile (waves_k=4): P(16) x Q(16) x K(256).
    {
        .waves_k  = 4,
        .wave_k16 = 4,
        .wave_p   = 8,
        .waves_q  = 1,
        .kh       = 3,
        .kw       = 3,
        .kd       = 3,
    },

    // 3x3x3 K256 arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 4,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .k_divisible = false,
    },

    // 3x3x3 K96.
    {
        .wave_k16 = 3,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .kd       = 3,
    },

    // 3x3x3 K64, in 3 steps of C(64).
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .kd       = 3,
        .c64_step = true,
    },

    // 3x3x3 K128 arbitrary-K fallback. Supports the c384/k32 VAE layer.
    {
        .wave_k16    = 4,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .k_divisible = false,
    },

    // 3x3x3 single_c K64: one C(64) iteration. Supports the c16/k384 VAE layer.
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .kd       = 3,
        .single_c = true,
        .c64_step = true,
    },

    // 3x3x3 single_c K64 arbitrary-K fallback.
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },

    // 3x3x3 K64 narrow: P(16) x Q(16) x K(64).
    //
    // The tile that keeps the machine busy when depth contributes no tiles. At
    // E == 1 the wider tiles can decompose a layer into far fewer tiles than
    // there are CUs; this one quarters the work each tile does. Its Q(16) also
    // divides output widths that are odd multiples of 16, which Q(32) does not.
    {
        .waves_k  = 4,
        .wave_k16 = 1,
        .wave_p   = 8,
        .waves_q  = 1,
        .kh       = 3,
        .kw       = 3,
        .kd       = 3,
    },

    // 3x3x3 K128 split-K: each tile's reduction shared by workgroups.
    //
    // Supports the K128 shapes whose unsplit tiles leave the machine in one underfilled
    // round. See docs/algorithms/direct/direct-3d.md.
    {
        .wave_k16 = 4,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .kd       = 3,
        .split_k  = true,
    },

    // ---- conv3d dgrad, stride 1 ----
    //
    // The fprop tiles above, run over dy with the dgrad-formatted weights (C<->K swap
    // and the tap flip in all three axes), as the 2D dgrad tiles are. Appended so every
    // earlier config keeps its shard.

    // 3x3x3 dgrad K128: P(16) x Q(32) x K(128). Supports the c384/k384 VAE layers.
    {
        .wave_k16  = 4,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .kd        = 3,
        .direction = hipconv::Direction::Dgrad,
    },

    // 3x3x3 dgrad K256 tile (waves_k=4): P(16) x Q(16) x K(256).
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_q   = 1,
        .kh        = 3,
        .kw        = 3,
        .kd        = 3,
        .direction = hipconv::Direction::Dgrad,
    },

    // 3x3x3 dgrad K256 arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 4,
        .wave_p      = 8,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },

    // 3x3x3 dgrad K96.
    {
        .wave_k16  = 3,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .kd        = 3,
        .direction = hipconv::Direction::Dgrad,
    },

    // 3x3x3 dgrad K64, in 3 steps of C(64).
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .kd        = 3,
        .direction = hipconv::Direction::Dgrad,
        .c64_step  = true,
    },

    // 3x3x3 dgrad K128 arbitrary-K fallback. Supports the c384/k32 VAE layer.
    {
        .wave_k16    = 4,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
    },

    // 3x3x3 dgrad single_c K64: one C(64) iteration. Supports the c16/k384 VAE layer.
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .kd        = 3,
        .direction = hipconv::Direction::Dgrad,
        .single_c  = true,
        .c64_step  = true,
    },

    // 3x3x3 dgrad single_c K64 arbitrary-K fallback.
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .c64_step    = true,
    },

    // 3x3x3 dgrad K64 narrow: P(16) x Q(16) x K(64).
    {
        .waves_k   = 4,
        .wave_k16  = 1,
        .wave_p    = 8,
        .waves_q   = 1,
        .kh        = 3,
        .kw        = 3,
        .kd        = 3,
        .direction = hipconv::Direction::Dgrad,
    },

    // 3x3x3 dgrad K128 split-K: each tile's reduction shared by workgroups.
    {
        .wave_k16  = 4,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .kd        = 3,
        .direction = hipconv::Direction::Dgrad,
        .split_k   = true,
    },

    // ---- conv2d, appended after conv3d for the same shard reason ----

    // 3x3 K64 narrow: P(16) x Q(16) x K(64), the conv2d copy of the 3x3x3 one.
    //
    // Supports the K128 shapes it runs in one round of the persistent grid, which K128
    // tiles leave underfilled: 101 us against K128's 159 at n1 c1152 k384 64x112.
    {
        .waves_k  = 4,
        .wave_k16 = 1,
        .wave_p   = 8,
        .waves_q  = 1,
        .kh       = 3,
        .kw       = 3,
    },

    // 3x3 K128 split-K: each tile's C/64 divided among workgroups.
    //
    // Supports the K128 shapes whose unsplit tiles leave the machine in one underfilled
    // round, as the 3x3x3 one does. See
    // docs/algorithms/direct/direct-l1-direct-flex-techniques.md.
    {
        .wave_k16 = 4,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .split_k  = true,
    },

    // 3x3 dgrad K128 split-K.
    {
        .wave_k16  = 4,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
        .split_k   = true,
    },

    // 3x3 K64 split-K, in 3 steps of C(64).
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .split_k  = true,
        .c64_step = true,
    },

    // 3x3 dgrad K64 split-K, in 3 steps of C(64).
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
        .split_k   = true,
        .c64_step  = true,
    },

    // 3x3x3 K64 split-K, in 3 steps of C(64).
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
        .kd       = 3,
        .split_k  = true,
        .c64_step = true,
    },

    // 3x3x3 dgrad K64 split-K, in 3 steps of C(64).
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .kd        = 3,
        .direction = hipconv::Direction::Dgrad,
        .split_k   = true,
        .c64_step  = true,
    },

    // 3x3 chunked unfold K256: P(8) x 4 slots x W(8) x K(256), a slot an 8-column chunk.
    //
    // See docs/algorithms/direct/direct-l1-chunked-unfold.md.
    {
        .waves_k       = 4,
        .wave_k16      = 4,
        .wave_p        = 8,
        .waves_p       = 1,
        .waves_q       = 2,
        .unfold_n      = 4,
        .kh            = 3,
        .kw            = 3,
        .unfold_chunks = true,
    },

    // 3x3 chunked unfold K128 (K32/wave).
    {
        .waves_k       = 4,
        .wave_k16      = 2,
        .wave_p        = 8,
        .waves_p       = 1,
        .waves_q       = 2,
        .unfold_n      = 4,
        .kh            = 3,
        .kw            = 3,
        .unfold_chunks = true,
    },

    // 3x3 chunked unfold K192 (K48/wave): 960 and 1920 output channels in whole blocks.
    {
        .waves_k       = 4,
        .wave_k16      = 3,
        .wave_p        = 8,
        .waves_p       = 1,
        .waves_q       = 2,
        .unfold_n      = 4,
        .kh            = 3,
        .kw            = 3,
        .unfold_chunks = true,
    },

    // 3x3 dgrad chunked unfold K256.
    {
        .waves_k       = 4,
        .wave_k16      = 4,
        .wave_p        = 8,
        .waves_p       = 1,
        .waves_q       = 2,
        .unfold_n      = 4,
        .kh            = 3,
        .kw            = 3,
        .direction     = hipconv::Direction::Dgrad,
        .unfold_chunks = true,
    },

    // 3x3 dgrad chunked unfold K128.
    {
        .waves_k       = 4,
        .wave_k16      = 2,
        .wave_p        = 8,
        .waves_p       = 1,
        .waves_q       = 2,
        .unfold_n      = 4,
        .kh            = 3,
        .kw            = 3,
        .direction     = hipconv::Direction::Dgrad,
        .unfold_chunks = true,
    },

    // 3x3 dgrad chunked unfold K192.
    {
        .waves_k       = 4,
        .wave_k16      = 3,
        .wave_p        = 8,
        .waves_p       = 1,
        .waves_q       = 2,
        .unfold_n      = 4,
        .kh            = 3,
        .kw            = 3,
        .direction     = hipconv::Direction::Dgrad,
        .unfold_chunks = true,
    },

    // 3x3 chunked unfold K256 split-K.
    {
        .waves_k       = 4,
        .wave_k16      = 4,
        .wave_p        = 8,
        .waves_p       = 1,
        .waves_q       = 2,
        .unfold_n      = 4,
        .kh            = 3,
        .kw            = 3,
        .split_k       = true,
        .unfold_chunks = true,
    },

    // 3x3 dgrad chunked unfold K256 split-K.
    {
        .waves_k       = 4,
        .wave_k16      = 4,
        .wave_p        = 8,
        .waves_p       = 1,
        .waves_q       = 2,
        .unfold_n      = 4,
        .kh            = 3,
        .kw            = 3,
        .direction     = hipconv::Direction::Dgrad,
        .split_k       = true,
        .unfold_chunks = true,
    },
    // Width-1 unfold tiles, which run 1×kw layers as kw×1 layers over an N·H × W × 1 × C input.
    //
    // Each of the unfold_n columns holds one image. See
    // docs/algorithms/details/direct/direct-l1-fold.md.

    // 3×1 fprop: K128 and K256, 16 and 32 images, and one C(64) step at K64.
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 2,
        .waves_q   = 1,
        .unfold_n  = 16,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Fprop,
    },
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 32,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Fprop,
    },
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_p   = 2,
        .waves_q   = 1,
        .unfold_n  = 16,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Fprop,
    },
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 32,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Fprop,
    },
    {
        .waves_k     = 2,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 2,
        .waves_q     = 2,
        .unfold_n    = 32,
        .kh          = 3,
        .kw          = 1,
        .direction   = hipconv::Direction::Fprop,
        .k_divisible = false,
        .single_c    = true,
    },

    // 5×1 fprop: K128, 32 images. K256 spills.
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 32,
        .kh        = 5,
        .kw        = 1,
        .direction = hipconv::Direction::Fprop,
    },

    // 3×1 dgrad: K128 and K256, 16 and 32 images, and one C(64) step at K64.
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 2,
        .waves_q   = 1,
        .unfold_n  = 16,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 32,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_p   = 2,
        .waves_q   = 1,
        .unfold_n  = 16,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },
    {
        .waves_k   = 4,
        .wave_k16  = 4,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 32,
        .kh        = 3,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },
    {
        .waves_k     = 2,
        .wave_k16    = 2,
        .wave_p      = 8,
        .waves_p     = 2,
        .waves_q     = 2,
        .unfold_n    = 32,
        .kh          = 3,
        .kw          = 1,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
    },

    // 5×1 dgrad: K128, 32 images. K256 spills.
    {
        .waves_k   = 4,
        .wave_k16  = 2,
        .wave_p    = 8,
        .waves_p   = 1,
        .waves_q   = 2,
        .unfold_n  = 32,
        .kh        = 5,
        .kw        = 1,
        .direction = hipconv::Direction::Dgrad,
    },

    // ---- conv3d tf32, appended after the conv2d tiles for the same shard reason ----
    //
    // The 3x3 tf32 tile over the depth fold, P(14) x Q(16): K128 in its four corners
    // (divisible or arbitrary output, single_c or not), K64 divisible and single_c in both
    // output corners, and split-K over K128 (divisible output) and K64 (divisible or arbitrary
    // output), each for fprop and dgrad. Each (tap, C(64)) step stages one input plane, so the
    // doubled LDS tile is the 2D one, block_h 16 x block_w 18 = 288.
    // relative_throughput_conv3d_tf32 ranks these. See docs/algorithms/direct/direct-3d.md.

    // 3x3x3 tf32 K128: P(14) x Q(16) x K(128).
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .elem_bytes = 4,
    },

    // 3x3x3 tf32 K128 arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .k_divisible = false,
        .elem_bytes  = 4,
    },

    // 3x3x3 tf32 single_c K128: one C(64) iteration.
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .single_c   = true,
        .elem_bytes = 4,
    },

    // 3x3x3 tf32 single_c K128 arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },

    // 3x3x3 tf32 K64: P(14) x Q(16) x K(64).
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .elem_bytes = 4,
    },

    // 3x3x3 tf32 dgrad K128: P(14) x Q(16) x K(128).
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },

    // 3x3x3 tf32 dgrad K128 arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .elem_bytes  = 4,
    },

    // 3x3x3 tf32 dgrad single_c K128.
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .single_c   = true,
        .elem_bytes = 4,
    },

    // 3x3x3 tf32 dgrad single_c K128 arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 2,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },

    // 3x3x3 tf32 dgrad K64.
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
    },

    // 3x3x3 tf32 K128 split-K: each tile's reduction shared by workgroups.
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .elem_bytes = 4,
        .split_k    = true,
    },

    // 3x3x3 tf32 dgrad K128 split-K.
    {
        .waves_k    = 4,
        .wave_k16   = 2,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
        .split_k    = true,
    },

    // 3x3x3 tf32 K64 split-K.
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .elem_bytes = 4,
        .split_k    = true,
    },

    // 3x3x3 tf32 dgrad K64 split-K.
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .elem_bytes = 4,
        .split_k    = true,
    },

    // 3x3x3 tf32 K64 split-K arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 1,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .k_divisible = false,
        .elem_bytes  = 4,
        .split_k     = true,
    },

    // 3x3x3 tf32 dgrad K64 split-K arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 1,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .elem_bytes  = 4,
        .split_k     = true,
    },

    // 3x3x3 tf32 single_c K64: one C(64) iteration.
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .single_c   = true,
        .elem_bytes = 4,
    },

    // 3x3x3 tf32 single_c K64 arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 1,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },

    // 3x3x3 tf32 dgrad single_c K64.
    {
        .waves_k    = 4,
        .wave_k16   = 1,
        .wave_p     = 7,
        .waves_q    = 1,
        .kh         = 3,
        .kw         = 3,
        .kd         = 3,
        .direction  = hipconv::Direction::Dgrad,
        .single_c   = true,
        .elem_bytes = 4,
    },

    // 3x3x3 tf32 dgrad single_c K64 arbitrary-K fallback.
    {
        .waves_k     = 4,
        .wave_k16    = 1,
        .wave_p      = 7,
        .waves_q     = 1,
        .kh          = 3,
        .kw          = 3,
        .kd          = 3,
        .direction   = hipconv::Direction::Dgrad,
        .k_divisible = false,
        .single_c    = true,
        .elem_bytes  = 4,
    },
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

// Index of the config matching the given key, or -1 if none exists.
//
// Lets callers look up an index by defining attributes instead of a hardcoded
// position that desyncs on reorder. The key is a config's full unique identity.
constexpr int config_index(int wave_k16,
                           int kh,
                           int kw,
                           bool k_divisible             = true,
                           bool single_c                = false,
                           hipconv::Direction direction = hipconv::Direction::Fprop,
                           int unfold_n                 = 1,
                           int waves_k                  = 2,
                           bool large_tensor            = false,
                           int elem_bytes               = 2,
                           int kd                       = 1,
                           bool split_k                 = false,
                           bool unfold_chunks           = false,
                           bool c64_step                = false)
{
    for(int i = 0; i < num_configs; ++i)
    {
        if(configs[i].wave_k16 == wave_k16 && configs[i].kh == kh && configs[i].kw == kw &&
           configs[i].k_divisible == k_divisible && configs[i].single_c == single_c &&
           configs[i].direction == direction && configs[i].unfold_n == unfold_n &&
           configs[i].waves_k == waves_k && configs[i].large_tensor == large_tensor &&
           configs[i].elem_bytes == elem_bytes && configs[i].kd == kd &&
           configs[i].split_k == split_k && configs[i].unfold_chunks == unfold_chunks &&
           configs[i].c64_step == c64_step)
        {
            return i;
        }
    }
    return -1;
}

constexpr bool config_index_keys_are_unique()
{
    for(int i = 0; i < num_configs; ++i)
    {
        const Config& c = configs[i];
        if(config_index(c.wave_k16,
                        c.kh,
                        c.kw,
                        c.k_divisible,
                        c.single_c,
                        c.direction,
                        c.unfold_n,
                        c.waves_k,
                        c.large_tensor,
                        c.elem_bytes,
                        c.kd,
                        c.split_k,
                        c.unfold_chunks,
                        c.c64_step) != i)
        {
            return false;
        }
    }
    return true;
}
static_assert(config_index_keys_are_unique(),
              "two configs share a config_index key; add the distinguishing field to it");

} // namespace hipconv::cdna4::direct_l1
