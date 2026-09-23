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
    // Serves the K256 window for K not a multiple of 256 (padded, straddling stores
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
    // Serves low-resolution maps (output width <= 8) by packing 4 batch images along
    // the columns. The 4-way K split halves per-wave weight traffic for more input
    // LDS reads, the right trade when the tiny input tile makes weights dominate.
    // Cedes K % 256 == 0 to the K256 unfold sibling below.
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
    // tile (proven spill-free). Serves K % 256 == 0; the rest stay on K128.
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
    // H32xW32). 5x5 has no K256 tile, so this serves every 5x5 fprop count.
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
    // Serves the W<=8 unfold window when the output channel count is not a multiple of
    // 128 (e.g. 96, 160, 224); the divisible unfold tiles above require output %
    // block_k == 0. The writer pads output to 128 and guards the straddling wave.
    // One per filter; 3x3 needs no K256 variant (%256 is a subset of %128, ceded to
    // the K256 divisible unfold).
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
    // 3x3
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 3,
        .kw       = 3,
    },
    // 2x2
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 2,
        .kw       = 2,
    },
    // 4x4
    {
        .wave_k16 = 2,
        .wave_p   = 7,
        .kh       = 4,
        .kw       = 4,
    },
    // 5x5
    {
        .wave_k16 = 2,
        .wave_p   = 6,
        .kh       = 5,
        .kw       = 5,
    },

    // 3x1 divisible tiles (asymmetric filter): K128/K96/K64, wave_p=8.
    //
    // The one-wide filter has no W halo (block_w == block_q), so its input tile is
    // strictly smaller than 3x3's and wave_p=8 is safe. num_load_steps = kw*2 = 2.
    // All the specialist cedes in is_valid_config (unfold, K256) gate on kw == kh, so
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
    // size gate makes these claim only shapes whose per-image span overflows int32, so
    // they never shadow the baseline tiles. The K256 tiles need no large_tensor variant:
    // is_valid_config exempts large_tensor from the K256 cede, so these keep serving the
    // shape even when the K256 overcompute comparison would otherwise prefer K256.
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
    // When the K256 selection claims the shape, a K not a multiple of 256 would have no
    // configuration without this padded tile (the divisible K256 sibling takes only
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
    // Small-C fallback (K64, single_c): serves C in [2, 64].
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
    },
    // 2x2
    {
        .wave_k16 = 2,
        .wave_p   = 8,
        .kh       = 2,
        .kw       = 2,
        .single_c = true,
    },
    // 4x4
    {
        .wave_k16 = 2,
        .wave_p   = 7,
        .kh       = 4,
        .kw       = 4,
        .single_c = true,
    },
    // 5x5
    {
        .wave_k16 = 2,
        .wave_p   = 6,
        .kh       = 5,
        .kw       = 5,
        .single_c = true,
    },
    // Small-C arbitrary-output fallback (K64, single_c, k_divisible=false).
    //
    // Serves C in [2, 64] combined with an output count not a multiple of 64 (e.g.
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
    },
    // 2x2
    {
        .wave_k16    = 2,
        .wave_p      = 8,
        .kh          = 2,
        .kw          = 2,
        .k_divisible = false,
        .single_c    = true,
    },
    // 4x4
    {
        .wave_k16    = 2,
        .wave_p      = 7,
        .kh          = 4,
        .kw          = 4,
        .k_divisible = false,
        .single_c    = true,
    },
    // 5x5
    {
        .wave_k16    = 2,
        .wave_p      = 6,
        .kh          = 5,
        .kw          = 5,
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
    // Cedes din % 256 == 0 to the K256 unfold sibling below (same split as fprop).
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
    // The deeper-K sibling of the K128 unfold above; serves din % 256 == 0 that the
    // K128 unfold cedes. Without it those counts fell through the direction-agnostic
    // cede in is_valid_config and no tile served the W<=8 dgrad window.
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
    // Mirrors the fprop arbitrary unfold for backward-data: serves the W<=8 window
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
    // Serve a dgrad output channel count (din) that is a multiple of 96 with exact
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
    // 3x3
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 3,
        .kw        = 3,
        .direction = hipconv::Direction::Dgrad,
    },
    // 2x2
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 2,
        .kw        = 2,
        .direction = hipconv::Direction::Dgrad,
    },
    // 4x4
    {
        .wave_k16  = 2,
        .wave_p    = 7,
        .kh        = 4,
        .kw        = 4,
        .direction = hipconv::Direction::Dgrad,
    },
    // 5x5
    {
        .wave_k16  = 2,
        .wave_p    = 6,
        .kh        = 5,
        .kw        = 5,
        .direction = hipconv::Direction::Dgrad,
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
    // Dgrad small-dout fallback (single_c): serves dout in [2, 64].
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
    },
    // 2x2
    {
        .wave_k16  = 2,
        .wave_p    = 8,
        .kh        = 2,
        .kw        = 2,
        .direction = hipconv::Direction::Dgrad,
        .single_c  = true,
    },
    // 4x4
    {
        .wave_k16  = 2,
        .wave_p    = 7,
        .kh        = 4,
        .kw        = 4,
        .direction = hipconv::Direction::Dgrad,
        .single_c  = true,
    },
    // 5x5
    {
        .wave_k16  = 2,
        .wave_p    = 6,
        .kh        = 5,
        .kw        = 5,
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
                           bool large_tensor            = false)
{
    for(int i = 0; i < num_configs; ++i)
    {
        if(configs[i].wave_k16 == wave_k16 && configs[i].kh == kh && configs[i].kw == kw &&
           configs[i].k_divisible == k_divisible && configs[i].single_c == single_c &&
           configs[i].direction == direction && configs[i].unfold_n == unfold_n &&
           configs[i].waves_k == waves_k && configs[i].large_tensor == large_tensor)
        {
            return i;
        }
    }
    return -1;
}

} // namespace hipconv::cdna4::direct_l1
