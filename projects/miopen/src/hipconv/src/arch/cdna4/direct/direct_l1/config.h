#pragma once

#include "hipconv/conv2d_params.hpp"

namespace hipconv::cdna4::direct_l1
{
// 64 threads per wave.
static constexpr int WAVE_SIZE = 64;

// LDS capacity per CU on MI355X (gfx950): the input tile + staging pad must fit.
static constexpr int LDS_BYTES_PER_CU = 160 * 1024;


// Configuration for the direct convolution kernel.
struct Config
{
    // Waves along the workgroup's K axis; the workgroup spans waves_k * wave_k().
    //
    // waves_k=2 is the standard split; waves_k=4 packs K(256) for small maps,
    // reusing one input tile across 4x the channels.
    int waves_k = 2;

    // Number of output channels / 16 per wave.
    int wave_k16;

    // Number of output columns / 16 per wave.
    int wave_q16 = 1;

    // Number of output rows per wave.
    int wave_p = 8;

    // Number of waves tiled along the p-axis.
    int waves_p = 2;

    // Number of waves tiled along the q-axis.
    int waves_q = 2;

    // Batch images packed along the Q output-column axis (1 = standard tile).
    //
    // When > 1, block_q() columns become unfold_n images of w_unfold() columns each.
    // The MFMA shape, accumulators, and weights path are unchanged; only the input
    // load, LDS column axis, output write, and grid N decode re-address.
    int unfold_n = 1;

    // Height of the convolution filter.
    int kh = 3;

    // Width of the convolution filter.
    int kw = 3;

    hipconv::Direction direction = hipconv::Direction::Fprop;

    // When true, K is an exact multiple of block_k: the writer takes the fast path.
    //
    // When false (arbitrary-K, K padded in the weights), the writer guards every
    // channel store. A separate config so the divisible ones omit the guard (K128 spill).
    bool k_divisible = true;

    // When true, C fits one C(64) iteration (the lone peel is both first and last).
    //
    // When false the loop needs C64 >= 2. A separate config so the C64 >= 2 configs
    // carry no single-C branch.
    bool single_c = false;

    // When true, address global memory with a per-tile row-origin fold (large tensors).
    //
    // The baseline folds only the image origin into the 64-bit base, capping one
    // image at INT32_MAX. This additionally folds the tile's output-row origin, so
    // the 32-bit buffer offset spans only the tile's rows (block_size_h input rows /
    // block_p output rows), letting a single image exceed 2 GiB. Gated so baseline
    // configs keep byte-for-byte identical addressing (the fold risks a K128 spill).
    bool large_tensor = false;

    constexpr int wave_k() const { return wave_k16 * 16; }

    constexpr int block_k() const { return waves_k * wave_k(); }
    constexpr int block_p() const { return waves_p * wave_p; }
    constexpr int block_q() const { return waves_q * wave_q16 * 16; }

    // Per-image output width packed along the column axis (= block_q() if unfold_n==1).
    constexpr int w_unfold() const { return block_q() / unfold_n; }

    // Images packed into one q-wave's 16-lane MFMA tile.
    //
    // Exact and >= 1 only for a well-formed unfold config; WaveComputeIndex asserts
    // that invariant, so a malformed one fails the build rather than miscomputing.
    constexpr int images_per_qwave() const { return wave_q16 * 16 / w_unfold(); }

    constexpr int block_h() const { return block_p() + kh - 1; }
    constexpr int block_w() const { return block_q() + kw - 1; }

    constexpr int num_waves() const { return waves_p * waves_q * waves_k; }
    constexpr int num_threads() const { return WAVE_SIZE * num_waves(); }
};

} // namespace hipconv::cdna4::direct_l1
