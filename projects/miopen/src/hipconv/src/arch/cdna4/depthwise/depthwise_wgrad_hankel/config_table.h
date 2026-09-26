#pragma once

#include "hipconv/conv_params.hpp"

// The compiled set of depthwise Hankel wgrad configs.
// Free of device code, so reading `num_configs` does not pull in kernel.h.

namespace hipconv::cdna4::depthwise_wgrad_hankel
{
using namespace hipconv;

constexpr int WAVE_SIZE = 64;

// V_MFMA_F32_4X4X4_F16 geometry: TAPS is the M and N extent, MFMA_K the contraction, and
// the 16 batches are channels. WAVES_C is 4 as ds_read_tr16_b64 transposes 64 columns.
constexpr int TAPS          = 4;
constexpr int MFMA_K        = 4;
constexpr int CHAN_PER_WAVE = 16;
constexpr int WAVES_C       = 4;
constexpr int BLOCK_C       = WAVES_C * CHAN_PER_WAVE;
constexpr int BLOCK_C8      = BLOCK_C / 8;

// The output tile a workgroup owns, how deep it stages, and which staging path it takes.
// q_tiles counts output columns in units of the 4-wide MFMA contraction, block_q() below.
struct Config
{
    int q_tiles         = 8;
    int kh              = 3;
    int kw              = 3;
    int rows_per_chunk  = 32;
    int stage_depth     = 2;
    int stride          = 1;
    bool narrow_c       = false;
    Direction direction = Direction::Wgrad;

    constexpr int block_q() const { return q_tiles * MFMA_K; }
    constexpr int threads() const { return WAVES_C * WAVE_SIZE; }
    // Channel-major row stride, padded away from the 64-byte LDS bank period.
    constexpr int main_stride() const { return block_q() + 4; }
};

// Widest q tile and tallest row chunk first; selection takes the first survivor.
//
// One row per config, past the column limit, so the families line up by eye.
// clang-format off
constexpr Config configs[] = {
    {.q_tiles = 8, .rows_per_chunk = 32},
    {.q_tiles = 8, .rows_per_chunk = 16},
    {.q_tiles = 8, .rows_per_chunk = 8},
    {.q_tiles = 4, .rows_per_chunk = 32},
    {.q_tiles = 4, .rows_per_chunk = 8},
    {.q_tiles = 8, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 8, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 8, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 4, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 4, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 4, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 32},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 16},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 8},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 16},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 32},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 16},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 8},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 16},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 32},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 16},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 8},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 16},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 32},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 16},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 8},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 16},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stride = 2},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stride = 2},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stride = 2},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stride = 2},
    // Narrow-C families, at the chunk-ladder ends only; each entry is an instantiation.
    // A narrow row lives in registers until it is written out, hence stage_depth 1.
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .narrow_c = true},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .narrow_c = true},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .narrow_c = true},
};
// clang-format on

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::depthwise_wgrad_hankel
