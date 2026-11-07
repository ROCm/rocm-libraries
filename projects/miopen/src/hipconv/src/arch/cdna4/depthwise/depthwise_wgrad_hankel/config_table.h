#pragma once

#include "hipconv/conv_params.hpp"

#include "../filter_buckets.h"

#include <span>

#include <array>
#include <cstddef>

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
    int q_tiles        = 8;
    int kh             = 3;
    int kw             = 3;
    int rows_per_chunk = 32;
    int stage_depth    = 2;
    int stride         = 1;
    // Channels a lane moves per access, which also names the staging path: 8 is the wide
    // path's uint4 and anything narrower goes register-to-LDS. Channels are contiguous
    // inside a pixel, so an access fits exactly when the count divides C.
    int chan_vec        = 8;
    int elem_bytes      = 2;
    Direction direction = Direction::Wgrad;

    constexpr int block_q() const { return q_tiles * MFMA_K; }
    constexpr int threads() const { return WAVES_C * WAVE_SIZE; }
    // Channel-major row stride, padded away from the 64-byte LDS bank period.
    constexpr int main_stride() const { return block_q() + 4; }
    constexpr bool is_tf32() const { return elem_bytes == 4; }
};

// Widest q tile and tallest row chunk first; selection takes the first survivor.
//
// One row per config, past the column limit, so the families line up by eye.
// clang-format off
constexpr Config square_configs[] = {
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
    // Narrow-C families, at the chunk-ladder ends only. stage_depth 1 leads because the
    // depth-2 buffer spills to scratch until the row loop is unrolled by DEPTH, and
    // chan_vec 1 leads to avoid full-grid regressions; wider accesses remain tuning options.
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 2},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 2, .chan_vec = 2},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 2, .stride = 2, .chan_vec = 1},
    // tf32 families, all stage_depth 1 (depth 2 spills): chan_vec 4 plus a chan_vec 1 twin for
    // C % 4 != 0. q4/32 leads, q_tiles 8 being short of residency, except at stride 1 kh 3 and
    // 7 and for stride 2 kh 9's chan_vec 4 entries, which keep the wide order.
    {.q_tiles = 8, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 5, .kw = 5, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 7, .kw = 7, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 9, .kw = 9, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 32, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 8, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 16, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 4, .elem_bytes = 4},
    {.q_tiles = 4, .kh = 11, .kw = 11, .rows_per_chunk = 8, .stage_depth = 1, .stride = 2, .chan_vec = 1, .elem_bytes = 4},
};
// clang-format on

using namespace depthwise_filters;

// Per 1-D filter, at stride 1 and 2: the 16-bit wide ladder, a narrow-C fallback and tf32.
// Rows 32 serves tall images and 8 any image; fallback sizes keep only q_tiles 4 at rows 8.
template <typename Emit>
constexpr void enumerate_bucket_configs(Emit&& emit)
{
    const auto family = [&](FilterShape f, bool fallback) {
        const std::array<int, 2> rows_set = {fallback ? 8 : 32, 8};
        const int n_rows                  = fallback ? 1 : 2;
        for(int st : {1, 2})
        {
            // A stride-1 delta ring holds kh rows: past kh 23 q_tiles 8 outgrows LDS.
            const bool tall = f.kh > 23 && st == 1;
            for(int q : {8, 4})
                for(int rows : std::span(rows_set.data(), n_rows))
                    if(q == 4 || !(tall || fallback))
                        emit(Config{.q_tiles        = q,
                                    .kh             = f.kh,
                                    .kw             = f.kw,
                                    .rows_per_chunk = rows,
                                    .stride         = st});
            // Scalar channel paths (C % 8 or C % 4 left over) take rows 8 only.
            emit(Config{.q_tiles        = 4,
                        .kh             = f.kh,
                        .kw             = f.kw,
                        .rows_per_chunk = 8,
                        .stage_depth    = 1,
                        .stride         = st,
                        .chan_vec       = 1});
            if(st == 1 && f.kh > tf32_max_col_kh_wgrad)
                continue;
            for(int cv : {4, 1})
                for(int rows : cv == 1 ? std::span(rows_set.data() + 1, 1)
                                       : std::span(rows_set.data(), n_rows))
                    emit(Config{.q_tiles        = 4,
                                .kh             = f.kh,
                                .kw             = f.kw,
                                .rows_per_chunk = rows,
                                .stage_depth    = 1,
                                .stride         = st,
                                .chan_vec       = cv,
                                .elem_bytes     = 4});
        }
    };
    for(const FilterShape& f : filter_shapes)
        if(is_one_d(f))
            family(f, is_fallback(f));
}

constexpr std::size_t num_bucket_configs()
{
    std::size_t n = 0;
    enumerate_bucket_configs([&](const Config&) { ++n; });
    return n;
}

constexpr auto configs = [] {
    constexpr std::size_t n_square = sizeof(square_configs) / sizeof(square_configs[0]);
    std::array<Config, n_square + num_bucket_configs()> all{};
    std::size_t i = 0;
    for(const Config& c : square_configs)
        all[i++] = c;
    enumerate_bucket_configs([&](const Config& c) { all[i++] = c; });
    return all;
}();

constexpr int num_configs = static_cast<int>(configs.size());

} // namespace hipconv::cdna4::depthwise_wgrad_hankel
