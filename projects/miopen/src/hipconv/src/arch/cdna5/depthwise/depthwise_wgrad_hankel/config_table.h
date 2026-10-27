#pragma once

#include "hipconv/conv_params.hpp"

#include <array>

namespace hipconv::cdna5::depthwise_wgrad_hankel
{

using namespace hipconv;

constexpr int WAVE_SIZE = 32;
constexpr int LDS_BYTES = 160 * 1024;

// Images a block folds into one accumulator. wgrad sums over the batch, so this
// is free arithmetically; what it buys is a longer row loop and fewer partials,
// against a grid divided by the same number. Measured at a fixed pixel count,
// c512: k3 wants one past h 7 (h4 355 -> 522 at four, h14 1943 -> 1102), k7 wants
// two wherever it was measured (h14 998 -> 1023, h56 1039 -> 1112, and 702 at four).
constexpr std::array IMAGES_PER_BLOCK{1, 2, 4};

// A tile is (q columns, channels). The channels fix the wave count and the staged
// tile both, and times the element size they are the row the engine issues.
struct Tile
{
    int q;
    int channels;
};
constexpr std::array TILES{Tile{32, 64},
                           Tile{32, 128},
                           Tile{32, 32},
                           Tile{32, 4},
                           Tile{16, 32},
                           Tile{16, 64},
                           Tile{16, 128}};
constexpr int MAX_COMPUTE_WAVES = 16;

// One rung: a filter size, a stride, a tile, and what the two derive from them.
struct Config
{
    int kh;
    int kw;
    int stride;
    int waves_per_wg       = 8;
    int q_cols             = 32;
    int prefetch_depth     = 4;
    int min_rows_per_chunk = 4;
    Direction direction    = Direction::Wgrad;

    int wmmas_per_wave = 1;

    int n_per_block = 1;

    // Channels one multiply covers. The WMMA's tile is 16x16 and a channel takes
    // kw of its rows and kh of its columns, so this is what fits in one.
    constexpr int channels_per_wmma() const
    {
        const int span = kh > kw ? kh : kw;
        return span <= 4 ? 4 : span <= 8 ? 2 : 1;
    }

    // Channels a wave carries: one multiply's worth, once per multiply it issues.
    constexpr int channels_per_wave() const { return channels_per_wmma() * wmmas_per_wave; }

    // Every wave holds channels, the two that also drive an engine included.
    constexpr int wg_channels() const { return waves_per_wg * channels_per_wave(); }

    // Channels the block stages. tile_ok turns away any tile whose channels the
    // block's waves cannot cover in one pass, so this is what they cover.
    constexpr int staged_channels() const { return wg_channels(); }

    // Whether A is built from the transposing read's own registers rather than
    // through LDS, which needs the tap shift to fit what the gather can name.
    constexpr bool a_from_tr16() const
    {
        return (stride == 1 || q_cols == 16) && channels_per_wave() <= 16 && kw <= 11;
    }

    // Whether B is carried in registers across steps. The delta geometry reads
    // this too, the window's depth being what holding B takes off it.
    constexpr bool b_in_reg() const { return a_from_tr16(); }

    // Whether every step fetches a new delta row, which lets the pool be no deeper
    // than the window. At stride 2 two steps share a row.
    constexpr bool b_rotates() const { return stride == 1; }

    constexpr int q_per_wave() const { return q_cols; }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
};

// Channels one multiply covers, as a free function so tile_ok can weigh a tile
// against the filter before a Config exists to ask.
constexpr int channels_per_wmma_for(int k)
{
    return k <= 4 ? 4 : k <= 8 ? 2 : 1;
}

// Multiplies a wave issues a step. Each carries a set of fragments, so registers
// go up with this and little else does; past kw 5 four of them spill.
constexpr int wmmas_for(int k)
{
    return k <= 7 ? 4 : 2;
}

// A sixteen-wave block puts four waves on a SIMD and caps each at 256 VGPRs. Past
// kw 5 a wave carrying four multiplies wants more, so those rungs stop at eight.
constexpr int max_waves_for(int k)
{
    return k > 5 && wmmas_for(k) == 4 ? 8 : MAX_COMPUTE_WAVES;
}

// Whether a rung is built at all. Only those that hold B in registers are; a
// shape matching none of them leaves the table empty and reports no-match.
constexpr bool tile_ok(int k, int stride, Tile t)
{
    const int channels_per_wave = channels_per_wmma_for(k) * wmmas_for(k);
    const int waves             = t.channels / channels_per_wave;
    if(t.channels % channels_per_wave != 0 || waves < 2 || waves > max_waves_for(k))
        return false;
    return (stride == 1 && (k <= 7 || t.q == 32)) || (stride == 2 && t.q == 16);
}

constexpr std::array FILTERS{3, 5, 7, 9, 11};
constexpr std::array STRIDES{1, 2};

constexpr int count_configs()
{
    int n = 0;
    for(int k : FILTERS)
        for(int stride : STRIDES)
            for(Tile t : TILES)
                if(tile_ok(k, stride, t))
                    n += static_cast<int>(IMAGES_PER_BLOCK.size());
    return n;
}

// Every filter, stride and tile the predicate lets through, with the wave count
// the tile's channels imply, once per batch fold. One image a block comes first,
// so a shape whose score cannot separate the folds keeps the unfolded one.
constexpr auto make_configs()
{
    std::array<Config, count_configs()> result{};
    int i = 0;
    for(int k : FILTERS)
        for(int stride : STRIDES)
            for(Tile t : TILES)
            {
                if(!tile_ok(k, stride, t))
                    continue;
                for(int npb : IMAGES_PER_BLOCK)
                {
                    Config c{
                        .kh = k, .kw = k, .stride = stride, .min_rows_per_chunk = k == 3 ? 8 : 4};
                    c.q_cols         = t.q;
                    c.wmmas_per_wave = wmmas_for(k);
                    c.n_per_block    = npb;
                    const int want   = t.channels / c.channels_per_wave();
                    c.waves_per_wg   = want < max_waves_for(k) ? want : max_waves_for(k);
                    result[i++]      = c;
                }
            }
    return result;
}

constexpr auto configs    = make_configs();
constexpr int num_configs = static_cast<int>(configs.size());

} // namespace hipconv::cdna5::depthwise_wgrad_hankel
