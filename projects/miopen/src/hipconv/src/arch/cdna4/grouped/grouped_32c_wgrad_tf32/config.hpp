#pragma once

namespace hipconv::cdna4::grouped_32c_wgrad_tf32
{

constexpr int WAVE_SIZE       = 64;
constexpr int WAVES_PER_GROUP = 4;
constexpr int GROUP_SIZE      = 32;

struct Config
{
    int groups_per_wg;
    int wave_q32 = 1;
    int wave_n   = 1;
    int kh       = 3;
    int kw       = 3;

    // Stride 2 runs as stride 1 over delta upsampled 2x (zeros between samples).
    int stride = 1;

    constexpr int waves_per_wg() const { return WAVES_PER_GROUP * groups_per_wg; }
    constexpr int block_c() const { return GROUP_SIZE * groups_per_wg; }
    constexpr int block_q() const { return wave_q32 * 32; }
    constexpr int block_size() const { return waves_per_wg() * WAVE_SIZE; }
};

} // namespace hipconv::cdna4::grouped_32c_wgrad_tf32
