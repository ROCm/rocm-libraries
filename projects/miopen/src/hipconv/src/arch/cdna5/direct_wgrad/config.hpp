#pragma once

#include "hipconv/conv_params.hpp"

namespace hipconv::cdna5::direct_wgrad
{

// Kernel configuration parameters.
struct Config
{
    // Assign default values to avoid -Wmissing-designated-field-initializers warnings.
    int kh           = 0;
    int kw           = 0;
    int wmma_size_c  = 16;
    int wmma_size_k  = 16;
    int wmma_size_q  = 32;
    int tile_size_c  = 64;
    int tile_size_k  = 128;
    int tiles_c      = 2;
    int tiles_k      = 4;
    int ring_size_p  = 16;
    int chunk_size_p = 4;
    constexpr auto tiles() const { return tiles_c * tiles_k; }
    constexpr auto reg_stride_c() const { return tile_size_c / tiles_c; }
    constexpr auto reg_stride_k() const { return tile_size_k / tiles_k; }
    constexpr auto reg_tiles_c() const { return reg_stride_c() / wmma_size_c; }
    constexpr auto reg_tiles_k() const { return reg_stride_k() / wmma_size_k; }
};

} // namespace hipconv::cdna5::direct_wgrad
