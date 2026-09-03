#pragma once

#include "hipconv/conv2d_params.hpp"

namespace hipconv::cdna4::direct
{

// Kernel configuration parameters.
struct Config
{
    // Assign default values to avoid -Wmissing-designated-field-initializers warnings.
    hipconv::Direction direction = hipconv::Direction::Fprop;
    hipconv::DataType type       = hipconv::DataType::fp16;
    int kh                       = 0;
    int kw                       = 0;
    int tile_size_k;
    int tile_size_c;
    int tile_size_n = 1;
    int tile_size_h = 16;
    int tile_size_w = 16;
    int tiles_j     = 2; // Note: the mode "j" is the product of modes n, h, and w
    int tiles_k     = 4;
    int wmma_size_j = 16;
    int wmma_size_k = 16;
    int wmma_size_c = 32;
    int k_pad       = 4; // Padding in DWORD, for output
    int k_parts     = 2; // Number of parts to split weight tensor into

    constexpr int tiles() const { return tiles_j * tiles_k; }
};

} // namespace hipconv::cdna4::direct
