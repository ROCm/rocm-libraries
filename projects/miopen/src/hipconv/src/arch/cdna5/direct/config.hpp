#pragma once

#include "hipconv/conv2d_params.hpp"
#include <limits>

namespace hipconv::cdna5::direct
{

// Kernel configuration parameters.
struct Config
{
    // Assign default values to avoid -Wmissing-designated-field-initializers warnings.
    hipconv::Direction direction = hipconv::Direction::Fprop;
    int kh                       = 0;
    int kw                       = 0;
    int tile_size_h              = 16;
    int tile_size_n              = 1;
    int tile_size_w              = 16;
    int tile_size_k              = 256;
    int tile_size_c              = 128;
    int wmma_size_j              = 16;
    int wmma_size_k              = 16;
    int wmma_size_c              = 32;
    int reg_tiles_c              = 2; // Must ensure tile_size_c % (wmma_size_c * reg_tiles_c) == 0
    int tile_size_c_pad_amount   = 4; // In dwords
    int tile_size_k_pad_amount   = 4; // In dwords; is added tiles_k times, so 8 dwords in total
    int tile_size_k_pad_out      = 4; // In dwords; used for staging output through LDS
    int tiles_j                  = 4;
    int tiles_k                  = 2;
    int max_px = std::numeric_limits<int>::max(); // Maximum allowed padding in x direction; INT_MAX
                                                  // means unlimited
    bool aligned = true; // True if fastest varying output mode is divible by 16 / sizeof(T)
    constexpr auto tiles() const { return tiles_j * tiles_k; }
};

} // namespace hipconv::cdna5::direct
