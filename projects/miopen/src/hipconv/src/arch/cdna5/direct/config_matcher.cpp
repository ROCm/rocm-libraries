#include "config_matcher.hpp"
#include "direction_field.h"

#include <limits>

namespace hipconv::cdna5::direct
{

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("tile_size_k", cfg.tile_size_k);
    int_field("tile_size_n", cfg.tile_size_n);
    int_field("tile_size_h", cfg.tile_size_h);
    int_field("tile_size_w", cfg.tile_size_w);
    bool_field("aligned", cfg.aligned);
    // Defaulted, so the 2-byte configs' descriptors are unchanged by tf32's arrival.
    int_field("elem_bytes", cfg.elem_bytes, 2);
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    direction_field(*this, cfg.direction, Direction::Fprop);
    int_field("tile_size_c", cfg.tile_size_c, /*default=*/128);
    int_field("wmma_size_j", cfg.wmma_size_j, /*default=*/16);
    int_field("wmma_size_k", cfg.wmma_size_k, /*default=*/16);
    int_field("wmma_size_c", cfg.wmma_size_c, /*default=*/32);
    int_field("reg_tiles_c", cfg.reg_tiles_c, /*default=*/2);
    int_field("tile_size_c_pad_amount", cfg.tile_size_c_pad_amount, /*default=*/4);
    int_field("tile_size_k_pad_amount", cfg.tile_size_k_pad_amount, /*default=*/4);
    int_field("tile_size_k_pad_out", cfg.tile_size_k_pad_out, /*default=*/4);
    int_field("tiles_j", cfg.tiles_j, /*default=*/4);
    int_field("tiles_k", cfg.tiles_k, /*default=*/2);
    int_field("max_px", cfg.max_px, /*default=*/std::numeric_limits<int>::max());
}

} // namespace hipconv::cdna5::direct
