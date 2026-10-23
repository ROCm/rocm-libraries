#include "config_matcher.hpp"
#include "data_type_field.h"
#include "direction_field.h"

namespace hipconv::cdna4::direct
{

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("tile_size_k", cfg.tile_size_k);
    int_field("tile_size_c", cfg.tile_size_c);
    int_field("tile_size_n", cfg.tile_size_n);
    int_field("tile_size_h", cfg.tile_size_h);
    int_field("tile_size_w", cfg.tile_size_w);
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    direction_field(*this, cfg.direction, Direction::Fprop);
    data_type_field(*this, "type", cfg.type, DataType::fp16);
    int_field("tiles_j", cfg.tiles_j, /*default=*/2);
    int_field("tiles_k", cfg.tiles_k, /*default=*/4);
    int_field("wmma_size_j", cfg.wmma_size_j, /*default=*/16);
    int_field("wmma_size_k", cfg.wmma_size_k, /*default=*/16);
    int_field("wmma_size_c", cfg.wmma_size_c, /*default=*/32);
    int_field("k_pad", cfg.k_pad, /*default=*/4);
    int_field("k_parts", cfg.k_parts, /*default=*/2);
}

} // namespace hipconv::cdna4::direct
