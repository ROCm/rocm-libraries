#include "config_matcher.hpp"

namespace hipconv::cdna5::direct
{

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("tile_size_k", cfg.tile_size_k);
    int_field("tile_size_n", cfg.tile_size_n);
    int_field("tile_size_h", cfg.tile_size_h);
    int_field("tile_size_w", cfg.tile_size_w);
    bool_field("aligned", cfg.aligned);
}

} // namespace hipconv::cdna5::direct
