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
    // Defaulted, so the 2-byte configs' descriptors are unchanged by tf32's arrival.
    int_field("elem_bytes", cfg.elem_bytes, 2);
}

} // namespace hipconv::cdna5::direct
