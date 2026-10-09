#include "config_matcher.hpp"

namespace hipconv::cdna5::direct_wgrad
{

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    int_field("tile_size_c", cfg.tile_size_c);
    int_field("tile_size_k", cfg.tile_size_k);

    int_field("wmma_size_c", cfg.wmma_size_c, /*default=*/16);
    int_field("wmma_size_k", cfg.wmma_size_k, /*default=*/16);
    int_field("wmma_size_q", cfg.wmma_size_q, /*default=*/32);
    int_field("tiles_c", cfg.tiles_c, /*default=*/2);
    int_field("tiles_k", cfg.tiles_k, /*default=*/4);
    int_field("ring_size_p", cfg.ring_size_p, /*default=*/16);
    int_field("chunk_size_p", cfg.chunk_size_p, /*default=*/4);
}

} // namespace hipconv::cdna5::direct_wgrad
