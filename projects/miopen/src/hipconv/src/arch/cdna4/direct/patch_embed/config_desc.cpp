#include "config_desc.h"

namespace hipconv::cdna4::patch_embed
{

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("m_tile16", cfg.m_tile16);
    int_field("n_tile16", cfg.n_tile16);
    int_field("k_chunk", cfg.k_chunk);
    int_field("waves_m", cfg.waves_m);
    int_field("waves_n", cfg.waves_n);
    bool_field("row_pad", cfg.row_pad, /*default=*/false);
}

} // namespace hipconv::cdna4::patch_embed
