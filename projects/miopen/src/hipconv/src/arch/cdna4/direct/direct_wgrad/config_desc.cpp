#include "config_desc.h"

namespace hipconv::cdna4::direct_wgrad
{

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    int_field("wave_c16", cfg.wave_c16);
    int_field("wave_k16", cfg.wave_k16);
    int_field("waves_c", cfg.waves_c);
    int_field("waves_k", cfg.waves_k);
    int_field("waves_q", cfg.waves_q, /*default=*/1);
    int_field("waves_g", cfg.waves_g, /*default=*/1);
    int_field("unfold_n", cfg.unfold_n, /*default=*/1);
    int_field("prefetch_rows", cfg.prefetch_rows, /*default=*/2);
}

} // namespace hipconv::cdna4::direct_wgrad
