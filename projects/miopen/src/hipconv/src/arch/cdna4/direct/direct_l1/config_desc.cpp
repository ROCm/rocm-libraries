#include "config_desc.h"
#include "direction_field.h"

namespace hipconv::cdna4::direct_l1
{

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("waves_k", cfg.waves_k);
    int_field("wave_k16", cfg.wave_k16);
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    direction_field(*this, cfg.direction);
    int_field("unfold_n", cfg.unfold_n, /*default=*/1);
    bool_field("k_divisible", cfg.k_divisible, /*default=*/true);
    bool_field("single_c", cfg.single_c, /*default=*/false);
    bool_field("large_tensor", cfg.large_tensor, /*default=*/false);
    int_field("elem_bytes", cfg.elem_bytes, /*default=*/2);
    int_field("wave_q16", cfg.wave_q16, /*default=*/1);
    int_field("wave_p", cfg.wave_p, /*default=*/8);
    int_field("waves_p", cfg.waves_p, /*default=*/2);
    int_field("waves_q", cfg.waves_q, /*default=*/2);
}

} // namespace hipconv::cdna4::direct_l1
