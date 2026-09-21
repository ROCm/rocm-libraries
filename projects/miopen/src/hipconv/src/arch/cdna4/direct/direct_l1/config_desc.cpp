#include "config_desc.h"

namespace hipconv::cdna4::direct_l1
{

namespace
{

// Render a Direction as its descriptor tag.
const char* direction_tag(hipconv::Direction d)
{
    switch(d)
    {
    case hipconv::Direction::Dgrad:
        return "dgrad";
    case hipconv::Direction::Wgrad:
        return "wgrad";
    case hipconv::Direction::Fprop:
    default:
        return "fprop";
    }
}

// Parse a direction tag. Returns false on an unknown tag.
bool to_direction(std::string_view s, hipconv::Direction& out)
{
    if(s == "fprop")
        return out = hipconv::Direction::Fprop, true;
    if(s == "dgrad")
        return out = hipconv::Direction::Dgrad, true;
    if(s == "wgrad")
        return out = hipconv::Direction::Wgrad, true;
    return false;
}

} // namespace

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("waves_k", cfg.waves_k);
    int_field("wave_k16", cfg.wave_k16);
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    custom_field("direction", cfg.direction, to_direction, direction_tag);
    int_field("unfold_n", cfg.unfold_n, /*default=*/1);
    bool_field("k_divisible", cfg.k_divisible, /*default=*/true);
    bool_field("single_c", cfg.single_c, /*default=*/false);
    bool_field("large_tensor", cfg.large_tensor, /*default=*/false);
}

} // namespace hipconv::cdna4::direct_l1
