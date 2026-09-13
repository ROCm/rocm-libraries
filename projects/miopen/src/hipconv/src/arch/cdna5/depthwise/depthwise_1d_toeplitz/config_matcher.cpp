#include "config_matcher.hpp"

namespace hipconv::cdna5::depthwise_1d_toeplitz
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

// Every knob in Config is registered, so describe() is a unique name for a table entry and
// --config can always pin exactly one.
//
// The fields carry the Config's own encoding, not the layer's: a stride-2 dgrad entry is
// stride=1,dilation=2 here (see the Config comment), so it answers to that rather than to
// stride=2.
ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    int_field("stride", cfg.stride);
    int_field("dilation", cfg.dilation, /*default=*/1);
    custom_field("direction", cfg.direction, to_direction, direction_tag);
    int_field("waves_per_wg", cfg.waves_per_wg);
    bool_field("narrow_c", cfg.narrow_c, /*default=*/false);
    int_field("w_fold", cfg.w_fold, /*default=*/1);
    int_field("prefetch_depth", cfg.prefetch_depth, /*default=*/3);
    int_field("n_fold", cfg.n_fold, /*default=*/8);
    int_field("elem_bytes", cfg.elem_bytes, /*default=*/2);
}

} // namespace hipconv::cdna5::depthwise_1d_toeplitz
