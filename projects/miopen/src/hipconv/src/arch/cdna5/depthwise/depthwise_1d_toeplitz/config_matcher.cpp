#include "config_matcher.hpp"
#include "direction_field.h"

namespace hipconv::cdna5::depthwise_1d_toeplitz
{

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
    direction_field(*this, cfg.direction);
    int_field("waves_per_wg", cfg.waves_per_wg);
    bool_field("narrow_c", cfg.narrow_c, /*default=*/false);
    int_field("w_fold", cfg.w_fold, /*default=*/1);
    int_field("prefetch_depth", cfg.prefetch_depth, /*default=*/3);
    int_field("n_fold", cfg.n_fold, /*default=*/8);
    int_field("elem_bytes", cfg.elem_bytes, /*default=*/2);
}

} // namespace hipconv::cdna5::depthwise_1d_toeplitz
