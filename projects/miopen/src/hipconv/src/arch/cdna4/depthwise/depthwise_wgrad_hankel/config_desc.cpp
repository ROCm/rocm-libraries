#include "config_desc.h"

namespace hipconv::cdna4::depthwise_wgrad_hankel
{

// Direction is omitted: it is Wgrad for every config here, so it selects nothing.
ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    int_field("q_tiles", cfg.q_tiles);
    int_field("rows_per_chunk", cfg.rows_per_chunk);
    int_field("stage_depth", cfg.stage_depth, /*default=*/2);
    int_field("stride", cfg.stride, /*default=*/1);
    bool_field("narrow_c", cfg.narrow_c, /*default=*/false);
}

} // namespace hipconv::cdna4::depthwise_wgrad_hankel
