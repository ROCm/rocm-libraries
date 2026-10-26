#include "config_desc.h"
#include "direction_field.h"

namespace hipconv::cdna4::depthwise_wgrad_hankel
{

ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    int_field("q_tiles", cfg.q_tiles);
    int_field("rows_per_chunk", cfg.rows_per_chunk);
    int_field("stage_depth", cfg.stage_depth, /*default=*/2);
    int_field("stride", cfg.stride, /*default=*/1);
    int_field("chan_vec", cfg.chan_vec, /*default=*/8);
    direction_field(*this, cfg.direction, Direction::Wgrad);
}

} // namespace hipconv::cdna4::depthwise_wgrad_hankel
