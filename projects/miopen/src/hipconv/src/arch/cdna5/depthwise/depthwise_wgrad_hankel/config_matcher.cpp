#include "config_matcher.hpp"

namespace hipconv::cdna5::depthwise_wgrad_hankel
{

// Direction is omitted: every entry here is Wgrad, so it selects nothing.
//
// make_configs varies kh, kw, stride, q_cols, wmmas_per_wave, waves_per_wg and
// n_per_block, so those seven pin exactly one entry. channels_per_wave follows from kh, kw and
// wmmas_per_wave; it is registered to be read rather than to select, a tile
// being easier to recognise by the channels a wave carries than by the two
// numbers behind them.
ConfigMatcher::ConfigMatcher(const Config& cfg)
{
    int_field("kh", cfg.kh);
    int_field("kw", cfg.kw);
    int_field("stride", cfg.stride);
    int_field("q_cols", cfg.q_cols);
    int_field("channels_per_wave", cfg.channels_per_wave());
    int_field("wmmas_per_wave", cfg.wmmas_per_wave);
    int_field("waves_per_wg", cfg.waves_per_wg);
    int_field("n_per_block", cfg.n_per_block, /*default=*/1);
    int_field("prefetch_depth", cfg.prefetch_depth, /*default=*/4);
    int_field("min_rows_per_chunk", cfg.min_rows_per_chunk, /*default=*/4);
}

} // namespace hipconv::cdna5::depthwise_wgrad_hankel
