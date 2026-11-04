#include "config_desc.h"
#include "direction_field.h"

namespace hipconv::cdna5::grouped_multi_g_wgrad
{

hipconv::KVDescriptor config_fields(const Config& cfg)
{
    hipconv::KVDescriptor d;
    d.int_field("group_size", cfg.group_size);
    d.int_field("waves_per_wg", cfg.waves_per_wg);
    d.bool_field("split_k", cfg.split_k, /*default=*/false);
    d.int_field("prefetch_depth", cfg.prefetch_depth, /*default=*/3);
    d.int_field("kh", cfg.kh, /*default=*/3);
    d.int_field("kw", cfg.kw, /*default=*/3);
    direction_field(d, cfg.direction, Direction::Wgrad);
    return d;
}

} // namespace hipconv::cdna5::grouped_multi_g_wgrad
