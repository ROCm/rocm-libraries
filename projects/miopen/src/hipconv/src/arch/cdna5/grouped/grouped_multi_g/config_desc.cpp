#include "direction_field.h"
#include "config_desc.h"

namespace hipconv::cdna5::grouped_multi_g
{

// A stride-2 dgrad entry is stride=1,dilation=2 here, the Config's encoding, not the layer's.
hipconv::KVDescriptor config_fields(const Config& cfg)
{
    hipconv::KVDescriptor d;
    d.int_field("group_size", cfg.group_size);
    d.int_field("waves_per_wg", cfg.waves_per_wg);
    d.int_field("stride", cfg.stride, /*default=*/1);
    d.int_field("dilation", cfg.dilation, /*default=*/1);
    direction_field(d, cfg.direction);
    d.int_field("prefetch_depth", cfg.prefetch_depth, /*default=*/2);
    d.int_field("kh", cfg.kh, /*default=*/3);
    d.int_field("kw", cfg.kw, /*default=*/3);
    return d;
}

} // namespace hipconv::cdna5::grouped_multi_g
