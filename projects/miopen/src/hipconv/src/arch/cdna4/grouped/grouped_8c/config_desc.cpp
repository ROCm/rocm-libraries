#include "config_desc.hpp"
#include "direction_field.h"

namespace hipconv::cdna4::grouped_8c
{

hipconv::KVDescriptor config_fields(const Config& cfg)
{
    KVDescriptor d;
    d.int_field("group_size", cfg.group_size);
    d.int_field("waves_per_wg", cfg.waves_per_wg);
    d.int_field("stride", cfg.stride, /*default=*/1);
    d.int_field("dilation", cfg.dilation, /*default=*/1);
    direction_field(d, cfg.direction);
    d.int_field("kh", cfg.kh, /*default=*/3);
    d.int_field("kw", cfg.kw, /*default=*/3);
    d.int_field("kd", cfg.kd, /*default=*/1);
    d.int_field("n_fold", cfg.n_fold, /*default=*/8);
    d.bool_field("per_row", cfg.per_row, /*default=*/false);
    return d;
}

} // namespace hipconv::cdna4::grouped_8c
