#include "direction_field.h"
#include "config_desc.h"

namespace hipconv::cdna4::depthwise_2d_toeplitz
{

hipconv::KVDescriptor config_fields(const Config& cfg)
{
    hipconv::KVDescriptor d;
    d.int_field("kh", cfg.kh);
    d.int_field("kw", cfg.kw);
    d.int_field("stride", cfg.stride);
    direction_field(d, cfg.direction);
    d.int_field("waves_per_wg", cfg.waves_per_wg);
    d.int_field("subgroups", cfg.subgroups, /*default=*/2);
    d.bool_field("dense", cfg.dense, /*default=*/false);
    d.int_field("tpr", cfg.tpr, /*default=*/16);
    return d;
}

} // namespace hipconv::cdna4::depthwise_2d_toeplitz
