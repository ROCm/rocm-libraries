#include "config_desc.hpp"
#include "direction_field.h"

namespace hipconv::cdna4::grouped_16c_wgrad
{

hipconv::KVDescriptor config_fields(const Config& cfg)
{
    KVDescriptor d;
    d.int_field("group_size", cfg.group_size);
    d.int_field("waves_per_wg", cfg.waves_per_wg);
    d.int_field("wave_q32", cfg.wave_q32, /*default=*/1);
    d.int_field("kh", cfg.kh, /*default=*/3);
    d.int_field("kw", cfg.kw, /*default=*/3);
    d.int_field("stride", cfg.stride, /*default=*/1);
    d.int_field("elem_bytes", 2, /*default=*/2);
    direction_field(d, cfg.direction, Direction::Wgrad);
    return d;
}

} // namespace hipconv::cdna4::grouped_16c_wgrad
