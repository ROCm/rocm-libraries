#include "config_desc.hpp"
#include "direction_field.h"

namespace hipconv::cdna4::grouped_4c_wgrad
{

hipconv::KVDescriptor config_fields(const Config& cfg)
{
    KVDescriptor d;
    d.int_field("group_size", GROUP_SIZE);
    d.int_field("waves_g", cfg.waves_g);
    d.int_field("waves_q", cfg.waves_q);
    d.int_field("wave_q4", cfg.wave_q4, /*default=*/8);
    d.int_field("kh", cfg.kh, /*default=*/3);
    d.int_field("kw", cfg.kw, /*default=*/3);
    d.int_field("stride", cfg.stride, /*default=*/1);
    d.int_field("elem_bytes", 2, /*default=*/2);
    direction_field(d, cfg.direction, Direction::Wgrad);
    return d;
}

} // namespace hipconv::cdna4::grouped_4c_wgrad
