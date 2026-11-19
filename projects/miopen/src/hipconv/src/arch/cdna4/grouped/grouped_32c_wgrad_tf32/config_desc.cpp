#include "config_desc.hpp"

namespace hipconv::cdna4::grouped_32c_wgrad_tf32
{

hipconv::KVDescriptor config_fields(const Config& cfg)
{
    KVDescriptor d;
    d.int_field("group_size", GROUP_SIZE);
    d.int_field("groups_per_wg", cfg.groups_per_wg);
    d.int_field("wave_q32", cfg.wave_q32, /*default=*/1);
    d.int_field("wave_n", cfg.wave_n, /*default=*/1);
    d.int_field("kh", cfg.kh, /*default=*/3);
    d.int_field("kw", cfg.kw, /*default=*/3);
    d.int_field("stride", cfg.stride, /*default=*/1);
    d.int_field("elem_bytes", 4, /*default=*/2);
    return d;
}

} // namespace hipconv::cdna4::grouped_32c_wgrad_tf32
