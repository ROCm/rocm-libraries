#include "direction_field.h"
#include "config_desc.h"

namespace hipconv::cdna4::depthwise_1d_toeplitz
{

// A stride-2 dgrad entry is stride=1,dilation=2 here, the Config's encoding, not the layer's.
hipconv::KVDescriptor config_fields(const Config& cfg)
{
    hipconv::KVDescriptor d;
    d.int_field("kh", cfg.kh);
    d.int_field("kw", cfg.kw);
    d.int_field("stride", cfg.stride);
    d.int_field("dilation", cfg.dilation, /*default=*/1);
    direction_field(d, cfg.direction);
    d.int_field("waves_per_wg", cfg.waves_per_wg);
    d.int_field("group_size", cfg.group_size, /*default=*/8);
    d.bool_field("narrow_c", cfg.narrow_c, /*default=*/false);
    d.int_field("w_fold", cfg.w_fold, /*default=*/1);
    d.int_field("lds_buffers", cfg.lds_buffers, /*default=*/3);
    d.int_field("n_fold", cfg.n_fold, /*default=*/8);
    d.int_field("elem_bytes", cfg.elem_bytes, /*default=*/2);
    return d;
}

} // namespace hipconv::cdna4::depthwise_1d_toeplitz
