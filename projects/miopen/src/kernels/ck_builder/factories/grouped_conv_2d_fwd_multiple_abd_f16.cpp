#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

std::vector<BaseOperatorPtr> DeviceOperationInstanceFactory<DeviceOpGFwdDefault<ck::half_t>>::GetInstances()
{
    // Adapted from GetInstances() in the composable_kernel project's file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp
    std::vector<BaseOperatorPtr> instances{};
    
    add_f16_merged_groups_instances(instances);
    add_f16_standard_instances(instances);
    add_f16_16x16_instances(instances);
    add_f16_comp_instances(instances);
    add_f16_comp_2x_instances(instances);
    add_f16_comp_part2_instances(instances);
    add_f16_mem_intra_instances(instances);
    add_f16_mem_inter_instances(instances);
    
    return instances;
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
