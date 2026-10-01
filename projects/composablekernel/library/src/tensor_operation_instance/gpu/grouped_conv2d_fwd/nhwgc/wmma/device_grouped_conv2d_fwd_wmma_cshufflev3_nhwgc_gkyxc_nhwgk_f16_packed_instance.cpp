// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <memory>
#include <vector>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_packed_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    // Register the block-diagonal candidate only on gfx1250.
    if(!ck::is_gfx125_supported())
        return;

    using PackedBase =
        DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<2,
                                                         NHWGC,
                                                         GKYXC,
                                                         Empty_Tuple,
                                                         NHWGK,
                                                         F16,
                                                         F16,
                                                         F32,
                                                         F32,
                                                         Empty_Tuple,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough,
                                                         ConvFwdDefault,
                                                         GemmMNKPadding,
                                                         64,
                                                         64,
                                                         16,
                                                         96,
                                                         8,
                                                         8,
                                                         16,
                                                         16,
                                                         2,
                                                         1,
                                                         S<4, 16, 1>,
                                                         S<1, 0, 2>,
                                                         S<1, 0, 2>,
                                                         2,
                                                         4,
                                                         8,
                                                         1,
                                                         S<4, 16, 1>,
                                                         S<1, 0, 2>,
                                                         S<1, 0, 2>,
                                                         2,
                                                         8,
                                                         8,
                                                         1,
                                                         1,
                                                         1,
                                                         S<1, 16, 1, 4>,
                                                         4,
                                                         BlockGemmPipelineScheduler::Intrawave,
                                                         BlockGemmPipelineVersion::v1,
                                                         true,
                                                         F16,
                                                         F16,
                                                         1,
                                                         4>;
    struct PackedInstance : PackedBase
    {
        bool IsSupportedArgument(const BaseArgument* p_arg) override
        {
            const auto* arg = dynamic_cast<const PackedBase::Argument*>(p_arg);
            return arg != nullptr && arg->b_g_k_c_xs_lengths_[2] == 4 &&
                   arg->b_g_k_c_xs_lengths_[1] == 4 && PackedBase::IsSupportedArgument(*arg);
        }
    };
    instances.emplace_back(std::make_unique<PackedInstance>());
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
