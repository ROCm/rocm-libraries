// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <memory>
#include <vector>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Channels selects tile/vector defaults, not an argument-admission shape gate.
// Width overrides allow scalar transfers for vector-incompatible channels.
template <index_t Channels,
          index_t GroupsPerWmma,
          index_t AVector = Channels == 2 ? 2 : 4,
          index_t BVector = GroupsPerWmma * Channels < 8 ? 4 : 8,
          index_t EVector = Channels == 2 ? 2 : 4,
          index_t NRepeat = (GroupsPerWmma * Channels + 15) / 16>
using PackedConvCandidate = DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<
    2,
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
    16 * NRepeat,
    96,
    8,
    8,
    16,
    16,
    2,
    NRepeat,
    S<4, 16, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    AVector,
    8,
    1,
    S<4, 16, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    BVector,
    8,
    1,
    1,
    1,
    S<1, 16, 1, 4>,
    EVector,
    BlockGemmPipelineScheduler::Intrawave,
    BlockGemmPipelineVersion::v1,
    true,
    F16,
    F16,
    1,
    GroupsPerWmma>;

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

    add_device_operation_instances(
        instances,
        std::tuple<PackedConvCandidate<4, 4>,
                   PackedConvCandidate<2, 4>,
                   PackedConvCandidate<2, 8>,
                   PackedConvCandidate<4, 8>,
                   PackedConvCandidate<8, 4>,
                   PackedConvCandidate<16, 4>>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
