// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test compilation of the XDL factory header

#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

// Test that the factory functions can be instantiated
[[maybe_unused]] constexpr auto test_xdl_bf16 =
    device_grouped_conv_fwd_xdl_bf16_instances(2,
                                               ckb::TensorLayout::GNHWC,
                                               ckb::TensorLayout::GKYXC,
                                               std::array<ckb::TensorLayout, 0>{},
                                               ckb::TensorLayout::GNHWK,
                                               ckb::ConvSpecialization::DEFAULT);

// Verify the returned arrays have the expected size
static_assert(test_xdl_bf16.size() == 16, "Expected 16 BF16 instances");

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
