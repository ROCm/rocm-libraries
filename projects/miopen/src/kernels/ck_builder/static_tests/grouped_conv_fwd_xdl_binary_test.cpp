// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test compilation of the XDL binary outelementop factory header

#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_binary_outelementop_instance.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

// Note: binary_outelementop requires CK_ENABLE_FP8 to have instances
// The function exists but returns empty array without FP8 support

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
