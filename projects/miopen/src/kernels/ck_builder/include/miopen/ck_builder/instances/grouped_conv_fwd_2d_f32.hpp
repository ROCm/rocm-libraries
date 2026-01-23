// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "xdl.hpp"
#include "xdl_v3.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
void add_grouped_conv_fwd_2d_f32(std::vector<BaseOperatorPtr>& instances);
} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
