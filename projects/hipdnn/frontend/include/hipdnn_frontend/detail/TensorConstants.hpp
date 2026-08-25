// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstdint>

namespace hipdnn_frontend::detail
{

// Default tensor byte alignment. Shared source for the TensorAttributes member
// default, the lowering guard in DescriptorHelpers, and their tests. Mirrors the
// schema default (tensor_attributes.fbs -> `alignment: long = 16`).
inline constexpr int64_t DEFAULT_TENSOR_ALIGNMENT = 16;

} // namespace hipdnn_frontend::detail
