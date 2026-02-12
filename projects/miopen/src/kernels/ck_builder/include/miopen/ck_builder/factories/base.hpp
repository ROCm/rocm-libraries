// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/shared.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

// This is the base factory template for all kernel instances supplied via CK builder. There should
// be an explicit specialization for each concrete DeviceOp that has instances. This base class
// intentionally omits a GetInstance() method to force a compilation error if a factory is
// specialized for a DeviceOp that has not been explicitly implemented.
template <typename DeviceOp>
struct DeviceOperationInstanceFactory
{
};

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
