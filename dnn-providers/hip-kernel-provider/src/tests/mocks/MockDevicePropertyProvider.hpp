// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include "device/IDevicePropertyProvider.hpp"

namespace hip_kernel_provider
{

class MockDevicePropertyProvider : public device::IDevicePropertyProvider
{
public:
    MOCK_METHOD(hipDeviceProp_t, getDeviceProperties, (), (const, override));
};

} // namespace hip_kernel_provider
