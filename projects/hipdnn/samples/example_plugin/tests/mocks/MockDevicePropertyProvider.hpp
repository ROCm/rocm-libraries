// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>

#include "hip/IDevicePropertyProvider.hpp"

namespace example_plugin
{

class MockDevicePropertyProvider : public IDevicePropertyProvider
{
public:
    MOCK_METHOD(hipDeviceProp_t, getDeviceProperties, (), (const, override));
};

} // namespace example_plugin
