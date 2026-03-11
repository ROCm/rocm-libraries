// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime_api.h>

namespace example_plugin
{

/// Interface for querying GPU device properties.
///
/// Abstracts device property access for testability. In production,
/// CurrentDevicePropertyProvider calls hipGetDeviceProperties().
/// In unit tests, MockDevicePropertyProvider returns preset values.
class IDevicePropertyProvider
{
public:
    virtual ~IDevicePropertyProvider() = default;

    virtual hipDeviceProp_t getDeviceProperties() const = 0;
};

} // namespace example_plugin
