// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/utilities/Tensor.hpp>

namespace test_helpers
{

template <typename T>
hipdnnPluginDeviceBuffer_t
    generateRandomDeviceBuffer(hipdnn_sdk::utilities::TensorBase<T>& tensor, int uid, T min, T max, unsigned int seed = 0)
{
    tensor.fillWithRandomValues(min, max, seed);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generateStaticDeviceBuffer(hipdnn_sdk::utilities::TensorBase<T>& tensor, int uid, T value)
{
    tensor.fillWithValue(value);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generateEmptyDeviceBuffer(hipdnn_sdk::utilities::TensorBase<T>& tensor, int uid)
{
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

} // namespace test_utils
