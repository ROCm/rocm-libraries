// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>

namespace hipdnn_data_sdk::helpers
{

template <typename T>
utilities::MigratableMemory<T> createBuffer(size_t size, T mult)
{
    utilities::MigratableMemory<T> buffer(size);

    T* data = buffer.hostData();

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<T>(static_cast<float>(i)) * mult;
    }

    return buffer;
}

template <typename T>
utilities::MigratableMemory<T> createConstantBuffer(size_t size, T value)
{
    utilities::MigratableMemory<T> buffer(size);

    T* data = buffer.hostData();

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = value;
    }

    return buffer;
}

} // namespace hipdnn_data_sdk::helpers
