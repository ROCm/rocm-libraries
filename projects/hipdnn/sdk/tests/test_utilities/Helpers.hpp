// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/MigratableMemory.hpp>

namespace helpers
{

using namespace hipdnn_sdk::utilities;

template <typename T>
MigratableMemory<T> createBuffer(size_t size, T mult)
{
    MigratableMemory<T> buffer(size);

    T* data = buffer.hostData();

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<T>(static_cast<float>(i)) * mult;
    }

    return buffer;
}

template <typename T>
MigratableMemory<T> createConstantBuffer(size_t size, T value)
{
    MigratableMemory<T> buffer(size);

    T* data = buffer.hostData();

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = value;
    }

    return buffer;
}

} // namespace helpers
