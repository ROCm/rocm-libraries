// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <hipdnn_sdk/utilities/ShallowHostOnlyMigratableMemory.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

namespace hipdnn_sdk
{
namespace utilities
{

template <class T>
class ShallowTensor : public TensorBase<T>
{
public:
    ShallowTensor(void* memory,
                  const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& strides)
        : _memory(memory)
        , _dims(dims)
        , _strides(strides)
    {
    }

    ShallowTensor(const ShallowTensor&) = delete;
    ShallowTensor& operator=(const ShallowTensor&) = delete;

    ShallowTensor(ShallowTensor&&) = default;
    ShallowTensor& operator=(ShallowTensor&&) = default;

    const std::vector<int64_t>& dims() const override
    {
        return _dims;
    }

    const std::vector<int64_t>& strides() const override
    {
        return _strides;
    }

    const IMigratableMemory<T>& memory() const override
    {
        return _memory;
    }

    IMigratableMemory<T>& memory() override
    {
        return _memory;
    }

    void fillWithValue([[maybe_unused]] T value) override
    {
        throwNotSupported();
    }

    void fillWithRandomValues([[maybe_unused]] T min,
                              [[maybe_unused]] T max,
                              [[maybe_unused]] unsigned int seed = std::random_device{}()) override
    {
        throwNotSupported();
    }

private:
    static void throwNotSupported()
    {
        throw std::runtime_error(
            "ShallowTensor does not support this operation.  Use the Tensor class instead.");
    }

    ShallowHostOnlyMigratableMemory<T> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
};

}
}
