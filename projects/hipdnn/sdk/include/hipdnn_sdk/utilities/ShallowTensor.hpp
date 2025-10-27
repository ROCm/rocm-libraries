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
        : _dims(dims)
        , _strides(strides)
        , _elementCount(TensorBase<T>::calculateItemCount(dims))
        , _packed(TensorBase<T>::computeIsPacked(dims, strides))
    {
        _memory = ShallowHostOnlyMigratableMemory<T>(
            memory, TensorBase<T>::calculateElementSpace(dims, strides));
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

    bool isPacked() const override
    {
        return _packed;
    }

    size_t elementCount() const override
    {
        return _elementCount;
    }

    size_t elementSpace() const override
    {
        return _memory.count();
    }

    const MigratableMemoryBase<T>& memory() const override
    {
        return _memory;
    }

    MigratableMemoryBase<T>& memory() override
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

    size_t fillWithData([[maybe_unused]] const void* data,
                        [[maybe_unused]] size_t maxBytesCopied) override
    {
        throwNotSupported();
        return 0;
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
    size_t _elementCount;
    bool _packed;
};

}
}
