// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_gpu_ref/detail/ShallowDeviceOnlyMigratableMemory.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace hipdnn_gpu_ref
{

/// Non-owning tensor that wraps an existing device pointer.
/// Mirrors hipdnn_data_sdk::utilities::ShallowTensor but for device memory.
template <class T>
class ShallowGpuTensor : public hipdnn_data_sdk::utilities::TensorBase<T>
{
    using Base = hipdnn_data_sdk::utilities::TensorBase<T>;

public:
    ShallowGpuTensor(void* deviceMemory,
                     const std::vector<int64_t>& dims,
                     const std::vector<int64_t>& strides)
        : _dims(dims)
        , _strides(strides)
        , _elementCount(Base::calculateItemCount(dims))
        , _packed(Base::computeIsPacked(dims, strides))
    {
        _memory = detail::ShallowDeviceOnlyMigratableMemory<T>(
            deviceMemory, Base::calculateElementSpace(dims, strides));
    }

    ShallowGpuTensor(const ShallowGpuTensor&) = delete;
    ShallowGpuTensor& operator=(const ShallowGpuTensor&) = delete;

    ShallowGpuTensor(ShallowGpuTensor&&) = default;
    ShallowGpuTensor& operator=(ShallowGpuTensor&&) = default;

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

    const hipdnn_data_sdk::utilities::MigratableMemoryBase<T>& memory() const override
    {
        return _memory;
    }

    hipdnn_data_sdk::utilities::MigratableMemoryBase<T>& memory() override
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
            "ShallowGpuTensor does not support this operation. Use the Tensor class instead.");
    }

    detail::ShallowDeviceOnlyMigratableMemory<T> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    size_t _elementCount;
    bool _packed;
};

} // namespace hipdnn_gpu_ref
