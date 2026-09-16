// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowHostOnlyMigratableMemory.hpp>

namespace hipdnn_data_sdk::utilities
{

/**
 * @brief Non-owning, ragged-aware peer to RaggedTensor<T>.
 *
 * Wraps a borrowed `void*` buffer (e.g. a variant-pack pointer) with the same ragged
 * addressing, traversal, geometry reporting, and structural validation as
 * RaggedTensor<T>; only memory ownership differs. Unlike RaggedTensor it carries no
 * allocator template parameter — pinned-vs-pageable is a property of the wrapped
 * buffer, not the wrapper.
 *
 * @note Host-only, matching ShallowTensor: device access throws.
 */
template <typename T>
class ShallowRaggedTensor : public RaggedTensorBase<T>
{
public:
    ShallowRaggedTensor(void* data,
                        std::vector<int64_t> paddedDims,
                        std::vector<int64_t> strides,
                        int seqAxis,
                        std::shared_ptr<ITensor> raggedOffset,
                        std::optional<size_t> physicalElementCount = std::nullopt)
        : RaggedTensorBase<T>(std::move(paddedDims),
                              std::move(strides),
                              seqAxis,
                              std::move(raggedOffset),
                              physicalElementCount)
    {
        _memory = ShallowHostOnlyMigratableMemory<T>(data, this->elementSpace());
    }

    ShallowRaggedTensor(const ShallowRaggedTensor&) = delete;
    ShallowRaggedTensor& operator=(const ShallowRaggedTensor&) = delete;

    ShallowRaggedTensor(ShallowRaggedTensor&&) = default;
    ShallowRaggedTensor& operator=(ShallowRaggedTensor&&) = default;

    const MigratableMemoryBase<T>& memory() const override
    {
        return _memory;
    }

    MigratableMemoryBase<T>& memory() override
    {
        return _memory;
    }

    void fillWithValue(T value) override
    {
        T* data = _memory.hostData();
        std::fill(data, data + _memory.count(), value);
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
        throw std::runtime_error("ShallowRaggedTensor does not support this operation. Use the "
                                 "RaggedTensor class instead.");
    }

    ShallowHostOnlyMigratableMemory<T> _memory;
};

} // namespace hipdnn_data_sdk::utilities
