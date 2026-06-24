// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_data_sdk::utilities
{

// NOLINTBEGIN(portability-template-virtual-member-function)

/**
 * @brief Shared, ragged-aware base for RaggedTensor<T> and ShallowRaggedTensor<T>.
 *
 * Holds the `ragged_offset` aux type-erased as a `std::shared_ptr<ITensor>` (both
 * int32 and int64 element types are accepted; the element type is discovered at
 * runtime). This intermediate carries everything the two concrete ragged types
 * share — offset reads, ragged addressing (`getIndexImpl`), the traversal hook
 * (`raggedRowOffsets`), geometry reporting, and structural validation — leaving the
 * concrete types responsible only for the memory carrier and fill operations.
 *
 * No CRTP is required because `memory()` is already virtual on `TensorBase<T>`.
 *
 * @note Physical memory layout is assumed BSHD with batch at logical index 0, so the
 * sequence (ragged) axis is the unique non-batch axis with the maximal stride. This
 * holds for all in-scope AITER FMHA configs (BSHD and BHSD logical orders). See RFC
 * 0014 §4.5/§4.6 and the ALMIOPEN-2124 implementation plan.
 */
template <typename T>
class RaggedTensorBase : public TensorBase<T>
{
public:
    RaggedTensorBase(std::vector<int64_t> paddedDims,
                     std::vector<int64_t> strides,
                     std::shared_ptr<ITensor> raggedOffset,
                     std::optional<size_t> physicalElementCount)
        : _paddedDims(std::move(paddedDims))
        , _strides(std::move(strides))
        , _raggedOffset(std::move(raggedOffset))
    {
        // Structural checks first, so readOffset below only sees a supported element size.
        validateRaggedStructure();

        // Snapshot and validate the B+1 offset table (RFC 0014 §4.5, plan §2.2). Reading
        // the aux may trigger a device->host sync if it lives in device memory.
        const std::vector<int64_t> offsets = collectRowOffsets();
        validateRaggedOffsets(offsets);

        _iteratedElementCount = static_cast<size_t>(offsets.back());

        // An explicit physicalElementCount is a redundant confirmation and must match.
        if(physicalElementCount.has_value() && *physicalElementCount != _iteratedElementCount)
        {
            throw std::invalid_argument(
                "physicalElementCount (" + std::to_string(*physicalElementCount)
                + ") must equal ragged_offset[B] (" + std::to_string(_iteratedElementCount) + ")");
        }
        _physicalElementCount = _iteratedElementCount;
    }

    const std::vector<int64_t>& dims() const override
    {
        return _paddedDims;
    }

    const std::vector<int64_t>& strides() const override
    {
        return _strides;
    }

    // Iterated elements == ragged_offset[B] (RFC §4.5.6).
    size_t elementCount() const override
    {
        return _iteratedElementCount;
    }

    // Allocated buffer size; sized to ragged_offset[B], so normally == elementCount().
    size_t elementSpace() const override
    {
        return _physicalElementCount;
    }

    // Diverges from the dense convention `_packed = (elementCount == elementSpace)`:
    // those are equal here, yet the buffer is neither prod(dims) elements long nor
    // regularly strided, so it must not be treated as a flat dense buffer (RFC §4.5.7).
    bool isPacked() const override
    {
        return false;
    }

    std::vector<int64_t> raggedRowOffsets() const override
    {
        return collectRowOffsets();
    }

    const ITensor* raggedOffset() const
    {
        return _raggedOffset.get();
    }

protected:
    // Ragged addressing: base each batch at ragged_offset[b], then add the remaining
    // indices against the padded strides. A bare index {b} bases at ragged_offset[b].
    int64_t getIndexImpl(const std::vector<int64_t>& indices) const override
    {
        if(indices.empty())
        {
            return 0;
        }

        const int64_t base = readOffset(static_cast<size_t>(indices[0]));
        return base + std::inner_product(std::next(indices.begin()),
                                         indices.end(),
                                         std::next(_strides.begin()),
                                         int64_t{0});
    }

    // Type-erased read of ragged_offset[b], widened to int64_t (RFC §4.5.1).
    int64_t readOffset(size_t b) const
    {
        const void* p = _raggedOffset->hostDataOffsetFromIndex(static_cast<int64_t>(b));
        switch(_raggedOffset->elementSize())
        {
        case 4: return *static_cast<const int32_t*>(p);
        case 8: return *static_cast<const int64_t*>(p);
        default: throw std::runtime_error("ragged_offset element size must be 4 or 8 bytes");
        }
    }

    // Constructor-time structural validation, shared by both concrete types (RFC §4.5.5).
    void validateRaggedStructure() const
    {
        if(_raggedOffset == nullptr)
        {
            throw std::invalid_argument("ragged_offset must not be null");
        }
        if(_paddedDims.empty())
        {
            throw std::invalid_argument("paddedDims must not be empty");
        }
        const size_t expectedCount = static_cast<size_t>(_paddedDims[0]) + 1;
        if(_raggedOffset->elementCount() != expectedCount)
        {
            throw std::invalid_argument(
                "ragged_offset elementCount (" + std::to_string(_raggedOffset->elementCount())
                + ") must equal paddedDims[0] + 1 (" + std::to_string(expectedCount) + ")");
        }
        if(_raggedOffset->dims().size() != 4)
        {
            throw std::invalid_argument("ragged_offset must have rank 4");
        }
        const size_t elementSize = _raggedOffset->elementSize();
        if(elementSize != 4 && elementSize != 8)
        {
            throw std::invalid_argument("ragged_offset element size must be 4 or 8 bytes (got "
                                        + std::to_string(elementSize) + ")");
        }
    }

    // Reads the B+1 offset table from the aux, each widened to int64_t.
    std::vector<int64_t> collectRowOffsets() const
    {
        const auto batchCount = static_cast<size_t>(_paddedDims[0]);
        std::vector<int64_t> offsets;
        offsets.reserve(batchCount + 1);
        for(size_t b = 0; b <= batchCount; ++b)
        {
            offsets.push_back(readOffset(b));
        }
        return offsets;
    }

    // Sequence (ragged) axis == the non-batch axis with the largest stride (plan §2.2).
    std::pair<int, int64_t> sequenceAxisAndStride() const
    {
        int seqAxis = 1;
        for(int j = 2; j < static_cast<int>(_strides.size()); ++j)
        {
            if(_strides[static_cast<size_t>(j)] > _strides[static_cast<size_t>(seqAxis)])
            {
                seqAxis = j;
            }
        }
        const int64_t seqStride =
            _strides.size() < 2 ? int64_t{1} : _strides[static_cast<size_t>(seqAxis)];
        return {seqAxis, seqStride};
    }

    // Validates offset contents: offset[0] == 0, monotonic non-decreasing, each per-batch
    // block a whole number of sequence rows, and per-batch extent <= S_max (RFC §4.5,
    // plan §2.2; mirrored as debug asserts in RaggedCompositeIndex).
    void validateRaggedOffsets(const std::vector<int64_t>& offsets) const
    {
        if(offsets.empty())
        {
            return;
        }
        if(offsets.front() != 0)
        {
            throw std::invalid_argument("ragged_offset[0] must be 0 (got "
                                        + std::to_string(offsets.front()) + ")");
        }

        const auto [seqAxis, seqStride] = sequenceAxisAndStride();
        const bool haveSeqAxis = _strides.size() >= 2;
        if(haveSeqAxis && seqStride <= 0)
        {
            throw std::invalid_argument("sequence-axis stride must be positive");
        }
        const int64_t sMax = haveSeqAxis ? _paddedDims[static_cast<size_t>(seqAxis)] : int64_t{0};

        const int64_t batchCount = static_cast<int64_t>(offsets.size()) - 1;
        for(int64_t b = 0; b < batchCount; ++b)
        {
            const auto bIdx = static_cast<size_t>(b);
            const int64_t block = offsets[bIdx + 1] - offsets[bIdx];
            if(block < 0)
            {
                throw std::invalid_argument("ragged_offset must be monotonic non-decreasing (batch "
                                            + std::to_string(b) + ")");
            }
            if(haveSeqAxis)
            {
                if(block % seqStride != 0)
                {
                    throw std::invalid_argument(
                        "per-batch block must be a whole number of sequence rows (batch "
                        + std::to_string(b) + ")");
                }
                if(block / seqStride > sMax)
                {
                    throw std::invalid_argument(
                        "per-batch sequence extent exceeds dims()[seqAxis] / S_max (batch "
                        + std::to_string(b) + ")");
                }
            }
        }
    }

    std::vector<int64_t> _paddedDims;
    std::vector<int64_t> _strides;
    size_t _iteratedElementCount{0}; ///< ragged_offset[B]
    size_t _physicalElementCount{0}; ///< allocated buffer size (== ragged_offset[B])
    std::shared_ptr<ITensor> _raggedOffset; ///< non-null, fixed at construction, never reseated
};

/**
 * @brief Memory-owning, ragged-aware tensor.
 *
 * Physical element count is independent of `prod(dims)`; the buffer is sized to
 * `ragged_offset[B]`. Holds a `shared_ptr` to its `ragged_offset` aux, fixed at
 * construction. Pinned host memory is available via
 * `RaggedTensor<T, PinnedHostAllocator<T>>` (alias `PinnedRaggedTensor<T>`).
 */
template <class T, class HostAlloc = HostAllocator<T>, class DeviceAlloc = DeviceAllocator<T>>
class RaggedTensor : public RaggedTensorBase<T>
{
public:
    RaggedTensor(std::vector<int64_t> paddedDims,
                 std::vector<int64_t> strides,
                 std::shared_ptr<ITensor> raggedOffset,
                 std::optional<size_t> physicalElementCount = std::nullopt)
        : RaggedTensorBase<T>(std::move(paddedDims),
                              std::move(strides),
                              std::move(raggedOffset),
                              physicalElementCount)
    {
        _memory = MigratableMemory<T, HostAlloc, DeviceAlloc>(this->elementSpace());
    }

    RaggedTensor(const RaggedTensor&) = delete;
    RaggedTensor& operator=(const RaggedTensor&) = delete;

    RaggedTensor(RaggedTensor&&) = default;
    RaggedTensor& operator=(RaggedTensor&&) = default;

    const MigratableMemoryBase<T>& memory() const override
    {
        return _memory;
    }

    MigratableMemoryBase<T>& memory() override
    {
        return _memory;
    }

    size_t fillWithData(const void* data, size_t maxBytesCopied) override
    {
        const size_t bytesCopied = std::min(maxBytesCopied, _memory.count() * sizeof(T));
        _memory.markHostModified();
        std::memcpy(_memory.hostData(), data, bytesCopied);
        return bytesCopied;
    }

    void fillWithValue(T value) override
    {
        _memory.markHostModified();
        for(auto valuePtr : (*this))
        {
            *static_cast<T*>(valuePtr) = value;
        }
    }

    void fillWithRandomValues(T min, T max, unsigned int seed = std::random_device{}()) override
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(min),
                                                           static_cast<float>(max));

        _memory.markHostModified();
        for(auto valuePtr : (*this))
        {
            *static_cast<T*>(valuePtr) = static_cast<T>(distribution(generator));
        }
    }

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
};

template <typename T>
using PinnedRaggedTensor = RaggedTensor<T, PinnedHostAllocator<T>>;

// NOLINTEND(portability-template-virtual-member-function)

} // namespace hipdnn_data_sdk::utilities
