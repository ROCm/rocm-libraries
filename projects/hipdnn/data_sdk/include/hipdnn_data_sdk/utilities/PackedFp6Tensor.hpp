// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_data_sdk::utilities
{

/// Device-side packed storage for an FP6 tensor (`fp6_e2m3` or `fp6_e3m2`): a
/// dense LSB-first 6-bit bitstream, four values per three bytes, element `i` at
/// bits `[6i, 6i+6)`. `Tensor<fp6_*>` instead stores one 6-bit code per *byte*
/// (unpacked) for element-wise CPU access.
///
/// Pair the two: this type for the GPU-side bundle, `Tensor<fp6_*>` for the
/// CPU-reference bundle. Filled with the same (seed, min, max) they agree
/// value-for-value.
///
/// Only dense (packed-stride) layouts are supported. Element-wise host access is
/// not provided.
template <typename T>
class PackedFp6Tensor : public ITensor
{
    static_assert(std::is_same_v<T, types::fp6_e2m3> || std::is_same_v<T, types::fp6_e3m2>,
                  "PackedFp6Tensor supports only fp6_e2m3 and fp6_e3m2");

public:
    PackedFp6Tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
        : _dims(dims)
        , _strides(strides)
    {
        if(dims.size() != strides.size())
        {
            throw std::invalid_argument("PackedFp6Tensor: dims and strides size mismatch");
        }
        validateAllPositive(dims, "dimension");
        validateAllPositive(strides, "stride");
        _elementCount = computeElementCount(dims);
        if(!isDensePacked(dims, strides, _elementCount))
        {
            throw std::invalid_argument(
                "PackedFp6Tensor requires dense (contiguous) strides for sub-byte packing");
        }
        _memory = MigratableMemory<uint8_t>(packedByteCount());
        // The final byte's unused high bits belong to no element: nothing reads
        // them, and packElementAt() never writes them. Clear them once so the
        // buffer is reproducible byte-for-byte. Indexing is safe because the
        // checks above leave _elementCount >= 1.
        static_cast<uint8_t*>(_memory.hostData())[packedByteCount() - 1] = 0;
        _memory.markHostModified();
    }

    const std::vector<int64_t>& dims() const override
    {
        return _dims;
    }
    const std::vector<int64_t>& strides() const override
    {
        return _strides;
    }

    void* rawHostData() override
    {
        return _memory.hostData();
    }
    void* rawDeviceData() override
    {
        return _memory.deviceData();
    }

    size_t elementCount() const override
    {
        return _elementCount;
    }
    size_t elementSpace() const override
    {
        return _elementCount;
    }
    size_t elementSize() const override
    {
        throw std::logic_error("PackedFp6Tensor: elementSize() is undefined for a 6-bit "
                               "packed type (four values per three bytes); derive byte size "
                               "from the packed buffer, not elementCount() * elementSize()");
    }

    void* hostDataOffsetFromIndex(int64_t /*index*/) override
    {
        throw std::logic_error("PackedFp6Tensor: per-element host access is not supported "
                               "(values are 6-bit packed, four per three bytes)");
    }
    const void* hostDataOffsetFromIndex(int64_t /*index*/) const override
    {
        throw std::logic_error("PackedFp6Tensor: per-element host access is not supported "
                               "(values are 6-bit packed, four per three bytes)");
    }

    // A repeated 6-bit code has a three-byte period, so there is no one-byte
    // fill shortcut (as there is for FP4): write element by element.
    void fillTensorWithValue(float value) override
    {
        const uint8_t code = encodeElement(value);
        auto* host = static_cast<uint8_t*>(_memory.hostData());
        for(size_t i = 0; i < _elementCount; ++i)
        {
            packElementAt(host, i, code);
        }
        _memory.markHostModified();
    }

    // Must match Tensor<T>::fillWithRandomValues draw-for-draw, or the packed and
    // unpacked bundles disagree silently. That includes round-tripping the bounds
    // through T before building the distribution, as TensorBase does.
    void fillTensorWithRandomValues(float min,
                                    float max,
                                    unsigned int seed = std::random_device{}()) override
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(T(min)),
                                                           static_cast<float>(T(max)));

        auto* host = static_cast<uint8_t*>(_memory.hostData());
        for(size_t i = 0; i < _elementCount; ++i)
        {
            packElementAt(host, i, encodeElement(distribution(generator)));
        }
        _memory.markHostModified();
    }

    void fillWithSentinelValue() override
    {
        // Neither FP6 encoding has a NaN, so TensorBase::fillWithSentinelValue
        // uses max() for these types; match it so packed and unpacked agree.
        fillTensorWithValue(static_cast<float>(std::numeric_limits<T>::max()));
    }

    size_t fillWithData(const void* data, size_t maxBytesCopied) override
    {
        const size_t bytesCopied = std::min(maxBytesCopied, packedByteCount());
        std::memcpy(_memory.hostData(), data, bytesCopied);
        _memory.markHostModified();
        return bytesCopied;
    }

    ITensorIterator<false> begin() override
    {
        return {*this, false};
    }
    ITensorIterator<false> end() override
    {
        return {*this, true};
    }
    ITensorIterator<true> cbegin() const override
    {
        return {*this, false};
    }
    ITensorIterator<true> cend() const override
    {
        return {*this, true};
    }

    bool isPacked() const override
    {
        return true;
    }

    void markHostModified() override
    {
        _memory.markHostModified();
    }
    void markDeviceModified() override
    {
        _memory.markDeviceModified();
    }

private:
    static constexpr size_t BITS_PER_ELEMENT = 6;
    static constexpr size_t BITS_PER_BYTE = 8;
    static constexpr uint8_t ELEMENT_CODE_MASK = static_cast<uint8_t>((1 << BITS_PER_ELEMENT) - 1);
    // Largest count for which packedByteCount()'s `count * 6 + 7` cannot wrap.
    static constexpr size_t MAX_ELEMENT_COUNT
        = (std::numeric_limits<size_t>::max() - (BITS_PER_BYTE - 1)) / BITS_PER_ELEMENT;

    // Size of the buffer, exactly: ceil(elementCount * 6 / 8).
    size_t packedByteCount() const
    {
        return (_elementCount * BITS_PER_ELEMENT + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    }

    // Writes the 6 bits of `code` at bit offset 6 * index, LSB first, leaving the
    // neighbouring elements' bits untouched.
    void packElementAt(uint8_t* host, size_t index, uint8_t code) const
    {
        const size_t bitOffset = index * BITS_PER_ELEMENT;
        const size_t byteIndex = bitOffset / BITS_PER_BYTE;
        const auto bitIndex = static_cast<unsigned>(bitOffset % BITS_PER_BYTE);

        const auto lowMask = static_cast<uint8_t>(ELEMENT_CODE_MASK << bitIndex);
        const auto lowBits = static_cast<uint8_t>(code << bitIndex);
        host[byteIndex] = static_cast<uint8_t>((host[byteIndex] & ~lowMask) | lowBits);

        // Guarded because the final element can end flush with the buffer, making
        // byteIndex + 1 one past the bitstream.
        if(bitIndex > BITS_PER_BYTE - BITS_PER_ELEMENT)
        {
            const size_t highShift = BITS_PER_BYTE - bitIndex;
            const auto highMask = static_cast<uint8_t>(ELEMENT_CODE_MASK >> highShift);
            const auto highBits = static_cast<uint8_t>(code >> highShift);
            host[byteIndex + 1]
                = static_cast<uint8_t>((host[byteIndex + 1] & ~highMask) | highBits);
        }
    }

    static uint8_t encodeElement(float value)
    {
        return static_cast<uint8_t>(T(value).data & ELEMENT_CODE_MASK);
    }

    static void validateAllPositive(const std::vector<int64_t>& values, const char* valueName)
    {
        for(const auto value : values)
        {
            if(value <= 0)
            {
                throw std::invalid_argument(std::string("PackedFp6Tensor: ") + valueName
                                            + " must be positive");
            }
        }
    }

    // Rejects an overflowing product rather than returning a wrapped one, which
    // would under-size the buffer. Dims must already be validated positive, or
    // the divisor below can be zero.
    static size_t computeElementCount(const std::vector<int64_t>& dims)
    {
        size_t count = 1;
        for(const auto d : dims)
        {
            const auto dim = static_cast<size_t>(d);
            if(count > MAX_ELEMENT_COUNT / dim)
            {
                throw std::invalid_argument(
                    "PackedFp6Tensor: element count exceeds the addressable packed size");
            }
            count *= dim;
        }
        return dims.empty() ? 0 : count;
    }

    // Dense == the element offsets span exactly [0, elementCount).
    static bool isDensePacked(const std::vector<int64_t>& dims,
                              const std::vector<int64_t>& strides,
                              size_t elementCount)
    {
        size_t span = 1;
        for(size_t i = 0; i < dims.size(); ++i)
        {
            // Widen first: (dim - 1) * stride can overflow int64_t (UB); strides
            // are only checked positive, never bounded.
            span += static_cast<size_t>(dims[i] - 1) * static_cast<size_t>(strides[i]);
        }
        return span == elementCount;
    }

    MigratableMemory<uint8_t> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    size_t _elementCount = 0;
};

} // namespace hipdnn_data_sdk::utilities
