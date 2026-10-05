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
#include <hipdnn_data_sdk/utilities/PackedElementTraits.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_data_sdk::utilities
{

/// Device-side packed storage for a sub-byte float tensor: `BitsPerElement`-wide
/// codes packed LSB-first into a dense byte buffer, matching the packed device
/// layout for the corresponding `HIP_R_*` type. `Tensor<T>` instead stores one
/// code per *byte* (unpacked) for element-wise CPU access.
///
/// Pair the two: this type for the GPU-side bundle, `Tensor<T>` for the
/// CPU-reference bundle. Filled with the same (seed, min, max) they agree
/// value-for-value, because randomization mirrors `Tensor<T>::fillWithRandomValues`
/// exactly, each draw packed at the offset its coordinates reach through the
/// strides — so the agreement holds for non-row-major layouts too.
///
/// Only dense (packed-stride) layouts are supported. Element-wise host access is
/// not provided.
template <typename T, size_t BitsPerElement>
class PackedSubByteTensor : public ITensor
{
    static_assert(PackedElementTraits<T>::BITS_PER_ELEMENT == BitsPerElement,
                  "PackedSubByteTensor<T, BitsPerElement>: BitsPerElement must match "
                  "PackedElementTraits<T>::BITS_PER_ELEMENT");

public:
    PackedSubByteTensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
        : _dims(dims)
        , _strides(strides)
    {
        if(dims.size() != strides.size())
        {
            throw std::invalid_argument(std::string(PackedElementTraits<T>::TYPE_NAME)
                                        + ": dims and strides size mismatch");
        }
        validateAllPositive(dims, "dimension");
        validateAllPositive(strides, "stride");
        _elementCount = computeElementCount(dims);
        if(!isDensePacked(dims, strides, _elementCount))
        {
            throw std::invalid_argument(
                std::string(PackedElementTraits<T>::TYPE_NAME)
                + " requires dense (contiguous) strides for sub-byte packing");
        }
        _memory = MigratableMemory<uint8_t>(packedByteCount());
        if(packedByteCount() > 0 && (_elementCount * BitsPerElement) % BITS_PER_BYTE != 0)
        {
            // The final byte's unused high bits belong to no element: nothing reads
            // them, and packElementAt() never writes them. Clear them once so the
            // buffer is reproducible byte-for-byte.
            static_cast<uint8_t*>(_memory.hostData())[packedByteCount() - 1] = 0;
        }
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
    // A sub-byte element has no integer byte size, and the common byte-size idiom
    // elementCount() * elementSize() would report a misleading value (multiple
    // values share a byte). Throw rather than hand back a misleading value; size
    // the buffer from the packed byte count instead.
    size_t elementSize() const override
    {
        throw std::logic_error(std::string(PackedElementTraits<T>::TYPE_NAME)
                               + ": elementSize() is undefined for a "
                               + std::to_string(BitsPerElement)
                               + "-bit packed type; derive byte size from the packed "
                                 "buffer, not elementCount() * elementSize()");
    }

    void* hostDataOffsetFromIndex(int64_t /*index*/) override
    {
        throw std::logic_error(std::string(PackedElementTraits<T>::TYPE_NAME)
                               + ": per-element host access is not supported (values are "
                               + std::to_string(BitsPerElement) + "-bit packed)");
    }
    const void* hostDataOffsetFromIndex(int64_t /*index*/) const override
    {
        throw std::logic_error(std::string(PackedElementTraits<T>::TYPE_NAME)
                               + ": per-element host access is not supported (values are "
                               + std::to_string(BitsPerElement) + "-bit packed)");
    }

    void fillTensorWithValue(float value) override
    {
        const uint8_t code = PackedElementTraits<T>::encode(value);
        auto* host = static_cast<uint8_t*>(_memory.hostData());
        if constexpr(BitsPerElement * 2 == BITS_PER_BYTE)
        {
            // Two codes tile one byte exactly (BitsPerElement == 4): every packed
            // byte holds the same repeated code in both halves, so memset each
            // byte once instead of writing element-by-element. Matches
            // PackedFp4Tensor's original fast path bit-for-bit.
            const auto packed = static_cast<uint8_t>(code | (code << BitsPerElement));
            for(size_t byte = 0; byte < packedByteCount(); ++byte)
            {
                host[byte] = packed;
            }
        }
        else
        {
            // No whole-byte repetition period for this BitsPerElement (e.g. 6-bit
            // codes repeat every 3 bytes) — write element by element.
            for(size_t i = 0; i < _elementCount; ++i)
            {
                packElementAt(host, i, code);
            }
        }
        _memory.markHostModified();
    }

    // Must match Tensor<T>::fillWithRandomValues draw-for-draw, or the packed and
    // unpacked bundles disagree silently: bounds rounded through T first (as
    // TensorBase does), and each draw placed via elementSlot rather than raw index i.
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
            packElementAt(
                host, elementSlot(i), PackedElementTraits<T>::encode(distribution(generator)));
        }
        _memory.markHostModified();
    }

    void fillWithSentinelValue() override
    {
        // Neither FP4 nor FP6 has a NaN, so TensorBase::fillWithSentinelValue uses
        // max() for these types; match it so packed and unpacked agree.
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
    static constexpr size_t BITS_PER_BYTE = 8;
    static constexpr uint8_t ELEMENT_CODE_MASK = PackedElementTraits<T>::CODE_MASK;
    // Largest count for which packedByteCount()'s `count * BitsPerElement + 7`
    // cannot wrap.
    static constexpr size_t MAX_ELEMENT_COUNT
        = (std::numeric_limits<size_t>::max() - (BITS_PER_BYTE - 1)) / BitsPerElement;

    // Size of the buffer, exactly: ceil(elementCount * BitsPerElement / 8).
    size_t packedByteCount() const
    {
        return (_elementCount * BitsPerElement + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    }

    // Writes the BitsPerElement bits of `code` at bit offset BitsPerElement * index,
    // LSB first, leaving the neighbouring elements' bits untouched.
    void packElementAt(uint8_t* host, size_t index, uint8_t code) const
    {
        const size_t bitOffset = index * BitsPerElement;
        const size_t byteIndex = bitOffset / BITS_PER_BYTE;
        const auto bitIndex = static_cast<unsigned>(bitOffset % BITS_PER_BYTE);

        const auto lowMask = static_cast<uint8_t>(ELEMENT_CODE_MASK << bitIndex);
        const auto lowBits = static_cast<uint8_t>(code << bitIndex);
        host[byteIndex] = static_cast<uint8_t>((host[byteIndex] & ~lowMask) | lowBits);

        // Guarded because the final element can end flush with the buffer, making
        // byteIndex + 1 one past the bitstream.
        if(bitIndex > BITS_PER_BYTE - BitsPerElement)
        {
            const size_t highShift = BITS_PER_BYTE - bitIndex;
            const auto highMask = static_cast<uint8_t>(ELEMENT_CODE_MASK >> highShift);
            const auto highBits = static_cast<uint8_t>(code >> highShift);
            host[byteIndex + 1]
                = static_cast<uint8_t>((host[byteIndex + 1] & ~highMask) | highBits);
        }
    }

    // Bit-packed slot holding logical element `index`: the offset its coordinates
    // reach through the strides. Equals `index` exactly when the strides are
    // row-major.
    size_t elementSlot(size_t index) const
    {
        size_t remaining = index;
        int64_t offset = 0;
        for(size_t axis = _dims.size(); axis-- > 0;)
        {
            const auto extent = static_cast<size_t>(_dims[axis]);
            offset += static_cast<int64_t>(remaining % extent) * _strides[axis];
            remaining /= extent;
        }
        return static_cast<size_t>(offset);
    }

    static void validateAllPositive(const std::vector<int64_t>& values, const char* valueName)
    {
        for(const auto value : values)
        {
            if(value <= 0)
            {
                throw std::invalid_argument(std::string(PackedElementTraits<T>::TYPE_NAME) + ": "
                                            + valueName + " must be positive");
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
                    std::string(PackedElementTraits<T>::TYPE_NAME)
                    + ": element count exceeds the addressable packed size");
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
