// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_data_sdk::utilities
{

/// Device-side packed storage for an FP4 (E2M1) tensor: two 4-bit values per
/// byte (low nibble = even logical index, high nibble = odd), matching the
/// packed `HIP_R_4F_E2M1` device layout. This is the representation a real GPU
/// kernel expects, unlike `Tensor<fp4_e2m1>` which stores one 4-bit code per
/// byte (unpacked) for element-wise CPU access.
///
/// Use this for the GPU-side bundle of an FP4 input; keep `Tensor<fp4_e2m1>` for
/// the CPU-reference bundle. Filled with the same (seed, min, max) the two agree
/// value-for-value, because randomization here mirrors
/// `Tensor<fp4_e2m1>::fillWithRandomValues` exactly: the same RNG sequence is
/// generated in linear index order and packed into nibbles.
///
/// Only dense (packed-stride) layouts are supported. Element-wise host access
/// (`operator()`, iteration dereference) is not provided; this type is a buffer
/// holder for the device variant pack.
class PackedFp4Tensor : public ITensor
{
public:
    PackedFp4Tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
        : _dims(dims)
        , _strides(strides)
        , _elementCount(computeElementCount(dims))
    {
        if(dims.size() != strides.size())
        {
            throw std::invalid_argument("PackedFp4Tensor: dims and strides size mismatch");
        }
        validateAllPositive(dims, "dimension");
        validateAllPositive(strides, "stride");
        if(!isDensePacked(dims, strides))
        {
            throw std::invalid_argument(
                "PackedFp4Tensor requires dense (contiguous) strides for sub-byte packing");
        }
        // Two FP4 values per byte; round up so an odd element count keeps its
        // final low nibble.
        _memory = MigratableMemory<uint8_t>((_elementCount + 1) / 2);
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
    // A 4-bit element has no integer byte size, and the common byte-size idiom
    // elementCount() * elementSize() would report double this buffer's real size
    // (two values share a byte). Throw rather than hand back a misleading value;
    // size the buffer from the packed byte count instead.
    size_t elementSize() const override
    {
        throw std::logic_error("PackedFp4Tensor: elementSize() is undefined for a 4-bit "
                               "packed type (two values per byte); derive byte size from "
                               "the packed buffer, not elementCount() * elementSize()");
    }

    void* hostDataOffsetFromIndex(int64_t /*index*/) override
    {
        throw std::logic_error("PackedFp4Tensor: per-element host access is not supported "
                               "(values are 4-bit packed two per byte)");
    }
    const void* hostDataOffsetFromIndex(int64_t /*index*/) const override
    {
        throw std::logic_error("PackedFp4Tensor: per-element host access is not supported "
                               "(values are 4-bit packed two per byte)");
    }

    void fillTensorWithValue(float value) override
    {
        const uint8_t nibble = nibbleFromFloat(value);
        const auto packed = static_cast<uint8_t>(nibble | (nibble << 4));
        auto* host = static_cast<uint8_t*>(_memory.hostData());
        for(size_t byte = 0; byte < packedByteCount(); ++byte)
        {
            host[byte] = packed;
        }
        _memory.markHostModified();
    }

    // Mirrors Tensor<fp4_e2m1>::fillWithRandomValues: identical RNG, distribution,
    // and linear traversal order, so a paired unpacked tensor (same seed) holds
    // the same logical values.
    void fillTensorWithRandomValues(float min,
                                    float max,
                                    unsigned int seed = std::random_device{}()) override
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(min, max);

        auto* host = static_cast<uint8_t*>(_memory.hostData());
        for(size_t byte = 0; byte < packedByteCount(); ++byte)
        {
            host[byte] = 0;
        }
        for(size_t i = 0; i < _elementCount; ++i)
        {
            const uint8_t nibble = nibbleFromFloat(distribution(generator));
            if((i % 2) == 0)
            {
                host[i / 2] = nibble;
            }
            else
            {
                host[i / 2] = static_cast<uint8_t>(host[i / 2] | (nibble << 4));
            }
        }
        _memory.markHostModified();
    }

    void fillWithSentinelValue() override
    {
        // FP4 E2M1 has no NaN; use the max-magnitude code in both nibbles.
        fillTensorWithValue(static_cast<float>(std::numeric_limits<types::fp4_e2m1>::max()));
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
    size_t packedByteCount() const
    {
        return (_elementCount + 1) / 2;
    }

    static uint8_t nibbleFromFloat(float value)
    {
        return static_cast<uint8_t>(types::fp4_e2m1(value).data & 0x0F);
    }

    static void validateAllPositive(const std::vector<int64_t>& values, const char* valueName)
    {
        for(const auto value : values)
        {
            if(value <= 0)
            {
                throw std::invalid_argument(std::string("PackedFp4Tensor: ") + valueName
                                            + " must be positive");
            }
        }
    }

    static size_t computeElementCount(const std::vector<int64_t>& dims)
    {
        size_t count = 1;
        for(const auto d : dims)
        {
            count *= static_cast<size_t>(d);
        }
        return dims.empty() ? 0 : count;
    }

    // Dense == the element offsets span exactly [0, elementCount): largest
    // stride * (its dim - 1) + 1 equals the product of dims.
    static bool isDensePacked(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
    {
        size_t span = 1;
        for(size_t i = 0; i < dims.size(); ++i)
        {
            span += static_cast<size_t>((dims[i] - 1) * strides[i]);
        }
        return span == computeElementCount(dims);
    }

    MigratableMemory<uint8_t> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    size_t _elementCount;
};

} // namespace hipdnn_data_sdk::utilities
