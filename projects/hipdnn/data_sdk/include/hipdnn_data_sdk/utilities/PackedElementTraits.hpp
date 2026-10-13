// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include <hipdnn_data_sdk/types.hpp>

namespace hipdnn_data_sdk::utilities
{

/// Per-type contract used by `PackedSubByteTensor<T, BitsPerElement>` to pack/encode
/// values without knowing which sub-byte float encoding `T` is. Deliberately declared
/// but not defined for the primary template: instantiating this for an unsupported `T`
/// fails to compile with an "incomplete type" error rather than silently doing nothing.
template <typename T>
struct PackedElementTraits;

template <>
struct PackedElementTraits<types::fp4_e2m1>
{
    static constexpr size_t BITS_PER_ELEMENT = 4;
    static constexpr uint8_t CODE_MASK = 0x0F;
    static constexpr const char* TYPE_NAME = "PackedFp4Tensor";
    static uint8_t encode(float value)
    {
        return static_cast<uint8_t>(types::fp4_e2m1(value).data & CODE_MASK);
    }
};

template <>
struct PackedElementTraits<types::fp6_e2m3>
{
    static constexpr size_t BITS_PER_ELEMENT = 6;
    static constexpr uint8_t CODE_MASK = 0x3F;
    static constexpr const char* TYPE_NAME = "PackedFp6Tensor";
    static uint8_t encode(float value)
    {
        return static_cast<uint8_t>(types::fp6_e2m3(value).data & CODE_MASK);
    }
};

template <>
struct PackedElementTraits<types::fp6_e3m2>
{
    static constexpr size_t BITS_PER_ELEMENT = 6;
    static constexpr uint8_t CODE_MASK = 0x3F;
    static constexpr const char* TYPE_NAME = "PackedFp6Tensor";
    static uint8_t encode(float value)
    {
        return static_cast<uint8_t>(types::fp6_e3m2(value).data & CODE_MASK);
    }
};

} // namespace hipdnn_data_sdk::utilities
