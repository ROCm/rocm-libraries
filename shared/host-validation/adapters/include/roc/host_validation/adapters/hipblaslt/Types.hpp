// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "datatype_interface.hpp"

namespace roc::host_validation::hipblaslt_adapter {
inline ScalarType scalarType(hipDataType type) {
    switch (type) {
        case HIP_R_8U:
            return ScalarType::UInt8;
        case HIP_R_8I:
            return ScalarType::Int8;
        case HIP_R_16U:
            return ScalarType::UInt16;
        case HIP_R_16I:
            return ScalarType::Int16;
        case HIP_R_32U:
            return ScalarType::UInt32;
        case HIP_R_32I:
            return ScalarType::Int32;
        case HIP_R_64U:
            return ScalarType::UInt64;
        case HIP_R_64I:
            return ScalarType::Int64;
        case HIP_R_16F:
            return ScalarType::Float16;
        case HIP_R_16BF:
            return ScalarType::BFloat16;
        case HIP_R_32F:
            return ScalarType::Float32;
        case HIP_R_64F:
            return ScalarType::Float64;
        case HIP_C_32F:
            return ScalarType::ComplexFloat32;
        case HIP_C_64F:
            return ScalarType::ComplexFloat64;
        case HIP_R_8F_E4M3:
            return ScalarType::Float8E4M3;
        case HIP_R_8F_E5M2:
            return ScalarType::Float8E5M2;
        case HIP_R_8F_E4M3_FNUZ:
            return ScalarType::Float8E4M3Fnuz;
        case HIP_R_8F_E5M2_FNUZ:
            return ScalarType::Float8E5M2Fnuz;
        case HIP_R_8F_UE8M0:
            return ScalarType::E8M0;
        default:
            break;
    }

    switch (static_cast<int>(type)) {
        case HIP_R_6F_E2M3_EXT:
            return ScalarType::Float6E2M3;
        case HIP_R_6F_E3M2_EXT:
            return ScalarType::Float6E3M2;
        case HIP_R_4F_E2M1_EXT:
            return ScalarType::Float4E2M1;
        case HIP_R_8F_E5M3_EXT:
            return ScalarType::E5M3;
        default:
            throw std::invalid_argument(
                "hipBLASLt data type has no host-validation scalar mapping.");
    }
}

template <typename T>
constexpr ScalarType scalarType() {
    if constexpr (std::is_same_v<T, uint8_t>)
        return ScalarType::UInt8;
    else if constexpr (std::is_same_v<T, hipblasLtInt8>)
        return ScalarType::Int8;
    else if constexpr (std::is_same_v<T, int32_t>)
        return ScalarType::Int32;
    else if constexpr (std::is_same_v<T, float>)
        return ScalarType::Float32;
    else if constexpr (std::is_same_v<T, double>)
        return ScalarType::Float64;
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        return ScalarType::ComplexFloat32;
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        return ScalarType::ComplexFloat64;
    else if constexpr (std::is_same_v<T, hipblasLtHalf>)
        return ScalarType::Float16;
    else if constexpr (std::is_same_v<T, hip_bfloat16>)
        return ScalarType::BFloat16;
    else if constexpr (std::is_same_v<T, hipblaslt_f8>)
        return ScalarType::Float8E4M3;
    else if constexpr (std::is_same_v<T, hipblaslt_bf8>)
        return ScalarType::Float8E5M2;
    else if constexpr (std::is_same_v<T, hipblaslt_f8_fnuz>)
        return ScalarType::Float8E4M3Fnuz;
    else if constexpr (std::is_same_v<T, hipblaslt_bf8_fnuz>)
        return ScalarType::Float8E5M2Fnuz;
    else
        static_assert(!sizeof(T), "C++ type has no host-validation scalar mapping.");
}

template <typename T>
TensorView tensorView(const T* data, size_t elements, Layout layout) {
    constexpr ScalarType type = scalarType<T>();
    static_assert(scalarTypeInfo(type).storageBits == sizeof(T) * 8,
                  "External C++ type does not store one scalar per object.");
    return TensorView(type, std::move(layout), std::as_bytes(std::span<const T>(data, elements)));
}

template <typename T>
MutableTensorView mutableTensorView(T* data, size_t elements, Layout layout) {
    constexpr ScalarType type = scalarType<T>();
    static_assert(scalarTypeInfo(type).storageBits == sizeof(T) * 8,
                  "External C++ type does not store one scalar per object.");
    return MutableTensorView(type, std::move(layout),
                             std::as_writable_bytes(std::span<T>(data, elements)));
}
}  // namespace roc::host_validation::hipblaslt_adapter
