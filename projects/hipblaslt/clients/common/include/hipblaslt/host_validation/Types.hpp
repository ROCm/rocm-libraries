// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <complex>
#include <cstdint>
#include <cstring>
#include <hip/hip_bfloat16.h>
#include <hip/library_types.h>
#include <hipblaslt/hipblaslt-export.h>
#include <hipblaslt/hipblaslt-types.h>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hipblaslt::host_validation
{
    using namespace ::roc::host_validation;

    // ScalarType is the conversion hub for host validation. Each external type
    // system maps to it once; runtime dispatch should not add pairwise
    // hipDataType-to-C++-type mappings.
    inline std::optional<ScalarType> tryScalarType(hipDataType type) noexcept
    {
        // hipBLASLt also defines integer-valued extension constants that are not
        // members of hipDataType (for example HIP_R_8F_E5M3_EXT). Normalize the
        // discriminant once so enum values and extension values share one table.
        switch(static_cast<int>(type))
        {
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
        case HIP_R_6F_E2M3_EXT:
            return ScalarType::Float6E2M3;
        case HIP_R_6F_E3M2_EXT:
            return ScalarType::Float6E3M2;
        case HIP_R_4F_E2M1_EXT:
            return ScalarType::Float4E2M1;
        case HIP_R_8F_E5M3_EXT:
            return ScalarType::E5M3;
        default:
            return std::nullopt;
        }
    }

    inline ScalarType scalarType(hipDataType type)
    {
        if(const auto mapped = tryScalarType(type))
            return *mapped;
        throw std::invalid_argument("hipBLASLt data type has no host-validation scalar mapping.");
    }

    template <typename T>
    constexpr ScalarType scalarType()
    {
        using Type = std::remove_cv_t<T>;

        if constexpr(std::is_same_v<Type, uint8_t> || std::is_same_v<Type, int8_t>
                     || std::is_same_v<Type, int32_t> || std::is_same_v<Type, float>
                     || std::is_same_v<Type, double> || std::is_same_v<Type, std::complex<float>>
                     || std::is_same_v<Type, std::complex<double>>)
            return nativeScalarType<Type>;
        else if constexpr(std::is_same_v<Type, char>)
            return std::is_signed_v<char> ? ScalarType::Int8 : ScalarType::UInt8;
        else if constexpr(std::is_same_v<Type, hipblasLtHalf>)
            return ScalarType::Float16;
        else if constexpr(std::is_same_v<Type, hip_bfloat16>)
            return ScalarType::BFloat16;
        else if constexpr(std::is_same_v<Type, hipblaslt_f8>)
            return ScalarType::Float8E4M3;
        else if constexpr(std::is_same_v<Type, hipblaslt_bf8>)
            return ScalarType::Float8E5M2;
        else if constexpr(std::is_same_v<Type, hipblaslt_f8_fnuz>)
            return ScalarType::Float8E4M3Fnuz;
        else if constexpr(std::is_same_v<Type, hipblaslt_bf8_fnuz>)
            return ScalarType::Float8E5M2Fnuz;
        else if constexpr(std::is_same_v<Type, hipblaslt_e8>)
            return ScalarType::E8M0;
        else
            static_assert(!sizeof(Type), "C++ type has no host-validation scalar mapping.");
    }

    template <typename T>
    ::roc::host_validation::Tensor tensorFromStorage(const T* data, size_t elements, Layout layout)
    {
        constexpr ScalarType type = scalarType<T>();
        static_assert(scalarTypeInfo(type).storageBits == sizeof(T) * 8,
                      "External C++ type does not store one scalar per object.");
        return ::roc::host_validation::Tensor(
            type, std::move(layout), std::as_bytes(std::span<const T>(data, elements)));
    }

    template <typename T>
    ::roc::host_validation::Tensor tensorFromMutableStorage(T* data, size_t elements, Layout layout)
    {
        constexpr ScalarType type = scalarType<T>();
        static_assert(scalarTypeInfo(type).storageBits == sizeof(T) * 8,
                      "External C++ type does not store one scalar per object.");
        return ::roc::host_validation::Tensor(
            type, std::move(layout), std::as_writable_bytes(std::span<T>(data, elements)));
    }

    inline ::roc::host_validation::Tensor
        tensorFromMutableStorage(void* data, size_t storageBytes, hipDataType type, Layout layout)
    {
        return ::roc::host_validation::Tensor(
            scalarType(type),
            std::move(layout),
            std::span<std::byte>(static_cast<std::byte*>(data), storageBytes));
    }

    inline ::roc::host_validation::Tensor
        tensorFromMutableStorage(void* data, ScalarType type, Layout layout)
    {
        const size_t storageBytes = storageBytesForLayout(type, layout);
        return ::roc::host_validation::Tensor(
            type,
            std::move(layout),
            std::span<std::byte>(static_cast<std::byte*>(data), storageBytes));
    }

    template <typename T>
    void copyTensorStorageTo(T* data, size_t elements, const ::roc::host_validation::Tensor& tensor)
    {
        const size_t bytes = elements * sizeof(T);
        if(tensor.storage().size() > bytes)
            throw std::invalid_argument("Tensor storage exceeds destination capacity.");
        if(!tensor.storage().empty())
            std::memcpy(static_cast<void*>(data),
                        tensor.storage().data(),
                        tensor.storage().size());
    }
} // namespace hipblaslt::host_validation
