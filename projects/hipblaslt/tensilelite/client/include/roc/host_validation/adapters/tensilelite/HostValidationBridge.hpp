// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private TensileLite adapter.

#include <roc/host_validation/validation.hpp>

#include <Tensile/Activation.hpp>
#include <Tensile/DataTypes.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace TensileLite::Client
{
    inline roc::host_validation::ScalarType toHostValidationScalarType(rocisa::DataType type)
    {
        using roc::host_validation::ScalarType;
        switch(type)
        {
        case rocisa::DataType::Float:
        case rocisa::DataType::XFloat32:
            return ScalarType::Float32;
        case rocisa::DataType::Double:
            return ScalarType::Float64;
        case rocisa::DataType::ComplexFloat:
            return ScalarType::ComplexFloat32;
        case rocisa::DataType::ComplexDouble:
            return ScalarType::ComplexFloat64;
        case rocisa::DataType::Half:
            return ScalarType::Float16;
        case rocisa::DataType::BFloat16:
            return ScalarType::BFloat16;
        case rocisa::DataType::Int8:
            return ScalarType::Int8;
        case rocisa::DataType::Int32:
            return ScalarType::Int32;
        case rocisa::DataType::Int64:
            return ScalarType::Int64;
        case rocisa::DataType::Float8:
            return ScalarType::Float8E4M3;
        case rocisa::DataType::BFloat8:
            return ScalarType::Float8E5M2;
        case rocisa::DataType::Float8_fnuz:
            return ScalarType::Float8E4M3Fnuz;
        case rocisa::DataType::BFloat8_fnuz:
            return ScalarType::Float8E5M2Fnuz;
        case rocisa::DataType::Float6:
            return ScalarType::Float6E2M3;
        case rocisa::DataType::BFloat6:
            return ScalarType::Float6E3M2;
        case rocisa::DataType::Float4:
            return ScalarType::Float4E2M1;
        case rocisa::DataType::E8:
            throw std::invalid_argument(
                "TensileLite E8 treats raw zero as numeric zero; convert it "
                "explicitly before using the OCP E8M0 tensor type.");
        case rocisa::DataType::E5M3:
            return ScalarType::E5M3;
        default:
            throw std::invalid_argument("rocisa data type has no scalar host-validation mapping.");
        }
    }

    inline roc::host_validation::ScalarType
        toHostValidationMxScaleType(rocisa::DataType type)
    {
        using roc::host_validation::ScalarType;
        switch(type)
        {
        case rocisa::DataType::Float8:
            return ScalarType::E4M3;
        case rocisa::DataType::E5M3:
            return ScalarType::E5M3;
        case rocisa::DataType::E8:
        case rocisa::DataType::None:
            return ScalarType::E8M0;
        default:
            throw std::invalid_argument(
                "rocisa data type has no MX scale host-validation mapping.");
        }
    }

    inline roc::host_validation::Activation
        toHostValidationActivation(ActivationType activation, bool gradientApplication = false)
    {
        switch(activation)
        {
        case ActivationType::None:
            return roc::host_validation::Activation::None;
        case ActivationType::Abs:
            return roc::host_validation::Activation::Absolute;
        case ActivationType::Clippedrelu:
            return roc::host_validation::Activation::ClippedRelu;
        case ActivationType::Relu:
            return roc::host_validation::Activation::Relu;
        case ActivationType::Gelu:
            return roc::host_validation::Activation::Gelu;
        case ActivationType::Geluscaling:
            return roc::host_validation::Activation::GeluScaling;
        case ActivationType::Leakyrelu:
            return roc::host_validation::Activation::LeakyRelu;
        case ActivationType::Sigmoid:
            return roc::host_validation::Activation::Sigmoid;
        case ActivationType::Tanh:
            return roc::host_validation::Activation::Tanh;
        case ActivationType::DGelu:
            return gradientApplication ? roc::host_validation::Activation::Gelu
                                       : roc::host_validation::Activation::GeluDerivative;
        case ActivationType::DRelu:
            return gradientApplication ? roc::host_validation::Activation::Relu
                                       : roc::host_validation::Activation::ReluDerivative;
        case ActivationType::Silu:
            return roc::host_validation::Activation::Silu;
        case ActivationType::Swish:
            return roc::host_validation::Activation::Swish;
        case ActivationType::Clamp:
            return roc::host_validation::Activation::Clamp;
        default:
            throw std::invalid_argument("Activation has no runtime host-validation mapping.");
        }
    }

    inline roc::host_validation::ComparisonOptions
        validationComparisonOptions(rocisa::DataType type, double threshold)
    {
        using namespace roc::host_validation;

        const ScalarType            scalarType = toHostValidationScalarType(type);
        const std::optional<double> toleranceOverride
            = scalarType == ScalarType::Float32 && threshold > 0.0
                  ? std::optional<double>(threshold)
                  : std::nullopt;
        return defaultComparisonOptions(scalarType, toleranceOverride);
    }

    inline roc::host_validation::ComparisonResult
        compareHostBuffers(rocisa::DataType                               type,
                           const void*                                    observed,
                           const void*                                    expected,
                           size_t                                         storageElements,
                           const roc::host_validation::Layout&            layout,
                           const roc::host_validation::ComparisonOptions& options)
    {
        const auto   scalarType = toHostValidationScalarType(type);
        const size_t bytes      = roc::host_validation::storageBytesForLayout(scalarType, layout);
        if(observed == nullptr && bytes != 0)
            throw std::invalid_argument("TensileLite observed comparison buffer is null.");
        if(expected == nullptr && bytes != 0)
            throw std::invalid_argument("TensileLite expected comparison buffer is null.");
        (void)storageElements;
        return roc::host_validation::compare(
            roc::host_validation::Tensor(
                scalarType,
                layout,
                std::span<const std::byte>(static_cast<const std::byte*>(observed), bytes)),
            roc::host_validation::Tensor(
                scalarType,
                layout,
                std::span<const std::byte>(static_cast<const std::byte*>(expected), bytes)),
            options);
    }
} // namespace TensileLite::Client
