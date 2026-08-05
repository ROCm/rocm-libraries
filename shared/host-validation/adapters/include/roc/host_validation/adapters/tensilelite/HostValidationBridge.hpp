// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_validation/validation.hpp>

#include <Tensile/Activation.hpp>
#include <Tensile/DataTypes.hpp>

#include <stdexcept>

namespace TensileLite::Client
{
inline roc::host_validation::ScalarType toHostValidationScalarType(rocisa::DataType type) {
    using roc::host_validation::ScalarType;
    switch (type) {
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

inline roc::host_validation::Activation toHostValidationActivation(ActivationType activation) {
    switch (activation) {
        case ActivationType::None:
            return roc::host_validation::Activation::None;
        case ActivationType::Relu:
            return roc::host_validation::Activation::Relu;
        case ActivationType::Gelu:
            return roc::host_validation::Activation::Gelu;
        case ActivationType::Silu:
        case ActivationType::Swish:
            return roc::host_validation::Activation::Silu;
        case ActivationType::Clamp:
            return roc::host_validation::Activation::Clamp;
        default:
            throw std::invalid_argument("Activation has no runtime host-validation mapping.");
    }
}
}  // namespace TensileLite::Client
