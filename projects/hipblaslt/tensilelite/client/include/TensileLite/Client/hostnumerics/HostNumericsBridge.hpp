// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TensileLite client-private adapter.

#include <roc/hostnumerics/comparison.hpp>
#include <roc/hostnumerics/epilogue.hpp>
#include <roc/hostnumerics/operation_types.hpp>
#include <roc/hostnumerics/tensor.hpp>

#include <Tensile/Activation.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/TensorDescriptor.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace TensileLite::Client
{
    template <typename Integer>
    inline ptrdiff_t checkedHostNumericsPtrdiff(Integer value)
    {
        if(!std::in_range<ptrdiff_t>(value))
            throw std::overflow_error("TensileLite adapter offset exceeds ptrdiff_t.");
        return static_cast<ptrdiff_t>(value);
    }

    inline roc::hostnumerics::Layout hostnumericsLayout(TensorDescriptor const& descriptor)
    {
        std::vector<ptrdiff_t> strides;
        strides.reserve(descriptor.strides().size());
        for(const size_t stride : descriptor.strides())
            strides.push_back(checkedHostNumericsPtrdiff(stride));
        return roc::hostnumerics::Layout(roc::hostnumerics::Shape(descriptor.sizes()),
                                            std::move(strides));
    }

    inline roc::hostnumerics::ScalarType toHostNumericsScalarType(rocisa::DataType type)
    {
        using roc::hostnumerics::ScalarType;
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
            return ScalarType::E8M0Zero;
        case rocisa::DataType::E5M3:
            return ScalarType::E5M3;
        default:
            throw std::invalid_argument("rocisa data type has no scalar HostNumerics mapping.");
        }
    }

    inline roc::hostnumerics::ScalarType
        toHostNumericsMxScaleType(rocisa::DataType type)
    {
        using roc::hostnumerics::ScalarType;
        switch(type)
        {
        case rocisa::DataType::Float8:
            return ScalarType::E4M3;
        case rocisa::DataType::E5M3:
            return ScalarType::E5M3;
        case rocisa::DataType::E8:
        case rocisa::DataType::None:
            return ScalarType::E8M0Zero;
        default:
            throw std::invalid_argument(
                "rocisa data type has no MX scale HostNumerics mapping.");
        }
    }

    inline roc::hostnumerics::ActivationFunction
        toHostNumericsActivation(ActivationType activation, bool gradientApplication = false)
    {
        switch(activation)
        {
        case ActivationType::None:
            return roc::hostnumerics::IdentityActivation{};
        case ActivationType::Abs:
            return roc::hostnumerics::AbsoluteActivation{};
        case ActivationType::Clippedrelu:
            return roc::hostnumerics::ClippedReluActivation{};
        case ActivationType::Relu:
            return roc::hostnumerics::ReluActivation{};
        case ActivationType::Gelu:
            return roc::hostnumerics::GeluActivation{};
        case ActivationType::Geluscaling:
            return roc::hostnumerics::GeluScalingActivation{};
        case ActivationType::Leakyrelu:
            return roc::hostnumerics::LeakyReluActivation{};
        case ActivationType::Sigmoid:
            return roc::hostnumerics::SigmoidActivation{};
        case ActivationType::Tanh:
            return roc::hostnumerics::TanhActivation{};
        case ActivationType::DGelu:
            return gradientApplication ? roc::hostnumerics::ActivationFunction(
                                             roc::hostnumerics::GeluActivation{})
                                       : roc::hostnumerics::ActivationFunction(
                                             roc::hostnumerics::GeluDerivativeActivation{});
        case ActivationType::DRelu:
            return gradientApplication ? roc::hostnumerics::ActivationFunction(
                                             roc::hostnumerics::ReluActivation{})
                                       : roc::hostnumerics::ActivationFunction(
                                             roc::hostnumerics::ReluDerivativeActivation{});
        case ActivationType::Silu:
            return roc::hostnumerics::SiluActivation{};
        case ActivationType::Swish:
            return roc::hostnumerics::SwishActivation{};
        case ActivationType::Clamp:
            return roc::hostnumerics::ClampActivation{};
        default:
            throw std::invalid_argument("Activation has no runtime HostNumerics mapping.");
        }
    }

    inline roc::hostnumerics::ComparisonOptions
        validationComparisonOptions(rocisa::DataType type, double threshold)
    {
        using roc::hostnumerics::ComparisonOptions;
        using roc::hostnumerics::ScalarType;

        const ScalarType scalarType = toHostNumericsScalarType(type);
        const double defaultTolerance = [&] {
            switch(scalarType)
            {
            case ScalarType::Float16:
                return 0.01;
            case ScalarType::BFloat16:
                return 0.1;
            case ScalarType::Float8E4M3:
            case ScalarType::Float8E4M3Fnuz:
                return 0.125;
            case ScalarType::Float8E5M2:
            case ScalarType::Float8E5M2Fnuz:
                return 0.25;
            case ScalarType::Float32:
            case ScalarType::ComplexFloat32:
                return 0.0002;
            case ScalarType::Float64:
            case ScalarType::ComplexFloat64:
                return 1e-12;
            case ScalarType::Boolean:
            case ScalarType::UInt8:
            case ScalarType::Int8:
            case ScalarType::UInt16:
            case ScalarType::Int16:
            case ScalarType::UInt32:
            case ScalarType::Int32:
            case ScalarType::UInt64:
            case ScalarType::Int64:
            case ScalarType::Float6E2M3:
            case ScalarType::Float6E3M2:
            case ScalarType::Float4E2M1:
            case ScalarType::Int4:
            case ScalarType::E8M0:
            case ScalarType::E8M0Zero:
            case ScalarType::E5M3:
            case ScalarType::E4M3:
                return 0.0;
            case ScalarType::Count:
                break;
            }
            throw std::invalid_argument("Invalid HostNumerics comparison scalar type.");
        }();
        const double tolerance
            = scalarType == ScalarType::Float32 && threshold > 0.0 ? threshold : defaultTolerance;

        ComparisonOptions options;
        // Preserve the scale of TensileLite's historical symmetric check while
        // expressing it through the NumPy allclose model. Near zero the old +1
        // term was an absolute tolerance; for similar nonzero operands its two
        // magnitude terms correspond to roughly twice that relative tolerance.
        options.absoluteTolerance = tolerance;
        options.relativeTolerance = 2.0 * tolerance;
        return options;
    }

    inline roc::hostnumerics::OutputSelection
        referenceOutputSelection(TensorDescriptor const& descriptor,
                                 size_t                  elementsToValidate)
    {
        return roc::hostnumerics::OutputSelection::primeStride(
            descriptor.totalLogicalElements(),
            descriptor.totalAllocatedElements(),
            elementsToValidate,
            roc::hostnumerics::IndexOrder::FirstDimensionFastest);
    }

} // namespace TensileLite::Client
