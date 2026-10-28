// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TensileLite client-private adapter.

#include <roc/hostnumerics/comparison.hpp>
#include <roc/hostnumerics/detail/prime.hpp>
#include <roc/hostnumerics/detail/shape_access.hpp>
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
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace TensileLite::Client
{
    namespace hostnumerics::detail
    {
        using roc::hostnumerics::detail::Expected;
        using roc::hostnumerics::detail::Unexpected;
        enum class DescriptorError
        {
            StrideRankMismatch,
            OffsetOutOfRange,
            DimensionOutOfRange,
            OffsetMultiplyOverflow,
            OffsetAddOverflow,
            BeforeBatchStorage,
            OffsetOverflow,
            NullStorage,
            ElementCountOverflow,
        };
        struct DescriptorFailure
        {
            std::variant<DescriptorError,
                         roc::hostnumerics::detail::LayoutFailure,
                         roc::hostnumerics::detail::StorageFailure>
                cause;
            DescriptorFailure(DescriptorError error) noexcept
                : cause(error)
            {
            }
            DescriptorFailure(roc::hostnumerics::detail::LayoutFailure error) noexcept
                : cause(error)
            {
            }
            DescriptorFailure(roc::hostnumerics::detail::StorageFailure error) noexcept
                : cause(error)
            {
            }
        };
        inline constexpr std::string_view message(DescriptorError error) noexcept
        {
            switch(error)
            {
            case DescriptorError::StrideRankMismatch:
                return "TensileLite descriptor stride count does not match its rank.";
            case DescriptorError::OffsetOutOfRange:
                return "TensileLite adapter offset exceeds ptrdiff_t.";
            case DescriptorError::DimensionOutOfRange:
                return "TensileLite adapter dimension is outside its descriptor.";
            case DescriptorError::OffsetMultiplyOverflow:
                return "TensileLite adapter offset multiplication overflow.";
            case DescriptorError::OffsetAddOverflow:
                return "TensileLite adapter offset addition overflow.";
            case DescriptorError::BeforeBatchStorage:
                return "TensileLite adapter layout addresses before its batch storage base.";
            case DescriptorError::OffsetOverflow:
                return "TensileLite adapter offset overflow.";
            case DescriptorError::NullStorage:
                return "Nonempty tensor storage is null.";
            case DescriptorError::ElementCountOverflow:
                return "Tensor shape element count exceeds size_t.";
            }
            return "Unknown descriptor failure.";
        }
        inline std::string message(const DescriptorFailure& failure)
        {
            return std::visit(
                [](const auto& error) -> std::string {
                    if constexpr(std::is_same_v<std::decay_t<decltype(error)>,
                                                roc::hostnumerics::detail::LayoutFailure>)
                        return error.message();
                    else
                        return std::string(message(error));
                },
                failure.cause);
        }
    }

    template <typename Integer>
    inline ptrdiff_t checkedHostNumericsPtrdiff(Integer value)
    {
        if(!std::in_range<ptrdiff_t>(value))
            throw std::runtime_error("TensileLite adapter offset exceeds ptrdiff_t.");
        return static_cast<ptrdiff_t>(value);
    }

    inline hostnumerics::detail::Expected<roc::hostnumerics::Layout,
                                          hostnumerics::detail::DescriptorFailure>
        getHostnumericsLayout(TensorDescriptor const& descriptor)
    {
        if(descriptor.sizes().size() != descriptor.strides().size())
        {
            return roc::hostnumerics::detail::Unexpected(hostnumerics::detail::DescriptorFailure(
                hostnumerics::detail::DescriptorError::StrideRankMismatch));
        }
        if(!roc::hostnumerics::detail::ShapeAccess::getElementCount(descriptor.sizes()))
        {
            return roc::hostnumerics::detail::Unexpected(hostnumerics::detail::DescriptorFailure(
                hostnumerics::detail::DescriptorError::ElementCountOverflow));
        }
        std::vector<ptrdiff_t> strides;
        strides.reserve(descriptor.strides().size());
        for(const size_t stride : descriptor.strides())
        {
            if(!std::in_range<ptrdiff_t>(stride))
            {
                return roc::hostnumerics::detail::Unexpected(
                    hostnumerics::detail::DescriptorFailure(
                        hostnumerics::detail::DescriptorError::OffsetOutOfRange));
            }
            strides.push_back(static_cast<ptrdiff_t>(stride));
        }
        return roc::hostnumerics::Layout(
            roc::hostnumerics::Shape(descriptor.sizes()), std::move(strides), 0);
    }

    inline roc::hostnumerics::Layout hostnumericsLayout(TensorDescriptor const& descriptor)
    {
        auto result = getHostnumericsLayout(descriptor);
        if(!result)
            throw std::runtime_error(std::string(hostnumerics::detail::message(result.error())));
        return std::move(*result);
    }

    inline std::optional<roc::hostnumerics::ScalarType>
        getHostNumericsScalarType(rocisa::DataType type)
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
            return std::nullopt;
        }
    }

    inline roc::hostnumerics::ScalarType toHostNumericsScalarType(rocisa::DataType type)
    {
        auto result = getHostNumericsScalarType(type);
        if(!result)
            throw std::runtime_error("rocisa data type has no scalar HostNumerics mapping.");
        return std::move(*result);
    }

    inline std::optional<roc::hostnumerics::ScalarType>
        getHostNumericsMxScaleType(rocisa::DataType type)
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
            return std::nullopt;
        }
    }

    inline roc::hostnumerics::ScalarType toHostNumericsMxScaleType(rocisa::DataType type)
    {
        auto result = getHostNumericsMxScaleType(type);
        if(!result)
            throw std::runtime_error("rocisa data type has no MX scale HostNumerics mapping.");
        return std::move(*result);
    }

    inline std::optional<roc::hostnumerics::ActivationFunction>
        getHostNumericsActivation(ActivationType activation, bool gradientApplication)
    {
        switch(activation)
        {
        case ActivationType::None:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::IdentityActivation{});
        case ActivationType::Abs:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::AbsoluteActivation{});
        case ActivationType::Clippedrelu:
            return roc::hostnumerics::ActivationFunction(
                roc::hostnumerics::ClippedReluActivation{});
        case ActivationType::Relu:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::ReluActivation{});
        case ActivationType::Gelu:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::GeluActivation{});
        case ActivationType::Geluscaling:
            return roc::hostnumerics::ActivationFunction(
                roc::hostnumerics::GeluScalingActivation{});
        case ActivationType::Leakyrelu:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::LeakyReluActivation{});
        case ActivationType::Sigmoid:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::SigmoidActivation{});
        case ActivationType::Tanh:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::TanhActivation{});
        case ActivationType::DGelu:
            return gradientApplication
                       ? roc::hostnumerics::ActivationFunction(roc::hostnumerics::GeluActivation{})
                       : roc::hostnumerics::ActivationFunction(
                           roc::hostnumerics::GeluDerivativeActivation{});
        case ActivationType::DRelu:
            return gradientApplication
                       ? roc::hostnumerics::ActivationFunction(roc::hostnumerics::ReluActivation{})
                       : roc::hostnumerics::ActivationFunction(
                           roc::hostnumerics::ReluDerivativeActivation{});
        case ActivationType::Silu:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::SiluActivation{});
        case ActivationType::Swish:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::SwishActivation{});
        case ActivationType::Clamp:
            return roc::hostnumerics::ActivationFunction(roc::hostnumerics::ClampActivation{});
        default:
            return std::nullopt;
        }
    }

    inline roc::hostnumerics::ActivationFunction toHostNumericsActivation(ActivationType activation,
                                                                          bool gradientApplication)
    {
        auto result = getHostNumericsActivation(activation, gradientApplication);
        if(!result)
            throw std::runtime_error("Activation has no runtime HostNumerics mapping.");
        return std::move(*result);
    }

    inline roc::hostnumerics::ComparisonOptions validationComparisonOptions(rocisa::DataType type,
                                                                            double threshold)
    {
        using roc::hostnumerics::ComparisonArithmeticPrecision;
        using roc::hostnumerics::ComparisonOptions;
        using roc::hostnumerics::ElementwiseToleranceModel;
        using roc::hostnumerics::ScalarType;

        const ScalarType scalarType       = toHostNumericsScalarType(type);
        const double     defaultTolerance = [&] {
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
            throw std::runtime_error("Invalid HostNumerics comparison scalar type.");
        }();
        const double tolerance
            = scalarType == ScalarType::Float32 && threshold > 0.0 ? threshold : defaultTolerance;

        ComparisonOptions options;
        if(tolerance > 0.0)
        {
            options.relativeTolerance          = tolerance;
            options.elementwiseToleranceModel  = ElementwiseToleranceModel::SymmetricUnitScaled;
            options.strictElementwiseTolerance = true;
            options.arithmeticPrecision
                = scalarType == ScalarType::Float64 || scalarType == ScalarType::ComplexFloat64
                      ? ComparisonArithmeticPrecision::Float64
                      : ComparisonArithmeticPrecision::Float32;
        }
        return options;
    }

    inline std::vector<size_t> referenceIndexOrder(const TensorDescriptor& descriptor)
    {
        std::vector<size_t> order(descriptor.dimensions());
        for(size_t axis = 0; axis < order.size(); ++axis)
            order[axis] = order.size() - 1 - axis;
        return order;
    }

    inline hostnumerics::detail::Expected<roc::hostnumerics::IndexRange,
                                          hostnumerics::detail::DescriptorFailure>
        getReferenceIndexRange(TensorDescriptor const& descriptor, size_t elementsToValidate)
    {
        const auto logicalCount
            = roc::hostnumerics::detail::ShapeAccess::getElementCount(descriptor.sizes());
        if(!logicalCount)
        {
            return roc::hostnumerics::detail::Unexpected(hostnumerics::detail::DescriptorFailure(
                hostnumerics::detail::DescriptorError::ElementCountOverflow));
        }
        // --num-elements-to-validate 0 turns validation off. A reference that is
        // still needed, for example to print tensors, covers every element, as the
        // client always has.
        if(elementsToValidate == 0)
            return roc::hostnumerics::IndexRange::all(*logicalCount);
        return roc::hostnumerics::primeIndexRange(
            *logicalCount, descriptor.totalAllocatedElements(), elementsToValidate);
    }

    inline roc::hostnumerics::IndexRange referenceIndexRange(TensorDescriptor const& descriptor,
                                                             size_t elementsToValidate)
    {
        auto selection = getReferenceIndexRange(descriptor, elementsToValidate);
        if(!selection)
            throw std::runtime_error(std::string(hostnumerics::detail::message(selection.error())));
        return std::move(*selection);
    }

} // namespace TensileLite::Client
