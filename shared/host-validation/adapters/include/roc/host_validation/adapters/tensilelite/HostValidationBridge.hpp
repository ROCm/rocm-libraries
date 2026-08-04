// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_validation/reference_gemm.hpp>

#include <Tensile/Activation.hpp>
#include <Tensile/DataTypes.hpp>

#include <stdexcept>

namespace TensileLite::Client
{
    inline roc::host_validation::Activation toHostValidationActivation(ActivationType activation)
    {
        switch(activation)
        {
        case ActivationType::None:
            return roc::host_validation::Activation::None;
        case ActivationType::Relu:
            return roc::host_validation::Activation::Relu;
        default:
            throw std::invalid_argument(
                "The host-validation POC bridge supports None and Relu activations.");
        }
    }

    template <typename Accumulator, typename Narrow>
    Accumulator quantizeForHostValidation(Accumulator value)
    {
        return static_cast<Accumulator>(static_cast<Narrow>(value));
    }

    template <typename Accumulator>
    roc::host_validation::QuantizeFunction<Accumulator>
        hostValidationQuantizerFor(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return &quantizeForHostValidation<Accumulator, float>;
        case rocisa::DataType::Double:
            return &quantizeForHostValidation<Accumulator, double>;
        case rocisa::DataType::Half:
            return &quantizeForHostValidation<Accumulator, Half>;
        case rocisa::DataType::BFloat16:
            return &quantizeForHostValidation<Accumulator, BFloat16>;
#ifdef TENSILE_USE_FP8_BF8
        case rocisa::DataType::Float8:
            return &quantizeForHostValidation<Accumulator, Float8>;
        case rocisa::DataType::BFloat8:
            return &quantizeForHostValidation<Accumulator, BFloat8>;
        case rocisa::DataType::Float8_fnuz:
            return &quantizeForHostValidation<Accumulator, Float8_fnuz>;
        case rocisa::DataType::BFloat8_fnuz:
            return &quantizeForHostValidation<Accumulator, BFloat8_fnuz>;
#endif
        default:
            throw std::invalid_argument(
                "Unsupported compute-input type in host-validation POC bridge.");
        }
    }

    template <typename InputA,
              typename InputB,
              typename InputC,
              typename Output,
              typename Accumulator>
    roc::host_validation::GemmInvocation<InputA, InputB, InputC, Output, Accumulator>
        makeHostValidationColumnMajorGemm(const InputA* a,
                                          const InputB* b,
                                          const InputC* c,
                                          Output*       d,
                                          size_t        m,
                                          size_t        n,
                                          size_t        k,
                                          bool          transA,
                                          bool          transB)
    {
        using roc::host_validation::ConstMatrixView;
        using roc::host_validation::GemmInvocation;
        using roc::host_validation::MatrixView;

        const ptrdiff_t strideARow    = transA ? static_cast<ptrdiff_t>(k) : 1;
        const ptrdiff_t strideAColumn = transA ? 1 : static_cast<ptrdiff_t>(m);
        const ptrdiff_t strideBRow    = transB ? static_cast<ptrdiff_t>(n) : 1;
        const ptrdiff_t strideBColumn = transB ? 1 : static_cast<ptrdiff_t>(k);

        return GemmInvocation<InputA, InputB, InputC, Output, Accumulator>{
            ConstMatrixView<InputA>(a, m, k, strideARow, strideAColumn),
            ConstMatrixView<InputB>(b, k, n, strideBRow, strideBColumn),
            ConstMatrixView<InputC>(c, m, n, 1, static_cast<ptrdiff_t>(m)),
            MatrixView<Output>(d, m, n, 1, static_cast<ptrdiff_t>(m))};
    }
}
