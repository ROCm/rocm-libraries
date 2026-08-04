// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace roc::host_validation {
enum class Activation {
    None,
    Relu,
    Gelu,
    Silu,
    Clamp,
};

template <typename T>
using QuantizeFunction = T (*)(T);

template <typename Accumulator>
struct BlockScaleView {
    using LoadFunction = Accumulator (*)(const void*, size_t);

    const void* data = nullptr;
    LoadFunction load = nullptr;
    size_t blockSize = 0;
    size_t freeStride = 0;
    size_t blockStride = 0;

    bool empty() const {
        return data == nullptr && load == nullptr && blockSize == 0 && freeStride == 0 &&
               blockStride == 0;
    }

    Accumulator value(size_t freeIndex, size_t blockIndex) const {
        return load(data, freeIndex * freeStride + blockIndex * blockStride);
    }
};

namespace detail {
template <typename T>
struct IsComplex : std::false_type {};

template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
T conjugateIfNeeded(const T& value, bool conjugate) {
    if constexpr (IsComplex<T>::value)
        return conjugate ? std::conj(value) : value;
    else
        return value;
}

template <typename Accumulator, typename Scale>
Accumulator loadScale(const void* data, size_t index) {
    return static_cast<Accumulator>(static_cast<const Scale*>(data)[index]);
}

template <typename Accumulator>
Accumulator applyActivation(Activation activation, Accumulator value, Accumulator parameter0,
                            Accumulator parameter1) {
    if constexpr (IsComplex<Accumulator>::value) {
        if (activation != Activation::None)
            throw std::invalid_argument("Complex reference GEMM does not support activation.");
        return value;
    } else {
        switch (activation) {
            case Activation::None:
                return value;
            case Activation::Relu:
                return std::max(Accumulator(0), value);
            case Activation::Gelu: {
                constexpr float coefficient0 = 0.7978845608028654f;
                constexpr float coefficient1 = 0.044715f;
                const float x = static_cast<float>(value);
                return static_cast<Accumulator>(
                    0.5f * x *
                    (1.0f + std::tanh(coefficient0 * x * (1.0f + coefficient1 * x * x))));
            }
            case Activation::Silu: {
                const float x = static_cast<float>(value);
                const float beta = static_cast<float>(parameter0);
                return static_cast<Accumulator>(x / (1.0f + std::exp(-beta * x)));
            }
            case Activation::Clamp:
                return std::max(parameter0, std::min(value, parameter1));
        }
    }

    throw std::invalid_argument("Unsupported reference GEMM activation.");
}
}  // namespace detail

template <typename Accumulator, typename Scale>
BlockScaleView<Accumulator> makeBlockScaleView(const Scale* data, size_t blockSize,
                                               size_t freeStride, size_t blockStride) {
    return {data, &detail::loadScale<Accumulator, Scale>, blockSize, freeStride, blockStride};
}

template <typename InputA, typename InputB, typename InputC, typename Output, typename Accumulator>
struct GemmInvocation {
    GemmInvocation(ConstMatrixView<InputA> aView, ConstMatrixView<InputB> bView,
                   ConstMatrixView<InputC> cView, MatrixView<Output> dView)
        : a(aView), b(bView), c(cView), d(dView) {}

    ConstMatrixView<InputA> a;
    ConstMatrixView<InputB> b;
    ConstMatrixView<InputC> c;
    MatrixView<Output> d;

    Accumulator alpha = Accumulator(1);
    Accumulator beta = Accumulator(0);

    std::optional<ConstVectorView<Accumulator>> bias;
    std::optional<ConstVectorView<Accumulator>> scaleAlpha;
    std::optional<ConstVectorView<Accumulator>> scaleA;
    std::optional<ConstVectorView<Accumulator>> scaleB;

    int factorDimension = 0;
    Activation activation = Activation::None;
    Accumulator activationParameter0 = Accumulator(0);
    Accumulator activationParameter1 = Accumulator(0);

    QuantizeFunction<Accumulator> quantizeA = nullptr;
    QuantizeFunction<Accumulator> quantizeB = nullptr;
    bool conjugateA = false;
    bool conjugateB = false;

    BlockScaleView<Accumulator> blockScaleA;
    BlockScaleView<Accumulator> blockScaleB;
};

template <typename InputA, typename InputB, typename InputC, typename Output, typename Accumulator>
void validate(const GemmInvocation<InputA, InputB, InputC, Output, Accumulator>& invocation) {
    const size_t m = invocation.a.rows();
    const size_t k = invocation.a.columns();
    const size_t n = invocation.b.columns();

    if (invocation.b.rows() != k)
        throw std::invalid_argument("Reference GEMM K dimension mismatch.");
    if (invocation.c.rows() != m || invocation.c.columns() != n)
        throw std::invalid_argument("Reference GEMM C shape mismatch.");
    if (invocation.d.rows() != m || invocation.d.columns() != n)
        throw std::invalid_argument("Reference GEMM D shape mismatch.");
    if (invocation.factorDimension != 0 && invocation.factorDimension != 1)
        throw std::invalid_argument("Reference GEMM factor dimension must be 0 or 1.");
    if (invocation.bias && invocation.bias->size() != m)
        throw std::invalid_argument("Reference GEMM bias length must equal M.");
    if (invocation.scaleA && invocation.scaleA->size() != m)
        throw std::invalid_argument("Reference GEMM scale-A length must equal M.");
    if (invocation.scaleB && invocation.scaleB->size() != n)
        throw std::invalid_argument("Reference GEMM scale-B length must equal N.");
    if (invocation.scaleAlpha) {
        const size_t expected = invocation.factorDimension == 0 ? m : n;
        if (invocation.scaleAlpha->size() != expected)
            throw std::invalid_argument(
                "Reference GEMM scale-alpha length does not match factor dimension.");
    }

    switch (invocation.activation) {
        case Activation::None:
        case Activation::Relu:
        case Activation::Gelu:
        case Activation::Silu:
        case Activation::Clamp:
            break;
        default:
            throw std::invalid_argument("Unsupported reference GEMM activation.");
    }

    const bool hasBlockScaleA = !invocation.blockScaleA.empty();
    const bool hasBlockScaleB = !invocation.blockScaleB.empty();
    if (hasBlockScaleA != hasBlockScaleB)
        throw std::invalid_argument(
            "Reference GEMM requires block scales for both operands or neither.");

    auto validateBlockScale = [](const char* name, const auto& scale) {
        if (scale.empty()) return;
        if (scale.data == nullptr || scale.load == nullptr)
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block scale has no data loader.");
        if (scale.blockSize == 0 || scale.freeStride == 0 || scale.blockStride == 0)
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block scale has invalid geometry.");
    };
    validateBlockScale("A", invocation.blockScaleA);
    validateBlockScale("B", invocation.blockScaleB);
}

template <typename OperandMath = void, typename InputA, typename InputB, typename InputC,
          typename Output, typename Accumulator>
void referenceGemm(const GemmInvocation<InputA, InputB, InputC, Output, Accumulator>& invocation) {
    validate(invocation);

    using MathType = std::conditional_t<std::is_void_v<OperandMath>, Accumulator, OperandMath>;

    const size_t m = invocation.a.rows();
    const size_t n = invocation.b.columns();
    const size_t k = invocation.a.columns();
    const bool hasBlockScale = !invocation.blockScaleA.empty();

    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            Accumulator sum = Accumulator(0);

            if (hasBlockScale) {
                const size_t step =
                    std::min(invocation.blockScaleA.blockSize, invocation.blockScaleB.blockSize);

                for (size_t blockBase = 0; blockBase < k; blockBase += step) {
                    Accumulator blockSum = Accumulator(0);
                    const size_t blockEnd = std::min(blockBase + step, k);
                    for (size_t reduction = blockBase; reduction < blockEnd; ++reduction) {
                        Accumulator aValue = static_cast<Accumulator>(detail::conjugateIfNeeded(
                            invocation.a(row, reduction), invocation.conjugateA));
                        Accumulator bValue = static_cast<Accumulator>(detail::conjugateIfNeeded(
                            invocation.b(reduction, column), invocation.conjugateB));
                        if (invocation.quantizeA) aValue = invocation.quantizeA(aValue);
                        if (invocation.quantizeB) bValue = invocation.quantizeB(bValue);

                        blockSum += static_cast<Accumulator>(static_cast<MathType>(aValue)) *
                                    static_cast<Accumulator>(static_cast<MathType>(bValue));
                    }

                    const size_t blockA = blockBase / invocation.blockScaleA.blockSize;
                    const size_t blockB = blockBase / invocation.blockScaleB.blockSize;
                    const Accumulator scale = invocation.blockScaleA.value(row, blockA) *
                                              invocation.blockScaleB.value(column, blockB);
                    sum += blockSum * scale;
                }
            } else {
                for (size_t reduction = 0; reduction < k; ++reduction) {
                    Accumulator aValue = static_cast<Accumulator>(detail::conjugateIfNeeded(
                        invocation.a(row, reduction), invocation.conjugateA));
                    Accumulator bValue = static_cast<Accumulator>(detail::conjugateIfNeeded(
                        invocation.b(reduction, column), invocation.conjugateB));
                    if (invocation.quantizeA) aValue = invocation.quantizeA(aValue);
                    if (invocation.quantizeB) bValue = invocation.quantizeB(bValue);

                    sum += static_cast<Accumulator>(static_cast<MathType>(aValue)) *
                           static_cast<Accumulator>(static_cast<MathType>(bValue));
                }
            }

            Accumulator effectiveAlpha = invocation.alpha;
            if (invocation.scaleA)
                effectiveAlpha *= static_cast<Accumulator>((*invocation.scaleA)[row]);
            if (invocation.scaleB)
                effectiveAlpha *= static_cast<Accumulator>((*invocation.scaleB)[column]);
            if (invocation.scaleAlpha) {
                effectiveAlpha *= static_cast<Accumulator>(
                    (*invocation.scaleAlpha)[invocation.factorDimension == 0 ? row : column]);
            }

            Accumulator result =
                effectiveAlpha * sum +
                invocation.beta * static_cast<Accumulator>(invocation.c(row, column));

            if (invocation.bias) result += static_cast<Accumulator>((*invocation.bias)[row]);

            result = detail::applyActivation(invocation.activation, result,
                                             invocation.activationParameter0,
                                             invocation.activationParameter1);

            invocation.d(row, column) = static_cast<Output>(result);
        }
    }
}
}  // namespace roc::host_validation
