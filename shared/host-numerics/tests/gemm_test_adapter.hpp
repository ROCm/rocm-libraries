// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../src/detail/gemm_invocation.hpp"
#include "../src/detail/reference_gemm.hpp"

#include <roc/host_numerics/backends/blas.hpp>
#include <roc/host_numerics/epilogue.hpp>
#include <roc/host_numerics/tensor_operations.hpp>

namespace roc::host_numerics {
// Test utility for product-style expressions. It composes the public matmul,
// tensor-arithmetic, and epilogue APIs; the production matmul backends never
// receive C, coefficients, bias, activation, or output conversion.
struct GemmTestOptions : MatmulOptions {
    explicit GemmTestOptions(ScalarType accumulator = ScalarType::Float32)
        : MatmulOptions(accumulator),
          alpha(Tensor::scalar(accumulator, 1)),
          beta(Tensor::scalar(accumulator, 0)),
          scaleC(Tensor::scalar(accumulator, 1)),
          outputScale(Tensor::scalar(accumulator, 1)),
          activationParameter0(Tensor::scalar(accumulator, 0)),
          activationParameter1(Tensor::scalar(accumulator, 0)) {}

    Tensor alpha;
    Tensor beta;
    Tensor scaleC;
    std::optional<Tensor> bias;
    std::optional<Tensor> scaleAlpha;
    std::optional<Tensor> scaleA;
    std::optional<Tensor> scaleB;
    Tensor outputScale;
    OutputConversion outputConversion = OutputConversion::Default;
    Activation activation = Activation::None;
    Tensor activationParameter0;
    Tensor activationParameter1;
    OutputSelection outputSelection = OutputSelection::all();
};

struct GemmTestCase : GemmTestOptions {
    GemmTestCase(Tensor aTensor, Tensor bTensor, Tensor cTensor, Tensor dTensor,
                 ScalarType accumulator)
        : GemmTestOptions(accumulator),
          a(std::move(aTensor)),
          b(std::move(bTensor)),
          c(std::move(cTensor)),
          d(std::move(dTensor)) {}

    GemmTestCase(Tensor aTensor, Tensor bTensor, Tensor cTensor, Tensor dTensor,
                 const GemmTestOptions& options)
        : GemmTestOptions(options),
          a(std::move(aTensor)),
          b(std::move(bTensor)),
          c(std::move(cTensor)),
          d(std::move(dTensor)) {
    }

    Tensor a;
    Tensor b;
    Tensor c;
    Tensor d;
};

using GemmSupportInfo = detail::GemmSupportInfo;
using GemmTestRunInfo = detail::GemmExecutionInfo;

inline ActivationFunction testActivation(const GemmTestOptions& options) {
    const double first = options.activationParameter0.item<double>();
    const double second = options.activationParameter1.item<double>();
    switch (options.activation) {
        case Activation::None:
            return IdentityActivation{};
        case Activation::Absolute:
            return AbsoluteActivation{};
        case Activation::ClippedRelu:
            return ClippedReluActivation{first, second};
        case Activation::Relu:
            return ReluActivation{};
        case Activation::Gelu:
            return GeluActivation{};
        case Activation::GeluDerivative:
            return GeluDerivativeActivation{};
        case Activation::GeluScaling:
            return GeluScalingActivation{first};
        case Activation::LeakyRelu:
            return LeakyReluActivation{first};
        case Activation::ReluDerivative:
            return ReluDerivativeActivation{};
        case Activation::Sigmoid:
            return SigmoidActivation{};
        case Activation::Tanh:
            return TanhActivation{first, second};
        case Activation::Silu:
            return SiluActivation{};
        case Activation::Swish:
            return SwishActivation{first};
        case Activation::Clamp:
            return ClampActivation{first, second};
    }
    throw std::invalid_argument("Unsupported test activation.");
}

inline MatmulOptions testMatmulOptions(const GemmTestOptions& options) {
    return static_cast<const MatmulOptions&>(options);
}

template <typename Execute>
GemmTestRunInfo composeGemmForTest(const GemmTestCase& request, Execute&& execute) {
    Tensor product(request.accumulatorType, request.d.layout());
    const detail::GemmInvocation invocation(
        request.a, request.b, product, testMatmulOptions(request), request.outputSelection);
    detail::validateRuntimeGemm(invocation);
    GemmTestRunInfo runInfo{};
    if (request.alpha.item<std::complex<double>>() != std::complex<double>(0.0, 0.0)) {
        runInfo = execute(invocation);
        if (request.a.shape()[1] != 0) {
            product = multiply(product, request.alpha, request.accumulatorType,
                               request.accumulatorType, request.d.layout());
            if (request.scaleA)
                product = multiply(product, *request.scaleA, request.accumulatorType,
                                   request.accumulatorType, request.d.layout());
            if (request.scaleB)
                product = multiply(product, *request.scaleB, request.accumulatorType,
                                   request.accumulatorType, request.d.layout());
            if (request.scaleAlpha)
                product = multiply(product, *request.scaleAlpha, request.accumulatorType,
                                   request.accumulatorType, request.d.layout());
        }
    } else {
        runInfo.backendUsed = GemmBackend::Blocked;
        runInfo.outputElementsWritten =
            request.outputSelection.selectedCount(request.d.elementCount());
        runInfo.outputElementsCovered = runInfo.outputElementsWritten;
    }

    if (request.beta.item<std::complex<double>>() != std::complex<double>(0.0, 0.0)) {
        const Tensor coefficient = multiply(request.beta, request.scaleC, request.accumulatorType,
                                            request.accumulatorType);
        const Tensor addend = multiply(request.c, coefficient, request.accumulatorType,
                                       request.accumulatorType, request.d.layout());
        product = add(product, addend, request.accumulatorType, request.accumulatorType,
                      request.d.layout());
    }

    EpilogueOptions epilogue(request.accumulatorType);
    epilogue.bias = request.bias;
    epilogue.outputScale = request.outputScale;
    epilogue.outputConversion = request.outputConversion;
    epilogue.activation = testActivation(request);
    epilogue.outputSelection = request.outputSelection;
    referenceEpilogueInto(product, {.output = request.d}, epilogue);
    return runInfo;
}

inline GemmSupportInfo queryGemmSupport(const GemmTestCase& request,
                                        GemmBackend backend = GemmBackend::Automatic) {
    return detail::queryGemmSupport(
        detail::GemmInvocation(request.a, request.b, request.d, testMatmulOptions(request),
                               request.outputSelection),
        backend);
}

inline GemmTestRunInfo referenceGemm(const GemmTestCase& request,
                                     GemmBackend backend = GemmBackend::Automatic) {
    return composeGemmForTest(request, [backend](const detail::GemmInvocation& invocation) {
        return detail::executeGemm(invocation, backend);
    });
}

inline GemmSupportInfo queryGemmSupportWithBlasBackend(
    const GemmTestCase& request, GemmBackend backend = GemmBackend::Automatic) {
    return detail::queryBlasGemmSupport(
        detail::GemmInvocation(request.a, request.b, request.d, testMatmulOptions(request),
                               request.outputSelection),
        backend);
}

inline GemmTestRunInfo referenceGemmWithBlasBackend(
    const GemmTestCase& request, GemmBackend backend = GemmBackend::Automatic) {
    return composeGemmForTest(request, [backend](const detail::GemmInvocation& invocation) {
        return detail::executeBlasGemm(invocation, backend);
    });
}

}  // namespace roc::host_numerics
