// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/operation_types.hpp>
#include <string>
#include <utility>
#include <vector>

namespace roc::host_validation {
enum class GemmBackend {
    Automatic,
    Canonical,
    Tiled,
    Blas,
};

enum class AccumulationRounding {
    TypeDefault,
    FullPrecision,
    AfterProductAndSum,
};

struct BlockScaleBinding {
    TensorView values;
    size_t blockSize;
};

struct GemmOperand {
    explicit GemmOperand(TensorView tensor) : values(std::move(tensor)) {}

    TensorView values;
    std::optional<ScalarType> computeType;
    std::vector<VectorBinding> preQuantizationScales;
    std::optional<BlockScaleBinding> blockScale;
    bool conjugate = false;
};

struct GemmEpilogue {
    std::complex<double> alpha = {1.0, 0.0};
    std::complex<double> beta = {0.0, 0.0};
    std::optional<VectorBinding> bias;
    std::optional<VectorBinding> scaleAlpha;
    std::optional<TensorView> scaleA;
    std::optional<TensorView> scaleB;
    std::complex<double> outputScale = {1.0, 0.0};
    OutputConversion outputConversion = OutputConversion::Default;
    Activation activation = Activation::None;
    double activationParameter0 = 0.0;
    double activationParameter1 = 0.0;
};

struct GemmProblem {
    GemmProblem(GemmOperand aOperand, GemmOperand bOperand, TensorView cTensor, ScalarType output,
                ScalarType accumulator)
        : a(std::move(aOperand)),
          b(std::move(bOperand)),
          c(std::move(cTensor)),
          outputType(output),
          accumulatorType(accumulator) {}

    GemmOperand a;
    GemmOperand b;
    TensorView c;
    ScalarType outputType;
    ScalarType accumulatorType;
    AccumulationRounding accumulationRounding = AccumulationRounding::TypeDefault;
    MathMode mathMode = MathMode::Default;
    GemmEpilogue epilogue;
};

struct GemmRequest : GemmProblem {
    GemmRequest(GemmOperand aOperand, GemmOperand bOperand, TensorView cTensor,
                MutableTensorView dTensor, ScalarType accumulator)
        : GemmProblem(std::move(aOperand), std::move(bOperand), std::move(cTensor), dTensor.type(),
                      accumulator),
          d(std::move(dTensor)) {}

    MutableTensorView d;
    OutputSelection outputSelection = OutputSelection::all();
};

struct GemmSupportInfo {
    bool supported = false;
    std::string reason;

    explicit operator bool() const {
        return supported;
    }
};

struct GemmRunInfo {
    GemmBackend backendUsed = GemmBackend::Canonical;
    std::optional<std::string> fallbackReason;
    size_t outputElementsComputed = 0;
};

struct GemmExecution {
    GemmBackend backend = GemmBackend::Automatic;
    bool requireRequestedBackend = false;
};

struct GemmResult {
    TensorView output;
    GemmRunInfo runInfo;
};

class GemmBackendImplementation {
   public:
    virtual ~GemmBackendImplementation() = default;

    virtual GemmBackend backend() const = 0;
    virtual GemmSupportInfo querySupport(const GemmRequest&) const = 0;
    virtual GemmRunInfo run(const GemmRequest&) const = 0;
};

GemmSupportInfo queryGemmSupport(const GemmRequest& request, const GemmExecution& execution = {},
                                 const GemmBackendImplementation* backendImplementation = nullptr);

GemmResult referenceGemm(const GemmRequest& request, const GemmExecution& execution = {},
                         const GemmBackendImplementation* backendImplementation = nullptr);
}  // namespace roc::host_validation
