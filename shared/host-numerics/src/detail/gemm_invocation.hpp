// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <roc/host_numerics/gemm.hpp>
#include <string>
#include <utility>

namespace roc::host_numerics::detail {
struct GemmOperand {
    explicit GemmOperand(Tensor tensor) : values(std::move(tensor)) {}

    GemmOperand(Tensor tensor, std::optional<ScalarType> compute,
                std::vector<Tensor> preQuantizationScaleTensors,
                std::optional<Tensor> blockScaleTensor, size_t reductionBlockSize,
                bool conjugateValues)
        : values(std::move(tensor)),
          computeType(compute),
          preQuantizationScales(std::move(preQuantizationScaleTensors)),
          blockScale(std::move(blockScaleTensor)),
          blockSize(reductionBlockSize),
          conjugate(conjugateValues) {}

    Tensor values;
    std::optional<ScalarType> computeType;
    std::vector<Tensor> preQuantizationScales;
    std::optional<Tensor> blockScale;
    size_t blockSize = 0;
    bool conjugate = false;
};

// Private numerical specification used to share validation and execution code
// between owning and caller-output entry points.
struct GemmSpecification {
    GemmSpecification(Tensor aTensor, Tensor bTensor, Tensor cTensor, ScalarType output,
                      ScalarType accumulator)
        : GemmSpecification(std::move(aTensor), std::move(bTensor), std::move(cTensor), output,
                            GemmOptions(accumulator)) {}

    GemmSpecification(Tensor aTensor, Tensor bTensor, Tensor cTensor, ScalarType output,
                      const GemmOptions& options)
        : GemmSpecification(
              GemmOperand(std::move(aTensor), options.computeTypeA, options.preQuantizationScalesA,
                          options.blockScaleA, options.blockSizeA, options.conjugateA),
              GemmOperand(std::move(bTensor), options.computeTypeB, options.preQuantizationScalesB,
                          options.blockScaleB, options.blockSizeB, options.conjugateB),
              std::move(cTensor), output, options) {}

    GemmSpecification(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, ScalarType output,
                      ScalarType accumulator)
        : GemmSpecification(std::move(aOperand), std::move(bOperand), std::move(cTensor), output,
                            GemmOptions(accumulator)) {}

    GemmSpecification(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, ScalarType output,
                      const GemmOptions& options)
        : a(std::move(aOperand)),
          b(std::move(bOperand)),
          c(std::move(cTensor)),
          outputType(output),
          accumulatorType(options.accumulatorType),
          accumulationRounding(options.accumulationRounding),
          mathMode(options.mathMode),
          epilogue(options.epilogue),
          outputSelection(options.outputSelection) {}

    GemmOperand a;
    GemmOperand b;
    Tensor c;
    ScalarType outputType;
    ScalarType accumulatorType;
    AccumulationRounding accumulationRounding;
    MathMode mathMode;
    GemmEpilogue epilogue;
    OutputSelection outputSelection;
};

// Private bound execution state shared by the built-in and optional BLAS
// implementations. Public callers use referenceGemm() or referenceGemmInto().
struct GemmInvocation : GemmSpecification {
    GemmInvocation(Tensor aTensor, Tensor bTensor, Tensor cTensor, Tensor dTensor,
                   ScalarType accumulator)
        : GemmInvocation(std::move(aTensor), std::move(bTensor), std::move(cTensor),
                         std::move(dTensor), GemmOptions(accumulator)) {}

    GemmInvocation(Tensor aTensor, Tensor bTensor, Tensor cTensor, Tensor dTensor,
                   const GemmOptions& options)
        : GemmInvocation(
              GemmOperand(std::move(aTensor), options.computeTypeA, options.preQuantizationScalesA,
                          options.blockScaleA, options.blockSizeA, options.conjugateA),
              GemmOperand(std::move(bTensor), options.computeTypeB, options.preQuantizationScalesB,
                          options.blockScaleB, options.blockSizeB, options.conjugateB),
              std::move(cTensor), std::move(dTensor), options) {}

    GemmInvocation(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, Tensor dTensor,
                   ScalarType accumulator)
        : GemmInvocation(std::move(aOperand), std::move(bOperand), std::move(cTensor),
                         std::move(dTensor), GemmOptions(accumulator)) {}

    GemmInvocation(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, Tensor dTensor,
                   const GemmOptions& options)
        : GemmSpecification(std::move(aOperand), std::move(bOperand), std::move(cTensor),
                            dTensor.type(), options),
          d(std::move(dTensor)) {}

    GemmInvocation(GemmSpecification specification, Tensor dTensor,
                   OutputSelection selection = OutputSelection::all())
        : GemmSpecification(std::move(specification)), d(std::move(dTensor)) {
        outputSelection = std::move(selection);
    }

    Tensor d;
};

struct GemmExecutionInfo {
    GemmBackend backendUsed = GemmBackend::Pointwise;
    std::optional<std::string> fallbackReason;
    size_t outputElementsWritten = 0;
    size_t outputElementsCovered = 0;
};

GemmSupportInfo queryGemmSupport(const GemmInvocation& invocation, GemmBackend backend);
GemmExecutionInfo executeGemm(const GemmInvocation& invocation, GemmBackend backend);
GemmSupportInfo queryBlasGemmSupport(const GemmInvocation& invocation, GemmBackend backend);
GemmExecutionInfo executeBlasGemm(const GemmInvocation& invocation, GemmBackend backend);
}  // namespace roc::host_numerics::detail

// Short implementation-only aliases used by compiled source files.
namespace roc::host_numerics {
using GemmSpecification = detail::GemmSpecification;
using GemmInvocation = detail::GemmInvocation;
using GemmExecutionInfo = detail::GemmExecutionInfo;
}  // namespace roc::host_numerics
