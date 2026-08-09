/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Product-private translation from hipBLASLt descriptors and host buffers to
// product-independent host-validation operations.

#include <roc/host_validation/adapters/hipblaslt/HipblasltReferenceGemm.hpp>
#include <roc/host_validation/adapters/hipblaslt/Types.hpp>
#include <roc/host_validation/backends/blas.hpp>
#include <roc/host_validation/validation.hpp>

#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace
{
    using namespace roc::host_validation;

    template <typename T>
    struct IsStdComplex : std::false_type
    {
    };

    template <typename T>
    struct IsStdComplex<std::complex<T>> : std::true_type
    {
    };

    template <typename T>
    std::complex<double> runtimeScalar(T value)
    {
        if constexpr(IsStdComplex<T>::value)
            return {static_cast<double>(value.real()), static_cast<double>(value.imag())};
        else
            return {static_cast<double>(value), 0.0};
    }

    template <typename Tc>
    constexpr ScalarType compatibilityAccumulatorType()
    {
        // Preserve the established client-reference policy while the public
        // compute-mode contract is clarified independently: I32 uses wide
        // host arithmetic and F16 compute uses an F32 host accumulator.
        if constexpr(std::is_same_v<Tc, int32_t>)
            return ScalarType::Float64;
        else if constexpr(std::is_same_v<Tc, hipblasLtHalf>)
            return ScalarType::Float32;
        else
            return roc::host_validation::hipblaslt_adapter::scalarType<Tc>();
    }

    ScalarType compatibilityComputeType(hipDataType type)
    {
        if(type == HIP_R_32I)
            return ScalarType::Float64;
        return roc::host_validation::hipblaslt_adapter::scalarType(type);
    }

    TensorView tensorView(const void* data, ScalarType type, Layout layout)
    {
        const size_t bytes = storageBytesForLayout(type, layout);
        if(data == nullptr && bytes != 0)
            throw std::invalid_argument("Null hipBLASLt reference input buffer.");
        return TensorView(type,
                          std::move(layout),
                          {static_cast<const std::byte*>(data), bytes});
    }

    MutableTensorView mutableTensorView(void* data, ScalarType type, Layout layout)
    {
        const size_t bytes = storageBytesForLayout(type, layout);
        if(data == nullptr && bytes != 0)
            throw std::invalid_argument("Null hipBLASLt reference output buffer.");
        return MutableTensorView(
            type, std::move(layout), {static_cast<std::byte*>(data), bytes});
    }

    template <typename Tc>
    TensorView scalarVector(const void* data, size_t elements)
    {
        const ScalarType type =
            roc::host_validation::hipblaslt_adapter::scalarType<Tc>();
        return tensorView(data, type, Layout::contiguous(Shape{elements}));
    }
}

template <typename Tc>
void hipblaslt_reference_gemm(hipblasOperation_t       transA,
                              hipblasOperation_t       transB,
                              int64_t                  m,
                              int64_t                  n,
                              int64_t                  k,
                              Tc                       alpha,
                              const void*              A,
                              int64_t                  lda,
                              const void*              B,
                              int64_t                  ldb,
                              Tc                       beta,
                              std::add_pointer_t<void> C,
                              int64_t                  ldc,
                              const void*              AlphaVec,
                              const void*              scaleAVec,
                              const void*              scaleBVec,
                              Tc                       scaleD,
                              bool                     isScaleAVec,
                              bool                     isScaleBVec,
                              hipDataType              TiA,
                              hipDataType              TiB,
                              hipDataType              To,
                              hipDataType              Tc_enum,
                              hipDataType              TciA,
                              hipDataType              TciB,
                              bool                     isScaleAMXFormat,
                              bool                     isScaleBMXFormat)
{
    if(m < 0 || n < 0 || k < 0 || lda < 0 || ldb < 0 || ldc < 0)
        throw std::invalid_argument(
            "hipBLASLt reference GEMM dimensions and strides must be nonnegative.");

    const size_t rows = static_cast<size_t>(m);
    const size_t columns = static_cast<size_t>(n);
    const size_t reduction = static_cast<size_t>(k);
    const ScalarType typeA =
        roc::host_validation::hipblaslt_adapter::scalarType(TiA);
    const ScalarType typeB =
        roc::host_validation::hipblaslt_adapter::scalarType(TiB);
    const ScalarType outputType =
        roc::host_validation::hipblaslt_adapter::scalarType(To);

    const ptrdiff_t aRowStride = transA == HIPBLAS_OP_N ? 1 : lda;
    const ptrdiff_t aColumnStride = transA == HIPBLAS_OP_N ? lda : 1;
    const ptrdiff_t bRowStride = transB == HIPBLAS_OP_N ? 1 : ldb;
    const ptrdiff_t bColumnStride = transB == HIPBLAS_OP_N ? ldb : 1;
    const Layout layoutA(Shape{rows, reduction}, {aRowStride, aColumnStride});
    const Layout layoutB(Shape{reduction, columns}, {bRowStride, bColumnStride});
    const Layout layoutC(Shape{rows, columns}, {1, ldc});

    GemmOperand operandA(tensorView(A, typeA, layoutA));
    GemmOperand operandB(tensorView(B, typeB, layoutB));
    operandA.conjugate = transA == HIPBLAS_OP_C;
    operandB.conjugate = transB == HIPBLAS_OP_C;

    const hipDataType computeTypeA =
        TciA == HIP_R_32I ? HIP_R_64F : TciA;
    const hipDataType computeTypeB =
        TciB == HIP_R_32I ? HIP_R_64F : TciB;
    // The former bridge only applied compute-input conversion when the
    // requested encoding was narrower than storage. Keep that observable
    // policy for this extraction; same-width cross-format quantization can be
    // corrected as a separately reviewed semantic change.
    if(realDataTypeSize(TiA) > realDataTypeSize(computeTypeA))
        operandA.computeType = compatibilityComputeType(computeTypeA);
    if(realDataTypeSize(TiB) > realDataTypeSize(computeTypeB))
        operandB.computeType = compatibilityComputeType(computeTypeB);

    if(scaleAVec != nullptr && !isScaleAMXFormat)
        operandA.preQuantizationScales.push_back(
            VectorBinding{scalarVector<Tc>(scaleAVec, isScaleAVec ? rows : 1),
                          MatrixAxis::Row});
    if(AlphaVec != nullptr)
        operandA.preQuantizationScales.push_back(
            VectorBinding{scalarVector<Tc>(AlphaVec, rows), MatrixAxis::Row});
    if(scaleBVec != nullptr && !isScaleBMXFormat)
        operandB.preQuantizationScales.push_back(
            VectorBinding{scalarVector<Tc>(scaleBVec, isScaleBVec ? columns : 1),
                          MatrixAxis::Column});

    GemmProblem problem(std::move(operandA),
                        std::move(operandB),
                        tensorView(C, outputType, layoutC),
                        mutableTensorView(C, outputType, layoutC),
                        compatibilityAccumulatorType<Tc>());
    problem.epilogue.alpha = runtimeScalar(alpha);
    problem.epilogue.beta = runtimeScalar(beta);
    problem.epilogue.outputScale = runtimeScalar(scaleD);
    if(outputType == ScalarType::Int8)
        problem.epilogue.outputConversion = GemmOutputConversion::SaturatingInt8;

    static constexpr int64_t blasThreshold = 600;
    const bool useBlas =
        rows != 0 && columns != 0 && reduction != 0 &&
        (m > blasThreshold || n > blasThreshold || k > blasThreshold ||
         lda > blasThreshold || ldb > blasThreshold || ldc > blasThreshold);
    GemmInvocation invocation(std::move(problem));
    if(useBlas)
    {
        static const TransformingBlasGemmBackend backend;
        invocation.execution = {
            .backend = GemmBackend::Blas,
            .requireRequestedBackend = true,
            .backendImplementation = &backend,
        };
    }
    referenceGemm(invocation);

    (void)Tc_enum;
}

#define CREATEFUNCTION(Tc)                                                                  \
    template void hipblaslt_reference_gemm<Tc>(hipblasOperation_t transA,                   \
                                                hipblasOperation_t transB,                   \
                                                int64_t            m,                        \
                                                int64_t            n,                        \
                                                int64_t            k,                        \
                                                Tc                 alpha,                    \
                                                const void*        A,                        \
                                                int64_t            lda,                      \
                                                const void*        B,                        \
                                                int64_t            ldb,                      \
                                                Tc                 beta,                     \
                                                std::add_pointer_t<void> C,                  \
                                                int64_t            ldc,                      \
                                                const void*        AlphaVec,                 \
                                                const void*        scaleAVec,                 \
                                                const void*        scaleBVec,                 \
                                                Tc                 scaleD,                   \
                                                bool               isScaleAVec,              \
                                                bool               isScaleBVec,              \
                                                hipDataType        TiA,                      \
                                                hipDataType        TiB,                      \
                                                hipDataType        To,                       \
                                                hipDataType        Tc_enum,                  \
                                                hipDataType        TciA,                     \
                                                hipDataType        TciB,                     \
                                                bool               isScaleAMXFormat,         \
                                                bool               isScaleBMXFormat);

CREATEFUNCTION(hipblasLtHalf)
CREATEFUNCTION(float)
CREATEFUNCTION(double)
CREATEFUNCTION(int32_t)
CREATEFUNCTION(std::complex<float>)
CREATEFUNCTION(std::complex<double>)
