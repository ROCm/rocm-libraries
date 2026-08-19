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

#include <hipblaslt/host_validation/HipblasltReferenceGemm.hpp>
#include <hipblaslt/host_validation/Types.hpp>
#include <roc/host_validation/backends/blas.hpp>
#include <roc/host_validation/validation.hpp>

#include "hipblaslt_ostream.hpp"

#include <complex>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace
{
    using namespace roc::host_validation;

    template <typename T>
    Scalar runtimeScalar(const T& value)
    {
        const ScalarType type = hipblaslt::host_validation::scalarType<T>();
        return Scalar::fromStorage(type, std::as_bytes(std::span<const T>(&value, 1)));
    }

    template <typename Tc>
    constexpr ScalarType referenceAccumulatorType()
    {
        // I32 reference GEMM uses wide host arithmetic to avoid intermediate
        // overflow. F16 coefficients use an F32 host accumulator.
        if constexpr(std::is_same_v<Tc, int32_t>)
            return ScalarType::Float64;
        else if constexpr(std::is_same_v<Tc, hipblasLtHalf>)
            return ScalarType::Float32;
        else
            return hipblaslt::host_validation::scalarType<Tc>();
    }

    ScalarType referenceComputeType(hipDataType type)
    {
        if(type == HIP_R_32I)
            return ScalarType::Float64;
        return hipblaslt::host_validation::scalarType(type);
    }

    Tensor tensorFromStorage(const void* data, ScalarType type, Layout layout)
    {
        const size_t bytes = storageBytesForLayout(type, layout);
        if(data == nullptr && bytes != 0)
            throw std::invalid_argument("Null hipBLASLt reference input buffer.");
        return Tensor(type,
                      std::move(layout),
                      std::span<const std::byte>(static_cast<const std::byte*>(data), bytes));
    }

    Tensor tensorFromMutableStorage(void* data, ScalarType type, Layout layout)
    {
        const size_t bytes = storageBytesForLayout(type, layout);
        if(data == nullptr && bytes != 0)
            throw std::invalid_argument("Null hipBLASLt reference output buffer.");
        return Tensor(
            type, std::move(layout), std::span<std::byte>(static_cast<std::byte*>(data), bytes));
    }

    template <typename Tc>
    Tensor scalarVector(const void* data, size_t elements)
    {
        const ScalarType type =
            hipblaslt::host_validation::scalarType<Tc>();
        return tensorFromStorage(data, type, Layout::contiguous(Shape{elements}));
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
                              const void*              C,
                              int64_t                  ldc,
                              std::add_pointer_t<void> D,
                              int64_t                  ldd,
                              const void*              AlphaVec,
                              const void*              scaleAVec,
                              const void*              scaleBVec,
                              Tc                       scaleD,
                              bool                     isScaleAVec,
                              bool                     isScaleBVec,
                              hipDataType              TiA,
                              hipDataType              TiB,
                              hipDataType              TiC,
                              hipDataType              To,
                              hipDataType              TciA,
                              hipDataType              TciB,
                              bool                     isScaleAMXFormat,
                              bool                     isScaleBMXFormat)
{
    if(m < 0 || n < 0 || k < 0 || lda < 0 || ldb < 0 || ldc < 0 || ldd < 0)
        throw std::invalid_argument(
            "hipBLASLt reference GEMM dimensions and strides must be nonnegative.");

    const size_t rows = static_cast<size_t>(m);
    const size_t columns = static_cast<size_t>(n);
    const size_t reduction = static_cast<size_t>(k);
    const ScalarType typeA =
        hipblaslt::host_validation::scalarType(TiA);
    const ScalarType typeB =
        hipblaslt::host_validation::scalarType(TiB);
    const ScalarType typeC =
        hipblaslt::host_validation::scalarType(TiC);
    const ScalarType outputType =
        hipblaslt::host_validation::scalarType(To);

    const ptrdiff_t aRowStride = transA == HIPBLAS_OP_N ? 1 : lda;
    const ptrdiff_t aColumnStride = transA == HIPBLAS_OP_N ? lda : 1;
    const ptrdiff_t bRowStride = transB == HIPBLAS_OP_N ? 1 : ldb;
    const ptrdiff_t bColumnStride = transB == HIPBLAS_OP_N ? ldb : 1;
    const Layout layoutA(Shape{rows, reduction}, {aRowStride, aColumnStride});
    const Layout layoutB(Shape{reduction, columns}, {bRowStride, bColumnStride});
    const Layout layoutC(Shape{rows, columns}, {1, ldc});
    const Layout layoutD(Shape{rows, columns}, {1, ldd});

    GemmOperand operandA(tensorFromStorage(A, typeA, layoutA));
    GemmOperand operandB(tensorFromStorage(B, typeB, layoutB));
    operandA.conjugate = transA == HIPBLAS_OP_C;
    operandB.conjugate = transB == HIPBLAS_OP_C;

    const hipDataType computeTypeA =
        TciA == HIP_R_32I ? HIP_R_64F : TciA;
    const hipDataType computeTypeB =
        TciB == HIP_R_32I ? HIP_R_64F : TciB;
    const ScalarType computeScalarTypeA = referenceComputeType(computeTypeA);
    const ScalarType computeScalarTypeB = referenceComputeType(computeTypeB);
    if(computeScalarTypeA != typeA)
        operandA.computeType = computeScalarTypeA;
    if(computeScalarTypeB != typeB)
        operandB.computeType = computeScalarTypeB;

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

    Tensor      output = tensorFromMutableStorage(D, outputType, layoutD);
    GemmRequest request(std::move(operandA),
                        std::move(operandB),
                        tensorFromStorage(C, typeC, layoutC),
                        output,
                        referenceAccumulatorType<Tc>());
    request.epilogue.alpha = runtimeScalar(alpha);
    request.epilogue.beta = runtimeScalar(beta);
    request.epilogue.outputScale = runtimeScalar(scaleD);
    if(outputType == ScalarType::Int8)
        request.epilogue.outputConversion = OutputConversion::SaturatingInt8;

    static constexpr int64_t blasThreshold = 600;
    const bool useBlas =
        rows != 0 && columns != 0 && reduction != 0 &&
        (m > blasThreshold || n > blasThreshold || k > blasThreshold ||
         lda > blasThreshold || ldb > blasThreshold || ldc > blasThreshold ||
         ldd > blasThreshold);
    if(useBlas)
    {
        static const TransformingBlasGemmBackend backend;
        referenceGemm(
            request,
            {
                .backend = GemmBackend::Blas,
                .requireRequestedBackend = true,
            },
            &backend);
    }
    else
    {
        referenceGemm(request);
    }
    const size_t outputBytes = storageBytesForLayout(outputType, layoutD);
    if(outputBytes != 0)
        std::memcpy(D, output.storage().data(), outputBytes);
}

void hipblaslt_reference_gemm(hipblasOperation_t   transA,
                              hipblasOperation_t   transB,
                              int64_t              m,
                              int64_t              n,
                              int64_t              k,
                              computeTypeInterface alpha,
                              const void*          A,
                              int64_t              lda,
                              const void*          B,
                              int64_t              ldb,
                              computeTypeInterface beta,
                              const void*          C,
                              int64_t              ldc,
                              void*                D,
                              int64_t              ldd,
                              const void*          AlphaVec,
                              const void*          scaleA,
                              const void*          scaleB,
                              const void*          scaleD,
                              bool                 isScaleAVec,
                              bool                 isScaleBVec,
                              hipDataType          tiA,
                              hipDataType          tiB,
                              hipDataType          tiC,
                              hipDataType          to,
                              hipDataType          tc,
                              hipDataType          tciA,
                              hipDataType          tciB,
                              bool                 isScaleAMXFormat,
                              bool                 isScaleBMXFormat)
{
    auto invoke = [&]<typename T>(T alphaValue,
                                  T betaValue,
                                  T scaleDValue,
                                  hipDataType computeInputA,
                                  hipDataType computeInputB) {
        hipblaslt_reference_gemm<T>(transA,
                                    transB,
                                    m,
                                    n,
                                    k,
                                    alphaValue,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    betaValue,
                                    C,
                                    ldc,
                                    D,
                                    ldd,
                                    AlphaVec,
                                    scaleA,
                                    scaleB,
                                    scaleDValue,
                                    isScaleAVec,
                                    isScaleBVec,
                                    tiA,
                                    tiB,
                                    tiC,
                                    to,
                                    computeInputA,
                                    computeInputB,
                                    isScaleAMXFormat,
                                    isScaleBMXFormat);
    };

    if(tiA == HIP_C_32F)
    {
        invoke(alpha.cf,
               beta.cf,
               *static_cast<const std::complex<float>*>(scaleD),
               tiA,
               tiB);
        return;
    }
    if(tiA == HIP_C_64F)
    {
        invoke(alpha.cd,
               beta.cd,
               *static_cast<const std::complex<double>*>(scaleD),
               tiA,
               tiB);
        return;
    }

    switch(tc)
    {
    case HIP_R_16F:
        invoke(alpha.f16, beta.f16, *static_cast<const hipblasLtHalf*>(scaleD), tciA, tciB);
        return;
    case HIP_R_32F:
        invoke(alpha.f32, beta.f32, *static_cast<const float*>(scaleD), tciA, tciB);
        return;
    case HIP_R_64F:
        invoke(alpha.f64, beta.f64, *static_cast<const double*>(scaleD), tciA, tciB);
        return;
    case HIP_R_32I:
        invoke(alpha.i32, beta.i32, *static_cast<const int32_t*>(scaleD), tciA, tciB);
        return;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_reference_gemm()" << std::endl;
        return;
    }
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
                                                const void*        C,                        \
                                                int64_t            ldc,                      \
                                                std::add_pointer_t<void> D,                  \
                                                int64_t            ldd,                      \
                                                const void*        AlphaVec,                 \
                                                const void*        scaleAVec,                 \
                                                const void*        scaleBVec,                 \
                                                Tc                 scaleD,                   \
                                                bool               isScaleAVec,              \
                                                bool               isScaleBVec,              \
                                                hipDataType        TiA,                      \
                                                hipDataType        TiB,                      \
                                                hipDataType        TiC,                      \
                                                hipDataType        To,                       \
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
