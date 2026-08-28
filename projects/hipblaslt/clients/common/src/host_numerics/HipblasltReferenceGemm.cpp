// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from hipBLASLt descriptors and host buffers to
// product-independent host-numerics operations.

#include <hipblaslt/host_numerics/HipblasltReferenceGemm.hpp>
#include <hipblaslt/host_numerics/Types.hpp>
#include <roc/host_numerics/backends/blas.hpp>
#include <roc/host_numerics/validation.hpp>

#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hipblaslt::host_numerics
{
    namespace
    {
        using namespace roc::host_numerics;

        template <typename T>
        Scalar runtimeScalar(const T& value)
        {
            const ScalarType type = scalarType<T>();
            return Scalar::fromStorage(type, std::as_bytes(std::span<const T>(&value, 1)));
        }

        template <typename Coefficient>
        constexpr ScalarType referenceAccumulatorType()
        {
            // I32 reference GEMM uses wide host arithmetic to avoid intermediate
            // overflow. F16 coefficients use an F32 host accumulator.
            if constexpr(std::is_same_v<Coefficient, int32_t>)
                return ScalarType::Float64;
            else if constexpr(std::is_same_v<Coefficient, hipblasLtHalf>)
                return ScalarType::Float32;
            else
                return scalarType<Coefficient>();
        }

        ScalarType referenceComputeType(hipDataType type)
        {
            if(type == HIP_R_32I)
                return ScalarType::Float64;
            return scalarType(type);
        }

        template <typename Coefficient>
        Tensor coefficientVector(const void* data, size_t elements)
        {
            const ScalarType type = scalarType<Coefficient>();
            return copyTensorFromEncodedStorage(
                data, type, Layout::contiguousLastDimensionFastest(Shape{elements}));
        }

        template <typename Coefficient>
        void runHipblasltReferenceGemm(const HipblasltReferenceGemmRequest& request,
                                       Coefficient                          alpha,
                                       Coefficient                          beta,
                                       Coefficient                          scaleC,
                                       Coefficient                          scaleD)
        {
            if(request.rows < 0 || request.columns < 0 || request.reduction < 0
               || request.leadingDimensionA < 0 || request.leadingDimensionB < 0
               || request.leadingDimensionC < 0 || request.leadingDimensionD < 0)
                throw std::invalid_argument(
                    "hipBLASLt reference GEMM dimensions and strides must be nonnegative.");

            const size_t     rows       = static_cast<size_t>(request.rows);
            const size_t     columns    = static_cast<size_t>(request.columns);
            const size_t     reduction  = static_cast<size_t>(request.reduction);
            const ScalarType typeA      = scalarType(request.typeA);
            const ScalarType typeB      = scalarType(request.typeB);
            const ScalarType typeC      = scalarType(request.typeC);
            const ScalarType outputType = scalarType(request.typeD);

            const ptrdiff_t aRowStride
                = request.operationA == HIPBLAS_OP_N ? 1 : request.leadingDimensionA;
            const ptrdiff_t aColumnStride
                = request.operationA == HIPBLAS_OP_N ? request.leadingDimensionA : 1;
            const ptrdiff_t bRowStride
                = request.operationB == HIPBLAS_OP_N ? 1 : request.leadingDimensionB;
            const ptrdiff_t bColumnStride
                = request.operationB == HIPBLAS_OP_N ? request.leadingDimensionB : 1;
            const Layout layoutA(Shape{rows, reduction}, {aRowStride, aColumnStride});
            const Layout layoutB(Shape{reduction, columns}, {bRowStride, bColumnStride});
            const Layout layoutC(Shape{rows, columns}, {1, request.leadingDimensionC});
            const Layout layoutD(Shape{rows, columns}, {1, request.leadingDimensionD});

            GemmOperand operandA(copyTensorFromEncodedStorage(request.a, typeA, layoutA));
            GemmOperand operandB(copyTensorFromEncodedStorage(request.b, typeB, layoutB));
            operandA.conjugate = request.operationA == HIPBLAS_OP_C;
            operandB.conjugate = request.operationB == HIPBLAS_OP_C;

            const ScalarType computeScalarTypeA = referenceComputeType(request.computeInputTypeA);
            const ScalarType computeScalarTypeB = referenceComputeType(request.computeInputTypeB);
            if(computeScalarTypeA != typeA)
                operandA.computeType = computeScalarTypeA;
            if(computeScalarTypeB != typeB)
                operandB.computeType = computeScalarTypeB;

            if(request.scaleA != nullptr && !request.scaleAIsMx)
                operandA.preQuantizationScales.push_back(
                    VectorBinding{coefficientVector<Coefficient>(request.scaleA,
                                                                 request.scaleAIsVector ? rows : 1),
                                  MatrixAxis::Row});
            if(request.alphaVector != nullptr)
                operandA.preQuantizationScales.push_back(VectorBinding{
                    coefficientVector<Coefficient>(request.alphaVector, rows), MatrixAxis::Row});
            if(request.scaleB != nullptr && !request.scaleBIsMx)
                operandB.preQuantizationScales.push_back(
                    VectorBinding{coefficientVector<Coefficient>(
                                      request.scaleB, request.scaleBIsVector ? columns : 1),
                                  MatrixAxis::Column});

            Tensor      output = copyTensorFromEncodedStorage(request.d, outputType, layoutD);
            GemmRequest gemm(std::move(operandA),
                             std::move(operandB),
                             copyTensorFromEncodedStorage(request.c, typeC, layoutC),
                             output,
                             referenceAccumulatorType<Coefficient>());
            gemm.epilogue.alpha       = runtimeScalar(alpha);
            gemm.epilogue.beta        = runtimeScalar(beta);
            gemm.epilogue.scaleC      = runtimeScalar(scaleC);
            gemm.epilogue.outputScale = runtimeScalar(scaleD);
            if(outputType == ScalarType::Int8)
                gemm.epilogue.outputConversion = OutputConversion::SaturatingInt8;

            referenceGemmWithBlasBackend(gemm);
            copyTensorEncodedBackingStorageToBuffer(
                request.d, storageBytesForLayout(outputType, layoutD), output);
        }
    } // namespace

    void hipblaslt_reference_gemm(const HipblasltReferenceGemmRequest& request)
    {
        auto invoke
            = [&]<typename Coefficient>(
                  Coefficient alpha, Coefficient beta, Coefficient scaleC, Coefficient scaleD) {
                  runHipblasltReferenceGemm(request, alpha, beta, scaleC, scaleD);
              };

        switch(request.coefficientType)
        {
        case HIP_C_32F:
        {
            // hipBLASLt's complex C scale is intentionally real-only.
            const auto scaleC
                = request.scaleC == nullptr
                      ? std::complex<float>(1.0f, 0.0f)
                      : std::complex<float>(
                            static_cast<const std::complex<float>*>(request.scaleC)->real(), 0.0f);
            invoke(request.alpha.cf,
                   request.beta.cf,
                   scaleC,
                   request.scaleD == nullptr
                       ? std::complex<float>(1.0f, 0.0f)
                       : *static_cast<const std::complex<float>*>(request.scaleD));
            return;
        }
        case HIP_C_64F:
        {
            // hipBLASLt's complex C scale is intentionally real-only.
            const auto scaleC
                = request.scaleC == nullptr
                      ? std::complex<double>(1.0, 0.0)
                      : std::complex<double>(
                            static_cast<const std::complex<double>*>(request.scaleC)->real(), 0.0);
            invoke(request.alpha.cd,
                   request.beta.cd,
                   scaleC,
                   request.scaleD == nullptr
                       ? std::complex<double>(1.0, 0.0)
                       : *static_cast<const std::complex<double>*>(request.scaleD));
            return;
        }
        case HIP_R_16F:
            invoke(request.alpha.f16,
                   request.beta.f16,
                   request.scaleC == nullptr ? hipblasLtHalf(1.0f)
                                             : *static_cast<const hipblasLtHalf*>(request.scaleC),
                   request.scaleD == nullptr ? hipblasLtHalf(1.0f)
                                             : *static_cast<const hipblasLtHalf*>(request.scaleD));
            return;
        case HIP_R_32F:
            invoke(request.alpha.f32,
                   request.beta.f32,
                   request.scaleC == nullptr ? 1.0f : *static_cast<const float*>(request.scaleC),
                   request.scaleD == nullptr ? 1.0f : *static_cast<const float*>(request.scaleD));
            return;
        case HIP_R_64F:
            invoke(request.alpha.f64,
                   request.beta.f64,
                   request.scaleC == nullptr ? 1.0 : *static_cast<const double*>(request.scaleC),
                   request.scaleD == nullptr ? 1.0 : *static_cast<const double*>(request.scaleD));
            return;
        case HIP_R_32I:
            invoke(request.alpha.i32,
                   request.beta.i32,
                   request.scaleC == nullptr ? int32_t{1}
                                             : *static_cast<const int32_t*>(request.scaleC),
                   request.scaleD == nullptr ? int32_t{1}
                                             : *static_cast<const int32_t*>(request.scaleD));
            return;
        default:
            throw std::invalid_argument(
                "hipBLASLt reference GEMM coefficient type is unsupported.");
        }
    }
} // namespace hipblaslt::host_numerics
