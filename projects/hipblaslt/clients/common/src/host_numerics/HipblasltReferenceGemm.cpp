// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from hipBLASLt descriptors and host buffers to
// product-independent host-numerics operations.

#include <hipblaslt/host_numerics/HipblasltReferenceGemm.hpp>
#include <hipblaslt/host_numerics/Types.hpp>
#include <roc/host_numerics/backends/blas.hpp>
#include <roc/host_numerics/validation.hpp>

#include <utility>

namespace hipblaslt::host_numerics
{
    roc::host_numerics::GemmRunInfo
        referenceMatmulGemm(const hipblaslt::client::MatmulProblem&         problem,
                            const hipblaslt::client::MatmulDataTypes&       dataTypes,
                            const hipblaslt::client::PreparedMatmulProblem& preparation,
                            MatmulReferenceInputs                            inputs,
                            hipblaslt_scaling_format                         scaleAMode,
                            hipblaslt_scaling_format                         scaleBMode)
    {
        using namespace roc::host_numerics;

        GemmOperand operandA(std::move(inputs.a));
        GemmOperand operandB(std::move(inputs.b));
        operandA.conjugate = problem.operationA == HIPBLAS_OP_C;
        operandB.conjugate = problem.operationB == HIPBLAS_OP_C;

        const ScalarType computeTypeA
            = isBlockScaling(scaleAMode) ? operandA.values.type()
                                         : referenceComputeType(dataTypes.computeInputA);
        const ScalarType computeTypeB
            = isBlockScaling(scaleBMode) ? operandB.values.type()
                                         : referenceComputeType(dataTypes.computeInputB);
        if(computeTypeA != operandA.values.type())
            operandA.computeType = computeTypeA;
        if(computeTypeB != operandB.values.type())
            operandB.computeType = computeTypeB;

        if(inputs.scaleA && !isBlockScaling(scaleAMode))
            operandA.preQuantizationScales.emplace_back(*inputs.scaleA, MatrixAxis::Row);
        if(inputs.alphaVector)
            operandA.preQuantizationScales.emplace_back(*inputs.alphaVector, MatrixAxis::Row);
        if(inputs.scaleB && !isBlockScaling(scaleBMode))
            operandB.preQuantizationScales.emplace_back(*inputs.scaleB, MatrixAxis::Column);

        GemmRequest request(std::move(operandA),
                            std::move(operandB),
                            std::move(inputs.c),
                            std::move(inputs.d),
                            referenceAccumulatorType(dataTypes.computeScalar));
        request.epilogue.alpha  = scalarValue(preparation.alpha, dataTypes.computeScalar);
        request.epilogue.beta   = scalarValue(preparation.beta, dataTypes.computeScalar);
        request.epilogue.scaleC = inputs.scaleC.value_or(
            Scalar::one(scalarType(dataTypes.computeScalar)));
        request.epilogue.outputScale = inputs.scaleD.value_or(
            Scalar::one(scalarType(dataTypes.computeScalar)));
        if(request.d.type() == ScalarType::Int8)
            request.epilogue.outputConversion = OutputConversion::SaturatingInt8;
        return referenceGemmWithBlasBackend(request);
    }
} // namespace hipblaslt::host_numerics
