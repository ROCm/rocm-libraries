// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from hipBLASLt descriptors and host buffers to
// product-independent host-numerics operations.

#include <hipblaslt/host_numerics/HipblasltReferenceGemm.hpp>
#include <hipblaslt/host_numerics/Types.hpp>
#include <roc/host_numerics/backends/blas.hpp>

#include <array>
#include <utility>

namespace hipblaslt::host_numerics
{
    roc::host_numerics::Layout referenceBatchLayout(const hipblaslt::client::MatmulMatrix& matrix,
                                                    size_t                                 rows,
                                                    size_t                                 columns,
                                                    hipblasOperation_t operation,
                                                    size_t             batch,
                                                    bool               separateBatchStorage)
    {
        using roc::host_numerics::Layout;
        using roc::host_numerics::Shape;

        const ptrdiff_t rowStride    = operation == HIPBLAS_OP_N ? 1 : matrix.layout.stride(1);
        const ptrdiff_t columnStride = operation == HIPBLAS_OP_N ? matrix.layout.stride(1) : 1;

        ptrdiff_t offset = 0;
        if(!separateBatchStorage)
        {
            // A zero matrix extent has no valid {0, 0, batch} coordinate, but
            // its batch base is still well-defined. Use a non-empty address
            // layout to retain Layout's checked offset arithmetic without
            // pretending that the empty matrix contains an element.
            const Layout batchAddressLayout(Shape{1, 1, matrix.layout.shape().extent(2)},
                                            {0, 0, matrix.layout.stride(2)},
                                            matrix.layout.offset());
            const std::array<size_t, 3> batchCoordinates{0, 0, batch};
            offset = batchAddressLayout.elementOffset(batchCoordinates);
        }

        return Layout(Shape{rows, columns}, {rowStride, columnStride}, offset);
    }

    void referenceMatmulGemm(const hipblaslt::client::MatmulProblem&         problem,
                             const hipblaslt::client::MatmulDataTypes&       dataTypes,
                             const hipblaslt::client::PreparedMatmulProblem& preparation,
                             MatmulReferenceInputs                           inputs,
                             hipblaslt_scaling_format                        scaleAMode,
                             hipblaslt_scaling_format                        scaleBMode)
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

        const ScalarType accumulatorType = referenceAccumulatorType(dataTypes.coefficient);
        GemmOptions      options(accumulatorType);
        options.epilogue.alpha       = scalarValue(preparation.alpha, dataTypes.coefficient);
        options.epilogue.beta        = scalarValue(preparation.beta, dataTypes.coefficient);
        options.epilogue.scaleC      = inputs.scaleC.value_or(Scalar::one(accumulatorType));
        options.epilogue.outputScale = inputs.scaleD.value_or(Scalar::one(accumulatorType));
        if(inputs.d.type() == ScalarType::Int8)
            options.epilogue.outputConversion = OutputConversion::SaturatingInt8;
        (void)referenceGemmIntoWithBlasBackend(std::move(operandA),
                                               std::move(operandB),
                                               std::move(inputs.c),
                                               std::move(inputs.d),
                                               options);
    }
} // namespace hipblaslt::host_numerics
