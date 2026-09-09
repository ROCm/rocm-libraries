// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from hipBLASLt descriptors and host buffers to
// product-independent host-numerics operations.

#include <hipblaslt/host_numerics/HipblasltReferenceGemm.hpp>
#include <hipblaslt/host_numerics/Types.hpp>
#include <roc/host_numerics/backends/blas.hpp>
#include <roc/host_numerics/epilogue.hpp>
#include <roc/host_numerics/tensor_operations.hpp>

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
                             const MatmulReferenceInputs&                    inputs,
                             hipblaslt_scaling_format                        scaleAMode,
                             hipblaslt_scaling_format                        scaleBMode)
    {
        using namespace roc::host_numerics;

        const ScalarType computeTypeA = isBlockScaling(scaleAMode)
                                            ? inputs.a.type()
                                            : referenceComputeType(dataTypes.computeInputA);
        const ScalarType computeTypeB = isBlockScaling(scaleBMode)
                                            ? inputs.b.type()
                                            : referenceComputeType(dataTypes.computeInputB);

        const ScalarType accumulatorType = referenceAccumulatorType(dataTypes.coefficient);
        MatmulOptions    options(accumulatorType);
        options.conjugateA = problem.operationA == HIPBLAS_OP_C;
        options.conjugateB = problem.operationB == HIPBLAS_OP_C;
        if(computeTypeA != inputs.a.type())
            options.computeTypeA = computeTypeA;
        if(computeTypeB != inputs.b.type())
            options.computeTypeB = computeTypeB;
        if(inputs.scaleA && !isBlockScaling(scaleAMode))
            options.preQuantizationScalesA.push_back(inputs.scaleA->expandDims(1));
        if(inputs.alphaVector)
            options.preQuantizationScalesA.push_back(inputs.alphaVector->expandDims(1));
        if(inputs.scaleB && !isBlockScaling(scaleBMode))
            options.preQuantizationScalesB.push_back(inputs.scaleB->expandDims(0));
        const Tensor alpha = scalarValue(preparation.alpha, dataTypes.coefficient);
        const Tensor beta  = scalarValue(preparation.beta, dataTypes.coefficient);
        const Tensor scaleC = inputs.scaleC.value_or(Tensor::scalar(accumulatorType, 1));
        const Tensor outputScale = inputs.scaleD.value_or(Tensor::scalar(accumulatorType, 1));

        const Shape outputShape{inputs.a.shape()[0], inputs.b.shape()[1]};
        std::optional<Tensor> result;
        if(alpha.item<std::complex<double>>() != std::complex<double>(0.0, 0.0)
           && inputs.a.shape()[1] != 0)
        {
            const Tensor product
                = matmulWithBlasBackend(inputs.a, inputs.b, accumulatorType, options);
            result = multiply(product, alpha, accumulatorType, accumulatorType);
        }
        if(beta.item<std::complex<double>>() != std::complex<double>(0.0, 0.0))
        {
            const Tensor cScale = multiply(beta, scaleC, accumulatorType, accumulatorType);
            Tensor       addend = multiply(inputs.c, cScale, accumulatorType, accumulatorType);
            result              = result ? add(*result, addend, accumulatorType, accumulatorType)
                                         : std::move(addend);
        }
        if(!result)
            result.emplace(accumulatorType, outputShape);

        EpilogueOptions epilogue(accumulatorType);
        epilogue.outputScale = outputScale;
        if(inputs.d.type() == ScalarType::Int8)
            epilogue.outputConversion = OutputConversion::SaturatingInt8;
        referenceEpilogueInto(*result, {.output = inputs.d}, epilogue);
    }
} // namespace hipblaslt::host_numerics
