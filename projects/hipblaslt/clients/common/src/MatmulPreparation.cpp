// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/client/MatmulPreparation.hpp>

#include "hipblaslt_datatype2string.hpp"

#include <stdexcept>

namespace hipblaslt::client
{
    namespace
    {
        size_t divideRoundUp(size_t value, size_t divisor)
        {
            return value / divisor + static_cast<size_t>(value % divisor != 0);
        }
    } // namespace

    hipblasLtEpilogue_t matmulEpilogue(const Arguments& arguments)
    {
        hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
        switch(arguments.activation_type)
        {
        case hipblaslt_activation_type::relu:
            epilogue
                = arguments.bias_vector ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_RELU;
            break;
        case hipblaslt_activation_type::gelu:
            epilogue
                = arguments.bias_vector ? HIPBLASLT_EPILOGUE_GELU_BIAS : HIPBLASLT_EPILOGUE_GELU;
            break;
        case hipblaslt_activation_type::swish:
            epilogue = arguments.bias_vector ? HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT
                                             : HIPBLASLT_EPILOGUE_SWISH_EXT;
            break;
        case hipblaslt_activation_type::clamp:
            epilogue = arguments.bias_vector ? HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT
                                             : HIPBLASLT_EPILOGUE_CLAMP_EXT;
            break;
        default:
            if(arguments.bias_vector)
                epilogue = HIPBLASLT_EPILOGUE_BIAS;
            break;
        }

        if(arguments.gradient)
        {
            switch(epilogue)
            {
            case HIPBLASLT_EPILOGUE_BIAS:
                if(arguments.bias_source == hipblaslt_bias_source::a)
                    epilogue = HIPBLASLT_EPILOGUE_BGRADA;
                else if(arguments.bias_source == hipblaslt_bias_source::b)
                    epilogue = HIPBLASLT_EPILOGUE_BGRADB;
                break;
            case HIPBLASLT_EPILOGUE_GELU:
                epilogue = HIPBLASLT_EPILOGUE_DGELU;
                break;
            case HIPBLASLT_EPILOGUE_GELU_BIAS:
                epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
                break;
            case HIPBLASLT_EPILOGUE_RELU:
                epilogue = HIPBLASLT_EPILOGUE_DRELU;
                break;
            case HIPBLASLT_EPILOGUE_RELU_BIAS:
                epilogue = HIPBLASLT_EPILOGUE_DRELU_BGRAD;
                break;
            default:
                break;
            }
            if((epilogue == HIPBLASLT_EPILOGUE_DGELU || epilogue == HIPBLASLT_EPILOGUE_DGELU_BGRAD
                || epilogue == HIPBLASLT_EPILOGUE_DRELU
                || epilogue == HIPBLASLT_EPILOGUE_DRELU_BGRAD)
               && !arguments.use_e)
            {
                throw std::invalid_argument(
                    "Gradient ReLU/GELU matmul requires auxiliary E storage.");
            }
        }

        if(!arguments.use_e)
            return epilogue;
        switch(epilogue)
        {
        case HIPBLASLT_EPILOGUE_RELU:
            return HIPBLASLT_EPILOGUE_RELU_AUX;
        case HIPBLASLT_EPILOGUE_RELU_BIAS:
            return HIPBLASLT_EPILOGUE_RELU_AUX_BIAS;
        case HIPBLASLT_EPILOGUE_GELU:
            return HIPBLASLT_EPILOGUE_GELU_AUX;
        case HIPBLASLT_EPILOGUE_GELU_BIAS:
            return HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
        case HIPBLASLT_EPILOGUE_CLAMP_EXT:
            return HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT;
        case HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
            return HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT;
        case HIPBLASLT_EPILOGUE_DGELU:
        case HIPBLASLT_EPILOGUE_DGELU_BGRAD:
        case HIPBLASLT_EPILOGUE_DRELU:
        case HIPBLASLT_EPILOGUE_DRELU_BGRAD:
            return epilogue;
        default:
            throw std::invalid_argument("Selected matmul epilogue does not support auxiliary E.");
        }
    }

    MatmulSwizzleParameters matmulSwizzleParameters(hipDataType          dataType,
                                                    hipblasComputeType_t computeType)
    {
        MatmulSwizzleParameters parameters{};
        switch(dataType)
        {
        case HIP_R_32F:
            if(computeType == HIPBLAS_COMPUTE_32F_FAST_TF32)
            {
                parameters.innerBlock  = 8;
                parameters.vectorWidth = 2;
            }
            else
            {
                parameters.innerBlock  = 4;
                parameters.vectorWidth = 1;
            }
            break;
        case HIP_R_64F:
            parameters.innerBlock  = 4;
            parameters.vectorWidth = 1;
            break;
        case HIP_R_16F:
        case HIP_R_16BF:
            parameters.innerBlock  = 16;
            parameters.vectorWidth = 4;
            break;
        case HIP_R_8I:
        case HIP_R_8F_E5M2_FNUZ:
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E4M3:
        case HIP_R_8F_E5M2:
            parameters.innerBlock  = 32;
            parameters.vectorWidth = 8;
            break;
        case HIP_R_4F_E2M1:
            parameters.innerBlock  = 16;
            parameters.vectorWidth = 8;
            break;
        default:
            throw std::runtime_error("Unsupported datatype for matmul swizzling.");
        }

        parameters.packingFactor = 16 / parameters.vectorWidth / realDataTypeSize(dataType);
        return parameters;
    }

    MatmulPreparation prepareMatmulCases(const Arguments&                arguments,
                                         std::span<const MatmulTestCase> matmulCases,
                                         hipDataType                     inputTypeA,
                                         hipDataType                     inputTypeB,
                                         hipDataType                     inputTypeC,
                                         hipDataType                     outputType,
                                         hipDataType                     computeScalarType,
                                         hipDataType                     coefficientType,
                                         hipDataType                     biasType,
                                         bool                            swizzleA,
                                         bool                            swizzleB,
                                         bool                            useRocrollerMxLayout)
    {
        MatmulPreparation preparation;
        preparation.cases.resize(matmulCases.size());

        for(size_t index = 0; index < matmulCases.size(); ++index)
        {
            const auto& testCase     = matmulCases[index];
            auto&       preparedCase = preparation.cases[index];

            set_alpha_type(preparedCase.alpha, arguments, computeScalarType, inputTypeA);
            set_beta_type(preparedCase.beta, arguments, computeScalarType, inputTypeA);
            if(arguments.scaleAlpha_vector)
                set_computeInterface(preparedCase.alpha, 1.0, computeScalarType, inputTypeA);

            preparedCase.a.elements    = testCase.a.allocationElements;
            preparedCase.a.batchStride = testCase.a.batchStride();
            if(swizzleA)
            {
                const auto parameters = matmulSwizzleParameters(inputTypeA, arguments.compute_type);
                constexpr int64_t microRows      = 16;
                const int64_t     reductionBlock = parameters.innerBlock * parameters.packingFactor;
                const int64_t     swizzledStride
                    = ((testCase.m + microRows - 1) / microRows) * microRows
                      * ((testCase.k + reductionBlock - 1) / reductionBlock) * reductionBlock;
                if(testCase.batchCount > 1 && testCase.a.batchStride() != 0)
                {
                    preparedCase.a.batchStride = swizzledStride;
                    preparedCase.a.replacedUnsupportedBatchStride
                        = testCase.a.batchStride()
                              != testCase.a.leadingDimension() * testCase.a.columns()
                          && testCase.a.batchStride() != swizzledStride;
                }
                preparedCase.a.elements = testCase.batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY
                                              ? swizzledStride
                                              : testCase.batchCount * swizzledStride;
            }

            preparedCase.b.elements    = testCase.b.allocationElements;
            preparedCase.b.batchStride = testCase.b.batchStride();
            if(swizzleB)
            {
                const auto parameters = matmulSwizzleParameters(inputTypeB, arguments.compute_type);
                constexpr int64_t microColumns   = 16;
                const int64_t     reductionBlock = parameters.innerBlock * parameters.packingFactor;
                const int64_t     swizzledStride
                    = ((testCase.n + microColumns - 1) / microColumns) * microColumns
                      * ((testCase.k + reductionBlock - 1) / reductionBlock) * reductionBlock;
                if(testCase.batchCount > 1 && testCase.b.batchStride() != 0)
                {
                    preparedCase.b.batchStride = swizzledStride;
                    preparedCase.b.replacedUnsupportedBatchStride
                        = testCase.b.batchStride()
                              != testCase.b.leadingDimension() * testCase.b.columns()
                          && testCase.b.batchStride() != swizzledStride;
                }
                preparedCase.b.elements = testCase.batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY
                                              ? swizzledStride
                                              : testCase.batchCount * swizzledStride;
            }

            preparedCase.outputCopyElements
                = arguments.unit_check || arguments.norm_check || arguments.allclose_check
                      ? testCase.d.allocationElements
                      : 0;
            preparedCase.scaleAlphaElements = arguments.scaleAlpha_vector ? testCase.m : 0;

            if(testCase.batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                if(arguments.scaleA == hipblaslt_scaling_format::Scalar)
                    preparedCase.a.scaleElements = 1;
                else if(arguments.scaleA == hipblaslt_scaling_format::Vector)
                    preparedCase.a.scaleElements = testCase.m;
                else if(isBlockScaling(arguments.scaleA))
                {
                    if(useRocrollerMxLayout)
                    {
                        preparedCase.a.scaleElements = scaleBufferSize(
                            testCase.a.rows(), testCase.a.columns(), arguments.scaleA);
                    }
                    else
                    {
                        const size_t scaleBlock = blockSize(arguments.scaleA);
                        const size_t tileCount  = 128 / scaleBlock;
                        const size_t scaleRows  = testCase.operationA == HIPBLAS_OP_T
                                                      ? divideRoundUp(testCase.a.rows(), scaleBlock)
                                                      : testCase.a.rows();
                        const size_t scaleColumns
                            = testCase.operationA == HIPBLAS_OP_T
                                  ? testCase.a.columns()
                                  : divideRoundUp(testCase.a.columns(), scaleBlock);
                        const bool   reductionAlongRows = testCase.operationA == HIPBLAS_OP_T;
                        const size_t reductionExtent
                            = reductionAlongRows ? scaleRows : scaleColumns;
                        const size_t outputExtent = reductionAlongRows ? scaleColumns : scaleRows;
                        const size_t paddedExtent
                            = divideRoundUp(reductionAlongRows ? reductionExtent : outputExtent,
                                            tileCount)
                              * tileCount;
                        preparedCase.a.scaleElements = reductionAlongRows
                                                           ? outputExtent * paddedExtent
                                                           : reductionExtent * paddedExtent;
                    }
                }

                if(arguments.scaleB == hipblaslt_scaling_format::Scalar)
                    preparedCase.b.scaleElements = 1;
                else if(arguments.scaleB == hipblaslt_scaling_format::Vector)
                    preparedCase.b.scaleElements = testCase.n;
                else if(isBlockScaling(arguments.scaleB))
                {
                    if(useRocrollerMxLayout)
                    {
                        preparedCase.b.scaleElements = scaleBufferSize(
                            testCase.b.rows(), testCase.b.columns(), arguments.scaleB);
                    }
                    else
                    {
                        const size_t scaleBlock = blockSize(arguments.scaleB);
                        const size_t tileCount  = 128 / scaleBlock;
                        const size_t scaleRows = testCase.operationB == HIPBLAS_OP_T
                                                     ? testCase.b.rows()
                                                     : divideRoundUp(testCase.b.rows(), scaleBlock);
                        const size_t scaleColumns
                            = testCase.operationB == HIPBLAS_OP_T
                                  ? divideRoundUp(testCase.b.columns(), scaleBlock)
                                  : testCase.b.columns();
                        const bool   reductionAlongRows = testCase.operationB == HIPBLAS_OP_N;
                        const size_t reductionExtent
                            = reductionAlongRows ? scaleRows : scaleColumns;
                        const size_t outputExtent = reductionAlongRows ? scaleColumns : scaleRows;
                        const size_t paddedExtent
                            = divideRoundUp(reductionAlongRows ? reductionExtent : outputExtent,
                                            tileCount)
                              * tileCount;
                        preparedCase.b.scaleElements = reductionAlongRows
                                                           ? outputExtent * paddedExtent
                                                           : reductionExtent * paddedExtent;
                    }
                }

                if(arguments.bias_vector)
                {
                    if(arguments.bias_source == hipblaslt_bias_source::a
                       || arguments.bias_source == hipblaslt_bias_source::d)
                        preparedCase.biasElements = testCase.m;
                    else if(arguments.bias_source == hipblaslt_bias_source::b)
                        preparedCase.biasElements = testCase.n;

                    if(arguments.bias_stride > 0)
                        preparedCase.biasElements = arguments.bias_stride * testCase.batchCount;
                }

                preparedCase.epilogue        = matmulEpilogue(arguments);
                preparedCase.epilogueEnabled = preparedCase.epilogue != HIPBLASLT_EPILOGUE_DEFAULT
                                               || arguments.scaleAlpha_vector || arguments.amaxD;
                if(preparedCase.epilogueEnabled)
                {
                    preparedCase.activation0 = arguments.activation_arg1;
                    preparedCase.activation1 = arguments.activation_arg2;
                }
            }
            else
            {
                if(arguments.scaleA == hipblaslt_scaling_format::Scalar)
                    preparedCase.a.scaleElements = 1;
                if(arguments.scaleB == hipblaslt_scaling_format::Scalar)
                    preparedCase.b.scaleElements = 1;
            }

            const size_t biasBytes = preparedCase.biasElements * realDataTypeSize(biasType);
            const size_t inputCBytes
                = get_computeInterface(preparedCase.beta, computeScalarType) == 0
                      ? 0
                      : testCase.c.allocationElements * realDataTypeSize(inputTypeC);
            if(testCase.batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                preparation.rotatingBytes
                    += preparedCase.a.elements * realDataTypeSize(inputTypeA)
                       + preparedCase.b.elements * realDataTypeSize(inputTypeB) + inputCBytes
                       + testCase.d.allocationElements * realDataTypeSize(outputType)
                       + testCase.auxiliaryAllocationElements() * realDataTypeSize(outputType)
                       + biasBytes
                       + preparedCase.scaleAlphaElements * realDataTypeSize(coefficientType)
                       + preparedCase.a.scaleElements * realDataTypeSize(coefficientType)
                       + preparedCase.b.scaleElements * realDataTypeSize(coefficientType);
            }
            else
            {
                preparation.rotatingBytes
                    += preparedCase.a.elements * realDataTypeSize(inputTypeA) * testCase.batchCount
                       + preparedCase.b.elements * realDataTypeSize(inputTypeB)
                             * testCase.batchCount
                       + inputCBytes * testCase.batchCount
                       + testCase.d.allocationElements * realDataTypeSize(outputType)
                             * testCase.batchCount
                       + biasBytes
                       + preparedCase.scaleAlphaElements * realDataTypeSize(coefficientType)
                       + preparedCase.a.scaleElements * realDataTypeSize(coefficientType)
                       + preparedCase.b.scaleElements * realDataTypeSize(coefficientType);
            }
        }

        return preparation;
    }
} // namespace hipblaslt::client
