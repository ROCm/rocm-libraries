// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/client/MatmulPreparation.hpp>

#include "hipblaslt_datatype2string.hpp"
#include "utility.hpp"

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

    bool supportsMatmulSwizzle(hipDataType dataType)
    {
        switch(dataType)
        {
        case HIP_R_16BF:
        case HIP_R_16F:
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_4F_E2M1:
            return true;
        default:
            return false;
        }
    }

    bool usesRocrollerMxLayout()
    {
#ifdef HIPBLASLT_USE_ROCROLLER
        return hipblaslt_get_arch() != 1250;
#else
        return false;
#endif
    }

    hipblasLtOrder_t matmulOrderForDataType(hipDataType dataType)
    {
        switch(dataType)
        {
        case HIP_R_16F:
        case HIP_R_16BF:
            return HIPBLASLT_ORDER_COL16_4R8;
        case HIP_R_8F_E4M3_FNUZ:
            return HIPBLASLT_ORDER_COL16_4R16;
        case HIP_R_4F_E2M1:
            return HIPBLASLT_ORDER_COL16_4R32;
        default:
            throw std::runtime_error("Unsupported datatype for a swizzled matmul layout.");
        }
    }

    hipblasLtMatmulMatrixScale_t matmulScaleMode(hipblaslt_scaling_format format)
    {
        switch(format)
        {
        case hipblaslt_scaling_format::Vector:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        case hipblaslt_scaling_format::Block_32_UE8M0:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        case hipblaslt_scaling_format::Block_16_UE8M0:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE8M0_EXT;
        case hipblaslt_scaling_format::Block_32_UE4M3:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE4M3_EXT;
        case hipblaslt_scaling_format::Block_16_UE4M3:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        case hipblaslt_scaling_format::Block_32_UE5M3:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE5M3_EXT;
        case hipblaslt_scaling_format::Block_16_UE5M3:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE5M3_EXT;
        case hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT;
        default:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        }
    }

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

    MatmulPreparation prepareMatmulProblems(const Arguments&               arguments,
                                            std::span<const MatmulProblem> matmulProblems,
                                            hipDataType                    inputTypeA,
                                            hipDataType                    inputTypeB,
                                            hipDataType                    inputTypeC,
                                            hipDataType                    outputType,
                                            hipDataType                    computeScalarType,
                                            hipDataType                    coefficientType,
                                            hipDataType                    biasType,
                                            bool                           swizzleA,
                                            bool                           swizzleB,
                                            bool                           useRocrollerMxLayout)
    {
        MatmulPreparation preparation;
        preparation.problems.resize(matmulProblems.size());

        for(size_t index = 0; index < matmulProblems.size(); ++index)
        {
            const auto& problem         = matmulProblems[index];
            auto&       preparedProblem = preparation.problems[index];

            set_alpha_type(preparedProblem.alpha, arguments, computeScalarType, inputTypeA);
            set_beta_type(preparedProblem.beta, arguments, computeScalarType, inputTypeA);
            if(arguments.scaleAlpha_vector)
                set_computeInterface(preparedProblem.alpha, 1.0, computeScalarType, inputTypeA);

            preparedProblem.a.elements    = problem.a.allocationElements;
            preparedProblem.a.batchStride = problem.a.batchStride();
            if(swizzleA)
            {
                const auto parameters = matmulSwizzleParameters(inputTypeA, arguments.compute_type);
                constexpr int64_t microRows      = 16;
                const int64_t     reductionBlock = parameters.innerBlock * parameters.packingFactor;
                const int64_t swizzledStride = ((problem.m + microRows - 1) / microRows) * microRows
                                               * ((problem.k + reductionBlock - 1) / reductionBlock)
                                               * reductionBlock;
                if(problem.batchCount > 1 && problem.a.batchStride() != 0)
                {
                    preparedProblem.a.batchStride = swizzledStride;
                    preparedProblem.a.replacedUnsupportedBatchStride
                        = problem.a.batchStride()
                              != problem.a.leadingDimension() * problem.a.columns()
                          && problem.a.batchStride() != swizzledStride;
                }
                preparedProblem.a.elements = problem.batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY
                                                 ? swizzledStride
                                                 : problem.batchCount * swizzledStride;
            }

            preparedProblem.b.elements    = problem.b.allocationElements;
            preparedProblem.b.batchStride = problem.b.batchStride();
            if(swizzleB)
            {
                const auto parameters = matmulSwizzleParameters(inputTypeB, arguments.compute_type);
                constexpr int64_t microColumns   = 16;
                const int64_t     reductionBlock = parameters.innerBlock * parameters.packingFactor;
                const int64_t     swizzledStride
                    = ((problem.n + microColumns - 1) / microColumns) * microColumns
                      * ((problem.k + reductionBlock - 1) / reductionBlock) * reductionBlock;
                if(problem.batchCount > 1 && problem.b.batchStride() != 0)
                {
                    preparedProblem.b.batchStride = swizzledStride;
                    preparedProblem.b.replacedUnsupportedBatchStride
                        = problem.b.batchStride()
                              != problem.b.leadingDimension() * problem.b.columns()
                          && problem.b.batchStride() != swizzledStride;
                }
                preparedProblem.b.elements = problem.batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY
                                                 ? swizzledStride
                                                 : problem.batchCount * swizzledStride;
            }

            preparedProblem.outputCopyElements
                = arguments.unit_check || arguments.norm_check || arguments.allclose_check
                      ? problem.d.allocationElements
                      : 0;
            preparedProblem.scaleAlphaElements = arguments.scaleAlpha_vector ? problem.m : 0;

            if(problem.batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                if(arguments.scaleA == hipblaslt_scaling_format::Scalar)
                    preparedProblem.a.scaleElements = 1;
                else if(arguments.scaleA == hipblaslt_scaling_format::Vector)
                    preparedProblem.a.scaleElements = problem.m;
                else if(isBlockScaling(arguments.scaleA))
                {
                    if(useRocrollerMxLayout)
                    {
                        preparedProblem.a.scaleElements = scaleBufferSize(
                            problem.a.rows(), problem.a.columns(), arguments.scaleA);
                    }
                    else
                    {
                        const size_t scaleBlock = blockSize(arguments.scaleA);
                        const size_t tileCount  = 128 / scaleBlock;
                        const size_t scaleRows  = problem.operationA == HIPBLAS_OP_T
                                                      ? divideRoundUp(problem.a.rows(), scaleBlock)
                                                      : problem.a.rows();
                        const size_t scaleColumns
                            = problem.operationA == HIPBLAS_OP_T
                                  ? problem.a.columns()
                                  : divideRoundUp(problem.a.columns(), scaleBlock);
                        const bool   reductionAlongRows = problem.operationA == HIPBLAS_OP_T;
                        const size_t reductionExtent
                            = reductionAlongRows ? scaleRows : scaleColumns;
                        const size_t outputExtent = reductionAlongRows ? scaleColumns : scaleRows;
                        const size_t paddedExtent
                            = divideRoundUp(reductionAlongRows ? reductionExtent : outputExtent,
                                            tileCount)
                              * tileCount;
                        preparedProblem.a.scaleElements = reductionAlongRows
                                                              ? outputExtent * paddedExtent
                                                              : reductionExtent * paddedExtent;
                    }
                }

                if(arguments.scaleB == hipblaslt_scaling_format::Scalar)
                    preparedProblem.b.scaleElements = 1;
                else if(arguments.scaleB == hipblaslt_scaling_format::Vector)
                    preparedProblem.b.scaleElements = problem.n;
                else if(isBlockScaling(arguments.scaleB))
                {
                    if(useRocrollerMxLayout)
                    {
                        preparedProblem.b.scaleElements = scaleBufferSize(
                            problem.b.rows(), problem.b.columns(), arguments.scaleB);
                    }
                    else
                    {
                        const size_t scaleBlock = blockSize(arguments.scaleB);
                        const size_t tileCount  = 128 / scaleBlock;
                        const size_t scaleRows  = problem.operationB == HIPBLAS_OP_T
                                                      ? problem.b.rows()
                                                      : divideRoundUp(problem.b.rows(), scaleBlock);
                        const size_t scaleColumns
                            = problem.operationB == HIPBLAS_OP_T
                                  ? divideRoundUp(problem.b.columns(), scaleBlock)
                                  : problem.b.columns();
                        const bool   reductionAlongRows = problem.operationB == HIPBLAS_OP_N;
                        const size_t reductionExtent
                            = reductionAlongRows ? scaleRows : scaleColumns;
                        const size_t outputExtent = reductionAlongRows ? scaleColumns : scaleRows;
                        const size_t paddedExtent
                            = divideRoundUp(reductionAlongRows ? reductionExtent : outputExtent,
                                            tileCount)
                              * tileCount;
                        preparedProblem.b.scaleElements = reductionAlongRows
                                                              ? outputExtent * paddedExtent
                                                              : reductionExtent * paddedExtent;
                    }
                }

                if(arguments.bias_vector)
                {
                    if(arguments.bias_source == hipblaslt_bias_source::a
                       || arguments.bias_source == hipblaslt_bias_source::d)
                        preparedProblem.biasElements = problem.m;
                    else if(arguments.bias_source == hipblaslt_bias_source::b)
                        preparedProblem.biasElements = problem.n;

                    if(arguments.bias_stride > 0)
                        preparedProblem.biasElements = arguments.bias_stride * problem.batchCount;
                }

                preparedProblem.epilogue = matmulEpilogue(arguments);
                preparedProblem.epilogueEnabled
                    = preparedProblem.epilogue != HIPBLASLT_EPILOGUE_DEFAULT
                      || arguments.scaleAlpha_vector || arguments.amaxD;
                if(preparedProblem.epilogueEnabled)
                {
                    preparedProblem.activation0 = arguments.activation_arg1;
                    preparedProblem.activation1 = arguments.activation_arg2;
                }
            }
            else
            {
                if(arguments.scaleA == hipblaslt_scaling_format::Scalar)
                    preparedProblem.a.scaleElements = 1;
                if(arguments.scaleB == hipblaslt_scaling_format::Scalar)
                    preparedProblem.b.scaleElements = 1;
            }

            const size_t biasBytes = preparedProblem.biasElements * realDataTypeSize(biasType);
            const size_t inputCBytes
                = get_computeInterface(preparedProblem.beta, computeScalarType) == 0
                      ? 0
                      : problem.c.allocationElements * realDataTypeSize(inputTypeC);
            if(problem.batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                preparation.rotatingBytes
                    += preparedProblem.a.elements * realDataTypeSize(inputTypeA)
                       + preparedProblem.b.elements * realDataTypeSize(inputTypeB) + inputCBytes
                       + problem.d.allocationElements * realDataTypeSize(outputType)
                       + problem.auxiliaryAllocationElements() * realDataTypeSize(outputType)
                       + biasBytes
                       + preparedProblem.scaleAlphaElements * realDataTypeSize(coefficientType)
                       + preparedProblem.a.scaleElements * realDataTypeSize(coefficientType)
                       + preparedProblem.b.scaleElements * realDataTypeSize(coefficientType);
            }
            else
            {
                preparation.rotatingBytes
                    += preparedProblem.a.elements * realDataTypeSize(inputTypeA)
                           * problem.batchCount
                       + preparedProblem.b.elements * realDataTypeSize(inputTypeB)
                             * problem.batchCount
                       + inputCBytes * problem.batchCount
                       + problem.d.allocationElements * realDataTypeSize(outputType)
                             * problem.batchCount
                       + biasBytes
                       + preparedProblem.scaleAlphaElements * realDataTypeSize(coefficientType)
                       + preparedProblem.a.scaleElements * realDataTypeSize(coefficientType)
                       + preparedProblem.b.scaleElements * realDataTypeSize(coefficientType);
            }
        }

        return preparation;
    }
} // namespace hipblaslt::client
