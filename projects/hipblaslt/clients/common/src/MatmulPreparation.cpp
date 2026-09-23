// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/client/MatmulPreparation.hpp>

#include "hipblaslt_datatype2string.hpp"
#include "utility.hpp"

#include <array>
#include <stdexcept>

namespace hipblaslt::client
{
    namespace
    {
        size_t divideRoundUp(size_t value, size_t divisor)
        {
            return value / divisor + static_cast<size_t>(value % divisor != 0);
        }

        roc::host_numerics::amd_gpu_layout::MxScaleStoragePlan mxScaleStoragePlan(
            const MatmulMatrix&                                      matrix,
            size_t                                                   blockAxis,
            size_t                                                   blockSize,
            roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout physicalLayout)
        {
            const size_t                blockedExtent = matrix.layout.shape()[blockAxis];
            const size_t                freeExtent    = matrix.layout.shape()[1 - blockAxis];
            const std::array<size_t, 2> naturalShape
                = blockAxis == 0
                      ? std::array<size_t, 2>{freeExtent, divideRoundUp(blockedExtent, blockSize)}
                      : std::array<size_t, 2>{divideRoundUp(blockedExtent, blockSize), freeExtent};
            return roc::host_numerics::amd_gpu_layout::planMxScaleStorage(
                naturalShape, blockSize, physicalLayout);
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
        case hipblaslt_scaling_format::none:
        case hipblaslt_scaling_format::Scalar:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        default:
            throw std::invalid_argument("Invalid hipBLASLt matrix scaling format.");
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
        case hipblaslt_activation_type::none:
        case hipblaslt_activation_type::sigmoid:
            // hipBLASLt has no sigmoid epilogue. Preserve the existing client
            // behavior: sigmoid selects no activation but may still select bias.
            if(arguments.bias_vector)
                epilogue = HIPBLASLT_EPILOGUE_BIAS;
            break;
        default:
            throw std::invalid_argument("Invalid hipBLASLt activation type.");
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

    MatmulPreparation
        prepareMatmulProblems(const Arguments&               arguments,
                              std::span<const MatmulProblem> matmulProblems,
                              const MatmulDataTypes&         dataTypes,
                              bool                           swizzleA,
                              bool                           swizzleB,
                              roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout scaleLayoutA,
                              roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout scaleLayoutB)
    {
        MatmulPreparation preparation;
        preparation.problems.resize(matmulProblems.size());

        for(size_t index = 0; index < matmulProblems.size(); ++index)
        {
            const auto& problem         = matmulProblems[index];
            auto&       preparedProblem = preparation.problems[index];

            set_alpha_type(
                preparedProblem.alpha, arguments, dataTypes.computeScalar, problem.a.apiType);
            set_beta_type(
                preparedProblem.beta, arguments, dataTypes.computeScalar, problem.a.apiType);
            if(arguments.scaleAlpha_vector)
                set_compute_type_value_from_double(
                    preparedProblem.alpha, 1.0, dataTypes.computeScalar, problem.a.apiType);

            preparedProblem.a.elements    = problem.a.allocationElements;
            preparedProblem.a.batchStride = problem.a.batchStride();
            if(swizzleA)
            {
                const auto parameters
                    = matmulSwizzleParameters(problem.a.apiType, arguments.compute_type);
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
                const auto parameters
                    = matmulSwizzleParameters(problem.b.apiType, arguments.compute_type);
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
                    const size_t scaleBlock = blockSize(arguments.scaleA);
                    const size_t blockAxis  = problem.operationA == HIPBLAS_OP_T ? 0 : 1;
                    preparedProblem.a.mxScaleStorage
                        = mxScaleStoragePlan(problem.a, blockAxis, scaleBlock, scaleLayoutA);
                    preparedProblem.a.scaleElements
                        = preparedProblem.a.mxScaleStorage->physicalByteCount;
                }

                if(arguments.scaleB == hipblaslt_scaling_format::Scalar)
                    preparedProblem.b.scaleElements = 1;
                else if(arguments.scaleB == hipblaslt_scaling_format::Vector)
                    preparedProblem.b.scaleElements = problem.n;
                else if(isBlockScaling(arguments.scaleB))
                {
                    const size_t scaleBlock = blockSize(arguments.scaleB);
                    const size_t blockAxis  = problem.operationB == HIPBLAS_OP_N ? 0 : 1;
                    preparedProblem.b.mxScaleStorage
                        = mxScaleStoragePlan(problem.b, blockAxis, scaleBlock, scaleLayoutB);
                    preparedProblem.b.scaleElements
                        = preparedProblem.b.mxScaleStorage->physicalByteCount;
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
            }
            else
            {
                if(arguments.scaleA == hipblaslt_scaling_format::Scalar)
                    preparedProblem.a.scaleElements = 1;
                if(arguments.scaleB == hipblaslt_scaling_format::Scalar)
                    preparedProblem.b.scaleElements = 1;
            }

            const size_t biasBytes
                = preparedProblem.biasElements * realDataTypeSize(dataTypes.biasStorage);
            const size_t inputCBytes
                = compute_type_value_as_double(preparedProblem.beta, dataTypes.computeScalar) == 0
                      ? 0
                      : problem.c.allocationElements * realDataTypeSize(problem.c.apiType);
            const auto scaleBytes = [&](const PreparedMatmulOperand& operand,
                                        hipblaslt_scaling_format      format) {
                return isBlockScaling(format)
                           ? operand.scaleElements * static_cast<size_t>(problem.batchCount)
                           : operand.scaleElements * realDataTypeSize(dataTypes.coefficient);
            };
            if(problem.batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                preparation.rotatingBytes
                    += preparedProblem.a.elements * realDataTypeSize(problem.a.apiType)
                       + preparedProblem.b.elements * realDataTypeSize(problem.b.apiType)
                       + inputCBytes
                       + problem.d.allocationElements * realDataTypeSize(problem.d.apiType)
                       + problem.auxiliaryAllocationElements()
                             * realDataTypeSize(dataTypes.auxiliary)
                       + biasBytes
                       + preparedProblem.scaleAlphaElements
                             * realDataTypeSize(dataTypes.coefficient)
                       + scaleBytes(preparedProblem.a, arguments.scaleA)
                       + scaleBytes(preparedProblem.b, arguments.scaleB);
            }
            else
            {
                preparation.rotatingBytes
                    += preparedProblem.a.elements * realDataTypeSize(problem.a.apiType)
                           * problem.batchCount
                       + preparedProblem.b.elements * realDataTypeSize(problem.b.apiType)
                             * problem.batchCount
                       + inputCBytes * problem.batchCount
                       + problem.d.allocationElements * realDataTypeSize(problem.d.apiType)
                             * problem.batchCount
                       + biasBytes
                       + preparedProblem.scaleAlphaElements
                             * realDataTypeSize(dataTypes.coefficient)
                       + scaleBytes(preparedProblem.a, arguments.scaleA)
                       + scaleBytes(preparedProblem.b, arguments.scaleB);
            }
        }

        return preparation;
    }
} // namespace hipblaslt::client
