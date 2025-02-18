/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/
#include <hiptensor/hiptensor.hpp>
#include <set>
#include <unordered_set>

#include "handle.hpp"
#include "hip_device.hpp"
#include "logger.hpp"

#include "ck/ck.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_reduce.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "reduction_solution.hpp"
#include "reduction_solution_instances.hpp"
#include "reduction_solution_registry.hpp"

#include "hiptensor_options.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

namespace
{
    hiptensorStatus_t checkReductionInputData(const hiptensorHandle_t*           handle,
                                              const void*                        alpha,
                                              const void*                        A,
                                              const hiptensorTensorDescriptor_t* descA,
                                              const int32_t*                     modeA,
                                              const void*                        beta,
                                              const void*                        C,
                                              const hiptensorTensorDescriptor_t* descC,
                                              const int32_t*                     modeC,
                                              void*                              D,
                                              const hiptensorTensorDescriptor_t* descD,
                                              const int32_t*                     modeD,
                                              hiptensorOperator_t                opReduce,
                                              hiptensorComputeType_t             typeCompute,
                                              void*                              workspace,
                                              uint64_t                           workspaceSize)
    {
        // Log API access
        using hiptensor::Logger;
        auto& logger = Logger::instance();
        char  msg[2048];
        if(!handle || !alpha || !A || !descA || !modeA || !beta || !descC || !D || !descD)
        {
            auto errorCode         = HIPTENSOR_STATUS_NOT_INITIALIZED;
            auto printErrorMessage = [&logger, errorCode](const std::string& paramName) {
                char msg[512];
                snprintf(msg,
                         sizeof(msg),
                         "Initialization Error : %s = nullptr (%s)",
                         paramName.c_str(),
                         hiptensorGetErrorString(errorCode));
                logger->logError("hiptensorReduction", msg);
            };
            if(!handle)
            {
                printErrorMessage("handle");
            }
            if(!alpha)
            {
                printErrorMessage("alpha");
            }
            if(!A)
            {
                printErrorMessage("A");
            }
            if(!descA)
            {
                printErrorMessage("descA");
            }
            if(!modeA)
            {
                printErrorMessage("modeA");
            }
            if(!beta)
            {
                printErrorMessage("beta");
            }
            if(!descC)
            {
                printErrorMessage("descC");
            }
            if(!D)
            {
                printErrorMessage("D");
            }
            if(!descD)
            {
                printErrorMessage("descD");
            }
            return errorCode;
        }

        const hiptensor::Hash            hashGenerator;
        const std::unordered_set<size_t> supportedTypes = {
            hashGenerator(HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPTENSOR_COMPUTE_16F),
            hashGenerator(HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPTENSOR_COMPUTE_32F),
            hashGenerator(HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPTENSOR_COMPUTE_16BF),
            hashGenerator(HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPTENSOR_COMPUTE_32F),
            hashGenerator(HIP_R_32F, HIP_R_32F, HIP_R_32F, HIPTENSOR_COMPUTE_32F),
            hashGenerator(HIP_R_64F, HIP_R_64F, HIP_R_64F, HIPTENSOR_COMPUTE_64F),
        };

        if(supportedTypes.find(hashGenerator(descA->mType, descC->mType, descD->mType, typeCompute))
           == supportedTypes.end())
        {
            auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
            snprintf(msg,
                     sizeof(msg),
                     "Unsupported Data Type Error : The combination of data types of A, C and D "
                     "and compute type is not supported. (%s)",
                     hiptensorGetErrorString(errorCode));
            logger->logError("hiptensorReduction", msg);
            return errorCode;
        }

        auto modeSetA = std::set<int32_t>(modeA, modeA + descA->mLengths.size());
        auto modeSetC = std::set<int32_t>(modeC, modeC + descC->mLengths.size());
        if(descA->mLengths.size() < descC->mLengths.size() || !(*descC == *descD)
           || !std::includes(
               modeSetA.cbegin(), modeSetA.cend(), modeSetC.cbegin(), modeSetC.cend()))
        {
            auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
            snprintf(msg,
                     sizeof(msg),
                     "Unsupported Data Error : The descriptor of C and D should be same and "
                     " modes of C should be subset of modes A. (%s)",
                     hiptensorGetErrorString(errorCode));
            logger->logError("hiptensorReduction", msg);
            return errorCode;
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }
}

hiptensorStatus_t hiptensorReduction(const hiptensorHandle_t*           handle,
                                     const void*                        alpha,
                                     const void*                        A,
                                     const hiptensorTensorDescriptor_t* descA,
                                     const int32_t                      modeA[],
                                     const void*                        beta,
                                     const void*                        C,
                                     const hiptensorTensorDescriptor_t* descC,
                                     const int32_t                      modeC[],
                                     void*                              D,
                                     const hiptensorTensorDescriptor_t* descD,
                                     const int32_t                      modeD[],
                                     hiptensorOperator_t                opReduce,
                                     hiptensorComputeType_t             typeCompute,
                                     void*                              workspace,
                                     uint64_t                           workspaceSize,
                                     hipStream_t                        stream)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();
    char  msg[2048];

    snprintf(msg,
             sizeof(msg),
             "hiptensorReduction: handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, beta=%p, C=%p, "
             "descC=%p, modeC=%p, D=%p, descD=%p, modeD=%p, opReduce=%d, typeCompute=%d, "
             "workspace=%p, workspaceSize=%lu, stream=%p",
             handle,
             alpha,
             A,
             descA,
             modeA,
             beta,
             C,
             descC,
             modeC,
             D,
             descD,
             modeD,
             (int)opReduce,
             (int)typeCompute,
             workspace,
             workspaceSize,
             stream);

    logger->logAPITrace("hiptensorReduction", msg);

    if(auto errorCode = checkReductionInputData(handle,
                                                alpha,
                                                A,
                                                descA,
                                                modeA,
                                                beta,
                                                C,
                                                descC,
                                                modeC,
                                                D,
                                                descD,
                                                modeD,
                                                opReduce,
                                                typeCompute,
                                                workspace,
                                                workspaceSize);
       errorCode != HIPTENSOR_STATUS_SUCCESS)
    {
        return errorCode;
    }

    auto& instances = hiptensor::ReductionSolutionInstances::instance();
    if(instances->solutionCount() == 0)
    {
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Internal Error : ReductionSolutionInstances is empty (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorReduction", msg);
        return errorCode;
    }

    int  rankA        = descA->mLengths.size();
    int  numReduceDim = descA->mLengths.size() - descD->mLengths.size();
    auto ADataType    = descA->mType;
    auto DDataType    = descD->mType;

    auto internalTypeCompute = typeCompute;
    if(typeCompute == HIPTENSOR_COMPUTE_16F || typeCompute == HIPTENSOR_COMPUTE_16BF)
    {
        // CK does not support f16 or bf16 as compute type
        internalTypeCompute = HIPTENSOR_COMPUTE_32F;
    }

    // Query reduction solutions for the correct reduction operation and type
    auto solutionQ = instances->querySolutions(ADataType,
                                               internalTypeCompute,
                                               DDataType,
                                               rankA,
                                               numReduceDim,
                                               opReduce,
                                               true, // @TODO hardcode
                                               false); // @TODO hardcode

    if(solutionQ.solutionCount() == 0)
    {
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Internal Error : querySolutions returns 0 kernel. (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorReduction", msg);
        return errorCode;
    }

    double alphaD;
    if(alpha != nullptr)
    {
        alphaD = hiptensor::readVal<double>(alpha, typeCompute);
    }
    double betaD;
    if(beta != nullptr)
    {
        betaD = hiptensor::readVal<double>(beta, typeCompute);
    }

    if(C && C != D)
    {
        // CK API can only process $D = alpha * reduce(A) + beta * D$
        // Need to copy C to D if C != D
        CHECK_HIP_ERROR(hipMemcpy(D,
                                  C,
                                  hiptensor::elementsFromLengths(descC->mLengths)
                                      * hiptensor::hipDataTypeSize(descC->mType),
                                  hipMemcpyDeviceToDevice));
    }

    for(auto [_, pSolution] : solutionQ.solutions())
    {
        using hiptensor::HiptensorOptions;
        auto& options = HiptensorOptions::instance();

        // Perform reduction with timing if LOG_LEVEL_PERF_TRACE
        auto streamConfig =
            (logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE) ?
            StreamConfig{
                stream, // stream id
                true, // time_kernel
                0, // log_level
                options->coldRuns(), // cold_niters
                options->hotRuns(), // nrepeat
            }:
        StreamConfig{stream, false};
        auto [isSupported, time] = (*pSolution)(descA->mLengths,
                                                descA->mStrides,
                                                {modeA, modeA + descA->mLengths.size()},
                                                descD->mLengths,
                                                descD->mStrides,
                                                {modeD, modeD + descD->mLengths.size()},
                                                alphaD,
                                                betaD,
                                                A,
                                                D,
                                                opReduce,
                                                streamConfig);
        if(isSupported)
        {
            if(time < 0)
            {
                return HIPTENSOR_STATUS_CK_ERROR;
            }
            if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
            {

                int  n     = pSolution->problemDim();
                auto flops = std::size_t(2) * n;
                auto bytes = pSolution->problemBytes();

                hiptensor::PerfMetrics metrics = {
                    pSolution->uid(), // id
                    pSolution->kernelName(), // name
                    time, // avg time
                    static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                    static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
                };

                // log perf metrics (not name/id)
                snprintf(msg,
                         sizeof(msg),
                         "KernelId: %lu KernelName: %s, %0.3f ms, %0.3f TFlops/s, %0.3f GB/s",
                         metrics.mKernelUid,
                         metrics.mKernelName.c_str(),
                         metrics.mAvgTimeMs,
                         metrics.mTflops,
                         metrics.mBandwidth);
                logger->logPerformanceTrace("hiptensorReduction", msg);
            }

            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
    snprintf(msg,
             sizeof(msg),
             "No kernel is able to solve the problem (%s)",
             hiptensorGetErrorString(errorCode));
    logger->logError("hiptensorReduction", msg);
    return errorCode;
}

hiptensorStatus_t hiptensorReductionGetWorkspaceSize(const hiptensorHandle_t*           handle,
                                                     const void*                        A,
                                                     const hiptensorTensorDescriptor_t* descA,
                                                     const int32_t                      modeA[],
                                                     const void*                        C,
                                                     const hiptensorTensorDescriptor_t* descC,
                                                     const int32_t                      modeC[],
                                                     const void*                        D,
                                                     const hiptensorTensorDescriptor_t* descD,
                                                     const int32_t                      modeD[],
                                                     hiptensorOperator_t                opReduce,
                                                     hiptensorComputeType_t             typeCompute,
                                                     uint64_t* workspaceSize)
{
    *workspaceSize = 0;
    return HIPTENSOR_STATUS_SUCCESS;
}
