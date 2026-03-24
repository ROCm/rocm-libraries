/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <set>
#include <unordered_set>
#include <vector>

#include <hiptensor/hiptensor.h>

#include <hiptensor/hiptensor.h>

#include "handle.hpp"
#include "hip_device.hpp"
#include "logger.hpp"

#include "reduction_solution.hpp"
#include "reduction_solution_instances.hpp"
#include "reduction_solution_registry.hpp"

#include "hiptensor_options.hpp"
#include "plancache_autotune.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

namespace
{
    hiptensorStatus_t checkReductionInputData(const hiptensorHandle_t           handle,
                                              const void*                       alpha,
                                              const void*                       A,
                                              const hiptensorTensorDescriptor_t descA,
                                              const int32_t*                    modeA,
                                              const void*                       beta,
                                              const void*                       C,
                                              const hiptensorTensorDescriptor_t descC,
                                              const int32_t*                    modeC,
                                              void*                             D,
                                              const hiptensorTensorDescriptor_t descD,
                                              const int32_t*                    modeD,
                                              hiptensorOperator_t               opReduce,
                                              hiptensorComputeDescriptor_t      typeCompute,
                                              void*                             workspace,
                                              uint64_t                          workspaceSize)
    {
        // Log API access
        using hiptensor::Logger;
        auto& logger = Logger::instance();
        char  msg[2048];

        hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, alpha);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, A);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeA);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, beta);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, C);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descC);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, D);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descD);
        if(descD->mLengths.size() != 0)
        {
            // modeC and modeD are nullptr if rank of output is 0
            CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeC);
            CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeD);
        }
        // hipTensor does not use `workspace`. Do not check `workspace`.

        if(checkResult != HIPTENSOR_STATUS_SUCCESS)
        {
            return checkResult;
        }

        const hiptensor::Hash            hashGenerator;
        const std::unordered_set<size_t> supportedTypes = {
            hashGenerator(
                HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_16F),
            hashGenerator(
                HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F),
            hashGenerator(
                HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_16BF),
            hashGenerator(
                HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F),
            hashGenerator(
                HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F),
            hashGenerator(
                HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F),
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
            logger->logError("hiptensorReduce", msg);
            return errorCode;
        }

        auto modeSetA = std::set<int32_t>(modeA, modeA + descA->mLengths.size());
        auto modeSetC = std::set<int32_t>(modeC, modeC + descC->mLengths.size());

        auto compareDescCD = [](auto&& left, auto&& right) {
            auto copyLeft = left;
            copyLeft.mAlignmentRequirement
                = right.mAlignmentRequirement; // compare these 2 desc except mAlignmentRequirement
            return copyLeft == right;
        };

        if(descA->mLengths.size() < descC->mLengths.size() || !compareDescCD(*descC, *descD)
           || !std::includes(
               modeSetA.cbegin(), modeSetA.cend(), modeSetC.cbegin(), modeSetC.cend()))
        {
            auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
            snprintf(msg,
                     sizeof(msg),
                     "Unsupported Data Error : The descriptor of C and D should be same and "
                     " modes of C should be subset of modes A. (%s)",
                     hiptensorGetErrorString(errorCode));
            logger->logError("hiptensorReduce", msg);
            return errorCode;
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }
}

hiptensorStatus_t hiptensorReduce(const hiptensorHandle_t handle,
                                  const hiptensorPlan_t   plan,
                                  const void*             alpha,
                                  const void*             A,
                                  const void*             beta,
                                  const void*             C,
                                  void*                   D,
                                  void*                   workspace,
                                  uint64_t                workspaceSize,
                                  hipStream_t             stream)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    using hiptensor::PlancacheAutotuneMgr;
    auto& autotuneMgr = PlancacheAutotuneMgr::instance();
    autotuneMgr->startAutotune(hiptensor::AutotuneOps::Autotune_Redution);

    hiptensorOperationDescriptor_t    opDes       = plan->mOpDesc;
    const hiptensorTensorDescriptor_t descA       = opDes->mDescA;
    const int32_t*                    modeA       = opDes->mModeA.data();
    const hiptensorTensorDescriptor_t descC       = opDes->mDescC;
    const int32_t*                    modeC       = opDes->mModeC.data();
    const hiptensorTensorDescriptor_t descD       = opDes->mDescD;
    const int32_t*                    modeD       = opDes->mModeD.data();
    hiptensorOperator_t               opA         = opDes->mOpA;
    hiptensorOperator_t               opC         = opDes->mOpC;
    hiptensorOperator_t               opReduce    = opDes->mOpAC;
    hiptensorComputeDescriptor_t      typeCompute = opDes->mDescCompute;

    char msg[2048];
    snprintf(msg,
             sizeof(msg),
             "hiptensorReduce: handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, beta=%p, C=%p, "
             "descC=%p, modeC=%p, D=%p, descD=%p, modeD=%p, opReduce=%s, typeCompute=%s, "
             "workspace=%p, workspaceSize=%llu, stream=%p",
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
             hiptensor::opTypeToString(opReduce).c_str(),
             hiptensor::computeTypeToString(typeCompute).c_str(),
             workspace,
             static_cast<unsigned long long>(workspaceSize),
             stream);

    logger->logAPITrace("hiptensorReduce", msg);

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

    if(descA->mLengths.size() == descD->mLengths.size())
    {
        // Composable Kernels (CK) does not handle reductions where the input and
        // output tensors maintain the same rank. For those scenarios, employ
        // elementwise binary operations.
        plan->mOpDesc->mOpAC = HIPTENSOR_OP_ADD;
        return hiptensorElementwiseBinaryExecute(handle, plan, alpha, A, beta, C, D, stream);
    }

    auto& instances = hiptensor::ReductionSolutionInstances::instance();
    if(instances->solutionCount() == 0)
    {
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Internal Error : ReductionSolutionInstances is empty (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorReduce", msg);
        return errorCode;
    }

    int  rankA        = descA->mLengths.size();
    int  numReduceDim = descA->mLengths.size() - descD->mLengths.size();
    auto ADataType    = descA->mType;
    auto DDataType    = descD->mType;

    auto internalTypeCompute = typeCompute;
    if(typeCompute == HIPTENSOR_COMPUTE_DESC_16F || typeCompute == HIPTENSOR_COMPUTE_DESC_16BF)
    {
        // CK does not support f16 or bf16 as compute type
        internalTypeCompute = HIPTENSOR_COMPUTE_DESC_32F;
    }

    autotuneMgr->setAutotune<hiptensor::ReductionSolution>(
        hiptensor::AutotuneOps::Autotune_Redution, handle, plan);

    std::vector<hiptensor::ReductionSolution*> solutions;
    if(plan->mPref->mSolution != nullptr)
        solutions.push_back((hiptensor::ReductionSolution*)plan->mPref->mSolution);
    else
    {
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
            logger->logError("hiptensorReduce", msg);
            return errorCode;
        }

        for(auto [_, pSolution] : solutionQ.solutions())
            solutions.push_back(pSolution);
    }

    double alphaValue;
    if(alpha != nullptr)
    {
        alphaValue = hiptensor::readVal<double>(alpha, typeCompute);
    }
    double betaValue;
    if(beta != nullptr)
    {
        betaValue = hiptensor::readVal<double>(beta, typeCompute);
    }

    if(C && C != D)
    {
        // CK API can only process $D = alpha * reduce(A) + beta * D$
        // Need to copy C to D if C != D
        CHECK_HIP_ERROR(hipMemcpy(D,
                                  C,
                                  hiptensor::elementsFromLengths(descC->mLengths)
                                      * hiptensor::hiptensorDataTypeSize(descC->mType),
                                  hipMemcpyDeviceToDevice));
    }

    // CK requires all the modes of the reduced tensors (C/D) to be sorted according to the
    // non-reduced tensor mode (modeA).
    //
    // By permuting the reduced tensor modes and their lengths, if the strides are permuted
    // following the same order, then it's possible to satisfy CK's requirement without
    // physically permuting the data.
    //
    // For example:
    //  - A lengths are [3, 5, 8], C/D modes are [2, 1].
    //  - CK requires sorted modes as [1, 2] for C/D, and the respective lengths [5, 8].
    //  - Strides of output are generated in this way:
    //    - output dims are [2, 1]
    //      => corresponding lengths are [8(2), 5(1)]
    //      => strides are [5(2), 1(1)]
    //      => permuted strides are [1(1), 5(2)]
    //
    // The code below implements this logic.
    hiptensorTensorDescriptor sortedDescC    = *descC;
    std::vector<int>          sortingIndices = {};

    CHECK_HIPTENSOR_ERROR(hiptensor::getSortingIndices(
        modeA, descA->mLengths.size(), modeC, sortedDescC.mLengths.size(), sortingIndices));
    CHECK_HIPTENSOR_ERROR(hiptensor::applySortingIndices(sortingIndices, sortedDescC.mLengths));
    CHECK_HIPTENSOR_ERROR(hiptensor::applySortingIndices(sortingIndices, sortedDescC.mStrides));

    for(auto pSolution : solutions)
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
                                                sortedDescC.mLengths,
                                                sortedDescC.mStrides,
                                                {modeC, modeC + sortedDescC.mLengths.size()},
                                                opA,
                                                opC,
                                                alphaValue,
                                                betaValue,
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
                         "KernelId: %zu KernelName: %s, %0.3f ms, %0.3f TFlops/s, %0.3f GB/s",
                         metrics.mKernelUid,
                         metrics.mKernelName.c_str(),
                         metrics.mAvgTimeMs,
                         metrics.mTflops,
                         metrics.mBandwidth);
                logger->logPerformanceTrace("hiptensorReduce", msg);
            }

            plan->mPref->mSolution = pSolution;
            autotuneMgr->saveAutotune<hiptensor::ReductionSolution>(
                hiptensor::AutotuneOps::Autotune_Redution, time, handle, plan);

            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
    snprintf(msg,
             sizeof(msg),
             "No kernel is able to solve the problem (%s)",
             hiptensorGetErrorString(errorCode));
    logger->logError("hiptensorReduce", msg);
    return errorCode;
}
