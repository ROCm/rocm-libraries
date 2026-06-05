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
#include <hiptensor/hiptensor.h>

#include "elementwise_solution.hpp"
#include "elementwise_solution_instances.hpp"
#include "elementwise_solution_registry.hpp"
#include "logger.hpp"

#include "hiptensor_options.hpp"
#include "plancache_autotune.hpp"

hiptensorStatus_t hiptensorElementwiseTrinaryExecute(const hiptensorHandle_t handle,
                                                     const hiptensorPlan_t   plan,
                                                     const void*             alpha,
                                                     const void*             A,
                                                     const void*             beta,
                                                     const void*             B,
                                                     const void*             gamma,
                                                     const void*             C,
                                                     void*                   D,
                                                     hipStream_t             stream)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    using hiptensor::PlancacheAutotuneMgr;
    auto& autotuneMgr = PlancacheAutotuneMgr::instance();
    autotuneMgr->startAutotune(hiptensor::AutotuneOps::Autotune_TrinaryOp);

    hiptensorOperationDescriptor_t    opDes      = plan->mOpDesc;
    const hiptensorTensorDescriptor_t descA      = opDes->mDescA;
    const int32_t*                    modeA      = opDes->mModeA.data();
    const hiptensorTensorDescriptor_t descB      = opDes->mDescB;
    const int32_t*                    modeB      = opDes->mModeB.data();
    const hiptensorTensorDescriptor_t descC      = opDes->mDescC;
    const int32_t*                    modeC      = opDes->mModeC.data();
    const hiptensorTensorDescriptor_t descD      = opDes->mDescD;
    const int32_t*                    modeD      = opDes->mModeD.data();
    hiptensorOperator_t               opA        = opDes->mOpA;
    hiptensorOperator_t               opB        = opDes->mOpB;
    hiptensorOperator_t               opC        = opDes->mOpC;
    hiptensorOperator_t               opAB       = opDes->mOpAC;
    hiptensorOperator_t               opABC      = opDes->mOpABC;
    const hiptensorDataType_t         typeScalar = opDes->mScalarType;

    // Log API access
    char msg[2048];
    snprintf(msg,
             sizeof(msg),
             "handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, "
             "beta=%p, B=%p, descB=%p, modeB=%p, "
             "gamma=%p, C=%p, descC=%p, modeC=%p, "
             "D=%p, descD=%p, modeD=%p, "
             "opAB=0x%02X, opABC=0x%02X, typeScalar=0x%02X, stream=%p",
             handle,
             alpha,
             A,
             descA,
             modeA,
             beta,
             B,
             descB,
             modeB,
             gamma,
             C,
             descC,
             modeC,
             D,
             descD,
             modeD,
             opAB,
             opABC,
             (unsigned int)typeScalar,
             stream);

    logger->logAPITrace("hiptensorElementwiseTrinaryExecute", msg);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, alpha);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, A);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, beta);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, B);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descB);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeB);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, gamma);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, C);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descC);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeC);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, D);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descD);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeD);

    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    constexpr std::array<std::array<hiptensorDataType_t, 4>, 3> validDataTypes
        = {{// typeA, typeB, typeC, typeScalar
            {HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
            {HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
            {HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F}}};

    std::array<hiptensorDataType_t, 4> inputTensorTypes
        = {descA->mType, descB->mType, descC->mType, typeScalar};
    if(std::none_of(validDataTypes.cbegin(),
                    validDataTypes.cend(),
                    [&inputTensorTypes](auto&& types) { return types == inputTensorTypes; }))
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
        snprintf(msg,
                 sizeof(msg),
                 "Unsupported Data Type Error : The combination of data types for input tensors A, "
                 "B, C, and D is not supported. "
                 "See the link for details "
                 "https://rocm.docs.amd.com/projects/hipTensor/en/docs-6.5.0/api-reference/"
                 "api-reference.html "
                 "(%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorElementwiseTrinaryExecute", msg);
        return errorCode;
    }

    float alphaF;
    if(alpha != nullptr)
    {
        alphaF = hiptensor::readVal<float>(alpha, hiptensor::convertToComputeType(typeScalar));
    }
    float betaF;
    if(beta != nullptr)
    {
        betaF = hiptensor::readVal<float>(beta, hiptensor::convertToComputeType(typeScalar));
    }
    float gammaF;
    if(gamma != nullptr)
    {
        gammaF = hiptensor::readVal<float>(gamma, hiptensor::convertToComputeType(typeScalar));
    }

    autotuneMgr->setAutotune<hiptensor::ElementwiseSolution>(
        hiptensor::AutotuneOps::Autotune_TrinaryOp, handle, plan);

    std::vector<hiptensor::ElementwiseSolution*> solutions;
    if(plan->mPref->mSolution != nullptr)
        solutions.push_back((hiptensor::ElementwiseSolution*)plan->mPref->mSolution);
    else
    {
        auto& instances = hiptensor::ElementwiseSolutionInstances::instance();
        solutions       = instances->query({alphaF, betaF, gammaF},
                                     descA->mLengths,
                                           {descA->mType, descB->mType, descC->mType},
                                           {descD->mType},
                                           {{modeA, modeA + descA->mLengths.size()},
                                            {modeB, modeB + descB->mLengths.size()},
                                            {modeC, modeC + descC->mLengths.size()}},
                                           {{modeD, modeD + descD->mLengths.size()}},
                                           {opABC, opAB, opA, opB, opC},
                                     hiptensor::ElementwiseExecutionSpaceType_t::DEVICE);
    }

    bool canRun = false;
    for(auto pSolution : solutions)
    {
        canRun = pSolution->initArgs({alphaF, betaF, gammaF},
                                     {descA->mLengths, descB->mLengths, descC->mLengths},
                                     {descA->mStrides, descB->mStrides, descC->mStrides},
                                     {std::vector<int32_t>(modeA, modeA + descA->mLengths.size()),
                                      std::vector<int32_t>(modeB, modeB + descB->mLengths.size()),
                                      std::vector<int32_t>(modeC, modeC + descC->mLengths.size())},
                                     {descD->mLengths},
                                     {descD->mStrides},
                                     {std::vector<int32_t>(modeD, modeD + descD->mLengths.size())},
                                     {opABC, opAB, opA, opB, opC},
                                     {A, B, C},
                                     {D});

        if(canRun)
        {
            float time = 0.0f;
            // Perform elementwise trinary with timing if LOG_LEVEL_PERF_TRACE
            if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
            {
                using hiptensor::HiptensorOptions;
                auto& options = HiptensorOptions::instance();

                time = (*pSolution)(StreamConfig{
                    stream, // stream id
                    true, // time_kernel
                    0, // log_level
                    options->coldRuns(), // cold_niters
                    options->hotRuns(), // nrepeat
                });
                if(time < 0)
                {
                    return HIPTENSOR_STATUS_CK_ERROR;
                }

                auto flops = std::size_t(8) * pSolution->problemSize();
                auto bytes = (hiptensor::hiptensorDataTypeSize(descA->mType)
                              + hiptensor::hiptensorDataTypeSize(descB->mType)
                              + hiptensor::hiptensorDataTypeSize(descC->mType)
                              + hiptensor::hiptensorDataTypeSize(descD->mType))
                             * pSolution->problemSize();

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
                logger->logPerformanceTrace("hiptensorElementwiseTrinaryExecute", msg);
            }
            // Perform elementwise trinary without timing
            else
            {
                time = (*pSolution)(StreamConfig{stream, false});
                if(time < 0)
                {
                    return HIPTENSOR_STATUS_CK_ERROR;
                }
            }

            plan->mPref->mSolution = pSolution;
            autotuneMgr->saveAutotune<hiptensor::ElementwiseSolution>(
                hiptensor::AutotuneOps::Autotune_TrinaryOp, time, handle, plan);

            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
    snprintf(msg,
             sizeof(msg),
             "Selected kernel is unable to solve the problem (%s)",
             hiptensorGetErrorString(errorCode));
    logger->logError("hiptensorElementwiseTrinaryExecute", msg);
    return errorCode;
}
