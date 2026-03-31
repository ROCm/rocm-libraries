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

hiptensorStatus_t hiptensorPermute(const hiptensorHandle_t handle,
                                   const hiptensorPlan_t   plan,
                                   const void*             alpha,
                                   const void*             A,
                                   void*                   B,
                                   const hipStream_t       stream)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    using hiptensor::PlancacheAutotuneMgr;
    auto& autotuneMgr = PlancacheAutotuneMgr::instance();
    autotuneMgr->startAutotune(hiptensor::AutotuneOps::Autotune_Permutation);

    hiptensorOperationDescriptor_t    opDes      = plan->mOpDesc;
    const hiptensorTensorDescriptor_t descA      = opDes->mDescA;
    const int32_t*                    modeA      = opDes->mModeA.data();
    const hiptensorTensorDescriptor_t descB      = opDes->mDescB;
    const int32_t*                    modeB      = opDes->mModeB.data();
    const hiptensorDataType_t         typeScalar = opDes->mScalarType;

    // Log API access
    char msg[2048];
    snprintf(msg,
             sizeof(msg),
             "handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, B=%p, descB=%p, modeB=%p, "
             "typeScalar=0x%02X, stream=%p",
             handle,
             alpha,
             A,
             descA,
             modeA,
             B,
             descB,
             modeB,
             (unsigned int)typeScalar,
             stream);

    logger->logAPITrace("hiptensorPermute", msg);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, alpha);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, A);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, B);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descB);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeB);

    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    constexpr std::array<std::array<hiptensorDataType_t, 2>, 3> validDataTypes
        = {{// typeA, typeC, typeScalar
            {HIPTENSOR_R_16F, HIPTENSOR_R_16F},
            {HIPTENSOR_R_16F, HIPTENSOR_R_32F},
            {HIPTENSOR_R_32F, HIPTENSOR_R_32F}}};

    std::array<hiptensorDataType_t, 2> inputTensorTypes = {descA->mType, typeScalar};
    if(std::none_of(validDataTypes.cbegin(),
                    validDataTypes.cend(),
                    [&inputTensorTypes](auto&& types) { return types == inputTensorTypes; }))
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
        snprintf(msg,
                 sizeof(msg),
                 "Unsupported Data Type Error : The combination of data types for input tensors A, "
                 "and B is not supported. "
                 "See the link for details "
                 "https://rocm.docs.amd.com/projects/hipTensor/en/docs-6.5.0/api-reference/"
                 "api-reference.html "
                 "(%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorPermute", msg);
        return errorCode;
    }

    float alphaF;
    if(alpha != nullptr)
    {
        alphaF = hiptensor::readVal<float>(alpha, hiptensor::convertToComputeType(typeScalar));
    }

    autotuneMgr->setAutotune<hiptensor::ElementwiseSolution>(
        hiptensor::AutotuneOps::Autotune_Permutation, handle, plan);

    std::vector<hiptensor::ElementwiseSolution*> solutions;
    if(plan->mPref->mSolution != nullptr)
        solutions.push_back((hiptensor::ElementwiseSolution*)plan->mPref->mSolution);
    else
    {
        auto& instances = hiptensor::ElementwiseSolutionInstances::instance();
        solutions       = instances->query({alphaF},
                                     descA->mLengths,
                                           {descA->mType},
                                           {descB->mType},
                                           {{modeA, modeA + descA->mLengths.size()}},
                                           {{modeB, modeB + descB->mLengths.size()}},
                                           {plan->mOpDesc->mOpA, plan->mOpDesc->mOpB},
                                     hiptensor::ElementwiseExecutionSpaceType_t::DEVICE);
    }

    bool canRun = false;
    for(auto pSolution : solutions)
    {
        canRun = pSolution->initArgs({alphaF},
                                     {descA->mLengths},
                                     {descA->mStrides},
                                     {std::vector<int32_t>(modeA, modeA + descA->mLengths.size())},
                                     {descB->mLengths},
                                     {descB->mStrides},
                                     {std::vector<int32_t>(modeB, modeB + descB->mLengths.size())},
                                     {plan->mOpDesc->mOpA},
                                     {A},
                                     {B});

        if(canRun)
        {
            float time = 0.0f;
            // Perform permutation with LOG_LEVEL_PERF_TRACE
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

                auto flops = std::size_t(2) * pSolution->problemSize();
                auto bytes = (hiptensor::hiptensorDataTypeSize(descA->mType)
                              + hiptensor::hiptensorDataTypeSize(descB->mType))
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
                logger->logPerformanceTrace("hiptensorPermute", msg);
            }
            // Perform permutation without timing
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
                hiptensor::AutotuneOps::Autotune_Permutation, time, handle, plan);

            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
    snprintf(msg,
             sizeof(msg),
             "Selected kernel is unable to solve the problem (%s)",
             hiptensorGetErrorString(errorCode));
    logger->logError("hiptensorPermute", msg);
    return errorCode;
}
