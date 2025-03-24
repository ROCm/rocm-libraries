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

#include "logger.hpp"
#include "permutation_solution.hpp"
#include "permutation_solution_instances.hpp"
#include "permutation_solution_registry.hpp"

#include "hiptensor_options.hpp"

hiptensorStatus_t hiptensorElementwiseBinary(const hiptensorHandle_t*           handle,
                                             const void*                        alpha,
                                             const void*                        A,
                                             const hiptensorTensorDescriptor_t* descA,
                                             const int32_t                      modeA[],
                                             const void*                        gamma,
                                             const void*                        C,
                                             const hiptensorTensorDescriptor_t* descC,
                                             const int32_t                      modeC[],
                                             void*                              D,
                                             const hiptensorTensorDescriptor_t* descD,
                                             const int32_t                      modeD[],
                                             hiptensorOperator_t                opAC,
                                             hipDataType                        typeScalar,
                                             hipStream_t                        stream)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[2048];
    snprintf(msg,
             sizeof(msg),
             "handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, "
             "gamma=%p, C=%p, descC=%p, modeC=%p, "
             "D=%p, descD=%p, modeD=%p, "
             "opAC=0x%02X, typeScalar=0x%02X, stream=%p",
             handle,
             alpha,
             A,
             descA,
             modeA,
             gamma,
             C,
             descC,
             modeC,
             D,
             descD,
             modeD,
             opAC,
             (unsigned int)typeScalar,
             stream);

    logger->logAPITrace("hiptensorElementwiseBinary", msg);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, alpha);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, A);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeA);
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

    constexpr std::array<std::array<hipDataType, 3>, 3> validDataTypes
        = {{// typeA, typeC, typeScalar
            {HIP_R_16F, HIP_R_16F, HIP_R_16F},
            {HIP_R_32F, HIP_R_32F, HIP_R_32F},
            {HIP_R_64F, HIP_R_64F, HIP_R_64F}}};

    std::array<hipDataType, 3> inputTensorTypes = {descA->mType, descC->mType, typeScalar};
    if(descC->mType != descD->mType
       || std::none_of(validDataTypes.cbegin(),
                       validDataTypes.cend(),
                       [&inputTensorTypes](auto&& types) { return types == inputTensorTypes; }))
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
        snprintf(msg,
                 sizeof(msg),
                 "Unsupported Data Type Error : The combination of data types for input tensors A, "
                 "C, and D is not supported. "
                 "See the link for details "
                 "https://rocm.docs.amd.com/projects/hipTensor/en/docs-6.5.0/api-reference/"
                 "api-reference.html "
                 "(%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorElementwiseBinary", msg);
        return errorCode;
    }

    float alphaF;
    if(alpha != nullptr)
    {
        alphaF = hiptensor::readVal<float>(alpha, hiptensor::convertToComputeType(typeScalar));
    }
    float gammaF;
    if(gamma != nullptr)
    {
        gammaF = hiptensor::readVal<float>(gamma, hiptensor::convertToComputeType(typeScalar));
    }

    auto& instances = hiptensor::PermutationSolutionInstances::instance();
    auto  solutions = instances->query(
        {alphaF, gammaF},
        descA->mLengths,
        {descA->mType, descC->mType},
        {descD->mType},
        {{modeA, modeA + descA->mLengths.size()}, {modeC, modeC + descC->mLengths.size()}},
        {{modeD, modeD + descD->mLengths.size()}},
        {opAC, descA->mUnaryOp, descC->mUnaryOp},
        hiptensor::ElementwiseExecutionSpaceType_t::DEVICE);

    bool canRun = false;
    for(auto pSolution : solutions)
    {
        canRun = pSolution->initArgs({alphaF, gammaF},
                                     {descA->mLengths, descC->mLengths},
                                     {descA->mStrides, descC->mStrides},
                                     {std::vector<int32_t>(modeA, modeA + descA->mLengths.size()),
                                      std::vector<int32_t>(modeC, modeC + descC->mLengths.size())},
                                     {descD->mLengths},
                                     {descD->mStrides},
                                     {std::vector<int32_t>(modeD, modeD + descD->mLengths.size())},
                                     {opAC, descA->mUnaryOp, descC->mUnaryOp},
                                     {A, C},
                                     {D});

        if(canRun)
        {
            // Perform permutation with timing if LOG_LEVEL_PERF_TRACE
            if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
            {
                using hiptensor::HiptensorOptions;
                auto& options = HiptensorOptions::instance();

                auto time = (*pSolution)(StreamConfig{
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

                auto flops = std::size_t(5) * pSolution->problemSize();
                auto bytes = (hiptensor::hipDataTypeSize(descA->mType)
                              + hiptensor::hipDataTypeSize(descC->mType)
                              + hiptensor::hipDataTypeSize(descD->mType))
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
                         "KernelId: %lu KernelName: %s, %0.3f ms, %0.3f TFlops/s, %0.3f GB/s",
                         metrics.mKernelUid,
                         metrics.mKernelName.c_str(),
                         metrics.mAvgTimeMs,
                         metrics.mTflops,
                         metrics.mBandwidth);
                logger->logPerformanceTrace("hiptensorElementwiseBinary", msg);
            }
            // Perform permutation without timing
            else
            {
                if((*pSolution)(StreamConfig{stream, false}) < 0)
                {
                    return HIPTENSOR_STATUS_CK_ERROR;
                }
            }

            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
    snprintf(msg,
             sizeof(msg),
             "Selected kernel is unable to solve the problem (%s)",
             hiptensorGetErrorString(errorCode));
    logger->logError("hiptensorElementwiseBinary", msg);
    return errorCode;
}
