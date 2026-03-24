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

#include "contraction_selection.hpp"
#include "contraction_solution.hpp"
#include "contraction_solution_instances.hpp"
#include "contraction_solution_registry.hpp"
#include "handle.hpp"
#include "hip_device.hpp"
#include "logger.hpp"
#include "util.hpp"

#include "hiptensor_options.hpp"
#include "plancache_autotune.hpp"

// Convert between vectors of void ptrs stored in opaque API objects
// to vectors of ContractionSolution ptrs with simple cast.
inline auto toContractionSolutionVec(std::vector<void*> const& v)
{
    auto result = std::vector<hiptensor::ContractionSolution*>(v.size());
    std::transform(v.begin(), v.end(), result.begin(), [](auto* p) {
        return (hiptensor::ContractionSolution*)p;
    });
    return result;
}

inline auto toContractionSolutionVec(
    std::unordered_map<std::size_t, hiptensor::ContractionSolution*> const& map)
{
    auto result = std::vector<hiptensor::ContractionSolution*>(map.size());
    transform(map.begin(), map.end(), result.begin(), [](auto p) { return p.second; });
    return result;
}

inline auto toVoidVec(std::unordered_map<std::size_t, hiptensor::ContractionSolution*> const& map)
{
    auto result = std::vector<void*>(map.size());
    transform(map.begin(), map.end(), result.begin(), [](auto p) { return (void*)p.second; });
    return result;
}

hiptensorStatus_t contractionGetWorkspaceSize(const hiptensorHandle_t              handle,
                                              const hiptensorOperationDescriptor_t desc,
                                              const hiptensorPlanPreference_t      planPref,
                                              const hiptensorWorksizePreference_t  workspacePref,
                                              uint64_t*                            workspaceSize)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[512];
    snprintf(
        msg,
        sizeof(msg),
        "handle=0x%0*llX, desc=0x%llX, planPref=0x%llX, workspacePref=0x%u, workspaceSize=0x%llX",
        2 * (int)sizeof(void*),
        (unsigned long long)handle,
        (unsigned long long)desc,
        (unsigned long long)planPref,
        (unsigned int)workspacePref,
        (unsigned long long)workspaceSize);

    logger->logAPITrace("contractionGetWorkspaceSize", msg);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, desc);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, planPref);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, workspaceSize);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    *workspaceSize = 0u;

    for(auto* candidate : planPref->mCandidates)
    {
        auto* solution = (hiptensor::ContractionSolution*)candidate;
        if(solution->initArgs(nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              hiptensor::getTensorLengths(desc->mDescA),
                              hiptensor::getTensorStrides(desc->mDescA),
                              desc->mModeA,
                              hiptensor::getTensorLengths(desc->mDescB),
                              hiptensor::getTensorStrides(desc->mDescB),
                              desc->mModeB,
                              hiptensor::getTensorLengths(desc->mDescC),
                              hiptensor::getTensorStrides(desc->mDescC),
                              desc->mModeD,
                              hiptensor::getTensorLengths(desc->mDescD),
                              hiptensor::getTensorStrides(desc->mDescD),
                              desc->mModeD,
                              {},
                              nullptr))
        {
            if(*workspaceSize == 0)
            {
                *workspaceSize = solution->workspaceSize();
            }
            else
            {
                if(workspacePref == HIPTENSOR_WORKSPACE_MIN)
                {
                    *workspaceSize = std::min(*workspaceSize, solution->workspaceSize());
                }
                else
                {
                    *workspaceSize = std::max(*workspaceSize, solution->workspaceSize());
                }
            }
        }
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorCreateContraction(const hiptensorHandle_t            handle,
                                             hiptensorOperationDescriptor_t*    desc,
                                             const hiptensorTensorDescriptor_t  descA,
                                             const int32_t                      modeA[],
                                             hiptensorOperator_t                opA,
                                             const hiptensorTensorDescriptor_t  descB,
                                             const int32_t                      modeB[],
                                             hiptensorOperator_t                opB,
                                             const hiptensorTensorDescriptor_t  descC,
                                             const int32_t                      modeC[],
                                             hiptensorOperator_t                opC,
                                             const hiptensorTensorDescriptor_t  descD,
                                             const int32_t                      modeD[],
                                             const hiptensorComputeDescriptor_t descCompute)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[2048];

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, desc);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descB);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descD);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeB);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeD);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    if(!handle->getDevice().matrixCoreSupport(descCompute))
    {
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
    }

    *desc = new hiptensorOperationDescriptor();

    int                  nModeA = descA->mLengths.size();
    std::vector<int32_t> modeAV(modeA, modeA + nModeA);
    int                  nModeB = descB->mLengths.size();
    std::vector<int32_t> modeBV(modeB, modeB + nModeB);
    int nModeC = (descC == nullptr || modeC == nullptr) ? 0 : descC->mLengths.size();
    std::vector<int32_t> modeCV = (descC == nullptr || modeC == nullptr)
                                      ? std::vector<int32_t>()
                                      : std::vector<int32_t>(modeC, modeC + nModeC);
    int                  nModeD = descD->mLengths.size();
    std::vector<int32_t> modeDV(modeD, modeD + nModeD);

    bool hasUnaryOp = (opA != HIPTENSOR_OP_IDENTITY) || (opB != HIPTENSOR_OP_IDENTITY)
                      || (opC != HIPTENSOR_OP_IDENTITY);

    auto contractionOp = descC ? (descCompute == HIPTENSOR_COMPUTE_DESC_C32F
                                          || descCompute == HIPTENSOR_COMPUTE_DESC_C64F
                                      ? hiptensor::ContractionOpId_t::BILINEAR_COMPLEX
                                      : (hasUnaryOp ? hiptensor::ContractionOpId_t::BILINEAR_UNARY
                                                    : hiptensor::ContractionOpId_t::BILINEAR))
                               : (descCompute == HIPTENSOR_COMPUTE_DESC_C32F
                                          || descCompute == HIPTENSOR_COMPUTE_DESC_C64F
                                      ? hiptensor::ContractionOpId_t::SCALE_COMPLEX
                                      : hiptensor::ContractionOpId_t::SCALE);

    (*desc)->mTag          = 0u;
    (*desc)->mScalarType   = *hiptensor::convertToHipTensorDataType(descCompute);
    (*desc)->mFlops        = 0;
    (*desc)->mMovedBytes   = 0;
    (*desc)->mPaddingLeft  = 0;
    (*desc)->mPaddingRighT = 0;
    (*desc)->mPaddingValue = nullptr;

    (*desc)->mOperationType   = HIPTENSOR_CONTRACTION;
    (*desc)->mContractionOpId = static_cast<int32_t>(contractionOp);
    (*desc)->mDescA           = descA;
    (*desc)->mModeA           = modeAV;
    (*desc)->mOpA             = opA;

    (*desc)->mDescB = descB;
    (*desc)->mModeB = modeBV;
    (*desc)->mOpB   = opB;

    (*desc)->mDescC = descC;
    (*desc)->mModeC = modeCV;
    (*desc)->mOpC   = opC;

    (*desc)->mDescD = descD;
    (*desc)->mModeD = modeDV;

    (*desc)->mDescCompute = descCompute;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorContract(const hiptensorHandle_t handle,
                                    const hiptensorPlan_t   plan,
                                    const void*             alpha,
                                    const void*             A,
                                    const void*             B,
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
    autotuneMgr->startAutotune(hiptensor::AutotuneOps::Autotune_Contraction);

    // Log API access
    char msg[512];
    char alphaMsg[32];
    char betaMsg[32];

    if(plan != nullptr)
    {
        if(alpha == nullptr)
        {
            snprintf(alphaMsg, sizeof(alphaMsg), "alpha=NULL");
        }
        else
        {
            auto alphaValue
                = hiptensor::readVal<hiptensor::ScalarData>(alpha, plan->mOpDesc->mDescCompute);
            snprintf(alphaMsg, sizeof(alphaMsg), "alpha=%s", std::to_string(alphaValue).c_str());
        }

        if(beta == nullptr)
        {
            snprintf(betaMsg, sizeof(betaMsg), "beta=NULL");
        }
        else
        {
            auto betaValue
                = hiptensor::readVal<hiptensor::ScalarData>(beta, plan->mOpDesc->mDescCompute);
            snprintf(betaMsg, sizeof(betaMsg), "beta=%s", std::to_string(betaValue).c_str());
        }

        autotuneMgr->setAutotune<hiptensor::ContractionSolution>(
            hiptensor::AutotuneOps::Autotune_Contraction, handle, plan);
    }
    else
    {
        snprintf(alphaMsg, sizeof(alphaMsg), "alpha=NULL");
        snprintf(betaMsg, sizeof(betaMsg), "beta=NULL");
    }

    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, plan=0x%llX, %s, A=0x%llX, B=0x%llX, %s, "
             "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)plan,
             alphaMsg,
             (unsigned long long)A,
             (unsigned long long)B,
             betaMsg,
             (unsigned long long)C,
             (unsigned long long)D,
             (unsigned long long)workspace,
             (unsigned long)workspaceSize,
             (unsigned long long)stream);

    logger->logAPITrace("hiptensorContraction", msg);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, plan);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, alpha);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, A);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, B);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, D);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != handle->getDevice().getDeviceId())
    {
        auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
        snprintf(msg,
                 sizeof(msg),
                 "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                 (int)currentDevice.getDeviceId(),
                 (int)handle->getDevice().getDeviceId(),
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContraction", msg);
        return errorCode;
    }

    auto*             cSolution = (hiptensor::ContractionSolution*)(plan->mPref->mSolution);
    hiptensorStatus_t errorCode = HIPTENSOR_STATUS_SUCCESS;
    float             time      = 0.0f;

    // Perform contraction with timing if LOG_LEVEL_PERF_TRACE
    if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
    {
        using hiptensor::HiptensorOptions;
        auto& options = HiptensorOptions::instance();

        std::tie(errorCode, time)
            = (*cSolution)(alpha,
                           A,
                           B,
                           beta,
                           C,
                           D,
                           hiptensor::getTensorLengths(plan->mOpDesc->mDescA),
                           hiptensor::getTensorStrides(plan->mOpDesc->mDescA),
                           plan->mOpDesc->mModeA,
                           hiptensor::getTensorLengths(plan->mOpDesc->mDescB),
                           hiptensor::getTensorStrides(plan->mOpDesc->mDescB),
                           plan->mOpDesc->mModeB,
                           hiptensor::getTensorLengths(plan->mOpDesc->mDescC),
                           hiptensor::getTensorStrides(plan->mOpDesc->mDescC),
                           plan->mOpDesc->mModeC,
                           hiptensor::getTensorLengths(plan->mOpDesc->mDescD),
                           hiptensor::getTensorStrides(plan->mOpDesc->mDescD),
                           plan->mOpDesc->mModeD,
                           {plan->mOpDesc->mOpA, plan->mOpDesc->mOpB, plan->mOpDesc->mOpC},
                           workspace,
                           workspaceSize,
                           StreamConfig{
                               stream, // stream id
                               true, // time_kernel
                               0, // log_level
                               options->coldRuns(), // cold_niters
                               options->hotRuns(), // nrepeat
                           });

        if(errorCode == HIPTENSOR_STATUS_SUCCESS)
        {
            int32_t m, n, k;
            std::tie(m, n, k) = cSolution->problemDims();
            auto flops        = std::size_t(2) * m * n * k;
            auto bytes        = cSolution->problemBytes();

            hiptensor::PerfMetrics metrics = {
                cSolution->uid(), // id
                cSolution->kernelName(), // name
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
            logger->logPerformanceTrace("hiptensorContraction", msg);
        }
    }
    else // Perform contraction without timing
    {
        std::tie(errorCode, time)
            = (*cSolution)(alpha,
                           A,
                           B,
                           beta,
                           C,
                           D,
                           hiptensor::getTensorLengths(plan->mOpDesc->mDescA),
                           hiptensor::getTensorStrides(plan->mOpDesc->mDescA),
                           plan->mOpDesc->mModeA,
                           hiptensor::getTensorLengths(plan->mOpDesc->mDescB),
                           hiptensor::getTensorStrides(plan->mOpDesc->mDescB),
                           plan->mOpDesc->mModeB,
                           hiptensor::getTensorLengths(plan->mOpDesc->mDescC),
                           hiptensor::getTensorStrides(plan->mOpDesc->mDescC),
                           plan->mOpDesc->mModeC,
                           hiptensor::getTensorLengths(plan->mOpDesc->mDescD),
                           hiptensor::getTensorStrides(plan->mOpDesc->mDescD),
                           plan->mOpDesc->mModeD,
                           {plan->mOpDesc->mOpA, plan->mOpDesc->mOpB, plan->mOpDesc->mOpC},
                           workspace,
                           workspaceSize,
                           StreamConfig{stream, false});
    }

    if(errorCode == HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
    {
        snprintf(msg,
                 sizeof(msg),
                 "Insufficient workspace: req: %zu alloc: %llu (%s)",
                 cSolution->workspaceSize(),
                 static_cast<unsigned long long>(workspaceSize),
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContraction", msg);
    }
    else if(errorCode == HIPTENSOR_STATUS_INTERNAL_ERROR)
    {
        snprintf(msg,
                 sizeof(msg),
                 "Selected kernel is unable to solve the problem (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContraction", msg);
    }

    autotuneMgr->saveAutotune<hiptensor::ContractionSolution>(
        hiptensor::AutotuneOps::Autotune_Contraction, time, handle, plan);

    return errorCode;
}

hiptensorStatus_t contractionCreatePlanPreference(const hiptensorHandle_t   handle,
                                                  hiptensorPlanPreference_t pref,
                                                  hiptensorAlgo_t           algo,
                                                  hiptensorJitMode_t        jitMode)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    char              msg[512];
    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, pref);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != handle->getDevice().getDeviceId())
    {
        auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
        snprintf(msg,
                 sizeof(msg),
                 "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                 (int)currentDevice.getDeviceId(),
                 (int)handle->getDevice().getDeviceId(),
                 hiptensorGetErrorString(errorCode));

        logger->logError("contractionCreatePlanPreference", msg);
        return errorCode;
    }

    if(algo == HIPTENSOR_ALGO_DEFAULT || algo == HIPTENSOR_ALGO_DEFAULT_PATIENT
       || algo == HIPTENSOR_ALGO_ACTOR_CRITIC)
    {
        // Update the stored selection algorithm
        pref->mSelectionAlgorithm = algo;

        // For now, enumerate all known contraction kernels.
        // Using the hipDevice, determine if the device supports F64
        auto& instances = hiptensor::ContractionSolutionInstances::instance();
        auto  solnQ     = instances->allSolutions();

        // Can do more checking for scale / bilinear, etc. if we need to.

        if(solnQ.solutionCount() == 0)
        {
            // No kernels found!
            auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
            snprintf(msg,
                     sizeof(msg),
                     "Internal Error : No Kernels Found (%s)",
                     hiptensorGetErrorString(errorCode));
            logger->logError("hiptensorInitContractionFind", msg);
            return errorCode;
        }

        // Extract the solutions to the candidates vector.
        pref->mCandidates = toVoidVec(solnQ.solutions());

        return HIPTENSOR_STATUS_SUCCESS;
    }
    else
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        snprintf(msg, sizeof(msg), "Invalid Algo Value (%s)", hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorInitContractionFind", msg);
        return errorCode;
    }
}

hiptensorStatus_t contractionInitPlan(const hiptensorHandle_t              handle,
                                      hiptensorPlan_t                      plan,
                                      const hiptensorOperationDescriptor_t desc,
                                      const hiptensorPlanPreference_t      pref,
                                      uint64_t                             workspaceSizeLimit)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access

    char msg[256];

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, plan);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, desc);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, pref);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != handle->getDevice().getDeviceId())
    {
        auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
        snprintf(msg,
                 sizeof(msg),
                 "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                 (int)currentDevice.getDeviceId(),
                 (int)handle->getDevice().getDeviceId(),
                 hiptensorGetErrorString(errorCode));
        logger->logError("contractionInitPlan", msg);
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
    }

    // At this point, we need to format inputs for kernels as they will be tested via selection model.
    // Brute force method currently uses CK kernel format, so we will adjust inputs to that style.

    // Convert to concrete contraction solutions
    auto& instances = hiptensor::ContractionSolutionInstances::instance();
    auto  solnQ     = instances->allSolutions();

    // Can do more checking for scale / bilinear, etc. if we need to.

    if(solnQ.solutionCount() == 0)
    {
        // No kernels found!
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Internal Error : No Kernels Found (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorInitContractionFind", msg);
        return errorCode;
    }

    // Extract the solutions to the candidates vector.
    pref->mCandidates = toVoidVec(solnQ.solutions());
    auto candidates   = toContractionSolutionVec(pref->mCandidates);

    auto computeType = desc->mDescCompute;
    auto ADataType   = desc->mDescA->mType;
    auto BDataType   = desc->mDescB->mType;
    auto DDataType   = desc->mDescC ? desc->mDescC->mType : hiptensor::NONE_TYPE;
    auto EDataType   = desc->mDescD->mType;

    // Query contraction solutions for the correct contraction operation and type
    auto solutionQ = hiptensor::ContractionSolutionRegistry::Query{candidates}
                         .query((hiptensor::ContractionOpId_t)desc->mContractionOpId)
                         .query(ADataType, BDataType, DDataType, EDataType, computeType);

    bool hasUnaryOp = (desc->mOpA != HIPTENSOR_OP_IDENTITY) || (desc->mOpB != HIPTENSOR_OP_IDENTITY)
                      || (desc->mOpC != HIPTENSOR_OP_IDENTITY);
    if(hasUnaryOp)
        solutionQ = solutionQ.query(HIPTENSOR_OP_UNKNOWN, HIPTENSOR_OP_UNKNOWN);
    else
        solutionQ = solutionQ.query(HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_IDENTITY);

    candidates = toContractionSolutionVec(solutionQ.solutions());

    // Measure timing for solution selection
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));

    // Launch selection algorithm
    hiptensor::ContractionSolution* winner = nullptr;
    auto                            result = HIPTENSOR_STATUS_INTERNAL_ERROR;

    //First to look for solution from memory cache (Plan Cache)
    //If there is a solution in Plan Cache, set that solution and skip solution finding
    if(handle->getPlanCache() && pref->mCacheMode == HIPTENSOR_CACHE_MODE_PEDANTIC)
    {
        auto Uid = handle->getPlanCache()->querySolutionUid(desc);
        if(Uid > 0ull)
        {
            winner = findSolutionByUid(candidates, Uid);
            if(winner != nullptr)
                result = HIPTENSOR_STATUS_SUCCESS;
        }
    }

    if(result != HIPTENSOR_STATUS_SUCCESS)
    {
        if(pref->mSelectionAlgorithm == HIPTENSOR_ALGO_DEFAULT
           || pref->mSelectionAlgorithm == HIPTENSOR_ALGO_DEFAULT_PATIENT)
        {
            result = hiptensor::bruteForceModel(
                &winner,
                candidates,
                ADataType,
                hiptensor::getTensorLengths(desc->mDescA),
                hiptensor::getTensorStrides(desc->mDescA),
                desc->mModeA,
                BDataType,
                hiptensor::getTensorLengths(desc->mDescB),
                hiptensor::getTensorStrides(desc->mDescB),
                desc->mModeB,
                DDataType,
                hiptensor::getTensorLengths(desc->mDescC),
                hiptensor::getTensorStrides(desc->mDescC),
                desc->mModeC,
                EDataType,
                hiptensor::getTensorLengths(desc->mDescD),
                hiptensor::getTensorStrides(desc->mDescD),
                desc->mModeD,
                desc->mDescCompute,
                {plan->mOpDesc->mOpA, plan->mOpDesc->mOpB, plan->mOpDesc->mOpC},
                workspaceSizeLimit);
            //Save solutions (from fastest to slowest) for plan cache autotune
            pref->mCandidates.clear();
            for(auto candidate : candidates)
                pref->mCandidates.push_back(candidate);
        }
        else if(pref->mSelectionAlgorithm == HIPTENSOR_ALGO_ACTOR_CRITIC)
        {
            if(hasUnaryOp)
                result
                    = hiptensor::actorCriticModelUnaryOps(&winner,
                                                          solutionQ.solutions(),
                                                          ADataType,
                                                          hiptensor::getTensorLengths(desc->mDescA),
                                                          hiptensor::getTensorStrides(desc->mDescA),
                                                          desc->mModeA,
                                                          BDataType,
                                                          hiptensor::getTensorLengths(desc->mDescB),
                                                          hiptensor::getTensorStrides(desc->mDescB),
                                                          desc->mModeB,
                                                          DDataType,
                                                          hiptensor::getTensorLengths(desc->mDescC),
                                                          hiptensor::getTensorStrides(desc->mDescC),
                                                          desc->mModeC,
                                                          EDataType,
                                                          hiptensor::getTensorLengths(desc->mDescD),
                                                          hiptensor::getTensorStrides(desc->mDescD),
                                                          desc->mModeC,
                                                          desc->mDescCompute,
                                                          workspaceSizeLimit);
            else
                result = hiptensor::actorCriticModel(&winner,
                                                     solutionQ.solutions(),
                                                     ADataType,
                                                     hiptensor::getTensorLengths(desc->mDescA),
                                                     hiptensor::getTensorStrides(desc->mDescA),
                                                     desc->mModeA,
                                                     BDataType,
                                                     hiptensor::getTensorLengths(desc->mDescB),
                                                     hiptensor::getTensorStrides(desc->mDescB),
                                                     desc->mModeB,
                                                     DDataType,
                                                     hiptensor::getTensorLengths(desc->mDescC),
                                                     hiptensor::getTensorStrides(desc->mDescC),
                                                     desc->mModeC,
                                                     EDataType,
                                                     hiptensor::getTensorLengths(desc->mDescD),
                                                     hiptensor::getTensorStrides(desc->mDescD),
                                                     desc->mModeC,
                                                     desc->mDescCompute,
                                                     workspaceSizeLimit);
        }
    }

    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));

    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    if(result != HIPTENSOR_STATUS_SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "Init contraction plan not successful (%s)",
                 hiptensorGetErrorString(result));
        logger->logError("hiptensorInitContractionPlan", msg);
        return result;
    }

    // Log the selected contraction solution and selection timing
    snprintf(msg,
             sizeof(msg),
             "Algo: %d, KernelId: %zu, KernelName: %s, SelectionTime: %0.3f ms",
             pref->mSelectionAlgorithm,
             winner->uid(),
             winner->kernelName().c_str(),
             elapsedTimeMs);
    logger->logPerformanceTrace("hiptensorInitContractionPlan", msg);

    // Assign the contraction descriptor
    pref->mSolution          = winner;
    plan->mRequiredWorkspace = workspaceSizeLimit;
    plan->mOpDesc            = desc;
    plan->mPref              = pref;

    return HIPTENSOR_STATUS_SUCCESS;
}
