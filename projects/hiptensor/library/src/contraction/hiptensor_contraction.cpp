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
#include <unordered_map>

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
                              desc->mModeC,
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
    (*desc)->mOpD   = HIPTENSOR_OP_IDENTITY;

    (*desc)->mDescE = nullptr;
    (*desc)->mModeE = {};

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
                                                          desc->mModeD,
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
                                                     desc->mModeD,
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

// ============================================================================
// Trinary Contraction
// ============================================================================
//
// Implements E = alpha * op(A) * op(B) * op(C) + beta * op(D), where A, B, C
// are input tensors, D is an optional bias tensor, and E is the output.
//
// CK (Composable Kernel) does not provide a native 3-input contraction kernel,
// so we decompose the operation into two successive binary contractions.
//
//   Step 1:  T = A * B               (alpha=1, beta=0, no bias)
//   Step 2:  E = alpha * T * C + beta * D
//
// where T is a temporary intermediate tensor whose storage comes from the
// user-provided workspace.
//
// The plan-creation phase selects a single "winner" kernel by driving the
// selection heuristic with the step 2 problem shape (T * C -> E, with bias D
// in the bilinear case).  The same kernel is then reused for step 1 at
// execution time with alpha=1 and beta=0.
// ============================================================================

// Determine which modes the intermediate tensor T must carry.
//
// Contracted-away modes (shared by A and B but absent from C and E) are
// removed; all surviving modes of A come first (in order), followed by
// modes of B that are not in A (i.e. !setA.count(m)).  A's order is
// preserved verbatim and B contributes only its A-disjoint modes.
inline std::vector<int32_t> computeTrinaryContractionIntermediateModes(
    const std::vector<int32_t>& modeA,
    const std::vector<int32_t>& modeB,
    const std::vector<int32_t>& modeC,
    const std::vector<int32_t>& modeE)
{
    std::set<int32_t> setA(modeA.begin(), modeA.end());
    std::set<int32_t> setB(modeB.begin(), modeB.end());
    std::set<int32_t> setC(modeC.begin(), modeC.end());
    std::set<int32_t> setE(modeE.begin(), modeE.end());

    // Modes contracted in step 1: shared between A and B, absent from C and E
    std::set<int32_t> contracted1;
    for(auto m : setA)
    {
        if(setB.count(m) && !setC.count(m) && !setE.count(m))
        {
            contracted1.insert(m);
        }
    }

    std::vector<int32_t> modeT;
    for(auto m : modeA)
    {
        if(!contracted1.count(m))
        {
            modeT.push_back(m);
        }
    }
    for(auto m : modeB)
    {
        if(!setA.count(m) && !contracted1.count(m))
        {
            modeT.push_back(m);
        }
    }
    return modeT;
}

// Build a tensor descriptor for T by looking up each mode's extent from A or B,
// then computing packed (column-major) strides.  The data type matches E so
// that step 2 can consume T without a type conversion.
inline hiptensorTensorDescriptor buildTrinaryContractionIntermediateDescriptor(
    const std::vector<int32_t>&        modeT,
    const hiptensorTensorDescriptor_t  descA,
    const std::vector<int32_t>&        modeA,
    const hiptensorTensorDescriptor_t  descB,
    const std::vector<int32_t>&        modeB,
    hiptensorDataType_t                dataType)
{
    std::unordered_map<int32_t, std::size_t> lengthMap;
    for(std::size_t i = 0; i < modeA.size(); i++)
    {
        lengthMap[modeA[i]] = descA->mLengths[i];
    }
    for(std::size_t i = 0; i < modeB.size(); i++)
    {
        if(lengthMap.find(modeB[i]) == lengthMap.end())
        {
            lengthMap[modeB[i]] = descB->mLengths[i];
        }
    }

    std::vector<std::size_t> lengthsT;
    lengthsT.reserve(modeT.size());
    for(auto m : modeT)
    {
        lengthsT.push_back(lengthMap.at(m));
    }

    auto stridesT = hiptensor::stridesFromLengths(lengthsT, true);

    hiptensorTensorDescriptor descT;
    descT.mType                 = dataType;
    descT.mLengths              = std::move(lengthsT);
    descT.mStrides              = std::move(stridesT);
    descT.mAlignmentRequirement = 256;
    return descT;
}

// Total byte count of the tensor (element count * element size).
inline std::size_t tensorByteSize(const hiptensorTensorDescriptor& desc)
{
    auto elements = hiptensor::elementsFromLengths(desc.mLengths);
    return elements * hiptensor::hiptensorDataTypeSize(desc.mType);
}

// Round up to the next 256-byte boundary (GPU alignment requirement).
inline std::size_t alignUp256(std::size_t v)
{
    return (v + 255) & ~std::size_t(255);
}

// Create the operation descriptor for a trinary contraction.
//
// Five tensors are involved:
//   A, B, C  -- multiplicative inputs (all required)
//   D        -- additive bias         (optional, may be nullptr for beta=0)
//   E        -- output                (required)
//
// Each input may carry a per-element unary operator (opA..opD).
// The descriptor is later consumed by contractionTrinaryInitPlan().
hiptensorStatus_t hiptensorCreateContractionTrinary(
    const hiptensorHandle_t            handle,
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
    hiptensorOperator_t                opD,
    const hiptensorTensorDescriptor_t  descE,
    const int32_t                      modeE[],
    const hiptensorComputeDescriptor_t descCompute)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, desc);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descB);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descC);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descE);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeB);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeC);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, modeE);
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
    int                  nModeC = descC->mLengths.size();
    std::vector<int32_t> modeCV(modeC, modeC + nModeC);
    int nModeD = (descD == nullptr || modeD == nullptr) ? 0 : descD->mLengths.size();
    std::vector<int32_t> modeDV = (descD == nullptr || modeD == nullptr)
                                      ? std::vector<int32_t>()
                                      : std::vector<int32_t>(modeD, modeD + nModeD);
    int                  nModeE = descE->mLengths.size();
    std::vector<int32_t> modeEV(modeE, modeE + nModeE);

    bool hasUnaryOp = (opA != HIPTENSOR_OP_IDENTITY) || (opB != HIPTENSOR_OP_IDENTITY)
                    ||(opC != HIPTENSOR_OP_IDENTITY) || (opD != HIPTENSOR_OP_IDENTITY);

    auto contractionOp = descD ? (descCompute == HIPTENSOR_COMPUTE_DESC_C32F
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

    (*desc)->mOperationType   = HIPTENSOR_CONTRACTION_TRINARY;
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
    (*desc)->mOpD   = opD;

    (*desc)->mDescE = descE;
    (*desc)->mModeE = modeEV;

    (*desc)->mOpAC  = HIPTENSOR_OP_IDENTITY;
    (*desc)->mOpABC = HIPTENSOR_OP_IDENTITY;

    (*desc)->mDescCompute = descCompute;

    logger->logAPITrace("hiptensorCreateContractionTrinary", "Created trinary contraction descriptor");

    return HIPTENSOR_STATUS_SUCCESS;
}

// Estimate the minimum workspace required for a trinary contraction.
//
// The workspace must be large enough to hold the intermediate tensor T
// (256-byte aligned).  Additional space beyond this minimum may be used
// by individual kernels for internal scratchpads.
hiptensorStatus_t contractionTrinaryGetWorkspaceSize(
    const hiptensorHandle_t              handle,
    const hiptensorOperationDescriptor_t desc,
    const hiptensorPlanPreference_t      planPref,
    const hiptensorWorksizePreference_t  workspacePref,
    uint64_t*                            workspaceSizeEstimate)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    char msg[512];
    snprintf(msg, sizeof(msg),
             "handle=0x%0*llX, desc=0x%llX, planPref=0x%llX, "
             "workspacePref=0x%u, workspaceSizeEstimate=0x%llX",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)desc,
             (unsigned long long)planPref,
             (unsigned int)workspacePref,
             (unsigned long long)workspaceSizeEstimate);
    logger->logAPITrace("contractionTrinaryGetWorkspaceSize", msg);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, desc);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, planPref);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, workspaceSizeEstimate);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS) { return checkResult; }
    
    *workspaceSizeEstimate = 0u;

    auto modeT = computeTrinaryContractionIntermediateModes(desc->mModeA, desc->mModeB,
                                          desc->mModeC, desc->mModeE);

    auto tDataType = desc->mDescE->mType;
    auto intermediateDesc = buildTrinaryContractionIntermediateDescriptor(
        modeT, desc->mDescA, desc->mModeA, desc->mDescB, desc->mModeB,
        tDataType);

    auto tBytes = tensorByteSize(intermediateDesc);
    auto tBytesAligned = alignUp256(tBytes);

    uint64_t kernelWs = 0u;

    for(auto* candidate : planPref->mCandidates)
    {
        auto* solution = (hiptensor::ContractionSolution*)candidate;
        if(solution->initArgs(nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              intermediateDesc.mLengths,
                              intermediateDesc.mStrides,
                              modeT,
                              hiptensor::getTensorLengths(desc->mDescC),
                              hiptensor::getTensorStrides(desc->mDescC),
                              desc->mModeC,
                              hiptensor::getTensorLengths(desc->mDescD),
                              hiptensor::getTensorStrides(desc->mDescD),
                              desc->mModeD,
                              hiptensor::getTensorLengths(desc->mDescE),
                              hiptensor::getTensorStrides(desc->mDescE),
                              desc->mModeE,
                              {},
                              nullptr))
        {
            if(kernelWs == 0)
            {
                kernelWs = solution->workspaceSize();
            }
            else
            {
                if(workspacePref == HIPTENSOR_WORKSPACE_MIN)
                {
                    kernelWs = std::min(kernelWs, solution->workspaceSize());
                }
                else
                {
                    kernelWs = std::max(kernelWs, solution->workspaceSize());
                }
            }
        }
    }

    *workspaceSizeEstimate = tBytesAligned + kernelWs;

    return HIPTENSOR_STATUS_SUCCESS;
}

// Initialize the execution plan for a trinary contraction.
//
// This function:
//  1. Computes the intermediate tensor T's shape from the mode analysis.
//  2. Queries the solution registry for kernels matching desc->mContractionOpId
//     (BILINEAR / BILINEAR_UNARY when descD is provided, SCALE / SCALE_COMPLEX
//     when descD is null).  The same kernel handles both steps by varying
//     alpha/beta and the tensor arguments.
//  3. Selects a single winner kernel via bruteForceModel (DEFAULT) or
//     actorCriticModel (ACTOR_CRITIC), using the step 2 problem shape
//     (T * C -> E, with bias D in the bilinear case).
//  4. Stores the winner in pref->mSolution for use by
//     hiptensorContractTrinary().
//
// The single winner solution is used for both step 1 (T = A*B) and
// step 2 (E = alpha*T*C + beta*D, or E = alpha*T*C in the scale case)
// at execution time.
hiptensorStatus_t contractionTrinaryInitPlan(
    const hiptensorHandle_t              handle,
    hiptensorPlan_t                      plan,
    const hiptensorOperationDescriptor_t desc,
    const hiptensorPlanPreference_t      pref,
    uint64_t                             workspaceSizeLimit)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

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

    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != handle->getDevice().getDeviceId())
    {
        auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
        snprintf(msg, sizeof(msg),
                 "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                 (int)currentDevice.getDeviceId(),
                 (int)handle->getDevice().getDeviceId(),
                 hiptensorGetErrorString(errorCode));
        logger->logError("contractionTrinaryInitPlan", msg);
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
    }

    // --- Derive the intermediate tensor T from mode analysis ---
    auto modeT = computeTrinaryContractionIntermediateModes(desc->mModeA, desc->mModeB,
                                          desc->mModeC, desc->mModeE);

    // T shares E's data type to avoid a type conversion between steps.
    auto tDataType = desc->mDescE->mType;
    auto intermediateDesc = buildTrinaryContractionIntermediateDescriptor(
        modeT, desc->mDescA, desc->mModeA, desc->mDescB, desc->mModeB, tDataType);

    auto tBytes = tensorByteSize(intermediateDesc);
    auto tBytesAligned = alignUp256(tBytes);

    // Workspace beyond what T needs is available for kernel scratchpads.
    uint64_t kernelWsLimit = (workspaceSizeLimit > tBytesAligned)
                                     ? (workspaceSizeLimit - tBytesAligned)
                                     : 0;

    // --- Load all registered contraction solutions ---
    auto& instances = hiptensor::ContractionSolutionInstances::instance();
    auto  solnQ     = instances->allSolutions();

    if(solnQ.solutionCount() == 0)
    {
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg, sizeof(msg), "Internal Error : No Kernels Found (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("contractionTrinaryInitPlan", msg);
        return errorCode;
    }

    auto allCandidatesVoid = toVoidVec(solnQ.solutions());
    auto allCandidates     = toContractionSolutionVec(allCandidatesVoid);

    auto computeType = desc->mDescCompute;
    auto ADataType   = desc->mDescA->mType;
    auto BDataType   = desc->mDescB->mType;
    auto CDataType   = desc->mDescC->mType;
    auto DDataType   = desc->mDescD ? desc->mDescD->mType : hiptensor::NONE_TYPE;
    auto EDataType   = desc->mDescE->mType;

    auto solutionQ = hiptensor::ContractionSolutionRegistry::Query{allCandidates}
                         .query((hiptensor::ContractionOpId_t)desc->mContractionOpId)
                         .query(ADataType, BDataType, DDataType, EDataType, computeType);

    bool hasUnaryOp = (desc->mOpA != HIPTENSOR_OP_IDENTITY)
                      || (desc->mOpB != HIPTENSOR_OP_IDENTITY)
                      || (desc->mOpC != HIPTENSOR_OP_IDENTITY)
                      || (desc->mOpD != HIPTENSOR_OP_IDENTITY);
    if(hasUnaryOp)
        solutionQ = solutionQ.query(HIPTENSOR_OP_UNKNOWN, HIPTENSOR_OP_UNKNOWN);
    else
        solutionQ = solutionQ.query(HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_IDENTITY);

    auto candidates = toContractionSolutionVec(solutionQ.solutions());

    // --- Select a single winner kernel ---
    // The winner is chosen using the step 2 problem shape (T * C -> E
    // with bias D).  The same kernel is reused for step 1 at execution
    // time with different alpha/beta and tensor arguments.
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
    CHECK_HIP_ERROR(hipEventRecord(startEvent));

    hiptensor::ContractionSolution* winner = nullptr;
    auto                            result = HIPTENSOR_STATUS_INTERNAL_ERROR;

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
        hiptensor::ContractionUnaryOps step2Ops
            = {HIPTENSOR_OP_IDENTITY, desc->mOpC, desc->mOpD};

        if(pref->mSelectionAlgorithm == HIPTENSOR_ALGO_DEFAULT
           || pref->mSelectionAlgorithm == HIPTENSOR_ALGO_DEFAULT_PATIENT)
        {
            result = hiptensor::bruteForceModel(
                &winner,
                candidates,
                tDataType,
                intermediateDesc.mLengths,
                intermediateDesc.mStrides,
                modeT,
                CDataType,
                hiptensor::getTensorLengths(desc->mDescC),
                hiptensor::getTensorStrides(desc->mDescC),
                desc->mModeC,
                DDataType,
                hiptensor::getTensorLengths(desc->mDescD),
                hiptensor::getTensorStrides(desc->mDescD),
                desc->mModeD,
                EDataType,
                hiptensor::getTensorLengths(desc->mDescE),
                hiptensor::getTensorStrides(desc->mDescE),
                desc->mModeE,
                computeType,
                step2Ops,
                kernelWsLimit);

            pref->mCandidates.clear();
            for(auto candidate : candidates)
                pref->mCandidates.push_back(candidate);
        }
        else if(pref->mSelectionAlgorithm == HIPTENSOR_ALGO_ACTOR_CRITIC)
        {
            auto tLengths = hiptensor::getTensorLengths(&intermediateDesc);
            auto tStrides = hiptensor::getTensorStrides(&intermediateDesc);
            auto cLengths = hiptensor::getTensorLengths(desc->mDescC);
            auto cStrides = hiptensor::getTensorStrides(desc->mDescC);

            if(hasUnaryOp)
                result = hiptensor::actorCriticModelUnaryOps(
                    &winner,
                    solutionQ.solutions(),
                    tDataType, tLengths, tStrides, modeT,
                    CDataType, cLengths, cStrides, desc->mModeC,
                    DDataType,
                    hiptensor::getTensorLengths(desc->mDescD),
                    hiptensor::getTensorStrides(desc->mDescD),
                    desc->mModeD,
                    EDataType,
                    hiptensor::getTensorLengths(desc->mDescE),
                    hiptensor::getTensorStrides(desc->mDescE),
                    desc->mModeE,
                    computeType,
                    kernelWsLimit);
            else
                result = hiptensor::actorCriticModel(
                    &winner,
                    solutionQ.solutions(),
                    tDataType, tLengths, tStrides, modeT,
                    CDataType, cLengths, cStrides, desc->mModeC,
                    DDataType,
                    hiptensor::getTensorLengths(desc->mDescD),
                    hiptensor::getTensorStrides(desc->mDescD),
                    desc->mModeD,
                    EDataType,
                    hiptensor::getTensorLengths(desc->mDescE),
                    hiptensor::getTensorStrides(desc->mDescE),
                    desc->mModeE,
                    computeType,
                    kernelWsLimit);
        }
    }

    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    if(result != HIPTENSOR_STATUS_SUCCESS || winner == nullptr)
    {
        snprintf(msg, sizeof(msg),
                 "Failed to select kernel for trinary contraction (%s)",
                 hiptensorGetErrorString(result));
        logger->logError("contractionTrinaryInitPlan", msg);
        return result;
    }

    snprintf(msg, sizeof(msg),
             "Algo: %d, KernelId: %zu, KernelName: %s, SelectionTime: %0.3f ms",
             pref->mSelectionAlgorithm,
             winner->uid(),
             winner->kernelName().c_str(),
             elapsedTimeMs);
    logger->logPerformanceTrace("contractionTrinaryInitPlan", msg);

    pref->mSolution          = winner;
    plan->mRequiredWorkspace = workspaceSizeLimit;
    plan->mOpDesc            = desc;
    plan->mPref              = pref;

    return HIPTENSOR_STATUS_SUCCESS;
}

// Execute a trinary contraction:  E = alpha * op(A) * op(B) * op(C) + beta * op(D)
//
// The operation is decomposed into two sequential binary contractions
// launched on the same HIP stream (no explicit host sync between them):
//
//   Step 1  T = A * B             (alpha=1, beta=0, no bias)
//   Step 2  E = alpha*T*C + beta*D
//
// A single kernel (pref->mSolution) selected at plan-creation time is used
// for both steps.
//
// Workspace layout:
//   [ intermediate tensor T (256-byte aligned) | sub-workspace for kernels ]
hiptensorStatus_t hiptensorContractTrinary(const hiptensorHandle_t handle,
                                           const hiptensorPlan_t   plan,
                                           const void*             alpha,
                                           const void*             A,
                                           const void*             B,
                                           const void*             C,
                                           const void*             beta,
                                           const void*             D,
                                           void*                   E,
                                           void*                   workspace,
                                           uint64_t                workspaceSize,
                                           hipStream_t             stream)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Plan Cache autotune: begin a new round (or continue an existing one).
    using hiptensor::PlancacheAutotuneMgr;
    auto& autotuneMgr = PlancacheAutotuneMgr::instance();
    autotuneMgr->startAutotune(hiptensor::AutotuneOps::Autotune_ContractionTrinary);

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
            hiptensor::AutotuneOps::Autotune_ContractionTrinary, handle, plan);
    }
    else
    {
        snprintf(alphaMsg, sizeof(alphaMsg), "alpha=NULL");
        snprintf(betaMsg, sizeof(betaMsg), "beta=NULL");
    }

    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, plan=0x%llX, %s, A=0x%llX, B=0x%llX, C=0x%llX, %s, "
             "D=0x%llX, E=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)plan,
             alphaMsg,
             (unsigned long long)A,
             (unsigned long long)B,
             (unsigned long long)C,
             betaMsg,
             (unsigned long long)D,
             (unsigned long long)E,
             (unsigned long long)workspace,
             (unsigned long)workspaceSize,
             (unsigned long long)stream);

    logger->logAPITrace("hiptensorContractTrinary", msg);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, plan);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, alpha);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, A);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, B);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, C);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_INVALID_VALUE, E);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != handle->getDevice().getDeviceId())
    {
        auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
        snprintf(msg, sizeof(msg),
                 "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                 (int)currentDevice.getDeviceId(),
                 (int)handle->getDevice().getDeviceId(),
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContractTrinary", msg);
        return errorCode;
    }

    auto* solution = (hiptensor::ContractionSolution*)(plan->mPref->mSolution);

    if(!solution)
    {
        snprintf(msg, sizeof(msg), "Trinary contraction plan has null solution");
        logger->logError("hiptensorContractTrinary", msg);
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }

    // Recompute the intermediate tensor T's shape from the operation descriptor.
    auto modeT = computeTrinaryContractionIntermediateModes(plan->mOpDesc->mModeA, plan->mOpDesc->mModeB,
                                          plan->mOpDesc->mModeC, plan->mOpDesc->mModeE);
    auto tDataType = plan->mOpDesc->mDescE->mType;
    auto intermediateDesc = buildTrinaryContractionIntermediateDescriptor(
        modeT, plan->mOpDesc->mDescA, plan->mOpDesc->mModeA,
        plan->mOpDesc->mDescB, plan->mOpDesc->mModeB, tDataType);

    // Partition the workspace: [T (aligned)] [sub-workspace for kernels]
    auto tBytes        = tensorByteSize(intermediateDesc);
    auto tBytesAligned = alignUp256(tBytes);

    if(workspaceSize < tBytesAligned)
    {
        snprintf(msg, sizeof(msg),
                 "Insufficient workspace for intermediate tensor: req: %zu alloc: %llu (%s)",
                 tBytesAligned,
                 static_cast<unsigned long long>(workspaceSize),
                 hiptensorGetErrorString(HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE));
        logger->logError("hiptensorContractTrinary", msg);
        return HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if(tBytesAligned > 0 && workspace == nullptr)
    {
        snprintf(msg, sizeof(msg),
                 "Null workspace pointer for trinary contraction (intermediate tensor requires %zu bytes)",
                 tBytesAligned);
        logger->logError("hiptensorContractTrinary", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    void*    T             = workspace;
    void*    kernelWs      = static_cast<char*>(workspace) + tBytesAligned;
    uint64_t kernelWsSize  = workspaceSize - tBytesAligned;

    using hiptensor::HiptensorOptions;
    auto& options = HiptensorOptions::instance();

    const bool perfTrace =
        (logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE) != 0;
    StreamConfig streamCfg = perfTrace
        ? StreamConfig{stream, true, 0, options->coldRuns(), options->hotRuns()}
        : StreamConfig{stream, false};

    // --- Step 1: T = A * B (alpha=1, no bias) ---
    hiptensor::ScalarData oneScalar;
    hiptensor::ScalarData zeroScalar;
    hiptensor::writeVal(&oneScalar, plan->mOpDesc->mDescCompute,
                        hiptensor::ScalarData(plan->mOpDesc->mDescCompute, 1.0, 0.0));
    hiptensor::writeVal(&zeroScalar, plan->mOpDesc->mDescCompute,
                        hiptensor::ScalarData(plan->mOpDesc->mDescCompute, 0.0, 0.0));

    hiptensorStatus_t errorCode = HIPTENSOR_STATUS_SUCCESS;
    float             time1      = 0.0f;
    std::tie(errorCode, time1)
        = (*solution)(&oneScalar,
                      A,
                      B,
                      &zeroScalar,
                      T,
                      T,
                      hiptensor::getTensorLengths(plan->mOpDesc->mDescA),
                      hiptensor::getTensorStrides(plan->mOpDesc->mDescA),
                      plan->mOpDesc->mModeA,
                      hiptensor::getTensorLengths(plan->mOpDesc->mDescB),
                      hiptensor::getTensorStrides(plan->mOpDesc->mDescB),
                      plan->mOpDesc->mModeB,
                      intermediateDesc.mLengths,
                      intermediateDesc.mStrides,
                      modeT,
                      intermediateDesc.mLengths,
                      intermediateDesc.mStrides,
                      modeT,
                      {plan->mOpDesc->mOpA, plan->mOpDesc->mOpB, HIPTENSOR_OP_IDENTITY},
                      kernelWs,
                      kernelWsSize,
                      streamCfg);

    // Capture step 1's problemDims before step 2 overwrites them (for comparison).
    int32_t m1, n1, k1;
    std::tie(m1, n1, k1) = solution->problemDims();
    auto bytes1 = solution->problemBytes();

    if(errorCode != HIPTENSOR_STATUS_SUCCESS)
    {
        snprintf(msg, sizeof(msg),
                 "Trinary contraction step 1 (T = A*B) failed (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContractTrinary", msg);
        return errorCode;
    }

    // --- Step 2: E = alpha * T * C + beta * D ---
    errorCode = HIPTENSOR_STATUS_SUCCESS;
    float             time2      = 0.0f;
    std::tie(errorCode, time2)
        = (*solution)(alpha,
                       T,
                       C,
                       beta,
                       D,
                       E,
                       intermediateDesc.mLengths,
                       intermediateDesc.mStrides,
                       modeT,
                       hiptensor::getTensorLengths(plan->mOpDesc->mDescC),
                       hiptensor::getTensorStrides(plan->mOpDesc->mDescC),
                       plan->mOpDesc->mModeC,
                       hiptensor::getTensorLengths(plan->mOpDesc->mDescD),
                       hiptensor::getTensorStrides(plan->mOpDesc->mDescD),
                       plan->mOpDesc->mModeD,
                       hiptensor::getTensorLengths(plan->mOpDesc->mDescE),
                       hiptensor::getTensorStrides(plan->mOpDesc->mDescE),
                       plan->mOpDesc->mModeE,
                       {HIPTENSOR_OP_IDENTITY, plan->mOpDesc->mOpC, plan->mOpDesc->mOpD},
                       kernelWs,
                       kernelWsSize,
                       streamCfg);

    if(errorCode != HIPTENSOR_STATUS_SUCCESS)
    {
        snprintf(msg, sizeof(msg),
                 "Trinary contraction step 2 (E = alpha*T*C + beta*D) failed (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContractTrinary", msg);
        return errorCode;
    }

    auto totalTime = time1 + time2;
    if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
    {
        int32_t m2, n2, k2;
        std::tie(m2, n2, k2) = solution->problemDims();
        auto bytes2 = solution->problemBytes();

        auto totalFlops = std::size_t(2) * m1 * n1 * k1
                        + std::size_t(2) * m2 * n2 * k2;
        auto totalBytes = bytes1 + bytes2;

        hiptensor::PerfMetrics metrics = {
            solution->uid(),
            solution->kernelName(),
            totalTime,
            static_cast<float>(totalFlops) / static_cast<float>(1.E9) / totalTime,
            static_cast<float>(totalBytes) / static_cast<float>(1.E6) / totalTime
        };

        snprintf(msg,
                 sizeof(msg),
                 "KernelId: %zu KernelName: %s, %0.3f ms, %0.3f TFlops/s, %0.3f GB/s",
                 metrics.mKernelUid,
                 metrics.mKernelName.c_str(),
                 metrics.mAvgTimeMs,
                 metrics.mTflops,
                 metrics.mBandwidth);
        logger->logPerformanceTrace("hiptensorContractTrinary", msg);
    }

    // Plan Cache autotune: record execution time and, once enough
    // samples have been collected, persist the fastest solution to the cache.
    autotuneMgr->saveAutotune<hiptensor::ContractionSolution>(
        hiptensor::AutotuneOps::Autotune_ContractionTrinary, totalTime, handle, plan);

    return errorCode;
}
