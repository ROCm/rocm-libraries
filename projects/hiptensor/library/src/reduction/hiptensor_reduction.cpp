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

#include <set>

#include <hiptensor/hiptensor.h>

#include "handle.hpp"
#include "hip_device.hpp"
#include "hip_error.hpp"
#include "logger.hpp"

#include "reduction_epilogue.hpp"
#include "reduction_solution.hpp"
#include "reduction_solution_instances.hpp"
#include "reduction_solution_registry.hpp"

#include "backend/registry.hpp"
#include "backends/ck/ck_backend.hpp"
#include "hiptensor_options.hpp"
#include "plancache_autotune.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

namespace
{
    // CK does not take f16 or bf16 as the compute type of a reduction; it accumulates in f32.
    hiptensorComputeDescriptor_t internalComputeType(hiptensorComputeDescriptor_t typeCompute)
    {
        return (typeCompute == HIPTENSOR_COMPUTE_DESC_16F
                || typeCompute == HIPTENSOR_COMPUTE_DESC_16BF)
                   ? HIPTENSOR_COMPUTE_DESC_32F
                   : typeCompute;
    }

    // The data types and the shape of a reduction: checked when the plan is created and when it
    // runs.
    //  - The data types of A, C and D with the compute type must be implemented
    //    (HIPTENSOR_STATUS_NOT_SUPPORTED).
    //  - C and D must have the same descriptor and the same modes, and carry a subset of A's
    //    modes, each at most once (HIPTENSOR_STATUS_NOT_SUPPORTED), with A's extents
    //    (HIPTENSOR_STATUS_INVALID_VALUE).
    hiptensorStatus_t checkReductionShape(char const*                         apiFuncName,
                                          hiptensorOperationDescriptor const& desc)
    {
        using hiptensor::Logger;
        auto& logger = Logger::instance();
        char  msg[512];

        auto const* descA = desc.mDescA;
        auto const* descC = desc.mDescC;
        auto const* descD = desc.mDescD;

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

        if(supportedTypes.find(
               hashGenerator(descA->mType, descC->mType, descD->mType, desc.mDescCompute))
           == supportedTypes.end())
        {
            auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
            snprintf(msg,
                     sizeof(msg),
                     "Unsupported Data Type Error : The combination of data types of A, C and D "
                     "and compute type is not supported. (%s)",
                     hiptensorGetErrorString(errorCode));
            logger->logError(apiFuncName, msg);
            return errorCode;
        }

        auto const& modeA    = desc.mModeA;
        auto const& modeC    = desc.mModeC;
        auto const  modeSetA = std::set<int32_t>(modeA.cbegin(), modeA.cend());
        auto const  modeSetC = std::set<int32_t>(modeC.cbegin(), modeC.cend());

        auto compareDescCD = [](auto&& left, auto&& right) {
            auto copyLeft = left;
            copyLeft.mAlignmentRequirement
                = right.mAlignmentRequirement; // compare these 2 desc except mAlignmentRequirement
            return copyLeft == right;
        };

        if(modeSetA.size() != modeA.size() || modeSetC.size() != modeC.size()
           || descA->mLengths.size() < descC->mLengths.size() || !compareDescCD(*descC, *descD)
           || desc.mModeD != modeC
           || !std::includes(
               modeSetA.cbegin(), modeSetA.cend(), modeSetC.cbegin(), modeSetC.cend()))
        {
            auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
            snprintf(msg,
                     sizeof(msg),
                     "Unsupported Data Error : The descriptors and modes of C and D should be the "
                     "same, the modes of C should be a subset of the modes of A, and no mode may "
                     "repeat within a tensor. (%s)",
                     hiptensorGetErrorString(errorCode));
            logger->logError(apiFuncName, msg);
            return errorCode;
        }

        for(std::size_t i = 0; i < modeC.size(); i++)
        {
            auto it = std::find(modeA.cbegin(), modeA.cend(), modeC[i]);
            if(descA->mLengths[std::distance(modeA.cbegin(), it)] != descC->mLengths[i])
            {
                snprintf(msg,
                         sizeof(msg),
                         "Mode '%c' (%d) has extent %zu in A but extent %zu in C and D",
                         static_cast<char>(modeC[i]),
                         modeC[i],
                         descA->mLengths[std::distance(modeA.cbegin(), it)],
                         descC->mLengths[i]);
                logger->logError(apiFuncName, msg);
                return HIPTENSOR_STATUS_INVALID_VALUE;
            }
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }

    // A's strides along the modes of C and D, for a reduction whose output keeps every mode of A.
    std::vector<std::size_t> stridesAlongOutputModes(hiptensorTensorDescriptor const& descA,
                                                     std::vector<int32_t> const&      modeA,
                                                     std::vector<int32_t> const&      modeOut)
    {
        auto const& options = hiptensor::HiptensorOptions::instance();
        auto        stridesA
            = descA.mStrides.empty()
                  ? hiptensor::stridesFromLengths(descA.mLengths, options->isColMajorStrides())
                  : descA.mStrides;
        std::vector<std::size_t> result;
        for(auto mode : modeOut)
        {
            auto it = std::find(modeA.cbegin(), modeA.cend(), mode);
            result.push_back(stridesA[std::distance(modeA.cbegin(), it)]);
        }
        return result;
    }

    std::vector<std::size_t> stridesOf(hiptensorTensorDescriptor const& desc)
    {
        auto const& options = hiptensor::HiptensorOptions::instance();
        return desc.mStrides.empty()
                   ? hiptensor::stridesFromLengths(desc.mLengths, options->isColMajorStrides())
                   : desc.mStrides;
    }
}

namespace hiptensor
{
    hiptensorStatus_t reductionCheckPlan(hiptensorOperationDescriptor const& desc)
    {
        auto& logger = Logger::instance();
        char  msg[256];

        if(auto status = checkReductionShape("hiptensorCreatePlan", desc);
           status != HIPTENSOR_STATUS_SUCCESS)
        {
            return status;
        }

        auto const rankA = desc.mDescA->mLengths.size();
        auto const rankD = desc.mDescD->mLengths.size();

        // A reduction whose output keeps every mode of A reduces nothing and runs as the
        // elementwise epilogue.
        if(rankA == rankD)
        {
            if(rankD > kReductionEpilogueMaxRank)
            {
                snprintf(msg,
                         sizeof(msg),
                         "Reduction of a rank-%zu tensor that keeps every mode is not supported",
                         rankA);
                logger->logError("hiptensorCreatePlan", msg);
                return HIPTENSOR_STATUS_NOT_SUPPORTED;
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }

        auto& instances = ReductionSolutionInstances::instance();
        auto  solutionQ = instances->querySolutions(desc.mDescA->mType,
                                                   internalComputeType(desc.mDescCompute),
                                                   desc.mDescD->mType,
                                                   static_cast<int>(rankA),
                                                   static_cast<int>(rankA - rankD),
                                                   desc.mOpAC,
                                                   true,
                                                   false);
        if(solutionQ.solutionCount() == 0)
        {
            snprintf(msg,
                     sizeof(msg),
                     "No reduction kernel reduces %zu of %zu modes with these data types and "
                     "operator",
                     rankA - rankD,
                     rankA);
            logger->logError("hiptensorCreatePlan", msg);
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        }
        return HIPTENSOR_STATUS_SUCCESS;
    }
} // namespace hiptensor

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
try
{
    hiptensor::backend::LaunchConfig cfg;
    cfg.stream = stream;

    // Plans run on the backend selected when they were created.
    if(handle == nullptr || plan == nullptr || plan->mBackend == nullptr)
    {
        return hiptensor::ckbackend::reduce(
            handle, plan, alpha, A, beta, C, D, workspace, workspaceSize, cfg);
    }

    hiptensor::backend::Scalars scalars;
    scalars.alpha = alpha;
    scalars.beta  = beta;

    hiptensor::backend::Buffers buffers;
    buffers.A   = A;
    buffers.C   = C;
    buffers.out = D;

    return hiptensor::backend::executePlan(
        handle, plan, scalars, buffers, workspace, workspaceSize, cfg);
}
catch(...)
{
    return hiptensor::exceptionToStatus("hiptensorReduce");
}

hiptensorStatus_t hiptensor::ckbackend::reduce(hiptensorHandle_t            handle,
                                               hiptensorPlan_t              plan,
                                               void const*                  alpha,
                                               void const*                  A,
                                               void const*                  beta,
                                               void const*                  C,
                                               void*                        D,
                                               void*                        workspace,
                                               uint64_t                     workspaceSize,
                                               backend::LaunchConfig const& cfg)
{
    hipStream_t const stream = cfg.stream;

    using hiptensor::Logger;
    auto& logger = Logger::instance();

    using hiptensor::PlancacheAutotuneMgr;
    auto& autotuneMgr = PlancacheAutotuneMgr::instance();
    autotuneMgr->startAutotune(hiptensor::AutotuneOps::Autotune_Redution);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, plan);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    // The operation descriptor is only read here, so every execution of the plan sees the
    // reduction operator the plan was created with. The CK path still records the selected
    // kernel in the plan (mPref->mSolution) and the plan cache, so concurrent executions of one
    // plan are not safe.
    hiptensorOperationDescriptor const& opDes       = *plan->mOpDesc;
    const hiptensorTensorDescriptor_t   descA       = opDes.mDescA;
    const hiptensorTensorDescriptor_t   descC       = opDes.mDescC;
    const hiptensorTensorDescriptor_t   descD       = opDes.mDescD;
    auto const&                         modeA       = opDes.mModeA;
    auto const&                         modeC       = opDes.mModeC;
    hiptensorOperator_t                 opA         = opDes.mOpA;
    hiptensorOperator_t                 opC         = opDes.mOpC;
    hiptensorOperator_t                 opReduce    = opDes.mOpAC;
    hiptensorComputeDescriptor_t        typeCompute = opDes.mDescCompute;

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
             modeA.data(),
             beta,
             C,
             descC,
             modeC.data(),
             D,
             descD,
             opDes.mModeD.data(),
             hiptensor::opTypeToString(opReduce).c_str(),
             hiptensor::computeTypeToString(typeCompute).c_str(),
             workspace,
             static_cast<unsigned long long>(workspaceSize),
             stream);

    logger->logAPITrace("hiptensorReduce", msg);

    // hipTensor does not use `workspace`. Do not check `workspace`.
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, alpha);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, beta);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descC);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, D);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descD);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    if(auto errorCode = checkReductionShape("hiptensorReduce", opDes);
       errorCode != HIPTENSOR_STATUS_SUCCESS)
    {
        return errorCode;
    }

    double const alphaValue = hiptensor::readVal<double>(alpha, typeCompute);
    double const betaValue  = hiptensor::readVal<double>(beta, typeCompute);

    // cuTENSOR semantics: a tensor whose scaling factor is zero is neither read nor transformed
    // by its operator, and may be nullptr.
    bool const readsA = alphaValue != 0.0;
    bool const readsC = betaValue != 0.0;
    if((readsA && A == nullptr) || (readsC && C == nullptr))
    {
        snprintf(msg,
                 sizeof(msg),
                 "%s = nullptr with a non-zero scaling factor",
                 (readsA && A == nullptr) ? "A" : "C");
        logger->logError("hiptensorReduce", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    auto const stridesD = stridesOf(*descD);

    // Nothing to reduce: D = alpha * opA(A) + beta * opC(C), element by element. Also when
    // alpha is zero: D = beta * opC(C), without reading A.
    if(descA->mLengths.size() == descD->mLengths.size() || !readsA)
    {
        return hiptensor::reductionEpilogue(descD->mType,
                                            descD->mLengths,
                                            stridesD,
                                            D,
                                            readsA ? stridesAlongOutputModes(*descA, modeA, modeC)
                                                   : std::vector<std::size_t>{},
                                            readsA ? A : nullptr,
                                            opA,
                                            alphaValue,
                                            stridesOf(*descC),
                                            readsC ? C : nullptr,
                                            opC,
                                            betaValue,
                                            stream);
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

    autotuneMgr->setAutotune<hiptensor::ReductionSolution>(
        hiptensor::AutotuneOps::Autotune_Redution, handle, plan);

    std::vector<hiptensor::ReductionSolution*> solutions;
    if(plan->mPref->mSolution != nullptr)
        solutions.push_back((hiptensor::ReductionSolution*)plan->mPref->mSolution);
    else
    {
        // Query reduction solutions for the correct reduction operation and type
        auto solutionQ = instances->querySolutions(ADataType,
                                                   internalComputeType(typeCompute),
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

    // The reduction kernels compute D = alpha * reduce(opA(A)) + beta * opC(D), updating D in
    // place, so C is copied to D first, on the caller's stream and element by element (C and D
    // have the same layout, which may leave gaps between elements). With beta == 0 the kernels
    // do not read D, and C is not copied.
    if(readsC && C != D)
    {
        HIPTENSOR_RETURN_IF_ERROR(hiptensor::reductionCopyTensor(
            descC->mType, descC->mLengths, stridesOf(*descC), D, C, stream));
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

    HIPTENSOR_RETURN_IF_ERROR(hiptensor::getSortingIndices(modeA.data(),
                                                           descA->mLengths.size(),
                                                           modeC.data(),
                                                           sortedDescC.mLengths.size(),
                                                           sortingIndices));
    HIPTENSOR_RETURN_IF_ERROR(hiptensor::applySortingIndices(sortingIndices, sortedDescC.mLengths));
    HIPTENSOR_RETURN_IF_ERROR(hiptensor::applySortingIndices(sortingIndices, sortedDescC.mStrides));

    for(auto pSolution : solutions)
    {
        // Perform reduction with timing if LOG_LEVEL_PERF_TRACE
        auto streamConfig = hiptensor::ckbackend::toStreamConfig(
            (logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
                ? hiptensor::ckbackend::timedLaunch(cfg)
                : cfg);
        auto [isSupported, time]
            = (*pSolution)(descA->mLengths,
                           descA->mStrides,
                           modeA,
                           sortedDescC.mLengths,
                           sortedDescC.mStrides,
                           {modeC.data(), modeC.data() + sortedDescC.mLengths.size()},
                           opA,
                           opC,
                           alphaValue,
                           readsC ? betaValue : 0.0,
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
