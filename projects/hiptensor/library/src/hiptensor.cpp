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
#include <cstring>

#include <hip/hip_runtime_api.h>

#include <hiptensor/hiptensor.h>

#include "data_types.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "util.hpp"

hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* handle)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(
        msg, sizeof(msg), "handle=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)handle);
    logger->logAPITrace("hiptensorCreate", msg);

    (*handle) = new hiptensorHandle;

    if(*handle == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_ALLOC_FAILED;
        snprintf(msg,
                 sizeof(msg),
                 "Initialization Error : handle = nullptr (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorCreate", msg);
        return HIPTENSOR_STATUS_ALLOC_FAILED;
    }

    auto hip_status = hipInit(0);

    if(hip_status == hipErrorInvalidDevice)
    {
        auto errorCode = HIPTENSOR_STATUS_HIP_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Initialization error: invalid device (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorCreate", msg);
        return HIPTENSOR_STATUS_HIP_ERROR;
    }
    else if(hip_status == hipErrorInvalidValue)
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        snprintf(
            msg, sizeof(msg), "Initialization error: (%s)", hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorCreate", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    if (hiptensor::checkEnvironmentVariableEnabled("HIPTENSOR_DISABLE_PLAN_CACHE"))
    {
        snprintf(msg, sizeof(msg), "Plan Cache is disabled.");
        logger->logAPITrace("hiptensorCreate", msg);
    }
    else
    {
        hiptensor::PlanCache* planCache = new hiptensor::PlanCache;
        (*handle)->setPlanCache(planCache);
        snprintf(msg, sizeof(msg), "Plan Cache is enabled.");
        logger->logAPITrace("hiptensorCreate", msg);
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t handle)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(
        msg, sizeof(msg), "handle=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)handle);
    logger->logAPITrace("hiptensorDestroy", msg);

    delete handle;
    handle = nullptr;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorHandleResizePlanCache(hiptensorHandle_t handle,
                                                 const uint32_t    numEntries)
{
    if(handle->getPlanCache() == nullptr)
        return HIPTENSOR_STATUS_SUCCESS;

    if(numEntries < 1u)
        return HIPTENSOR_STATUS_INVALID_VALUE;

    handle->getPlanCache()->resize(numEntries);

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorHandleWritePlanCacheToFile(const hiptensorHandle_t handle,
                                                      const char              fileName[])
{
    if(handle->getPlanCache() == nullptr)
        return HIPTENSOR_STATUS_SUCCESS;

    return handle->getPlanCache()->writeFile(fileName);
}

hiptensorStatus_t hiptensorHandleReadPlanCacheFromFile(hiptensorHandle_t handle,
                                                       const char        fileName[],
                                                       uint32_t*         numCachelinesRead)
{
    if(handle->getPlanCache() == nullptr)
        return HIPTENSOR_STATUS_SUCCESS;

    hiptensorStatus_t retVal = handle->getPlanCache()->readFile(fileName);
    if(retVal == HIPTENSOR_STATUS_SUCCESS)
        (*numCachelinesRead) = handle->getPlanCache()->getCachelinesNum();
    return retVal;
}

hiptensorStatus_t hiptensorWriteKernelCacheToFile(const hiptensorHandle_t handle,
                                                  const char              fileName[])
{
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorReadKernelCacheFromFile(hiptensorHandle_t handle, const char fileName[])
{
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorCreateTensorDescriptor(const hiptensorHandle_t      handle,
                                                  hiptensorTensorDescriptor_t* desc,
                                                  const uint32_t               numModes,
                                                  const int64_t                lens[],
                                                  const int64_t                strides[],
                                                  hiptensorDataType_t          dataType,
                                                  uint32_t                     alignmentRequirement)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[256];
    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, numModes=0x%02X, lens=0x%llX, strides=0x%llX,"
             "dataType=0x%02X, alignmentRequirement=0x%02X",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned int)numModes,
             (unsigned long long)lens,
             (unsigned long long)strides,
             (unsigned int)dataType,
             (unsigned int)alignmentRequirement);
    logger->logAPITrace("hiptensorCreateTensorDescriptor", msg);

    if(handle == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_INITIALIZED;
        if(handle == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : handle = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : contraction descriptor = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorCreateTensorDescriptor", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    if((lens == nullptr && strides != nullptr)
       || ((dataType != HIPTENSOR_R_16F) && (dataType != HIPTENSOR_R_16BF)
           && (dataType != HIPTENSOR_R_32F) && (dataType != HIPTENSOR_R_64F)
           && (dataType != HIPTENSOR_C_32F) && (dataType != HIPTENSOR_C_64F)))
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        if(lens == nullptr && strides != nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Tensor Initialization Error : lens = nullptr and strides != nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Tensor Initialization Error : datatype should be float or double (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorCreateTensorDescriptor", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    if(dataType == HIPTENSOR_R_64F && !handle->getDevice().supportsF64())
    {
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
    }

    *desc = new hiptensorTensorDescriptor();

    if(strides)
    {
        // Construct with both given lengths and strides
        **desc = {dataType,
                  std::vector<std::size_t>(lens, lens + numModes),
                  std::vector<std::size_t>(strides, strides + numModes),
                  alignmentRequirement};
    }
    else
    {
        // Re-construct strides from lengths, assuming packed.
        if(numModes > 0)
        {
            using hiptensor::HiptensorOptions;
            auto& options = HiptensorOptions::instance();

            auto                     lensVector = std::vector<std::size_t>(lens, lens + numModes);
            std::vector<std::size_t> stridesVector
                = hiptensor::stridesFromLengths(lensVector, options->isColMajorStrides());
            **desc = {dataType, lensVector, stridesVector, alignmentRequirement};
        }
        else
        {
            **desc = {dataType,
                      std::vector<std::size_t>(),
                      std::vector<std::size_t>(),
                      alignmentRequirement};
        }
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorDestroyTensorDescriptor(hiptensorTensorDescriptor_t desc)
{
    delete desc;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorDestroyOperationDescriptor(hiptensorOperationDescriptor_t desc)
{
    delete desc;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t
    hiptensorOperationDescriptorSetAttribute(const hiptensorHandle_t                 handle,
                                             hiptensorOperationDescriptor_t          desc,
                                             hiptensorOperationDescriptorAttribute_t attr,
                                             const void*                             buf,
                                             size_t                                  sizeInBytes)
{
    hiptensorStatus_t retStatus = HIPTENSOR_STATUS_SUCCESS;

    switch(attr)
    {
    case HIPTENSOR_OPERATION_DESCRIPTOR_TAG:
        std::memcpy(&desc->mTag, buf, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE:
        std::memcpy(&desc->mScalarType, buf, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS:
        std::memcpy(&desc->mFlops, buf, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES:
        std::memcpy(&desc->mMovedBytes, buf, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT:
        std::memcpy(&desc->mPaddingLeft, buf, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT:
        std::memcpy(&desc->mPaddingRighT, buf, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE:
        std::memcpy(desc->mPaddingValue, buf, sizeInBytes);
        break;
    default:
        retStatus = HIPTENSOR_STATUS_NOT_SUPPORTED;
        break;
    }

    return retStatus;
}

hiptensorStatus_t
    hiptensorOperationDescriptorGetAttribute(const hiptensorHandle_t                 handle,
                                             hiptensorOperationDescriptor_t          desc,
                                             hiptensorOperationDescriptorAttribute_t attr,
                                             void*                                   buf,
                                             size_t                                  sizeInBytes)
{
    hiptensorStatus_t retStatus = HIPTENSOR_STATUS_SUCCESS;

    switch(attr)
    {
    case HIPTENSOR_OPERATION_DESCRIPTOR_TAG:
        std::memcpy(buf, &desc->mTag, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE:
        std::memcpy(buf, &desc->mScalarType, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS:
        std::memcpy(buf, &desc->mFlops, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES:
        std::memcpy(buf, &desc->mMovedBytes, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT:
        std::memcpy(buf, &desc->mPaddingLeft, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT:
        std::memcpy(buf, &desc->mPaddingRighT, sizeInBytes);
        break;
    case HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE:
        std::memcpy(buf, desc->mPaddingValue, sizeInBytes);
        break;
    default:
        retStatus = HIPTENSOR_STATUS_NOT_SUPPORTED;
        break;
    }

    return retStatus;
}

hiptensorStatus_t hiptensorCreatePermutation(const hiptensorHandle_t            handle,
                                             hiptensorOperationDescriptor_t*    desc,
                                             const hiptensorTensorDescriptor_t  descA,
                                             const int32_t                      modeA[],
                                             hiptensorOperator_t                opA,
                                             const hiptensorTensorDescriptor_t  descB,
                                             const int32_t                      modeB[],
                                             const hiptensorComputeDescriptor_t descCompute)
{
    *desc = new hiptensorOperationDescriptor();

    (*desc)->mTag          = 0u;
    (*desc)->mScalarType   = *hiptensor::convertToHipTensorDataType(descCompute);
    (*desc)->mFlops        = 0.0f;
    (*desc)->mMovedBytes   = 0.0f;
    (*desc)->mPaddingLeft  = 0u;
    (*desc)->mPaddingRighT = 0u;
    (*desc)->mPaddingValue = nullptr;

    (*desc)->mOperationType   = HIPTENSOR_PERMUTATION;
    (*desc)->mContractionOpId = 0;

    (*desc)->mDescA       = descA;
    (*desc)->mModeA       = std::vector<int32_t>(modeA, modeA + descA->mLengths.size());
    (*desc)->mOpA         = opA;
    (*desc)->mDescB       = descB;
    (*desc)->mModeB       = std::vector<int32_t>(modeB, modeB + descB->mLengths.size());
    (*desc)->mOpB         = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescC       = nullptr;
    (*desc)->mModeC       = {};
    (*desc)->mOpC         = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescD       = nullptr;
    (*desc)->mModeD       = {};
    (*desc)->mOpD         = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescE       = nullptr;
    (*desc)->mModeE       = {};
    (*desc)->mOpAC        = HIPTENSOR_OP_IDENTITY;
    (*desc)->mOpABC       = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescCompute = descCompute;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorCreateElementwiseBinary(const hiptensorHandle_t            handle,
                                                   hiptensorOperationDescriptor_t*    desc,
                                                   const hiptensorTensorDescriptor_t  descA,
                                                   const int32_t                      modeA[],
                                                   hiptensorOperator_t                opA,
                                                   const hiptensorTensorDescriptor_t  descC,
                                                   const int32_t                      modeC[],
                                                   hiptensorOperator_t                opC,
                                                   const hiptensorTensorDescriptor_t  descD,
                                                   const int32_t                      modeD[],
                                                   hiptensorOperator_t                opAC,
                                                   const hiptensorComputeDescriptor_t descCompute)
{
    *desc                  = new hiptensorOperationDescriptor();
    (*desc)->mTag          = 0u;
    (*desc)->mScalarType   = *hiptensor::convertToHipTensorDataType(descCompute);
    (*desc)->mFlops        = 0.0f;
    (*desc)->mMovedBytes   = 0.0f;
    (*desc)->mPaddingLeft  = 0u;
    (*desc)->mPaddingRighT = 0u;
    (*desc)->mPaddingValue = nullptr;

    (*desc)->mOperationType   = HIPTENSOR_ELEMENTWISE_BINARY;
    (*desc)->mContractionOpId = 0;

    (*desc)->mDescA       = descA;
    (*desc)->mModeA       = std::vector<int32_t>(modeA, modeA + descA->mLengths.size());
    (*desc)->mOpA         = opA;
    (*desc)->mDescB       = nullptr;
    (*desc)->mModeB       = {};
    (*desc)->mOpB         = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescC       = descC;
    (*desc)->mModeC       = std::vector<int32_t>(modeC, modeC + descC->mLengths.size());
    (*desc)->mOpC         = opC;
    (*desc)->mDescD       = descD;
    (*desc)->mModeD       = std::vector<int32_t>(modeD, modeD + descD->mLengths.size());
    (*desc)->mOpD         = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescE       = nullptr;
    (*desc)->mModeE       = {};
    (*desc)->mOpAC        = opAC;
    (*desc)->mOpABC       = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescCompute = descCompute;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorCreateElementwiseTrinary(const hiptensorHandle_t            handle,
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
                                                    hiptensorOperator_t                opAB,
                                                    hiptensorOperator_t                opABC,
                                                    const hiptensorComputeDescriptor_t descCompute)
{
    *desc                  = new hiptensorOperationDescriptor();
    (*desc)->mTag          = 0u;
    (*desc)->mScalarType   = *hiptensor::convertToHipTensorDataType(descCompute);
    (*desc)->mFlops        = 0.0f;
    (*desc)->mMovedBytes   = 0.0f;
    (*desc)->mPaddingLeft  = 0u;
    (*desc)->mPaddingRighT = 0u;
    (*desc)->mPaddingValue = nullptr;

    (*desc)->mOperationType   = HIPTENSOR_ELEMENTWISE_TRINARY;
    (*desc)->mContractionOpId = 0;

    (*desc)->mDescA       = descA;
    (*desc)->mModeA       = std::vector<int32_t>(modeA, modeA + descA->mLengths.size());
    (*desc)->mOpA         = opA;
    (*desc)->mDescB       = descB;
    (*desc)->mModeB       = std::vector<int32_t>(modeB, modeB + descB->mLengths.size());
    (*desc)->mOpB         = opB;
    (*desc)->mDescC       = descC;
    (*desc)->mModeC       = std::vector<int32_t>(modeC, modeC + descC->mLengths.size());
    (*desc)->mOpC         = opC;
    (*desc)->mDescD       = descD;
    (*desc)->mModeD       = std::vector<int32_t>(modeD, modeD + descD->mLengths.size());
    (*desc)->mOpD         = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescE       = nullptr;
    (*desc)->mModeE       = {};
    (*desc)->mOpAC        = opAB;
    (*desc)->mOpABC       = opABC;
    (*desc)->mDescCompute = descCompute;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorCreateReduction(const hiptensorHandle_t            handle,
                                           hiptensorOperationDescriptor_t*    desc,
                                           const hiptensorTensorDescriptor_t  descA,
                                           const int32_t                      modeA[],
                                           hiptensorOperator_t                opA,
                                           const hiptensorTensorDescriptor_t  descC,
                                           const int32_t                      modeC[],
                                           hiptensorOperator_t                opC,
                                           const hiptensorTensorDescriptor_t  descD,
                                           const int32_t                      modeD[],
                                           hiptensorOperator_t                opReduce,
                                           const hiptensorComputeDescriptor_t descCompute)
{
    *desc                  = new hiptensorOperationDescriptor();
    (*desc)->mTag          = 0u;
    (*desc)->mScalarType   = *hiptensor::convertToHipTensorDataType(descCompute);
    (*desc)->mFlops        = 0.0f;
    (*desc)->mMovedBytes   = 0.0f;
    (*desc)->mPaddingLeft  = 0u;
    (*desc)->mPaddingRighT = 0u;
    (*desc)->mPaddingValue = nullptr;

    (*desc)->mOperationType   = HIPTENSOR_REDUCTION;
    (*desc)->mContractionOpId = 0;

    (*desc)->mDescA       = descA;
    (*desc)->mModeA       = std::vector<int32_t>(modeA, modeA + descA->mLengths.size());
    (*desc)->mOpA         = opA;
    (*desc)->mDescB       = nullptr;
    (*desc)->mModeB       = {};
    (*desc)->mOpB         = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescC       = descC;
    (*desc)->mModeC       = std::vector<int32_t>(modeC, modeC + descC->mLengths.size());
    (*desc)->mOpC         = opC;
    (*desc)->mDescD       = descD;
    (*desc)->mModeD       = std::vector<int32_t>(modeD, modeD + descD->mLengths.size());
    (*desc)->mOpD         = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescE       = nullptr;
    (*desc)->mModeE       = {};
    (*desc)->mOpAC        = opReduce;
    (*desc)->mOpABC       = HIPTENSOR_OP_IDENTITY;
    (*desc)->mDescCompute = descCompute;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t contractionCreatePlanPreference(const hiptensorHandle_t   handle,
                                                  hiptensorPlanPreference_t pref,
                                                  hiptensorAlgo_t           algo,
                                                  hiptensorJitMode_t        jitMode);

hiptensorStatus_t hiptensorCreatePlanPreference(const hiptensorHandle_t    handle,
                                                hiptensorPlanPreference_t* pref,
                                                hiptensorAlgo_t            algo,
                                                hiptensorJitMode_t         jitMode)
{
    *pref = new hiptensorPlanPreference();

    (*pref)->mAutotuneMode     = HIPTENSOR_AUTOTUNE_MODE_NONE;
    (*pref)->mCacheMode        = HIPTENSOR_CACHE_MODE_PEDANTIC;
    (*pref)->mIncrementalCount = 0;
    (*pref)->mKernelRank       = 0;
    (*pref)->mJit              = jitMode;

    (*pref)->mSelectionAlgorithm = algo;
    (*pref)->mSolution           = nullptr;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorDestroyPlanPreference(hiptensorPlanPreference_t pref)
{
    delete pref;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorPlanPreferenceSetAttribute(const hiptensorHandle_t            handle,
                                                      hiptensorPlanPreference_t          pref,
                                                      hiptensorPlanPreferenceAttribute_t attr,
                                                      const void*                        buf,
                                                      size_t sizeInBytes)
{
    hiptensorStatus_t retStatus = HIPTENSOR_STATUS_SUCCESS;
    switch(attr)
    {
    case HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE:
        std::memcpy(&pref->mAutotuneMode, buf, sizeInBytes);
        break;
    case HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE:
        std::memcpy(&pref->mCacheMode, buf, sizeInBytes);
        break;
    case HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT:
        std::memcpy(&pref->mIncrementalCount, buf, sizeInBytes);
        break;
    case HIPTENSOR_PLAN_PREFERENCE_ALGO:
        std::memcpy(&pref->mSelectionAlgorithm, buf, sizeInBytes);
        break;
    case HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK:
        std::memcpy(&pref->mKernelRank, buf, sizeInBytes);
        break;
    case HIPTENSOR_PLAN_PREFERENCE_JIT:
        std::memcpy(&pref->mJit, buf, sizeInBytes);
        break;
    default:
        retStatus = HIPTENSOR_STATUS_NOT_SUPPORTED;
        break;
    }

    return retStatus;
}

hiptensorStatus_t hiptensorPlanGetAttribute(const hiptensorHandle_t  handle,
                                            const hiptensorPlan_t    plan,
                                            hiptensorPlanAttribute_t attr,
                                            void*                    buf,
                                            size_t                   sizeInBytes)
{
    hiptensorStatus_t retStatus = HIPTENSOR_STATUS_SUCCESS;
    switch(attr)
    {
    case HIPTENSOR_PLAN_REQUIRED_WORKSPACE:
        std::memcpy(buf, &plan->mRequiredWorkspace, sizeInBytes);
        break;
    default:
        retStatus = HIPTENSOR_STATUS_NOT_SUPPORTED;
        break;
    }

    return retStatus;
}

hiptensorStatus_t contractionGetWorkspaceSize(const hiptensorHandle_t              handle,
                                              const hiptensorOperationDescriptor_t desc,
                                              const hiptensorPlanPreference_t      planPref,
                                              const hiptensorWorksizePreference_t  workspacePref,
                                              uint64_t* workspaceSizeEstimate);
hiptensorStatus_t contractionTrinaryGetWorkspaceSize(const hiptensorHandle_t              handle,
                                                     const hiptensorOperationDescriptor_t desc,
                                                     const hiptensorPlanPreference_t      planPref,
                                                     const hiptensorWorksizePreference_t  workspacePref,
                                                     uint64_t* workspaceSizeEstimate);
hiptensorStatus_t hiptensorEstimateWorkspaceSize(const hiptensorHandle_t              handle,
                                                 const hiptensorOperationDescriptor_t desc,
                                                 const hiptensorPlanPreference_t      planPref,
                                                 const hiptensorWorksizePreference_t  workspacePref,
                                                 uint64_t* workspaceSizeEstimate)
{
    if(desc->mOperationType == HIPTENSOR_CONTRACTION)
    {
        return contractionGetWorkspaceSize(
            handle, desc, planPref, workspacePref, workspaceSizeEstimate);
    }
    if(desc->mOperationType == HIPTENSOR_CONTRACTION_TRINARY)
    {
        return contractionTrinaryGetWorkspaceSize(
            handle, desc, planPref, workspacePref, workspaceSizeEstimate);
    }
    *workspaceSizeEstimate = 0u;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t contractionInitPlan(const hiptensorHandle_t              handle,
                                      hiptensorPlan_t                      plan,
                                      const hiptensorOperationDescriptor_t desc,
                                      const hiptensorPlanPreference_t      pref,
                                      uint64_t                             workspaceSizeLimit);
hiptensorStatus_t contractionTrinaryInitPlan(const hiptensorHandle_t              handle,
                                             hiptensorPlan_t                      plan,
                                             const hiptensorOperationDescriptor_t desc,
                                             const hiptensorPlanPreference_t      pref,
                                             uint64_t                             workspaceSizeLimit);
hiptensorStatus_t hiptensorCreatePlan(const hiptensorHandle_t              handle,
                                      hiptensorPlan_t*                     plan,
                                      const hiptensorOperationDescriptor_t desc,
                                      const hiptensorPlanPreference_t      pref,
                                      uint64_t                             workspaceSizeLimit)
{
    (*plan)                     = new hiptensorPlan();
    hiptensorPlan_t             newPlan = *plan;
    newPlan->mRequiredWorkspace = workspaceSizeLimit;

    // Deep-copy the tensor descriptors referenced by the operation descriptor
    if(desc->mDescA)
    {
        newPlan->mOwnedDescA = std::make_unique<hiptensorTensorDescriptor>(*desc->mDescA);
    }
    if(desc->mDescB)
    {
        newPlan->mOwnedDescB = std::make_unique<hiptensorTensorDescriptor>(*desc->mDescB);
    }
    if(desc->mDescC)
    {
        newPlan->mOwnedDescC = std::make_unique<hiptensorTensorDescriptor>(*desc->mDescC);
    }
    if(desc->mDescD)
    {
        newPlan->mOwnedDescD = std::make_unique<hiptensorTensorDescriptor>(*desc->mDescD);
    }
    if(desc->mDescE)
    {
        newPlan->mOwnedDescE = std::make_unique<hiptensorTensorDescriptor>(*desc->mDescE);
    }

    newPlan->mOwnedOpDesc  = std::make_unique<hiptensorOperationDescriptor>(*desc);
    newPlan->mOwnedOpDesc->mDescA = newPlan->mOwnedDescA.get();
    newPlan->mOwnedOpDesc->mDescB = newPlan->mOwnedDescB.get();
    newPlan->mOwnedOpDesc->mDescC = newPlan->mOwnedDescC.get();
    newPlan->mOwnedOpDesc->mDescD = newPlan->mOwnedDescD.get();
    newPlan->mOwnedOpDesc->mDescE = newPlan->mOwnedDescE.get();

    newPlan->mOwnedPref = std::make_unique<hiptensorPlanPreference>(*pref);

    newPlan->mOpDesc = newPlan->mOwnedOpDesc.get();
    newPlan->mPref   = newPlan->mOwnedPref.get();

    if(desc->mOperationType == HIPTENSOR_CONTRACTION)
    {
        return contractionInitPlan(handle, newPlan, newPlan->mOpDesc, newPlan->mPref, workspaceSizeLimit);
    }
    if(desc->mOperationType == HIPTENSOR_CONTRACTION_TRINARY)
    {
        return contractionTrinaryInitPlan(handle, newPlan, newPlan->mOpDesc, newPlan->mPref, workspaceSizeLimit);
    }
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorDestroyPlan(hiptensorPlan_t plan)
{
    delete plan;
    return HIPTENSOR_STATUS_SUCCESS;
}

const char* hiptensorGetErrorString(const hiptensorStatus_t error)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(msg, sizeof(msg), "error=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)error);
    logger->logAPITrace("hiptensorGetErrorString", msg);

    if(error == HIPTENSOR_STATUS_SUCCESS)
        return "HIPTENSOR_STATUS_SUCCESS";
    else if(error == HIPTENSOR_STATUS_NOT_INITIALIZED)
        return "HIPTENSOR_STATUS_NOT_INITIALIZED";
    else if(error == HIPTENSOR_STATUS_ALLOC_FAILED)
        return "HIPTENSOR_STATUS_ALLOC_FAILED";
    else if(error == HIPTENSOR_STATUS_INVALID_VALUE)
        return "HIPTENSOR_STATUS_INVALID_VALUE";
    else if(error == HIPTENSOR_STATUS_ARCH_MISMATCH)
        return "HIPTENSOR_STATUS_ARCH_MISMATCH";
    else if(error == HIPTENSOR_STATUS_EXECUTION_FAILED)
        return "HIPTENSOR_STATUS_EXECUTION_FAILED";
    else if(error == HIPTENSOR_STATUS_INTERNAL_ERROR)
        return "HIPTENSOR_STATUS_INTERNAL_ERROR";
    else if(error == HIPTENSOR_STATUS_NOT_SUPPORTED)
        return "HIPTENSOR_STATUS_NOT_SUPPORTED";
    else if(error == HIPTENSOR_STATUS_CK_ERROR)
        return "HIPTENSOR_STATUS_CK_ERROR";
    else if(error == HIPTENSOR_STATUS_HIP_ERROR)
        return "HIPTENSOR_STATUS_HIP_ERROR";
    else if(error == HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
        return "HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    else if(error == HIPTENSOR_STATUS_INSUFFICIENT_DRIVER)
        return "HIPTENSOR_STATUS_INSUFFICIENT_DRIVER";
    else if(error == HIPTENSOR_STATUS_IO_ERROR)
        return "HIPTENSOR_STATUS_IO_ERROR";
    else
        return "HIPTENSOR_STATUS_UNKNOWN";
}

hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(msg,
             sizeof(msg),
             "callback=0x%0*llX",
             2 * (int)sizeof(void*),
             (unsigned long long)callback);
    logger->logAPITrace("hiptensorLoggerSetCallback", msg);

    // Check logger callback result
    auto loggerResult = logger->setCallback(callback);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "Error : logger set callback not successful (%s)",
                 logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetCallback", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerSetFile(FILE* file)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(msg, sizeof(msg), "file=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)file);
    logger->logAPITrace("hiptensorLoggerSetFile", msg);

    // Check logger callback result
    auto loggerResult = logger->writeToStream(file);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "Error : logger set file not successful (%s)",
                 logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetFile", msg);
        return HIPTENSOR_STATUS_IO_ERROR;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API trace
    char msg[2048];
    snprintf(msg, sizeof(msg), "logFile=%s", logFile);
    logger->logAPITrace("hiptensorLoggerOpenFile", msg);

    // Check logger open file result
    auto loggerResult = logger->openFileStream(logFile);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg, sizeof(msg), "fileName=%s (%s)", logFile, logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerOpenFile", msg);
        return HIPTENSOR_STATUS_IO_ERROR;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API trace
    char msg[128];
    snprintf(msg, sizeof(msg), "log level=0x%02X", (unsigned int)level);
    logger->logAPITrace("hiptensorLoggerSetLevel", msg);

    // Check logger level
    auto loggerResult = logger->setLogLevel(Logger::LogLevel_t(level));
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "level=0x%02X (%s)",
                 (unsigned int)level,
                 logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetLevel", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API trace
    char msg[128];
    snprintf(msg, sizeof(msg), "mask=0x%02X", (unsigned int)mask);
    logger->logAPITrace("hiptensorLoggerSetMask", msg);

    // Check for logger error
    auto loggerResult = logger->setLogMask(mask);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "mask=0x%02X (%s)",
                 (unsigned int)mask,
                 logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetMask", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerForceDisable()
{
    // Log API trace
    auto& logger = hiptensor::Logger::instance();
    logger->logAPITrace("hiptensorLoggerForceDisable", "Logging Disabled");
    logger->disable();
    return HIPTENSOR_STATUS_SUCCESS;
}

int hiptensorGetHiprtVersion()
{
    // Log API trace
    auto& logger = hiptensor::Logger::instance();
    logger->logAPITrace("hiptensorGetHiprtVersion", "");

    int  version   = 0;
    auto hipResult = hipRuntimeGetVersion(&version);
    if(hipResult != hipError_t::hipSuccess)
    {
        char msg[256];
        snprintf(msg, sizeof(msg), "Hip error: (%s)", hipGetErrorString(hipResult));
        logger->logError("hiptensorGetHiprtVersion", msg);
        return -1;
    }

    return version;
}

size_t hiptensorGetVersion()
{
    return HIPTENSOR_MAJOR_VERSION * 1e6 + HIPTENSOR_MINOR_VERSION * 1e3 + HIPTENSOR_PATCH_VERSION;
}
