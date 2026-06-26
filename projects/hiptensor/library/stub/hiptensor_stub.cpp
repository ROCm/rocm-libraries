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

// Stub implementation of the hipTensor public API.

#include <hiptensor/hiptensor.h>

hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorHandleResizePlanCache(hiptensorHandle_t, const uint32_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorHandleWritePlanCacheToFile(const hiptensorHandle_t, const char[])
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t
    hiptensorHandleReadPlanCacheFromFile(hiptensorHandle_t, const char[], uint32_t*)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorWriteKernelCacheToFile(const hiptensorHandle_t, const char[])
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorReadKernelCacheFromFile(hiptensorHandle_t, const char[])
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreateTensorDescriptor(const hiptensorHandle_t,
                                                  hiptensorTensorDescriptor_t*,
                                                  const uint32_t,
                                                  const int64_t[],
                                                  const int64_t[],
                                                  hiptensorDataType_t,
                                                  uint32_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorDestroyTensorDescriptor(hiptensorTensorDescriptor_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreateContraction(const hiptensorHandle_t,
                                             hiptensorOperationDescriptor_t*,
                                             const hiptensorTensorDescriptor_t,
                                             const int32_t[],
                                             hiptensorOperator_t,
                                             const hiptensorTensorDescriptor_t,
                                             const int32_t[],
                                             hiptensorOperator_t,
                                             const hiptensorTensorDescriptor_t,
                                             const int32_t[],
                                             hiptensorOperator_t,
                                             const hiptensorTensorDescriptor_t,
                                             const int32_t[],
                                             const hiptensorComputeDescriptor_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorDestroyOperationDescriptor(hiptensorOperationDescriptor_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t
    hiptensorOperationDescriptorSetAttribute(const hiptensorHandle_t,
                                             hiptensorOperationDescriptor_t,
                                             hiptensorOperationDescriptorAttribute_t,
                                             const void*,
                                             size_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t
    hiptensorOperationDescriptorGetAttribute(const hiptensorHandle_t,
                                             hiptensorOperationDescriptor_t,
                                             hiptensorOperationDescriptorAttribute_t,
                                             void*,
                                             size_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreatePlanPreference(const hiptensorHandle_t,
                                                hiptensorPlanPreference_t*,
                                                hiptensorAlgo_t,
                                                hiptensorJitMode_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorDestroyPlanPreference(hiptensorPlanPreference_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorPlanPreferenceSetAttribute(const hiptensorHandle_t,
                                                      hiptensorPlanPreference_t,
                                                      hiptensorPlanPreferenceAttribute_t,
                                                      const void*,
                                                      size_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorPlanGetAttribute(const hiptensorHandle_t,
                                            const hiptensorPlan_t,
                                            hiptensorPlanAttribute_t,
                                            void*,
                                            size_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorEstimateWorkspaceSize(const hiptensorHandle_t,
                                                 const hiptensorOperationDescriptor_t,
                                                 const hiptensorPlanPreference_t,
                                                 const hiptensorWorksizePreference_t,
                                                 uint64_t*)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreatePermutation(const hiptensorHandle_t,
                                             hiptensorOperationDescriptor_t*,
                                             const hiptensorTensorDescriptor_t,
                                             const int32_t[],
                                             hiptensorOperator_t,
                                             const hiptensorTensorDescriptor_t,
                                             const int32_t[],
                                             const hiptensorComputeDescriptor_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreatePlan(const hiptensorHandle_t,
                                      hiptensorPlan_t*,
                                      const hiptensorOperationDescriptor_t,
                                      const hiptensorPlanPreference_t,
                                      uint64_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorDestroyPlan(hiptensorPlan_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorContract(const hiptensorHandle_t,
                                    const hiptensorPlan_t,
                                    const void*,
                                    const void*,
                                    const void*,
                                    const void*,
                                    const void*,
                                    void*,
                                    void*,
                                    uint64_t,
                                    hipStream_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreateContractionTrinary(const hiptensorHandle_t,
                                                    hiptensorOperationDescriptor_t*,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    hiptensorOperator_t,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    hiptensorOperator_t,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    hiptensorOperator_t,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    hiptensorOperator_t,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    const hiptensorComputeDescriptor_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorContractTrinary(const hiptensorHandle_t,
                                           const hiptensorPlan_t,
                                           const void*,
                                           const void*,
                                           const void*,
                                           const void*,
                                           const void*,
                                           const void*,
                                           void*,
                                           void*,
                                           uint64_t,
                                           hipStream_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

const char* hiptensorGetErrorString(const hiptensorStatus_t error)
{
    switch(error)
    {
    case HIPTENSOR_STATUS_SUCCESS:
        return "HIPTENSOR_STATUS_SUCCESS";
    case HIPTENSOR_STATUS_NOT_INITIALIZED:
        return "HIPTENSOR_STATUS_NOT_INITIALIZED";
    case HIPTENSOR_STATUS_ALLOC_FAILED:
        return "HIPTENSOR_STATUS_ALLOC_FAILED";
    case HIPTENSOR_STATUS_INVALID_VALUE:
        return "HIPTENSOR_STATUS_INVALID_VALUE";
    case HIPTENSOR_STATUS_ARCH_MISMATCH:
        return "HIPTENSOR_STATUS_ARCH_MISMATCH";
    case HIPTENSOR_STATUS_EXECUTION_FAILED:
        return "HIPTENSOR_STATUS_EXECUTION_FAILED";
    case HIPTENSOR_STATUS_INTERNAL_ERROR:
        return "HIPTENSOR_STATUS_INTERNAL_ERROR";
    case HIPTENSOR_STATUS_NOT_SUPPORTED:
        return "HIPTENSOR_STATUS_NOT_SUPPORTED";
    case HIPTENSOR_STATUS_CK_ERROR:
        return "HIPTENSOR_STATUS_CK_ERROR";
    case HIPTENSOR_STATUS_HIP_ERROR:
        return "HIPTENSOR_STATUS_HIP_ERROR";
    case HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE:
        return "HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    case HIPTENSOR_STATUS_INSUFFICIENT_DRIVER:
        return "HIPTENSOR_STATUS_INSUFFICIENT_DRIVER";
    case HIPTENSOR_STATUS_IO_ERROR:
        return "HIPTENSOR_STATUS_IO_ERROR";
    default:
        return "HIPTENSOR_STATUS_UNKNOWN";
    }
}

hiptensorStatus_t hiptensorPermute(const hiptensorHandle_t,
                                   const hiptensorPlan_t,
                                   const void*,
                                   const void*,
                                   void*,
                                   const hipStream_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreateElementwiseBinary(const hiptensorHandle_t,
                                                   hiptensorOperationDescriptor_t*,
                                                   const hiptensorTensorDescriptor_t,
                                                   const int32_t[],
                                                   hiptensorOperator_t,
                                                   const hiptensorTensorDescriptor_t,
                                                   const int32_t[],
                                                   hiptensorOperator_t,
                                                   const hiptensorTensorDescriptor_t,
                                                   const int32_t[],
                                                   hiptensorOperator_t,
                                                   const hiptensorComputeDescriptor_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorElementwiseBinaryExecute(const hiptensorHandle_t,
                                                    const hiptensorPlan_t,
                                                    const void*,
                                                    const void*,
                                                    const void*,
                                                    const void*,
                                                    void*,
                                                    hipStream_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreateElementwiseTrinary(const hiptensorHandle_t,
                                                    hiptensorOperationDescriptor_t*,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    hiptensorOperator_t,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    hiptensorOperator_t,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    hiptensorOperator_t,
                                                    const hiptensorTensorDescriptor_t,
                                                    const int32_t[],
                                                    hiptensorOperator_t,
                                                    hiptensorOperator_t,
                                                    const hiptensorComputeDescriptor_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorElementwiseTrinaryExecute(const hiptensorHandle_t,
                                                     const hiptensorPlan_t,
                                                     const void*,
                                                     const void*,
                                                     const void*,
                                                     const void*,
                                                     const void*,
                                                     const void*,
                                                     void*,
                                                     hipStream_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorCreateReduction(const hiptensorHandle_t,
                                           hiptensorOperationDescriptor_t*,
                                           const hiptensorTensorDescriptor_t,
                                           const int32_t[],
                                           hiptensorOperator_t,
                                           const hiptensorTensorDescriptor_t,
                                           const int32_t[],
                                           hiptensorOperator_t,
                                           const hiptensorTensorDescriptor_t,
                                           const int32_t[],
                                           hiptensorOperator_t,
                                           const hiptensorComputeDescriptor_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorReduce(const hiptensorHandle_t,
                                  const hiptensorPlan_t,
                                  const void*,
                                  const void*,
                                  const void*,
                                  const void*,
                                  void*,
                                  void*,
                                  uint64_t,
                                  hipStream_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorLoggerSetFile(FILE*)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorLoggerOpenFile(const char*)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorLoggerSetMask(int32_t)
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

hiptensorStatus_t hiptensorLoggerForceDisable()
{
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}

int hiptensorGetHiprtVersion()
{
    return -1;
}

size_t hiptensorGetVersion()
{
    return HIPTENSOR_MAJOR_VERSION * 1e6 + HIPTENSOR_MINOR_VERSION * 1e3 + HIPTENSOR_PATCH_VERSION;
}
