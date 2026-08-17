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
#include <stdio.h>

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>

// This example assumes the device supports F32
typedef float ADataType;
typedef float CDataType;
typedef float floatTypeCompute;

#define HIPTENSOR_FREE_DEVICE(ptr)     \
    if(ptr != NULL)                    \
    {                                  \
        CHECK_HIP_ERROR(hipFree(ptr)); \
    }

#define HIPTENSOR_FREE_HOST(ptr)           \
    if(ptr != NULL)                        \
    {                                      \
        CHECK_HIP_ERROR(hipHostFree(ptr)); \
    }

int reductionSample(const void*                        alpha,
                    const void*                        beta,
                    const hiptensorDataType_t          typeA,
                    const hiptensorDataType_t          typeC,
                    const hiptensorComputeDescriptor_t typeCompute,
                    const hiptensorComputeDescriptor_t descCompute)
{
    /**********************
     * Computing: C_{k,v} = alpha * A_{m,h,k,v} + beta * C_{k,v}
     **********************/

    // Modes
    int32_t modeA[4] = {'m', 'h', 'k', 'v'};
    int32_t modeC[2] = {'k', 'v'};

    // Number of dimensions
    int32_t nmodeA = 4;
    int32_t nmodeC = 2;

    // Initializing length corresponging to each dimension
    int64_t extentC[2] = {58, 64};
    int64_t extentA[4] = {32, 46, 58, 64};

    /**********************
     * Allocating data
     **********************/
    printf("Initializing host data...\n");

    size_t elementsA = 1;
    for(int i = 0; i < nmodeA; i++)
    {
        elementsA *= extentA[i];
    }
    size_t elementsC = 1;
    for(int i = 0; i < nmodeC; i++)
    {
        elementsC *= extentC[i];
    }

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeC = sizeof(CDataType) * elementsC;

    void *A_d, *C_d;
    CHECK_HIP_ERROR(hipMalloc((void**)&A_d, sizeA));
    CHECK_HIP_ERROR(hipMalloc((void**)&C_d, sizeC));

    ADataType *A, *C;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA, hipDeviceScheduleAuto));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeC, hipDeviceScheduleAuto));

    /*******************
     * Initialize data
     *******************/

    for(size_t i = 0; i < elementsA; i++)
    {
        A[i] = (ADataType)i;
    }

    for(size_t i = 0; i < elementsC; i++)
    {
        C[i] = (CDataType)(i & 1);
    }

    /********************************************
     * Transfer the Host Tensor to Device Memory
     ********************************************/
    printf("Initializing device data...\n");

    CHECK_HIP_ERROR(hipMemcpy(A_d, A, sizeA, hipMemcpyDefault));
    CHECK_HIP_ERROR(hipMemcpy(C_d, C, sizeC, hipMemcpyDefault));

    /*************************
     * hipTensor
     *************************/
    hiptensorHandle_t handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /****************************
     * Create Tensor Descriptors
     ****************************/
    hiptensorTensorDescriptor_t descA = NULL;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
        handle, &descA, nmodeA, extentA, NULL /* stride */, typeA, 0 /* alignmentRequirement */));

    hiptensorTensorDescriptor_t descC = NULL;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
        handle, &descC, nmodeC, extentC, NULL /* stride */, typeC, 0 /* alignmentRequirement */));

    /*******************************
     * Create Reduction Descriptor
     *******************************/
    const hiptensorOperator_t opReduce = HIPTENSOR_OP_ADD;

    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateReduction(handle,
                                                   &desc,
                                                   descA,
                                                   modeA,
                                                   HIPTENSOR_OP_IDENTITY,
                                                   descC,
                                                   modeC,
                                                   HIPTENSOR_OP_IDENTITY,
                                                   descC,
                                                   modeC,
                                                   opReduce,
                                                   descCompute));

    /***************************
     * Set the algorithm to use
     ***************************/
    const hiptensorAlgo_t algo = HIPTENSOR_ALGO_DEFAULT;

    hiptensorPlanPreference_t planPref;
    CHECK_HIPTENSOR_ERROR(
        hiptensorCreatePlanPreference(handle, &planPref, algo, HIPTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/
    uint64_t worksize = 0;

    const hiptensorWorksizePreference_t workspacePref = HIPTENSOR_WORKSPACE_DEFAULT;
    CHECK_HIPTENSOR_ERROR(
        hiptensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, &worksize));
    void* work = NULL;
    if(worksize > 0)
    {
        if(hipSuccess != hipMalloc(&work, worksize))
        {
            work     = NULL;
            worksize = 0;
        }
    }

    /**************************
     * Create Plan
     **************************/
    printf("Initializing plan...\n");

    hiptensorPlan_t plan;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

    /**********************
     * Run
     **********************/
    printf("Launching reduction kernel...\n");

    CHECK_HIPTENSOR_ERROR(hiptensorReduce(
        handle, plan, (const void*)&alpha, A_d, (const void*)&beta, C_d, C_d, work, worksize, 0));

    printf("Reduction kernel finished...\n");

    CHECK_HIP_ERROR(hipMemcpy(C, C_d, sizeC, hipMemcpyDeviceToHost));

    /**************************
     * Destroying resources
     **************************/
    printf("Destroying resources...\n");

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descA));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descC));

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(C);
    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(work);

    printf("Finished!\n");

    return 0;
}

int main()
{
    hiptensorDataType_t                typeA       = HIPTENSOR_R_32F;
    hiptensorDataType_t                typeC       = HIPTENSOR_R_32F;
    hiptensorComputeDescriptor_t       typeCompute = HIPTENSOR_COMPUTE_DESC_32F;
    const hiptensorComputeDescriptor_t descCompute = HIPTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    return reductionSample(&alpha, &beta, typeA, typeC, typeCompute, descCompute);
}
