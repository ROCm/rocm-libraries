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
typedef float BDataType;
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

int bilinearContractionSample(const void*                        alpha,
                              const void*                        beta,
                              const hiptensorDataType_t          typeA,
                              const hiptensorDataType_t          typeB,
                              const hiptensorDataType_t          typeC,
                              const hiptensorComputeDescriptor_t typeCompute)
{
    /**********************
     * Computing: C_{m,n,u,v} = alpha * A_{m,n,h,k} B_{u,v,h,k} + beta * C_{m,n,u,v}
     **********************/

    // Modes
    int modeC[4] = {'m', 'n', 'u', 'v'};
    int modeA[4] = {'m', 'n', 'h', 'k'};
    int modeB[4] = {'u', 'v', 'h', 'k'};

    // Number of dimensions
    int nmode = 4;

    // Initializing length corresponging to each dimension
    int64_t c_ms_ns_lengths[4] = {4, 3, 4, 3};
    int64_t a_ms_ks_lengths[4] = {4, 3, 6, 5};
    int64_t b_ns_ks_lengths[4] = {4, 3, 6, 5};

    /**********************
     * Allocating data
     **********************/
    printf("Initializing host data...\n");

    size_t elementsA = 1;
    size_t elementsB = 1;
    size_t elementsC = 1;
    for(int i = 0; i < nmode; i++)
    {
        elementsA *= a_ms_ks_lengths[i];
        elementsB *= b_ns_ks_lengths[i];
        elementsC *= c_ms_ns_lengths[i];
    }

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeC = sizeof(CDataType) * elementsC;

    ADataType* A = NULL;
    BDataType* B = NULL;
    CDataType* C = NULL;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA, hipDeviceScheduleAuto));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&B, sizeB, hipDeviceScheduleAuto));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeC, hipDeviceScheduleAuto));

    void *A_d, *B_d, *C_d;

    CHECK_HIP_ERROR(hipMalloc((void**)(&A_d), sizeA));
    CHECK_HIP_ERROR(hipMalloc((void**)(&B_d), sizeB));
    CHECK_HIP_ERROR(hipMalloc((void**)(&C_d), sizeC));

    /*******************
     * Initialize data
     *******************/
    for(int64_t i = 0; i < elementsA; i++)
    {
        A[i] = (ADataType)((ADataType)(i) / 100);
    }

    for(int64_t i = 0; i < elementsB; i++)
    {
        B[i] = (BDataType)((BDataType)(i) / 100);
    }

    for(int64_t i = 0; i < elementsC; i++)
    {
        C[i] = (CDataType)((CDataType)(i) / 100);
    }

    /********************************************
     * Transfer the Host Tensor to Device Memory
     ********************************************/
    printf("Initializing device data...\n");

    CHECK_HIP_ERROR(hipMemcpy(A_d, (const void*)(A), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_d, (const void*)(B), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(C_d, (const void*)(C), sizeC, hipMemcpyHostToDevice));

    /************************************************
     * Retrieve the memory alignment for each tensor
     ************************************************/
    uint32_t          alignmentRequirement = 1;
    hiptensorHandle_t handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /********************************************
     * Initialize tensors with the input lengths
     ********************************************/
    hiptensorTensorDescriptor_t a_ms_ks = NULL;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &a_ms_ks,
                                                          nmode,
                                                          a_ms_ks_lengths,
                                                          NULL, /*stride*/
                                                          typeA,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t b_ns_ks = NULL;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &b_ns_ks,
                                                          nmode,
                                                          b_ns_ks_lengths,
                                                          NULL, /*stride*/
                                                          typeB,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t c_ms_ns = NULL;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &c_ms_ns,
                                                          nmode,
                                                          c_ms_ns_lengths,
                                                          NULL, /*stride*/
                                                          typeC,
                                                          alignmentRequirement));

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateContraction(handle,
                                                     &desc,
                                                     a_ms_ks,
                                                     modeA,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     b_ns_ks,
                                                     modeB,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     c_ms_ns,
                                                     modeC,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     c_ms_ns,
                                                     modeC,
                                                     typeCompute));

    /**************************
     * Set the algorithm to use
     ***************************/
    hiptensorPlanPreference_t planPref;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
        handle, &planPref, HIPTENSOR_ALGO_ACTOR_CRITIC, HIPTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace
     **********************/

    uint64_t worksize = 0;
    CHECK_HIPTENSOR_ERROR(hiptensorEstimateWorkspaceSize(
        handle, desc, planPref, HIPTENSOR_WORKSPACE_DEFAULT, &worksize));

    /**************************
     * Create Contraction Plan
     **************************/
    printf("Initializing contraction plan...\n");

    hiptensorPlan_t plan;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

    void* workspace = NULL;
    if(worksize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc((void**)(&workspace), worksize));
    }

    printf("Launching contraction kernel...\n");

    CHECK_HIPTENSOR_ERROR(hiptensorContract(
        handle, plan, alpha, A_d, B_d, beta, C_d, C_d, workspace, worksize, 0 /* stream */));

    printf("Contraction kernel finished...\n");

    CHECK_HIP_ERROR(hipMemcpy(C, C_d, sizeC, hipMemcpyDeviceToHost));

    /**************************
     * Destroying resources
     **************************/
    printf("Destroying resources...\n");

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));
    if(a_ms_ks)
    {
        hiptensorDestroyTensorDescriptor(a_ms_ks);
        a_ms_ks = NULL;
    }
    if(b_ns_ks)
    {
        hiptensorDestroyTensorDescriptor(b_ns_ks);
        b_ns_ks = NULL;
    }
    if(c_ms_ns)
    {
        hiptensorDestroyTensorDescriptor(c_ms_ns);
        c_ms_ns = NULL;
    }

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(C);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(workspace);

    printf("Finished!\n");

    return 0;
}

int main(int argc, char* argv[])
{
    const hiptensorDataType_t          typeA       = HIPTENSOR_R_32F;
    const hiptensorDataType_t          typeB       = HIPTENSOR_R_32F;
    const hiptensorDataType_t          typeC       = HIPTENSOR_R_32F;
    const hiptensorComputeDescriptor_t typeCompute = HIPTENSOR_COMPUTE_DESC_16F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)1.0f;

    return bilinearContractionSample(&alpha, &beta, typeA, typeB, typeC, typeCompute);
}
