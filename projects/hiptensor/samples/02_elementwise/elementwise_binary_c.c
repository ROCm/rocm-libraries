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
typedef float DDataType;
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

int elementwiseBinarySample(const void*                        alpha,
                            const void*                        gamma,
                            const hiptensorDataType_t          typeA,
                            const hiptensorDataType_t          typeC,
                            const hiptensorDataType_t          typeD,
                            const hiptensorComputeDescriptor_t typeCompute)
{
    /**********************
     * Computing: D_{c,w,h} = alpha * A_{w,h,c}  + gamma * C_{w,h,c}
     **********************/

    // Modes
    int modeA[4] = {'w', 'h', 'c'};
    int modeC[4] = {'w', 'h', 'c'};
    int modeD[4] = {'c', 'w', 'h'};

    // Number of dimensions
    int nmode = 3;

    // Initializing length corresponging to each dimension
    int64_t extentA[4] = {512, 512, 512};
    int64_t extentC[4] = {512, 512, 512};
    int64_t extentD[4] = {512, 512, 512};

    /**********************
     * Allocating data
     **********************/
    printf("Initializing host data...\n");

    size_t elementsA = 1;
    size_t elementsC = 1;
    size_t elementsD = 1;
    for(int i = 0; i < nmode; i++)
    {
        elementsA *= extentA[i];
        elementsC *= extentC[i];
        elementsD *= extentD[i];
    }

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeC = sizeof(CDataType) * elementsC;
    size_t sizeD = sizeof(DDataType) * elementsD;

    void *A_d, *C_d, *D_d;
    CHECK_HIP_ERROR(hipMalloc((void**)&A_d, sizeA));
    CHECK_HIP_ERROR(hipMalloc((void**)&C_d, sizeC));
    CHECK_HIP_ERROR(hipMalloc((void**)&D_d, sizeD));

    ADataType* A;
    CDataType* C;
    DDataType* D;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA, hipDeviceScheduleAuto));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeC, hipDeviceScheduleAuto));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&D, sizeD, hipDeviceScheduleAuto));

    /*******************
     * Initialize data
     *******************/

    for(size_t i = 0; i < elementsA; i++)
    {
        A[i] = (ADataType)i;
    }
    for(size_t i = 0; i < elementsC; i++)
    {
        C[i] = (CDataType)(i % 41);
    }

    /********************************************
     * Transfer the Host Tensor to Device Memory
     ********************************************/
    printf("Initializing device data...\n");

    CHECK_HIP_ERROR(hipMemcpy(A_d, A, sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(C_d, C, sizeC, hipMemcpyHostToDevice));

    /*************************
     * hipTensor
     *************************/
    hiptensorHandle_t handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /********************************************
     * Initialize tensors with the input lengths
     ********************************************/
    hiptensorTensorDescriptor_t descA = NULL;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
        handle, &descA, nmode, extentA, NULL /* stride */, typeA, 0 /* alignmentRequirement */));

    hiptensorTensorDescriptor_t descC = NULL;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
        handle, &descC, nmode, extentC, NULL /* stride */, typeC, 0 /* alignmentRequirement */));

    hiptensorTensorDescriptor_t descD = NULL;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
        handle, &descD, nmode, extentD, NULL /* stride */, typeD, 0 /* alignmentRequirement */));

    /***************************************
     * Create Elementwise Binary Descriptor
     ***************************************/
    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(
        hiptensorCreateElementwiseBinary(handle,
                                         &desc,
                                         descA,
                                         modeA,
                                         /* unary operator A  */ HIPTENSOR_OP_IDENTITY,
                                         descC,
                                         modeC,
                                         /* unary operator C  */ HIPTENSOR_OP_IDENTITY,
                                         descD,
                                         modeD,
                                         /* unary operator AC */ HIPTENSOR_OP_ADD,
                                         typeCompute));

    /***************************
     * Set the algorithm to use
     ***************************/
    const hiptensorAlgo_t algo = HIPTENSOR_ALGO_DEFAULT;

    hiptensorPlanPreference_t planPref;
    CHECK_HIPTENSOR_ERROR(
        hiptensorCreatePlanPreference(handle, &planPref, algo, HIPTENSOR_JIT_MODE_NONE));

    /**************************
     * Create Plan
     **************************/
    printf("Initializing plan...\n");

    hiptensorPlan_t plan;
    CHECK_HIPTENSOR_ERROR(
        hiptensorCreatePlan(handle, &plan, desc, planPref, 0 /* workspaceSizeLimit */));

    /**********************
     * Run
     **********************/
    printf("Launching elementwise binary kernel...\n");

    CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryExecute(
        handle, plan, (void*)&alpha, A_d, (void*)&gamma, C_d, D_d, 0 /* stream */));

    printf("Elementwise binary kernel finished...\n");

    CHECK_HIP_ERROR(hipMemcpy(D, D_d, sizeD, hipMemcpyDeviceToHost));

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
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descD));

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(C);
    HIPTENSOR_FREE_HOST(D);
    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(D_d);

    printf("Finished!\n");

    return 0;
}

int main()
{
    const hiptensorDataType_t          typeA       = HIPTENSOR_R_32F;
    const hiptensorDataType_t          typeC       = HIPTENSOR_R_32F;
    const hiptensorDataType_t          typeD       = HIPTENSOR_R_32F;
    const hiptensorComputeDescriptor_t typeCompute = HIPTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute gamma = (floatTypeCompute)2.0f;

    return elementwiseBinarySample(&alpha, &gamma, typeA, typeC, typeD, typeCompute);
}
