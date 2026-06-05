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
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <hiptensor/hiptensor.h>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include "common.hpp"

int main()
{
    if(!isF32Supported())
    {
        std::cout << "unsupported host device" << std::endl;
        exit(EXIT_FAILURE);
    }

    typedef float floatTypeA;
    typedef float floatTypeC;
    typedef float floatTypeD;
    typedef float floatTypeCompute;

    hiptensorDataType_t                typeA       = HIPTENSOR_R_32F;
    hiptensorDataType_t                typeC       = HIPTENSOR_R_32F;
    hiptensorDataType_t                typeD       = HIPTENSOR_R_32F;
    hiptensorComputeDescriptor_t const descCompute = HIPTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute gamma = (floatTypeCompute)2.0f;

    /**********************
     * Computing: D_{c,w,h} = alpha * A_{w,h,c}  + gamma * C_{w,h,c}
     **********************/

    std::vector<int> modeA{'w', 'h', 'c'};
    std::vector<int> modeC{'w', 'h', 'c'};
    std::vector<int> modeD{'c', 'w', 'h'};
    int              nmodeA = modeA.size();
    int              nmodeC = modeC.size();
    int              nmodeD = modeD.size();

    std::unordered_map<int, int64_t> extent;
    extent['h'] = 512;
    extent['w'] = 512;
    extent['c'] = 512;

    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentD;
    for(auto mode : modeD)
        extentD.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];
    size_t elementsD = 1;
    for(auto mode : modeD)
        elementsD *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    size_t sizeD = sizeof(floatTypeD) * elementsD;

    void *A_d, *C_d, *D_d;
    CHECK_HIP_ERROR(hipMalloc((void**)&A_d, sizeA));
    CHECK_HIP_ERROR(hipMalloc((void**)&C_d, sizeC));
    CHECK_HIP_ERROR(hipMalloc((void**)&D_d, sizeD));

    floatTypeA* A;
    floatTypeC* C;
    floatTypeD* D;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeC));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&D, sizeD));

    /*******************
     * Initialize data
     *******************/

    for(size_t i = 0; i < elementsA; i++)
    {
        A[i] = static_cast<floatTypeA>(i);
    }
    for(size_t i = 0; i < elementsC; i++)
    {
        C[i] = static_cast<floatTypeC>(i % 41);
    }

    /********************************************
     * Transfer the Host Tensor to Device Memory
     ********************************************/
    CHECK_HIP_ERROR(hipMemcpy(A_d, A, sizeA, hipMemcpyDefault));
    CHECK_HIP_ERROR(hipMemcpy(C_d, C, sizeC, hipMemcpyDefault));

    /*************************
     * hipTensor
     *************************/
    hiptensorHandle_t handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /********************************************
     * Initialize tensors with the input lengths
     ********************************************/
    hiptensorTensorDescriptor_t descA = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &descA,
                                                          nmodeA,
                                                          extentA.data(),
                                                          nullptr /* stride */,
                                                          typeA,
                                                          0 /* alignmentRequirement */));

    hiptensorTensorDescriptor_t descC = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &descC,
                                                          nmodeC,
                                                          extentC.data(),
                                                          nullptr /* stride */,
                                                          typeC,
                                                          0 /* alignmentRequirement */));

    hiptensorTensorDescriptor_t descD = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &descD,
                                                          nmodeD,
                                                          extentD.data(),
                                                          nullptr /* stride */,
                                                          typeD,
                                                          0 /* alignmentRequirement */));

    /***************************************
     * Create Elementwise Binary Descriptor
     ***************************************/
    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(
        hiptensorCreateElementwiseBinary(handle,
                                         &desc,
                                         descA,
                                         modeA.data(),
                                         /* unary operator A  */ HIPTENSOR_OP_IDENTITY,
                                         descC,
                                         modeC.data(),
                                         /* unary operator C  */ HIPTENSOR_OP_IDENTITY,
                                         descD,
                                         modeD.data(),
                                         /* unary operator AC */ HIPTENSOR_OP_ADD,
                                         descCompute));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    hiptensorDataType_t scalarType;
    CHECK_HIPTENSOR_ERROR(
        hiptensorOperationDescriptorGetAttribute(handle,
                                                 desc,
                                                 HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                 (void*)&scalarType,
                                                 sizeof(scalarType)));

    assert(scalarType == HIPTENSOR_R_32F);

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

    hiptensorPlan_t plan;
    CHECK_HIPTENSOR_ERROR(
        hiptensorCreatePlan(handle, &plan, desc, planPref, 0 /* workspaceSizeLimit */));

    /**********************
     * Run
     **********************/

    CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryExecute(
        handle, plan, (void*)&alpha, A_d, (void*)&gamma, C_d, D_d, nullptr /* stream */));
#if !NDEBUG
    bool printElements = false;
    bool storeElements = false;

    if(printElements || storeElements)
    {
        CHECK_HIP_ERROR(hipMemcpy(D, D_d, sizeD, hipMemcpyDefault));
    }

    if(printElements)
    {
        if(elementsA < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor A elements:\n";
            hiptensorPrintArrayElements(std::cout, A, elementsA);
            std::cout << std::endl;
        }

        if(elementsC < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor C elements:\n";
            hiptensorPrintArrayElements(std::cout, C, elementsC);
            std::cout << std::endl;
        }

        if(elementsD < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D elements:\n";
            hiptensorPrintArrayElements(std::cout, D, elementsD);
            std::cout << std::endl;
        }
    }

    if(storeElements)
    {
        std::ofstream tensorA, tensorC, tensorD;
        tensorA.open("tensor_A.txt");
        hiptensorPrintElementsToFile(tensorA, A, elementsA, ", ");
        tensorA.close();

        tensorC.open("tensor_C.txt");
        hiptensorPrintElementsToFile(tensorC, C, elementsC, ", ");
        tensorC.close();

        tensorD.open("tensor_D_scale_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorD, D, elementsD, ", ");
        tensorD.close();
    }

#endif

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

    std::cout << "Finished!" << std::endl;
    return 0;
}
