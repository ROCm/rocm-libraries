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
    typedef float floatTypeCompute;

    hiptensorDataType_t                typeA       = HIPTENSOR_R_32F;
    hiptensorDataType_t                typeC       = HIPTENSOR_R_32F;
    hiptensorComputeDescriptor_t const descCompute = HIPTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;

    /**********************
      B_{w, h, c, n} = 1.0 *  \textsl{IDENTITY}(A_{c, n, h, w})
     **********************/

    std::vector<int> modeA{'w', 'h', 'c'};
    std::vector<int> modeC{'c', 'w', 'h'};
    int              nmodeA = modeA.size();
    int              nmodeC = modeC.size();

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

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d, *C_d;
    CHECK_HIP_ERROR(hipMalloc((void**)&A_d, sizeA));
    CHECK_HIP_ERROR(hipMalloc((void**)&C_d, sizeC));

    floatTypeA *A, *C;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeof(floatTypeA) * elementsA));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeof(floatTypeC) * elementsC));

    /*******************
     * Initialize data
     *******************/

    for(size_t i = 0; i < elementsA; i++)
    {
        A[i] = (float)i;
    }

    CHECK_HIP_ERROR(hipMemcpy(A_d, A, sizeA, hipMemcpyDefault));

    /*************************
     * hipTensor
    *************************/

    hiptensorHandle_t handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /**********************
     * Create Tensor Descriptors
     **********************/

    hiptensorTensorDescriptor_t descA;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
        handle, &descA, nmodeA, extentA.data(), nullptr /* stride */, typeA, 0));

    hiptensorTensorDescriptor_t descC;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
        handle, &descC, nmodeC, extentC.data(), nullptr /* stride */, typeC, 0));

    /*******************************
     * Create Permutation Descriptor
     *******************************/

    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePermutation(handle,
                                                     &desc,
                                                     descA,
                                                     modeA.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     descC,
                                                     modeC.data(),
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

    /**************************
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

    CHECK_HIPTENSOR_ERROR(hiptensorPermute(handle, plan, &alpha, A_d, C_d, nullptr /* stream */))

#if !NDEBUG
    bool printElements = false;
    bool storeElements = false;

    if(printElements || storeElements)
    {
        CHECK_HIP_ERROR(hipMemcpy(C, C_d, sizeC, hipMemcpyDefault));
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
    }

    if(storeElements)
    {
        std::ofstream tensorA, tensorB, tensorC;
        tensorA.open("tensor_A.txt");
        hiptensorPrintElementsToFile(tensorA, A, elementsA, ", ");
        tensorA.close();

        tensorC.open("tensor_C_scale_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorC, C, elementsC, ", ");
        tensorC.close();
    }

#endif

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

    std::cout << "Finished!" << std::endl;
    return 0;
}
