/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hiptensor/hiptensor.hpp>
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
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    hipDataType            typeA       = HIP_R_32F;
    hipDataType            typeC       = HIP_R_32F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    /**********************
     C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}
     **********************/

    std::vector<int32_t> modeA{'m', 'h', 'k', 'v'};
    std::vector<int32_t> modeC{'k', 'v'};
    int32_t              nmodeA = modeA.size();
    int32_t              nmodeC = modeC.size();

    std::unordered_map<int32_t, int64_t> extent;
    extent['m'] = 3;
    extent['v'] = 4;
    extent['h'] = 5;
    extent['k'] = 6;

    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);

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

    for(size_t i = 0; i < elementsA; i++)
    {
        A[i] = (float)i;
    }
    for(size_t i = 0; i < elementsC; i++)
    {
        C[i] = (float)(i & 1);
    }

    CHECK_HIP_ERROR(hipMemcpy(A_d, A, sizeA, hipMemcpyDefault));
    CHECK_HIP_ERROR(hipMemcpy(C_d, C, sizeC, hipMemcpyDefault));

    hiptensorStatus_t  err;
    hiptensorHandle_t* handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    hiptensorTensorDescriptor_t descA;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descA, nmodeA, extentA.data(), NULL /* stride */, typeA, HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t descC;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descC, nmodeC, extentC.data(), NULL /* stride */, typeC, HIPTENSOR_OP_IDENTITY));

    const hiptensorOperator_t opReduce = HIPTENSOR_OP_ADD;

    uint64_t worksize = 0;
    CHECK_HIPTENSOR_ERROR(hiptensorReductionGetWorkspaceSize(handle,
                                                             A_d,
                                                             &descA,
                                                             modeA.data(),
                                                             C_d,
                                                             &descC,
                                                             modeC.data(),
                                                             C_d,
                                                             &descC,
                                                             modeC.data(),
                                                             opReduce,
                                                             typeCompute,
                                                             &worksize));
    void* work = nullptr;
    if(worksize > 0)
    {
        if(hipSuccess != hipMalloc(&work, worksize))
        {
            work     = nullptr;
            worksize = 0;
        }
    }

    CHECK_HIPTENSOR_ERROR(hiptensorReduction(handle,
                                             (const void*)&alpha,
                                             A_d,
                                             &descA,
                                             modeA.data(),
                                             (const void*)&beta,
                                             C_d,
                                             &descC,
                                             modeC.data(),
                                             C_d,
                                             &descC,
                                             modeC.data(),
                                             opReduce,
                                             typeCompute,
                                             work,
                                             worksize,
                                             0 /* stream */));

#if !NDEBUG
    bool printElements = true;
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
        std::ofstream tensorA, tensorC;
        tensorA.open("tensor_A.txt");
        hiptensorPrintElementsToFile(tensorA, A, elementsA, ", ");
        tensorA.close();

        tensorC.open("tensor_C_scale_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorC, C, elementsC, ", ");
        tensorC.close();
    }
#endif

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(C);
    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(work);

    std::cout << "Finished!" << std::endl;
    return 0;
}
