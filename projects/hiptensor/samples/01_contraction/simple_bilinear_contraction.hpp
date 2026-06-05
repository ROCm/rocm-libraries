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
#pragma once

#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>
#include <numeric>
#include <unordered_map>

#include "common.hpp"

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          hiptensorDataType_t          typeA,
          hiptensorDataType_t          typeB,
          hiptensorDataType_t          typeC,
          hiptensorComputeDescriptor_t typeCompute>
int bilinearContractionSample(void* alpha, void* beta)
{
    /**********************
     * Computing: C_{m,n,u,v} = alpha * A_{m,n,h,k} B_{u,v,h,k} + beta * C_{m,n,u,v}
     **********************/

    std::vector<int> modeC{'m', 'n', 'u', 'v'};
    std::vector<int> modeA{'m', 'n', 'h', 'k'};
    std::vector<int> modeB{'u', 'v', 'h', 'k'};

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;

    extent['m'] = 128;
    extent['n'] = 32;
    extent['u'] = 48;
    extent['v'] = 32;
    extent['h'] = 64;
    extent['k'] = 56;

    std::vector<int64_t> c_ms_ns_lengths;
    for(auto mode : modeC)
    {
        c_ms_ns_lengths.push_back(extent[mode]);
    }

    std::vector<int64_t> a_ms_ks_lengths;
    for(auto mode : modeA)
    {
        a_ms_ks_lengths.push_back(extent[mode]);
    }

    std::vector<int64_t> b_ns_ks_lengths;
    for(auto mode : modeB)
    {
        b_ns_ks_lengths.push_back(extent[mode]);
    }

    /**********************
     * Allocating data
     **********************/
    std::cout << "Initializing host data..." << std::endl;

    size_t elementsA = std::accumulate(
        a_ms_ks_lengths.begin(), a_ms_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsB = std::accumulate(
        b_ns_ks_lengths.begin(), b_ns_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsC = std::accumulate(
        c_ms_ns_lengths.begin(), c_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeC = sizeof(CDataType) * elementsC;

    ADataType* A = nullptr;
    BDataType* B = nullptr;
    CDataType* C = nullptr;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&B, sizeB));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeC));

    void *A_d, *B_d, *C_d;

    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&A_d), sizeA));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&B_d), sizeB));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&C_d), sizeC));

    /*******************
     * Initialize data
     *******************/
    int initMethod = 1; // TODO read value from commandline
    for(int64_t i = 0; i < elementsA; i++)
    {
        if(initMethod == 0)
        {
            A[i] = ADataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            A[i] = (ADataType)(float(i) / 100);
        }
    }

    for(int64_t i = 0; i < elementsB; i++)
    {
        if(initMethod == 0)
        {
            B[i] = BDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            B[i] = (BDataType)(float(i) / 100);
        }
    }

    for(int64_t i = 0; i < elementsC; i++)
    {
        if(initMethod == 0)
        {
            C[i] = CDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            C[i] = (CDataType)(float(i) / 100);
        }
    }

    /********************************************
     * Transfer the Host Tensor to Device Memory
     ********************************************/
    std::cout << "Initializing device data..." << std::endl;

    CHECK_HIP_ERROR(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(C_d, static_cast<const void*>(C), sizeC, hipMemcpyHostToDevice));

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
    hiptensorTensorDescriptor_t a_ms_ks = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &a_ms_ks,
                                                          nmodeA,
                                                          a_ms_ks_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeA,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t b_ns_ks = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &b_ns_ks,
                                                          nmodeB,
                                                          b_ns_ks_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeB,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t c_ms_ns = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &c_ms_ns,
                                                          nmodeC,
                                                          c_ms_ns_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeC,
                                                          alignmentRequirement));

    /********************************
     * Create Contraction Descriptor
     ********************************/

    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateContraction(handle,
                                                     &desc,
                                                     a_ms_ks,
                                                     modeA.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     b_ns_ks,
                                                     modeB.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     c_ms_ns,
                                                     modeC.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     c_ms_ns,
                                                     modeC.data(),
                                                     typeCompute));

    hiptensorDataType_t scalarType;
    CHECK_HIPTENSOR_ERROR(
        hiptensorOperationDescriptorGetAttribute(handle,
                                                 desc,
                                                 HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                 (void*)&scalarType,
                                                 sizeof(scalarType)));
    assert(scalarType == *hiptensor::convertToHipTensorDataType(typeCompute));

    /***************************
     * Set the algorithm to use
     ***************************/
    hiptensorPlanPreference_t planPref;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
        handle, &planPref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace
     **********************/
    uint64_t worksize = 0;
    CHECK_HIPTENSOR_ERROR(hiptensorEstimateWorkspaceSize(
        handle, desc, planPref, HIPTENSOR_WORKSPACE_DEFAULT, &worksize));

    /**************************
     * Create Contraction Plan
     **************************/
    std::cout << "Initializing contraction plan..." << std::endl;

    hiptensorPlan_t plan;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

    // TODO query actually used workspace
    void* workspace = nullptr;

    if(worksize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
    }

    std::cout << "Launching contraction kernel..." << std::endl;

    CHECK_HIPTENSOR_ERROR(hiptensorContract(
        handle, plan, alpha, A_d, B_d, beta, C_d, C_d, workspace, worksize, 0 /* stream */));

#if !NDEBUG
    bool printElements = false;
    bool storeElements = false;

    if(printElements)
    {
        if(elementsA < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor A elements:\n";
            hiptensorPrintArrayElements(std::cout, A, elementsA);
            std::cout << std::endl;
        }

        if(elementsB < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor B elements:\n";
            hiptensorPrintArrayElements(std::cout, B, elementsB);
            std::cout << std::endl;
        }

        if(elementsC < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor C elements:\n";
            hiptensorPrintArrayElements(std::cout, C, elementsC);
            std::cout << std::endl;
        }

        CHECK_HIP_ERROR(hipMemcpy(C, C_d, sizeC, hipMemcpyDeviceToHost));

        if(elementsC < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D elements:\n";
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

        tensorB.open("tensor_B.txt");
        hiptensorPrintElementsToFile(tensorB, B, elementsB, ", ");
        tensorB.close();

        tensorC.open("tensor_C.txt");
        hiptensorPrintElementsToFile(tensorC, C, elementsC, ", ");
        tensorC.close();

        CHECK_HIP_ERROR(hipMemcpy(C, C_d, sizeC, hipMemcpyDeviceToHost));

        tensorC.open("tensor_C_scale_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorC, C, elementsC, ", ");
        tensorC.close();
    }

#endif

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));
    if(a_ms_ks)
    {
        hiptensorDestroyTensorDescriptor(a_ms_ks);
        a_ms_ks = nullptr;
    }
    if(b_ns_ks)
    {
        hiptensorDestroyTensorDescriptor(b_ns_ks);
        b_ns_ks = nullptr;
    }
    if(c_ms_ns)
    {
        hiptensorDestroyTensorDescriptor(c_ms_ns);
        c_ms_ns = nullptr;
    }

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(C);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(workspace);

    std::cout << "Finished!" << std::endl;

    return 0;
}
