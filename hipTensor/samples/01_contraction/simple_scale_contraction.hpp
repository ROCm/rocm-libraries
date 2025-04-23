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
#include <algorithm>
#include <fstream>
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>
#include <iterator>
#include <numeric>
#include <unordered_map>

#include "common.hpp"

template <typename ADataType,
          typename BDataType,
          typename DDataType,
          hipDataType            typeA,
          hipDataType            typeB,
          hipDataType            typeD,
          hiptensorComputeType_t typeCompute>
int scaleContractionSample(void* alpha)
{
    /**********************
   * Computing: C_{m,n,u,v} = A_{m,n,h,k} B_{h,k,u,v}
   **********************/

    std::vector<int> modeD{'m', 'n', 'u', 'v'};
    std::vector<int> modeA{'m', 'n', 'h', 'k'};
    std::vector<int> modeB{'u', 'v', 'h', 'k'};

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeD = modeD.size();

    std::unordered_map<int, int64_t> extent;

    extent['m'] = 4;
    extent['n'] = 3;
    extent['u'] = 4;
    extent['v'] = 3;
    extent['h'] = 6;
    extent['k'] = 5;

    std::vector<int64_t> d_ms_ns_lengths;
    for(auto mode : modeD)
    {
        d_ms_ns_lengths.push_back(extent[mode]);
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

    hiptensorHandle_t* handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /********************************************
   * Initialize tensors with the input lengths *
   ********************************************/
    hiptensorTensorDescriptor_t a_ms_ks;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &a_ms_ks,
                                                        nmodeA,
                                                        a_ms_ks_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeA,
                                                        HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t b_ns_ks;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &b_ns_ks,
                                                        nmodeB,
                                                        b_ns_ks_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeB,
                                                        HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t d_ms_ns;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &d_ms_ns,
                                                        nmodeD,
                                                        d_ms_ns_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeD,
                                                        HIPTENSOR_OP_IDENTITY));

    /**********************
   * Allocating data
   **********************/
    std::cout << "Initializing host data..." << std::endl;

    size_t elementsA = std::accumulate(
        a_ms_ks_lengths.begin(), a_ms_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsB = std::accumulate(
        b_ns_ks_lengths.begin(), b_ns_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsD = std::accumulate(
        d_ms_ns_lengths.begin(), d_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeD = sizeof(DDataType) * elementsD;

    ADataType* A = nullptr;
    BDataType* B = nullptr;
    DDataType* D = nullptr;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&B, sizeB));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&D, sizeD));

    void *A_d, *B_d, *D_d;

    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&A_d), sizeA));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&B_d), sizeB));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&D_d), sizeD));

    /*******************
   * Initialize data
   *******************/
    int initMethod = 1; // TODO read the value from command line
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

    for(int64_t i = 0; i < elementsD; i++)
    {
        D[i] = std::numeric_limits<DDataType>::signaling_NaN();
    }

    /********************************************
   * Transfer the Host Tensor to Device Memory *
   ********************************************/
    std::cout << "Initializing device data..." << std::endl;

    CHECK_HIP_ERROR(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(D_d, 0, sizeD));

    /************************************************
   * Retrieve the memory alignment for each tensor
   ************************************************/
    uint32_t alignmentRequirementA;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, A_d, &a_ms_ks, &alignmentRequirementA));

    uint32_t alignmentRequirementB;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, B_d, &b_ns_ks, &alignmentRequirementB));

    uint32_t alignmentRequirementD;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, D_d, &d_ms_ns, &alignmentRequirementD));

    /*******************************
   * Create Contraction Descriptor
   *******************************/

    std::cout << "a_ms_ks: " << a_ms_ks << std::endl;
    std::cout << "b_ns_ks: " << b_ns_ks << std::endl;
    std::cout << "d_ms_ns: " << d_ms_ns << std::endl;

    hiptensorContractionDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorInitContractionDescriptor(handle,
                                                             &desc,
                                                             &a_ms_ks,
                                                             modeA.data(),
                                                             alignmentRequirementA,
                                                             &b_ns_ks,
                                                             modeB.data(),
                                                             alignmentRequirementB,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             &d_ms_ns,
                                                             modeD.data(),
                                                             alignmentRequirementD,
                                                             typeCompute));
    /**************************
   * Set the algorithm to use
   ***************************/

    hiptensorContractionFind_t find;
    CHECK_HIPTENSOR_ERROR(hiptensorInitContractionFind(handle, &find, HIPTENSOR_ALGO_ACTOR_CRITIC));

    /**********************
   * Query workspace
   **********************/

    uint64_t worksize = 0;
    CHECK_HIPTENSOR_ERROR(hiptensorContractionGetWorkspaceSize(
        handle, &desc, &find, HIPTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void* workspace = nullptr;

    if(worksize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
    }

    /**************************
   * Create Contraction Plan
   **************************/
    std::cout << "Initializing contraction plan..." << std::endl;

    hiptensorContractionPlan_t plan;
    CHECK_HIPTENSOR_ERROR(hiptensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

    std::cout << "Launching contraction kernel..." << std::endl;

    CHECK_HIPTENSOR_ERROR(hiptensorContraction(handle,
                                               &plan,
                                               alpha,
                                               A_d,
                                               B_d,
                                               nullptr,
                                               nullptr,
                                               D_d,
                                               workspace,
                                               worksize,
                                               0 /* stream */));

#if !NDEBUG
    bool printElements = false;
    bool storeElements = false;

    if(printElements || storeElements)
    {
        CHECK_HIP_ERROR(hipMemcpy(D, D_d, sizeD, hipMemcpyDeviceToHost));
    }

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

        if(elementsD < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D elements:\n";
            hiptensorPrintArrayElements(std::cout, D, elementsD);
            std::cout << std::endl;
        }
    }

    if(storeElements)
    {
        std::ofstream tensorA, tensorB, tensorD;
        tensorA.open("tensor_A.txt");
        hiptensorPrintElementsToFile(tensorA, A, elementsA, ", ");
        tensorA.close();

        tensorB.open("tensor_B.txt");
        hiptensorPrintElementsToFile(tensorB, B, elementsB, ", ");
        tensorB.close();

        tensorD.open("tensor_D_scale_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorD, D, elementsD, ", ");
        tensorD.close();
    }

#endif

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(D);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(D_d);
    HIPTENSOR_FREE_DEVICE(workspace);

    std::cout << "Finished!" << std::endl;

    return 0;
}
