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

#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>
#include <numeric>
#include <unordered_map>

#include "common.hpp"

int main(int argc, char* argv[])
{
    /***************************************
    * Check device support                 *
    **************************************/
    if(!isF32F16MatrixCoreSupported())
    {
        std::cout << "unsupported host device" << std::endl;
        exit(EXIT_FAILURE);
    }

    typedef float    ADataType;
    typedef float    BDataType;
    typedef float    CDataType;
    typedef float    DDataType;
    typedef float    EDataType;
    typedef _Float16 floatTypeCompute;

    constexpr hiptensorDataType_t          typeA       = HIPTENSOR_R_32F;
    constexpr hiptensorDataType_t          typeB       = HIPTENSOR_R_32F;
    constexpr hiptensorDataType_t          typeC       = HIPTENSOR_R_32F;
    constexpr hiptensorDataType_t          typeD       = HIPTENSOR_R_32F;
    constexpr hiptensorDataType_t          typeE       = HIPTENSOR_R_32F;
    constexpr hiptensorComputeDescriptor_t typeCompute = HIPTENSOR_COMPUTE_DESC_16F;

    /**********************
     * Computing: E_{m,n,b,r,a} = alpha * A_{m,k,a,j,b,i} B_{k,n,i} C_{r,j}
     *                             + beta * D_{m,n,b,r,a}
     **********************/

    std::vector<int> modeA{'m', 'k', 'a', 'j', 'b', 'i'};
    std::vector<int> modeB{'k', 'n', 'i'};
    std::vector<int> modeC{'r', 'j'};
    std::vector<int> modeD{'m', 'n', 'b', 'r', 'a'};
    std::vector<int> modeE{'m', 'n', 'b', 'r', 'a'};

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();
    int nmodeD = modeD.size();
    int nmodeE = modeE.size();

    std::unordered_map<int, int64_t> extent;
    extent['m'] = 64; //256;
    extent['a'] = 32;
    extent['b'] = 32;
    extent['n'] = 64;
    extent['r'] = 64;
    extent['k'] = 8;
    extent['i'] = 8;
    extent['j'] = 64;

    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);

    std::vector<int64_t> extentB;
    for(auto mode : modeB)
        extentB.push_back(extent[mode]);

    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);

    std::vector<int64_t> extentD;
    for(auto mode : modeD)
        extentD.push_back(extent[mode]);

    std::vector<int64_t> extentE;
    for(auto mode : modeE)
        extentE.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/
    std::cout << "Initializing host data..." << std::endl;

    size_t elementsA = std::accumulate(
        extentA.begin(), extentA.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsB = std::accumulate(
        extentB.begin(), extentB.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsC = std::accumulate(
        extentC.begin(), extentC.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsD = std::accumulate(
        extentD.begin(), extentD.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsE = std::accumulate(
        extentE.begin(), extentE.end(), size_t{1}, std::multiplies<size_t>());

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeC = sizeof(CDataType) * elementsC;
    size_t sizeD = sizeof(DDataType) * elementsD;
    size_t sizeE = sizeof(EDataType) * elementsE;

    printf("Total memory: %.2f GiB\n",
           (sizeA + sizeB + sizeC + sizeD + sizeE) / 1024. / 1024. / 1024);

    ADataType* A = nullptr;
    BDataType* B = nullptr;
    CDataType* C = nullptr;
    DDataType* D = nullptr;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&B, sizeB));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeC));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&D, sizeD));

    void *A_d, *B_d, *C_d, *D_d, *E_d;
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&A_d), sizeA));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&B_d), sizeB));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&C_d), sizeC));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&D_d), sizeD));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&E_d), sizeE));

    /*******************
     * Initialize data
     *******************/
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = (ADataType)(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = (BDataType)(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = (CDataType)(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
    for(int64_t i = 0; i < elementsD; i++)
        D[i] = (DDataType)(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;

    /********************************************
     * Transfer the Host Tensor to Device Memory
     ********************************************/
    std::cout << "Initializing device data..." << std::endl;

    CHECK_HIP_ERROR(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(C_d, static_cast<const void*>(C), sizeC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(D_d, static_cast<const void*>(D), sizeD, hipMemcpyHostToDevice));

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
    hiptensorTensorDescriptor_t descA = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &descA,
                                                          nmodeA,
                                                          extentA.data(),
                                                          NULL, /*stride*/
                                                          typeA,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t descB = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &descB,
                                                          nmodeB,
                                                          extentB.data(),
                                                          NULL, /*stride*/
                                                          typeB,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t descC = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &descC,
                                                          nmodeC,
                                                          extentC.data(),
                                                          NULL, /*stride*/
                                                          typeC,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t descD = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &descD,
                                                          nmodeD,
                                                          extentD.data(),
                                                          NULL, /*stride*/
                                                          typeD,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t descE = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &descE,
                                                          nmodeE,
                                                          extentE.data(),
                                                          NULL, /*stride*/
                                                          typeE,
                                                          alignmentRequirement));

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateContractionTrinary(handle,
                                                            &desc,
                                                            descA,
                                                            modeA.data(),
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descB,
                                                            modeB.data(),
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descC,
                                                            modeC.data(),
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descD,
                                                            modeD.data(),
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descE,
                                                            modeE.data(),
                                                            typeCompute));

    /*****************************
     * Optional: ensure that the scalar type is correct.
     *****************************/

    hiptensorDataType_t scalarType;
    CHECK_HIPTENSOR_ERROR(
        hiptensorOperationDescriptorGetAttribute(handle,
                                                 desc,
                                                 HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                 (void*)&scalarType,
                                                 sizeof(scalarType)));
    assert(scalarType == *hiptensor::convertToHipTensorDataType(typeCompute));

    floatTypeCompute alpha = 1.1f;
    floatTypeCompute beta  = 2.3f;

    /**************************
     * Set the algorithm to use
     ***************************/
    const hiptensorAlgo_t algo = HIPTENSOR_ALGO_DEFAULT;

    hiptensorPlanPreference_t planPref;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
        handle, &planPref, algo, HIPTENSOR_JIT_MODE_NONE));

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

    void* workspace = nullptr;
    if(worksize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
    }

    /**********************
     * Run
     **********************/
    std::cout << "Launching trinary contraction kernel..." << std::endl;

    CHECK_HIPTENSOR_ERROR(hiptensorContractTrinary(handle,
                                                   plan,
                                                   (void*)&alpha,
                                                   A_d,
                                                   B_d,
                                                   C_d,
                                                   (void*)&beta,
                                                   D_d,
                                                   E_d,
                                                   workspace,
                                                   worksize,
                                                   0 /* stream */));

    /**********************
     * Clean up
     **********************/
    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));

    if(descA)
    {
        hiptensorDestroyTensorDescriptor(descA);
        descA = nullptr;
    }
    if(descB)
    {
        hiptensorDestroyTensorDescriptor(descB);
        descB = nullptr;
    }
    if(descC)
    {
        hiptensorDestroyTensorDescriptor(descC);
        descC = nullptr;
    }
    if(descD)
    {
        hiptensorDestroyTensorDescriptor(descD);
        descD = nullptr;
    }
    if(descE)
    {
        hiptensorDestroyTensorDescriptor(descE);
        descE = nullptr;
    }

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(C);
    HIPTENSOR_FREE_HOST(D);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(D_d);
    HIPTENSOR_FREE_DEVICE(E_d);
    HIPTENSOR_FREE_DEVICE(workspace);

    std::cout << "Finished!" << std::endl;

    return 0;
}
