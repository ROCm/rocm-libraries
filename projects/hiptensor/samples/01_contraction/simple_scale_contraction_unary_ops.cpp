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

#include <algorithm>
#include <fstream>
#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>
#include <hiptensor/internal/hiptensor_utility.hpp>
#include <iterator>
#include <numeric>
#include <unordered_map>

#include "common.hpp"

template <typename ADataType,
          typename BDataType,
          typename DDataType,
          hiptensorDataType_t          typeA,
          hiptensorDataType_t          typeB,
          hiptensorDataType_t          typeD,
          hiptensorComputeDescriptor_t typeCompute>
void scaleContractionSampleUnaryOps(void* alpha, hiptensorOperator_t opA, hiptensorOperator_t opB)
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

    extent['m'] = 256;
    extent['n'] = 20;
    extent['u'] = 128;
    extent['v'] = 128;
    extent['h'] = 128;
    extent['k'] = 128;

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

    /**********************
   * Allocating data
   **********************/

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

    CHECK_HIP_ERROR(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(D_d, 0, sizeD));

    /************************************************
   * Retrieve the memory alignment for each tensor
   ************************************************/
    uint32_t          alignmentRequirement = 1;
    hiptensorHandle_t handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /********************************************
   * Initialize tensors with the input lengths *
   ********************************************/
    hiptensorTensorDescriptor_t a_ms_ks;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &a_ms_ks,
                                                          nmodeA,
                                                          a_ms_ks_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeA,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t b_ns_ks;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &b_ns_ks,
                                                          nmodeB,
                                                          b_ns_ks_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeB,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t d_ms_ns;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &d_ms_ns,
                                                          nmodeD,
                                                          d_ms_ns_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeD,
                                                          alignmentRequirement));

    /*******************************
   * Create Contraction Descriptor
   *******************************/

    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateContraction(handle,
                                                     &desc,
                                                     a_ms_ks,
                                                     modeA.data(),
                                                     opA,
                                                     b_ns_ks,
                                                     modeB.data(),
                                                     opB,
                                                     nullptr,
                                                     nullptr,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     d_ms_ns,
                                                     modeD.data(),
                                                     typeCompute));

    hiptensorDataType_t scalarType;
    CHECK_HIPTENSOR_ERROR(
        hiptensorOperationDescriptorGetAttribute(handle,
                                                 desc,
                                                 HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                 (void*)&scalarType,
                                                 sizeof(scalarType)));
    assert(scalarType == *hiptensor::convertToHipTensorDataType(typeCompute));

    /**************************
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

    hiptensorPlan_t plan;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

    // TODO query actually used workspace
    void* workspace = nullptr;

    if(worksize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
    }

    CHECK_HIPTENSOR_ERROR(hiptensorContract(
        handle, plan, alpha, A_d, B_d, nullptr, nullptr, D_d, workspace, worksize, 0 /* stream */));

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
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));
    if(a_ms_ks)
    {
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(a_ms_ks));
        a_ms_ks = nullptr;
    }
    if(b_ns_ks)
    {
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(b_ns_ks));
        b_ns_ks = nullptr;
    }
    if(d_ms_ns)
    {
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(d_ms_ns));
        d_ms_ns = nullptr;
    }

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(D);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(D_d);
    HIPTENSOR_FREE_DEVICE(workspace);
}

int main()
{
    typedef hip_bfloat16 DataType_BF16;
    typedef _Float16     DataType_F16;
    typedef float        DataType_F32;
    typedef double       DataType_F64;

    typedef float        floatTypeCompute;
    typedef hip_bfloat16 bf16TypeCompute;
    typedef _Float16     f16TypeCompute;
    typedef double       doubleTypeCompute;

    constexpr hiptensorDataType_t   typeInput_16BF = HIPTENSOR_R_16BF;
    constexpr hiptensorDataType_t   typeInput_16F  = HIPTENSOR_R_16F;
    constexpr hiptensorDataType_t   typeInput_32F  = HIPTENSOR_R_32F;
    constexpr hiptensorDataType_t   typeInput_64F  = HIPTENSOR_R_64F;

    constexpr hiptensorComputeDescriptor_t typeCompute_16BF = HIPTENSOR_COMPUTE_DESC_16BF;
    constexpr hiptensorComputeDescriptor_t typeCompute_16F = HIPTENSOR_COMPUTE_DESC_16F;
    constexpr hiptensorComputeDescriptor_t typeCompute_32F = HIPTENSOR_COMPUTE_DESC_32F;
    constexpr hiptensorComputeDescriptor_t typeCompute_64F = HIPTENSOR_COMPUTE_DESC_64F;

    floatTypeCompute alphaFloat{1.0f};
    bf16TypeCompute alphaBF16{1.0f};
    f16TypeCompute    alphaF16{1.0f};
    doubleTypeCompute alphaDouble{1.0};

    hiptensorOperator_t opA = HIPTENSOR_OP_SQRT;
    hiptensorOperator_t opB = HIPTENSOR_OP_LOG;

    // Example 1: BF16 input tensors and F32 compute type
    std::cout << "Running bilinear contraction sample with unary ops, BF16 input tensors and F32 compute type ..." << std::endl;
    scaleContractionSampleUnaryOps<DataType_BF16, DataType_BF16, DataType_BF16,
                                   typeInput_16BF, typeInput_16BF, typeInput_16BF,
                                   typeCompute_32F>(&alphaFloat, opA, opB);
    std::cout << std::endl;

    // Example 2: F16 input tensors and F32 compute type
    std::cout << "Running bilinear contraction sample with unary ops, F16 input tensors and F32 compute type ..." << std::endl;
    scaleContractionSampleUnaryOps<DataType_F16, DataType_F16, DataType_F16,
                                   typeInput_16F, typeInput_16F, typeInput_16F,
                                   typeCompute_32F>(&alphaFloat, opA, opB);
    std::cout << std::endl;

   // Example 3: F32 input tensors and BF16 compute type
    std::cout << "Running bilinear contraction sample with unary ops, F32 input tensors and BF16 compute type ..." << std::endl;
    scaleContractionSampleUnaryOps<DataType_F32, DataType_F32, DataType_F32,
                                   typeInput_32F, typeInput_32F, typeInput_32F,
                                   typeCompute_16BF>(&alphaBF16, opA, opB);
    std::cout << std::endl;

   // Example 4: F32 input tensors and F16 compute type
    std::cout << "Running bilinear contraction sample with unary ops, F32 input tensors and F16 compute type ..." << std::endl;
    scaleContractionSampleUnaryOps<DataType_F32, DataType_F32, DataType_F32,
                                   typeInput_32F, typeInput_32F, typeInput_32F,
                                   typeCompute_16F>(&alphaF16, opA, opB);
    std::cout << std::endl;

   // Example 5: F32 input tensors and F32 compute type
    std::cout << "Running bilinear contraction sample with unary ops, F32 input tensors and F32 compute type ..." << std::endl;
    scaleContractionSampleUnaryOps<DataType_F32, DataType_F32, DataType_F32,
                                   typeInput_32F, typeInput_32F, typeInput_32F,
                                   typeCompute_32F>(&alphaFloat, opA, opB);
    std::cout << std::endl;

   // Example 6: F64 input tensors and F32 compute type
    std::cout << "Running bilinear contraction sample with unary ops, F64 input tensors and F32 compute type ..." << std::endl;
    scaleContractionSampleUnaryOps<DataType_F64, DataType_F64, DataType_F64,
                                   typeInput_64F, typeInput_64F, typeInput_64F,
                                   typeCompute_32F>(&alphaFloat, opA, opB);
    std::cout << std::endl; 

   // Example 7: F64 input tensors and F64 compute type
    std::cout << "Running bilinear contraction sample with unary ops, F64 input tensors and F64 compute type ..." << std::endl;
    scaleContractionSampleUnaryOps<DataType_F64, DataType_F64, DataType_F64,
                                   typeInput_64F, typeInput_64F, typeInput_64F,
                                   typeCompute_64F>(&alphaDouble, opA, opB);
    std::cout << std::endl; 

    std::cout << "Finished!" << std::endl;

    return 0;
}
