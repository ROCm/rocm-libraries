/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <hiptensor/hiptensor.hpp>

#include "data_types.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "reduction/reduction_cpu_reference.hpp"
#include "reduction_test.hpp"
#include "utils.hpp"

template <typename floatTypeA, typename floatTypeC, typename floatTypeCompute>
auto reduceWithCpu(hipDataType typeA, hipDataType typeC, hiptensorComputeType_t typeCompute)
{
    floatTypeCompute alpha = (floatTypeCompute)1.2f;
    floatTypeCompute beta  = (floatTypeCompute)2.1f;

    std::vector<int32_t> modeA{'m', 'h', 'k', 'v'};
    std::vector<int32_t> modeC{'k', 'v'};
    int32_t              nmodeA = modeA.size();
    int32_t              nmodeC = modeC.size();

    std::unordered_map<int32_t, int64_t> extent;
    extent['m'] = 3;
    extent['h'] = 5;
    extent['k'] = 6;
    extent['v'] = 4;

    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);

    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    std::vector<floatTypeA> aArray(elementsA);
    std::vector<floatTypeC> cArray(elementsC, 1);
    std::iota(aArray.begin(), aArray.end(), 0);
    std::vector<floatTypeC> referenceArray;

    using hiptensor::HiptensorOptions;
    auto& options = HiptensorOptions::instance();

    if(options->isColMajorStrides())
    {
        referenceArray = {128.1,  398.1,  668.1,  938.1,  1208.1, 1478.1, 1748.1, 2018.1,
                          2288.1, 2558.1, 2828.1, 3098.1, 3368.1, 3638.1, 3908.1, 4178.1,
                          4448.1, 4718.1, 4988.1, 5258.1, 5528.1, 5798.1, 6068.1, 6338.1};
    }
    else
    {
        referenceArray = {3026.1, 3044.1, 3062.1, 3080.1, 3098.1, 3116.1, 3134.1, 3152.1,
                          3170.1, 3188.1, 3206.1, 3224.1, 3242.1, 3260.1, 3278.1, 3296.1,
                          3314.1, 3332.1, 3350.1, 3368.1, 3386.1, 3404.1, 3422.1, 3440.1};
    }

    hiptensorHandle_t* handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

    hiptensorTensorDescriptor_t descA;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descA, nmodeA, extentA.data(), NULL /* stride */, typeA, HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t descC;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descC, nmodeC, extentC.data(), NULL /* stride */, typeC, HIPTENSOR_OP_IDENTITY));

    const hiptensorOperator_t opReduce = HIPTENSOR_OP_ADD;

    uint64_t worksize = 0;
    CHECK_HIPTENSOR_ERROR(hiptensorReductionGetWorkspaceSize(handle,
                                                             aArray.data(),
                                                             &descA,
                                                             modeA.data(),
                                                             cArray.data(),
                                                             &descC,
                                                             modeC.data(),
                                                             cArray.data(),
                                                             &descC,
                                                             modeC.data(),
                                                             opReduce,
                                                             typeCompute,
                                                             &worksize));

    double alphaValue{};
    double betaValue{};
    hiptensor::writeVal(&alphaValue, typeCompute, {typeCompute, alpha});
    hiptensor::writeVal(&betaValue, typeCompute, {typeCompute, beta});
    CHECK_HIPTENSOR_ERROR(hiptensorReductionReference((const void*)&alphaValue,
                                                      aArray.data(),
                                                      &descA,
                                                      modeA.data(),
                                                      (const void*)&betaValue,
                                                      cArray.data(),
                                                      &descC,
                                                      modeC.data(),
                                                      cArray.data(),
                                                      &descC,
                                                      modeC.data(),
                                                      opReduce,
                                                      typeCompute,
                                                      0 /* stream */));

    return compareEqual(referenceArray.data(), cArray.data(), cArray.size(), typeCompute);
}

TEST(ReductionCpuImplTest, CompareF32ResultWithReference)
{
    using floatTypeA       = hiptensor::float32_t;
    using floatTypeC       = hiptensor::float32_t;
    using floatTypeCompute = hiptensor::float32_t;

    hipDataType            typeA       = HIP_R_32F;
    hipDataType            typeC       = HIP_R_32F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_32F;

    auto [result, maxRelativeError]
        = reduceWithCpu<floatTypeA, floatTypeC, floatTypeCompute>(typeA, typeC, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}

TEST(ReductionCpuImplTest, CompareF64ResultWithReference)
{
    using floatTypeA       = hiptensor::float64_t;
    using floatTypeC       = hiptensor::float64_t;
    using floatTypeCompute = hiptensor::float64_t;

    hipDataType            typeA       = HIP_R_64F;
    hipDataType            typeC       = HIP_R_64F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_64F;

    auto [result, maxRelativeError]
        = reduceWithCpu<floatTypeA, floatTypeC, floatTypeCompute>(typeA, typeC, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}

// TEST(ReductionCpuImplTest, CompareF16ResultWithReference)
// {
// typedef _Float16 floatTypeA;
// typedef _Float16 floatTypeC;
// typedef _Float16 floatTypeCompute;
//
// hipDataType typeA       = HIP_R_16F;
// hipDataType typeC       = HIP_R_16F;
// hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_16F;
//
// auto [result, maxRelativeError]
// = reduceWithCpu<floatTypeA, floatTypeC, floatTypeCompute>(typeA, typeC, typeCompute);
// EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
// }
