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
#include "permutation/permutation_cpu_reference.hpp"
#include "permutation_test.hpp"
#include "utils.hpp"

template <typename InputType, typename OutputType, typename ComputeType>
auto elementaryBinaryOpWithCpu(hipDataType inputType,
                               hipDataType outputType,
                               hipDataType typeCompute)
{
    std::vector<int> inMode{'w', 'h', 'c', 'n'};
    std::vector<int> outputMode{'c', 'n', 'h', 'w'};
    int              ninMode     = inMode.size();
    int              noutputMode = outputMode.size();

    std::unordered_map<int, int64_t> extent;
    extent['w'] = 3;
    extent['h'] = 2;
    extent['c'] = 4;
    extent['n'] = 5;

    std::vector<int64_t> inExtent;
    for(auto mode : inMode)
    {
        inExtent.push_back(extent[mode]);
    }
    std::vector<int64_t> outputExtent;
    for(auto mode : outputMode)
    {
        outputExtent.push_back(extent[mode]);
    }

    /**********************
     * Allocating data
     **********************/

    size_t inElements = 1;
    for(auto mode : inMode)
    {
        inElements *= extent[mode];
    }
    size_t outputElements = 1;
    for(auto mode : outputMode)
    {
        outputElements *= extent[mode];
    }

    // size_t sizeA = sizeof(InputType) * inElements;
    // size_t sizeB = sizeof(OutputType) * outputElements;

    std::vector<InputType> aArray(inElements);
    std::iota(aArray.begin(), aArray.end(), 0);
    std::vector<InputType>  cArray(aArray);
    std::vector<OutputType> dArray(outputElements);
    std::vector<OutputType> referenceArray;

    using hiptensor::HiptensorOptions;
    auto& options = HiptensorOptions::instance();

    if(options->isColMajorStrides())
    {
        referenceArray
            = {0.,    19.8,  39.6,  59.4,  79.2,  99.,   118.8, 138.6, 158.4, 178.2, 198.,  217.8,
               237.6, 257.4, 277.2, 297.,  316.8, 336.6, 356.4, 376.2, 9.9,   29.7,  49.5,  69.3,
               89.1,  108.9, 128.7, 148.5, 168.3, 188.1, 207.9, 227.7, 247.5, 267.3, 287.1, 306.9,
               326.7, 346.5, 366.3, 386.1, 3.3,   23.1,  42.9,  62.7,  82.5,  102.3, 122.1, 141.9,
               161.7, 181.5, 201.3, 221.1, 240.9, 260.7, 280.5, 300.3, 320.1, 339.9, 359.7, 379.5,
               13.2,  33.,   52.8,  72.6,  92.4,  112.2, 132.,  151.8, 171.6, 191.4, 211.2, 231.,
               250.8, 270.6, 290.4, 310.2, 330.,  349.8, 369.6, 389.4, 6.6,   26.4,  46.2,  66.,
               85.8,  105.6, 125.4, 145.2, 165.,  184.8, 204.6, 224.4, 244.2, 264.,  283.8, 303.6,
               323.4, 343.2, 363.,  382.8, 16.5,  36.3,  56.1,  75.9,  95.7,  115.5, 135.3, 155.1,
               174.9, 194.7, 214.5, 234.3, 254.1, 273.9, 293.7, 313.5, 333.3, 353.1, 372.9, 392.7};
    }
    else
    {
        referenceArray
            = {0.,   132.,  264.,  66.,   198.,  330.,  3.3,  135.3, 267.3, 69.3,  201.3, 333.3,
               6.6,  138.6, 270.6, 72.6,  204.6, 336.6, 9.9,  141.9, 273.9, 75.9,  207.9, 339.9,
               13.2, 145.2, 277.2, 79.2,  211.2, 343.2, 16.5, 148.5, 280.5, 82.5,  214.5, 346.5,
               19.8, 151.8, 283.8, 85.8,  217.8, 349.8, 23.1, 155.1, 287.1, 89.1,  221.1, 353.1,
               26.4, 158.4, 290.4, 92.4,  224.4, 356.4, 29.7, 161.7, 293.7, 95.7,  227.7, 359.7,
               33.,  165.,  297.,  99.,   231.,  363.,  36.3, 168.3, 300.3, 102.3, 234.3, 366.3,
               39.6, 171.6, 303.6, 105.6, 237.6, 369.6, 42.9, 174.9, 306.9, 108.9, 240.9, 372.9,
               46.2, 178.2, 310.2, 112.2, 244.2, 376.2, 49.5, 181.5, 313.5, 115.5, 247.5, 379.5,
               52.8, 184.8, 316.8, 118.8, 250.8, 382.8, 56.1, 188.1, 320.1, 122.1, 254.1, 386.1,
               59.4, 191.4, 323.4, 125.4, 257.4, 389.4, 62.7, 194.7, 326.7, 128.7, 260.7, 392.7};
    }

    const ComputeType  alphaValue = 2.1f;
    const ComputeType  gammaValue = 1.2f;
    hiptensorHandle_t* handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    hiptensorTensorDescriptor_t descA;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &descA,
                                                        ninMode,
                                                        inExtent.data(),
                                                        NULL /* stride */,
                                                        inputType,
                                                        HIPTENSOR_OP_IDENTITY));
    hiptensorTensorDescriptor_t descC;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &descC,
                                                        ninMode,
                                                        inExtent.data(),
                                                        NULL /* stride */,
                                                        inputType,
                                                        HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t descD;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &descD,
                                                        noutputMode,
                                                        outputExtent.data(),
                                                        NULL /* stride */,
                                                        outputType,
                                                        HIPTENSOR_OP_IDENTITY));

    hiptensorElementwiseBinaryOpReference(handle,
                                          &alphaValue,
                                          aArray.data(),
                                          &descA,
                                          inMode.data(),
                                          &gammaValue,
                                          cArray.data(),
                                          &descC,
                                          inMode.data(),
                                          dArray.data(),
                                          &descD,
                                          outputMode.data(),
                                          HIPTENSOR_OP_ADD,
                                          typeCompute,
                                          0);

    return compareEqual(referenceArray.data(),
                        dArray.data(),
                        dArray.size(),
                        hiptensor::convertToComputeType(typeCompute),
                        0);
}

TEST(ElementaryBinaryOpCpuImplTest, CompareF32ResultWithReference)
{
    typedef float InputType;
    typedef float OutputType;
    typedef float ComputeType;

    hipDataType inputType   = HIP_R_32F;
    hipDataType outputType  = HIP_R_32F;
    hipDataType typeCompute = HIP_R_32F;

    auto [result, maxRelativeError] = elementaryBinaryOpWithCpu<InputType, OutputType, ComputeType>(
        inputType, outputType, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}

TEST(ElementaryBinaryOpCpuImplTest, CompareF16ResultWithReference)
{
    typedef _Float16 InputType;
    typedef _Float16 OutputType;
    typedef _Float16 ComputeType;

    hipDataType inputType   = HIP_R_16F;
    hipDataType outputType  = HIP_R_16F;
    hipDataType typeCompute = HIP_R_16F;

    auto [result, maxRelativeError] = elementaryBinaryOpWithCpu<InputType, OutputType, ComputeType>(
        inputType, outputType, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}
