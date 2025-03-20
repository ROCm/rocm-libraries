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
auto elementaryTrinaryOpWithCpu(hipDataType inputType,
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

    std::vector<InputType> aArray(inElements);
    std::iota(aArray.begin(), aArray.end(), 0);
    std::vector<InputType>  bArray(aArray);
    std::vector<InputType>  cArray(aArray);
    std::vector<OutputType> dArray(outputElements);
    std::vector<OutputType> referenceArray;

    using hiptensor::HiptensorOptions;
    auto& options = HiptensorOptions::instance();

    if(options->isColMajorStrides())
    {
        referenceArray
            = {0.,    21.6,  43.2,  64.8,  86.4,  108.,  129.6, 151.2, 172.8, 194.4, 216.,  237.6,
               259.2, 280.8, 302.4, 324.,  345.6, 367.2, 388.8, 410.4, 10.8,  32.4,  54.,   75.6,
               97.2,  118.8, 140.4, 162.,  183.6, 205.2, 226.8, 248.4, 270.,  291.6, 313.2, 334.8,
               356.4, 378.,  399.6, 421.2, 3.6,   25.2,  46.8,  68.4,  90.,   111.6, 133.2, 154.8,
               176.4, 198.,  219.6, 241.2, 262.8, 284.4, 306.,  327.6, 349.2, 370.8, 392.4, 414.,
               14.4,  36.,   57.6,  79.2,  100.8, 122.4, 144.,  165.6, 187.2, 208.8, 230.4, 252.,
               273.6, 295.2, 316.8, 338.4, 360.,  381.6, 403.2, 424.8, 7.2,   28.8,  50.4,  72.,
               93.6,  115.2, 136.8, 158.4, 180.,  201.6, 223.2, 244.8, 266.4, 288.,  309.6, 331.2,
               352.8, 374.4, 396.,  417.6, 18.,   39.6,  61.2,  82.8,  104.4, 126.,  147.6, 169.2,
               190.8, 212.4, 234.,  255.6, 277.2, 298.8, 320.4, 342.,  363.6, 385.2, 406.8, 428.4};
    }
    else
    {
        referenceArray
            = {0.,   144.,  288.,  72.,   216.,  360.,  3.6,  147.6, 291.6, 75.6,  219.6, 363.6,
               7.2,  151.2, 295.2, 79.2,  223.2, 367.2, 10.8, 154.8, 298.8, 82.8,  226.8, 370.8,
               14.4, 158.4, 302.4, 86.4,  230.4, 374.4, 18.,  162.,  306.,  90.,   234.,  378.,
               21.6, 165.6, 309.6, 93.6,  237.6, 381.6, 25.2, 169.2, 313.2, 97.2,  241.2, 385.2,
               28.8, 172.8, 316.8, 100.8, 244.8, 388.8, 32.4, 176.4, 320.4, 104.4, 248.4, 392.4,
               36.,  180.,  324.,  108.,  252.,  396.,  39.6, 183.6, 327.6, 111.6, 255.6, 399.6,
               43.2, 187.2, 331.2, 115.2, 259.2, 403.2, 46.8, 190.8, 334.8, 118.8, 262.8, 406.8,
               50.4, 194.4, 338.4, 122.4, 266.4, 410.4, 54.,  198.,  342.,  126.,  270.,  414.,
               57.6, 201.6, 345.6, 129.6, 273.6, 417.6, 61.2, 205.2, 349.2, 133.2, 277.2, 421.2,
               64.8, 208.8, 352.8, 136.8, 280.8, 424.8, 68.4, 212.4, 356.4, 140.4, 284.4, 428.4};
    }

    const ComputeType alphaValue = 0.3f;
    const ComputeType betaValue  = 2.1f;
    const ComputeType gammaValue = 1.2f;

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

    hiptensorTensorDescriptor_t descB;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &descB,
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

    hiptensorElementwiseTrinaryOpReference(handle,
                                           &alphaValue,
                                           aArray.data(),
                                           &descA,
                                           inMode.data(),
                                           &betaValue,
                                           bArray.data(),
                                           &descB,
                                           inMode.data(),
                                           &gammaValue,
                                           cArray.data(),
                                           &descC,
                                           inMode.data(),
                                           dArray.data(),
                                           &descD,
                                           outputMode.data(),
                                           HIPTENSOR_OP_ADD,
                                           HIPTENSOR_OP_ADD,
                                           typeCompute,
                                           0);

    return compareEqual(referenceArray.data(),
                        dArray.data(),
                        dArray.size(),
                        hiptensor::convertToComputeType(typeCompute),
                        0);
}

TEST(ElementaryTrinaryOpCpuImplTest, CompareF32ResultWithReference)
{
    typedef float InputType;
    typedef float OutputType;
    typedef float ComputeType;

    hipDataType inputType   = HIP_R_32F;
    hipDataType outputType  = HIP_R_32F;
    hipDataType typeCompute = HIP_R_32F;

    auto [result, maxRelativeError]
        = elementaryTrinaryOpWithCpu<InputType, OutputType, ComputeType>(
            inputType, outputType, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}

TEST(ElementaryTrinaryOpCpuImplTest, CompareF16ResultWithReference)
{
    typedef _Float16 InputType;
    typedef _Float16 OutputType;
    typedef _Float16 ComputeType;

    hipDataType inputType   = HIP_R_16F;
    hipDataType outputType  = HIP_R_16F;
    hipDataType typeCompute = HIP_R_16F;

    auto [result, maxRelativeError]
        = elementaryTrinaryOpWithCpu<InputType, OutputType, ComputeType>(
            inputType, outputType, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}
