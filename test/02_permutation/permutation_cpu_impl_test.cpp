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

template <typename floatTypeA, typename floatTypeB, typename floatTypeCompute>
auto permuteWithCpu(hipDataType typeA, hipDataType typeB, hipDataType typeCompute)
{
    std::vector<int> modeA{'w', 'h', 'c', 'n'};
    std::vector<int> modeB{'c', 'n', 'h', 'w'};
    int              nmodeA = modeA.size();
    int              nmodeB = modeB.size();

    std::unordered_map<int, int64_t> extent;
    extent['w'] = 3;
    extent['h'] = 2;
    extent['c'] = 4;
    extent['n'] = 5;

    std::vector<int64_t> extentA;
    for(auto mode : modeA)
    {
        extentA.push_back(extent[mode]);
    }
    std::vector<int64_t> extentB;
    for(auto mode : modeB)
    {
        extentB.push_back(extent[mode]);
    }

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for(auto mode : modeA)
    {
        elementsA *= extent[mode];
    }
    size_t elementsB = 1;
    for(auto mode : modeB)
    {
        elementsB *= extent[mode];
    }

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;

    std::vector<floatTypeA> aArray(elementsA);
    std::vector<floatTypeB> bArray(elementsB);
    std::iota(aArray.begin(), aArray.end(), 0);
    std::vector<floatTypeB> referenceArray;

    using hiptensor::HiptensorOptions;
    auto& options = HiptensorOptions::instance();

    if(options->isColMajorStrides())
    {
        referenceArray
            = {0.,    12.6,  25.2,  37.8,  50.4,  63.,   75.6,  88.2,  100.8, 113.4, 126.,  138.6,
               151.2, 163.8, 176.4, 189.,  201.6, 214.2, 226.8, 239.4, 6.3,   18.9,  31.5,  44.1,
               56.7,  69.3,  81.9,  94.5,  107.1, 119.7, 132.3, 144.9, 157.5, 170.1, 182.7, 195.3,
               207.9, 220.5, 233.1, 245.7, 2.1,   14.7,  27.3,  39.9,  52.5,  65.1,  77.7,  90.3,
               102.9, 115.5, 128.1, 140.7, 153.3, 165.9, 178.5, 191.1, 203.7, 216.3, 228.9, 241.5,
               8.4,   21.,   33.6,  46.2,  58.8,  71.4,  84.,   96.6,  109.2, 121.8, 134.4, 147.,
               159.6, 172.2, 184.8, 197.4, 210.,  222.6, 235.2, 247.8, 4.2,   16.8,  29.4,  42.,
               54.6,  67.2,  79.8,  92.4,  105.,  117.6, 130.2, 142.8, 155.4, 168.,  180.6, 193.2,
               205.8, 218.4, 231.,  243.6, 10.5,  23.1,  35.7,  48.3,  60.9,  73.5,  86.1,  98.7,
               111.3, 123.9, 136.5, 149.1, 161.7, 174.3, 186.9, 199.5, 212.1, 224.7, 237.3, 249.9};
    }
    else
    {
        referenceArray
            = {0.,   84.,   168.,  42.,  126.,  210.,  2.1,  86.1,  170.1, 44.1, 128.1, 212.1,
               4.2,  88.2,  172.2, 46.2, 130.2, 214.2, 6.3,  90.3,  174.3, 48.3, 132.3, 216.3,
               8.4,  92.4,  176.4, 50.4, 134.4, 218.4, 10.5, 94.5,  178.5, 52.5, 136.5, 220.5,
               12.6, 96.6,  180.6, 54.6, 138.6, 222.6, 14.7, 98.7,  182.7, 56.7, 140.7, 224.7,
               16.8, 100.8, 184.8, 58.8, 142.8, 226.8, 18.9, 102.9, 186.9, 60.9, 144.9, 228.9,
               21.,  105.,  189.,  63.,  147.,  231.,  23.1, 107.1, 191.1, 65.1, 149.1, 233.1,
               25.2, 109.2, 193.2, 67.2, 151.2, 235.2, 27.3, 111.3, 195.3, 69.3, 153.3, 237.3,
               29.4, 113.4, 197.4, 71.4, 155.4, 239.4, 31.5, 115.5, 199.5, 73.5, 157.5, 241.5,
               33.6, 117.6, 201.6, 75.6, 159.6, 243.6, 35.7, 119.7, 203.7, 77.7, 161.7, 245.7,
               37.8, 121.8, 205.8, 79.8, 163.8, 247.8, 39.9, 123.9, 207.9, 81.9, 165.9, 249.9};
    }

    const floatTypeCompute alphaValue = 2.1f;
    hiptensorHandle_t*     handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    hiptensorTensorDescriptor_t descA;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descA, nmodeA, extentA.data(), NULL /* stride */, typeA, HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t descB;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descB, nmodeB, extentB.data(), NULL /* stride */, typeB, HIPTENSOR_OP_IDENTITY));

    hiptensorPermutationReference(&alphaValue,
                                  aArray.data(),
                                  &descA,
                                  modeA.data(),
                                  bArray.data(),
                                  &descB,
                                  modeB.data(),
                                  typeCompute,
                                  0);

    return compareEqual(referenceArray.data(),
                        bArray.data(),
                        bArray.size(),
                        hiptensor::convertToComputeType(typeCompute),
                        0);
}

TEST(PermutationCpuImplTest, CompareF32ResultWithReference)
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeCompute;

    hipDataType typeA       = HIP_R_32F;
    hipDataType typeB       = HIP_R_32F;
    hipDataType typeCompute = HIP_R_32F;

    auto [result, maxRelativeError]
        = permuteWithCpu<floatTypeA, floatTypeB, floatTypeCompute>(typeA, typeB, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}

TEST(PermutationCpuImplTest, CompareF16ResultWithReference)
{
    typedef _Float16 floatTypeA;
    typedef _Float16 floatTypeB;
    typedef _Float16 floatTypeCompute;

    hipDataType typeA       = HIP_R_16F;
    hipDataType typeB       = HIP_R_16F;
    hipDataType typeCompute = HIP_R_16F;

    auto [result, maxRelativeError]
        = permuteWithCpu<floatTypeA, floatTypeB, floatTypeCompute>(typeA, typeB, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}
