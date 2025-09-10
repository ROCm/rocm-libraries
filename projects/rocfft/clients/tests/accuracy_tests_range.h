// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ACCURACY_TESTS_RANGE_H
#define ACCURACY_TESTS_RANGE_H

#include <vector>

const static std::vector<std::vector<size_t>> stride_range = {{1}};

const static std::vector<std::vector<size_t>> ioffset_range_zero = {{0, 0}};
const static std::vector<std::vector<size_t>> ooffset_range_zero = {{0, 0}};

const static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}, {1, 1}};
const static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}, {1, 1}};

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// 1D test problems
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// TODO: handle special case where length=2 for real/complex transforms.
const static std::vector<size_t> pow2_range_1D
    = {2,        4,        8,         16,        32,        64,        128,     256,
       512,      1024,     2048,      4096,      8192,      16384,     32768,   65536,
       131072,   262144,   524288,    1048576,   2097152,   4194304,   8388608, 16777216,
       33554432, 67108864, 134217728, 268435456, 536870912, 1073741824};

const static std::vector<size_t> pow2_range_half_1D
    = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

const static std::vector<size_t> pow3_range_1D = {3,
                                                  9,
                                                  27,
                                                  81,
                                                  243,
                                                  729,
                                                  2187,
                                                  6561,
                                                  19683,
                                                  59049,
                                                  177147,
                                                  531441,
                                                  1594323,
                                                  4782969,
                                                  14348907,
                                                  43046721,
                                                  129140163,
                                                  387420489};

const static std::vector<size_t> pow5_range_1D
    = {5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125, 244140625};

// radix 7, 11, 13 sizes that are either pure powers or sizes people have wanted in the wild
const static std::vector<size_t> radX_range_1D
    = {7, 49, 84, 112, 11, 13, 52, 104, 208, 343, 2401, 16807};

const static std::vector<size_t> mix_range_1D
    = {6,   10,  12,   15,   20,   30,   56,   120,  150,  225,  240,  300,   336,   486,
       600, 900, 1250, 1500, 1875, 2160, 2187, 2250, 2500, 3000, 4000, 12000, 24000, 72000};

const static std::vector<size_t> prime_range_1D
    = {17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

static std::vector<size_t> small_1D_sizes()
{
    static const size_t SMALL_1D_MAX = 8192;

    // generate a list of sizes from 2 and up, skipping any sizes that are already covered
    std::vector<size_t> covered_sizes;
    std::copy(pow2_range_1D.begin(), pow2_range_1D.end(), std::back_inserter(covered_sizes));
    std::copy(pow3_range_1D.begin(), pow3_range_1D.end(), std::back_inserter(covered_sizes));
    std::copy(pow5_range_1D.begin(), pow5_range_1D.end(), std::back_inserter(covered_sizes));
    std::copy(radX_range_1D.begin(), radX_range_1D.end(), std::back_inserter(covered_sizes));
    std::copy(mix_range_1D.begin(), mix_range_1D.end(), std::back_inserter(covered_sizes));
    std::copy(prime_range_1D.begin(), prime_range_1D.end(), std::back_inserter(covered_sizes));
    std::sort(covered_sizes.begin(), covered_sizes.end());

    std::vector<size_t> output;
    for(size_t i = 2; i < SMALL_1D_MAX; ++i)
    {
        if(!std::binary_search(covered_sizes.begin(), covered_sizes.end(), i))
        {
            output.push_back(i);
        }
    }
    return output;
}

const static std::vector<size_t> batch_range_1D = {4, 2, 1};

const static std::vector<std::vector<size_t>> stride_range_for_prime_1D
    = {{1}, {2}, {3}, {64}, {65}}; //TODO: this will be merged back to stride_range

const static std::vector<size_t>              pow2_range_for_stride_1D      = {4096, 8192, 524288};
const static std::vector<size_t>              pow2_range_for_stride_half_1D = {4096, 8192};
const static std::vector<std::vector<size_t>> stride_range_for_pow2_1D      = {{2}, {3}};
const static std::vector<size_t>              batch_range_for_stride_1D     = {2, 1};

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// 2D test problems
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
const static std::vector<size_t> pow2_range_2D
    = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

// For the current configuration, half-precision has a fft size limit of 65536
const static std::vector<size_t> pow2_range_half_2D
    = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

const static std::vector<size_t> pow3_range_2D = {3, 9, 27, 81, 243, 729, 2187, 6561};

const static std::vector<size_t> pow5_range_2D = {5, 25, 125, 625, 3125, 15625};

const static std::vector<size_t> prime_range_2D = {7, 11, 13, 17, 19, 23, 29, 263, 269, 271, 277};

const static std::vector<size_t> mix_range_2D = {56, 120, 336, 2160, 5000, 6000, 8000};

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// 3D test problems
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
const static std::vector<size_t> pow2_range_3D = {4, 8, 16, 32, 128, 256};
// For the current configuration, half-precision has a fft size limit of 65536
const static std::vector<size_t> pow2_range_half_3D = {4, 8, 16, 32};

const static std::vector<size_t> pow3_range_3D = {3, 9, 27, 81, 243};

const static std::vector<size_t> pow5_range_3D = {5, 25, 125};

const static std::vector<size_t> prime_range_3D = {7, 11, 13, 17, 19, 23, 29};

// SBCC+SBRC as a sub-node of a 3D TRTRTR
const static std::vector<std::vector<size_t>> pow2_adhoc_3D = {{4, 4, 8192}};

// Test combinations of SBRC sizes, plus a non-SBRC size (10) to
// exercise fused SBRC+transpose kernels.
const static std::vector<size_t> sbrc_range_3D       = {50, 64, 81, 100, 200, 10, 128, 256};
const static std::vector<size_t> sbrc_batch_range_3D = {2, 1};

// pick small sizes that will exercise 2D_SINGLE and a couple of sizes that won't
const static std::vector<size_t> inner_batch_3D_range       = {4, 8, 16, 32, 20, 24, 64};
const static std::vector<size_t> inner_batch_3D_range_half  = {4, 8, 16, 32, 20, 24};
const static std::vector<size_t> inner_batch_3D_batch_range = {3, 2, 1};

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// partial pass test problems
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
const static std::vector<std::vector<size_t>> partial_pass_adhoc_3D = {
    {64, 64, 128},
    {64, 64, 64},
    {64, 64, 52},
    {60, 60, 60},
    {32, 32, 128},
    {32, 32, 64},
    {64, 32, 128},
};
const static std::vector<size_t> partial_pass_batch_range_3D = {1, 5, 10, 20, 50};

#endif // ACCURACY_TESTS_RANGE_H