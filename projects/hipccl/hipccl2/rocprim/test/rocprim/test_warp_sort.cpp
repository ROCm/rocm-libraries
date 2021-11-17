// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common_test_header.hpp"

// required rocprim headers
#include <rocprim/warp/warp_sort.hpp>

// required test headers
#include "test_utils_types.hpp"

// kernel definitions
#include "test_warp_sort.kernels.hpp"

// Start stamping out tests
struct RocprimWarpSortShuffleBasedTests;

struct Integral;
#define suite_name RocprimWarpSortShuffleBasedTests
#define warp_params WarpSortParamsIntegral
#define name_suffix Integral

#include "test_warp_sort.hpp"

#undef suite_name
#undef warp_params
#undef name_suffix

struct Integral;
#define suite_name RocprimWarpSortShuffleBasedTests
#define warp_params WarpSortParamsIntegralMultiThread
#define name_suffix IntegralMultiThread

#include "test_warp_sort.hpp"

#undef suite_name
#undef warp_params
#undef name_suffix

struct Floating;
#define suite_name RocprimWarpSortShuffleBasedTests
#define warp_params WarpSortParamsFloating
#define name_suffix Floating

#include "test_warp_sort.hpp"
