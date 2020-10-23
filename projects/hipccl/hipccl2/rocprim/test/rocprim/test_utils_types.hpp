// Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_TEST_UTILS_TYPES_HPP_
#define TEST_TEST_UTILS_TYPES_HPP_

#include "test_utils.hpp"

// required rocprim headers
#include <rocprim/config.hpp>

template<
    class T,
    unsigned int WarpSize
>
struct warp_params
{
    using type = T;
    static constexpr unsigned int warp_size = WarpSize;
};

template<
    class T,
    class U,
    unsigned int BlockSize = 256U
>
struct block_params
{
    using input_type = T;
    using output_type = U;
    static constexpr unsigned int block_size = BlockSize;
};

#if (defined(__gfx1030__))
    #define warp_param_type(type) \
        warp_params<type, 4U>, \
        warp_params<type, 8U>, \
        warp_params<type, 16U>, \
        warp_params<type, 32U>, \
        warp_params<type, 3U>, \
        warp_params<type, 7U>, \
        warp_params<type, 15U>
#else
    #define warp_param_type(type) \
       warp_params<type, 4U>, \
       warp_params<type, 8U>, \
       warp_params<type, 16U>, \
       warp_params<type, 32U>, \
       warp_params<type, 64U>, \
       warp_params<type, 3U>, \
       warp_params<type, 7U>, \
       warp_params<type, 15U>, \
       warp_params<type, 37U>, \
       warp_params<type, 61U>
#endif
#define block_param_type(input_type, output_type) \
    block_params<input_type, output_type, 64U>, \
    block_params<input_type, output_type, 128U>, \
    block_params<input_type, output_type, 192U>, \
    block_params<input_type, output_type, 256U>, \
    block_params<input_type, output_type, 129U>, \
    block_params<input_type, output_type, 162U>, \
    block_params<input_type, output_type, 255U>

typedef ::testing::Types<
    warp_param_type(int),
    warp_param_type(float),
    warp_param_type(uint8_t),
    warp_param_type(int8_t),
    warp_param_type(rocprim::half)
> WarpParams;

typedef ::testing::Types<
    block_param_type(int, test_utils::custom_test_type<int>),
    block_param_type(float, long),
    block_param_type(double, test_utils::custom_test_type<double>),
    block_param_type(uint8_t, short),
    block_param_type(int8_t, float),
    block_param_type(rocprim::half, rocprim::half)
> BlockParams;

static constexpr size_t n_items = 7;
static constexpr unsigned int items[n_items] = {
    1, 2, 4, 5, 7, 15, 32
};

template<class T, class BinaryOp>
T apply(BinaryOp binary_op, const T& a, const T& b)
{
    return binary_op(a, b);
}

#endif // TEST_TEST_UTILS_TYPES_HPP_
