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
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>

template<
    class T,
    unsigned int WarpSize,
    unsigned int ItemsPerThread = 1
>
struct warp_params
{
    using type = T;
    static constexpr unsigned int warp_size = WarpSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
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

template<
    class T,
    class U,
    unsigned int ItemsPerThread,
    bool ShouldBeVectorized
>
struct vector_params
{
    using type = T;
    using vector_type = U;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool should_be_vectorized = ShouldBeVectorized;
};

template<
    class Type,
    rocprim::block_load_method Load,
    rocprim::block_store_method Store,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct class_params
{
    using type = Type;
    static constexpr rocprim::block_load_method load_method = Load;
    static constexpr rocprim::block_store_method store_method = Store;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

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
    warp_param_type(int8_t),
    warp_param_type(uint8_t)
> WarpParamsIntegral;

typedef ::testing::Types<
    warp_param_type(float),
    warp_param_type(rocprim::half),
    warp_param_type(rocprim::bfloat16)
> WarpParamsFloating;

// Separate sort params (only power of two warp sizes)
#define warp_sort_param_type(type, items_per_thread) \
   warp_params<type, 2U, items_per_thread>, \
   warp_params<type, 4U, items_per_thread>, \
   warp_params<type, 8U, items_per_thread>, \
   warp_params<type, 16U, items_per_thread>, \
   warp_params<type, 32U, items_per_thread>, \
   warp_params<type, 64U, items_per_thread>

typedef ::testing::Types<
    warp_sort_param_type(int, 1),
    warp_sort_param_type(test_utils::custom_test_type<int>, 1),
    warp_sort_param_type(uint8_t, 1),
    warp_sort_param_type(int8_t, 1)
> WarpSortParamsIntegral;

typedef ::testing::Types<
    warp_sort_param_type(rocprim::half, 1),
    warp_sort_param_type(rocprim::bfloat16, 1)
> WarpSortParamsFloating;

typedef ::testing::Types<
    warp_sort_param_type(int, 2),
    warp_sort_param_type(test_utils::custom_test_type<int>, 2),
    warp_sort_param_type(uint8_t, 2),
    warp_sort_param_type(int8_t, 2),
    warp_sort_param_type(int, 4),
    warp_sort_param_type(test_utils::custom_test_type<int>, 4),
    warp_sort_param_type(uint8_t, 4),
    warp_sort_param_type(int8_t, 4)
> WarpSortParamsIntegralMultiThread;

typedef ::testing::Types<
    block_param_type(int, test_utils::custom_test_type<int>),
    block_param_type(uint8_t, short),
    block_param_type(int8_t, float)
> BlockParamsIntegral;

typedef ::testing::Types<
    block_param_type(float, long),
    block_param_type(double, test_utils::custom_test_type<double>),
    block_param_type(rocprim::half, rocprim::half),
    block_param_type(rocprim::bfloat16, rocprim::bfloat16)
> BlockParamsFloating;

typedef ::testing::Types<
    block_param_type(test_utils::custom_test_type<int>, int),
    block_param_type(uint8_t, bool),
    block_param_type(int8_t, bool)
> BlockDiscParamsIntegral;

typedef ::testing::Types<
    block_param_type(float, char),
    block_param_type(double, unsigned int),
    block_param_type(rocprim::half, int),
    block_param_type(rocprim::bfloat16, int)
> BlockDiscParamsFloating;

typedef ::testing::Types<
    block_param_type(unsigned int, unsigned int)
> BlockHistAtomicParamsIntegral;

typedef ::testing::Types<
    block_param_type(float, long),
    block_param_type(double, test_utils::custom_test_type<double>),
    block_param_type(rocprim::half, rocprim::half)
> BlockExchParamsFloating;

typedef ::testing::Types<
    block_param_type(float, float),
    block_param_type(float, unsigned int),
    block_param_type(float, unsigned long long),
    block_param_type(double, float),
    block_param_type(double, unsigned long long)
> BlockHistAtomicParamsFloating;

typedef ::testing::Types<
    block_param_type(int, uint8_t),
    block_param_type(uint8_t, uint8_t),
    block_param_type(short, uint8_t),
    block_param_type(int, int8_t)
> BlockHistSortParamsIntegral;

typedef ::testing::Types<
    block_param_type(rocprim::half, unsigned short),
    block_param_type(rocprim::half, unsigned int),
    block_param_type(rocprim::bfloat16, unsigned short),
    block_param_type(rocprim::bfloat16, unsigned int)
> BlockHistSortParamsFloating;

static constexpr size_t n_items = 7;
static constexpr unsigned int items[n_items] = {
    1, 2, 4, 5, 7, 15, 32
};

// Global utility defines
#define test_suite_type_def_helper(name, suffix) \
    template<class Params> \
    class name ## suffix : public ::testing::Test { \
    public: \
        using params = Params; \
    };

#define test_suite_type_def(name, suffix) test_suite_type_def_helper(name, suffix)

#define block_histo_test_suite_type_def_helper(name, suffix) \
    template<class Params> \
    class name ## suffix : public ::testing::Test { \
    public: \
        using type = typename Params::input_type; \
        using bin_type = typename Params::output_type; \
        static constexpr unsigned int block_size = Params::block_size; \
        static constexpr unsigned int bin_size = Params::block_size; \
    };

#define block_histo_test_suite_type_def(name, suffix) block_histo_test_suite_type_def_helper(name, suffix)

#define block_reduce_test_suite_type_def_helper(name, suffix) \
    template<class Params> \
    class name ## suffix : public ::testing::Test { \
    public: \
        using input_type = typename Params::input_type; \
        static constexpr unsigned int block_size = Params::block_size; \
    };

#define block_reduce_test_suite_type_def(name, suffix) block_reduce_test_suite_type_def_helper(name, suffix)

#define block_sort_test_suite_type_def_helper(name, suffix) \
    template<class Params> \
    class name ## suffix : public ::testing::Test { \
    public: \
        using key_type = typename Params::input_type; \
        using value_type = typename Params::output_type; \
        static constexpr unsigned int block_size = Params::block_size; \
    };

#define block_sort_test_suite_type_def(name, suffix) block_sort_test_suite_type_def_helper(name, suffix)

#define typed_test_suite_def_helper(name, suffix, params) TYPED_TEST_SUITE(name ## suffix, params)

#define typed_test_suite_def(name, suffix, params) typed_test_suite_def_helper(name, suffix, params)

#define typed_test_def_helper(suite, suffix, name) TYPED_TEST(suite ## suffix, name)

#define typed_test_def(suite, suffix, name) typed_test_def_helper(suite, suffix, name)

#endif // TEST_TEST_UTILS_TYPES_HPP_
