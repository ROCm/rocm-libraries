// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_BLOCK_LOAD_STORE_KERNELS_HPP_
#define TEST_BLOCK_LOAD_STORE_KERNELS_HPP_

typedef ::testing::Types<
    // block_load_direct
    class_params<int, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 64U, 1>,
    class_params<rocprim::half, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 64U, 7>,
    class_params<rocprim::bfloat16, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 64U, 7>,
    class_params<int, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 256U, 1>,
    class_params<char, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 256U, 4>,
    class_params<int, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 512U, 1>,
    class_params<int, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 512U, 3>,

    class_params<double, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 64U, 1>,
    class_params<long long, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 64U, 6>,
    class_params<double, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 256U, 1>,
    class_params<rocprim::half, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 256U, 3>,
    class_params<rocprim::bfloat16, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 256U, 3>,
    class_params<double, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 512U, 1>,
    class_params<double, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 512U, 2>,

    class_params<test_utils::custom_test_type<int>, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 64U, 5>,
    class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 256U, 4>,

    // block_load_vectorize
    class_params<int, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 64U, 1>,
    class_params<int, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 64U, 8>,
    class_params<rocprim::half, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 256U, 1>,
    class_params<rocprim::bfloat16, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 256U, 1>,
    class_params<int, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 256U, 4>,
    class_params<unsigned char, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 512U, 1>,
    class_params<int, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 512U, 4>

> ClassParamsFirstPart;

typedef ::testing::Types<

    class_params<double, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 64U, 1>,
    class_params<double, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 64U, 4>,
    class_params<double, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 256U, 1>,
    class_params<double, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 256U, 8>,
    class_params<double, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 512U, 1>,
    class_params<double, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 512U, 2>,

    class_params<test_utils::custom_test_type<int>, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 64U, 4>,
    class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 256U, 4>,

    // block_load_transpose
    class_params<int, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 64U, 1>,
    class_params<int, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 64U, 9>,
    class_params<int, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 256U, 1>,
    class_params<char, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 256U, 4>,
    class_params<int, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 512U, 1>,
    class_params<rocprim::half, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 512U, 4>,
    class_params<rocprim::bfloat16, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 512U, 4>,

    class_params<double, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 64U, 1>,
    class_params<double, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 64U, 7>,
    class_params<double, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 256U, 1>,
    class_params<double, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 256U, 4>,
    class_params<double, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 512U, 1>,
    class_params<double, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 512U, 3>,

    class_params<test_utils::custom_test_type<int>, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 64U, 5>,
    class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 256U, 4>

> ClassParamsSecondPart;

typedef ::testing::Types<
    vector_params<int, int, 3, false>,
    vector_params<int, rocprim::detail::int4, 4, true>,
    vector_params<int, int, 7, false>,
    vector_params<int, rocprim::detail::int4, 8, true>,
    vector_params<int, int, 11, false>,
    vector_params<int, rocprim::detail::int4, 16, true>,

    vector_params<char, char, 3, false>,
    vector_params<char, rocprim::detail::char4, 4, true>,
    vector_params<char, char, 7, false>,
    vector_params<char, rocprim::detail::char4, 8, true>,
    vector_params<char, char, 11, false>,
    vector_params<char, rocprim::detail::char4, 16, true>,

    vector_params<short, short, 3, false>,
    vector_params<short, rocprim::detail::short4, 4, true>,
    vector_params<short, short, 7, false>,
    vector_params<short, rocprim::detail::short4, 8, true>,
    vector_params<short, short, 11, false>,
    vector_params<short, rocprim::detail::short4, 16, true>,

    vector_params<float, int, 3, false>,
    vector_params<float, rocprim::detail::int4, 4, true>,
    vector_params<float, int, 7, false>,
    vector_params<float, rocprim::detail::int4, 8, true>,
    vector_params<float, int, 11, false>,
    vector_params<float, rocprim::detail::int4, 16, true>,

    vector_params<int2, rocprim::detail::int2, 3, false>,
    vector_params<int2, rocprim::detail::int4, 4, true>,
    vector_params<int2, rocprim::detail::int2, 7, false>,
    vector_params<int2, rocprim::detail::int4, 8, true>,
    vector_params<int2, rocprim::detail::int2, 11, false>,
    vector_params<int2, rocprim::detail::int4, 16, true>,

    vector_params<float2, rocprim::detail::int2, 3, false>,
    vector_params<float2, rocprim::detail::int4, 4, true>,
    vector_params<float2, rocprim::detail::int2, 7, false>,
    vector_params<float2, rocprim::detail::int4, 8, true>,
    vector_params<float2, rocprim::detail::int2, 11, false>,
    vector_params<float2, rocprim::detail::int4, 16, true>,

    vector_params<char4, int, 3, false>,
    vector_params<char4, rocprim::detail::int4, 4, true>,
    vector_params<char4, int, 7, false>,
    vector_params<char4, rocprim::detail::int4, 8, true>,
    vector_params<char4, int, 11, false>,
    vector_params<char4, rocprim::detail::int4, 16, true>
> VectorParams;

template<
    class Type,
    rocprim::block_load_method LoadMethod,
    rocprim::block_store_method StoreMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void load_store_kernel(Type* device_input, Type* device_output)
{
    Type _items[ItemsPerThread];
    auto offset = blockIdx.x * BlockSize * ItemsPerThread;
    rocprim::block_load<Type, BlockSize, ItemsPerThread, LoadMethod> load;
    rocprim::block_store<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.load(device_input + offset, _items);
    store.store(device_output + offset, _items);
}

template<
    class Type,
    rocprim::block_load_method LoadMethod,
    rocprim::block_store_method StoreMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void load_store_valid_kernel(Type* device_input, Type* device_output, size_t valid)
{
    Type _items[ItemsPerThread];
    auto offset = blockIdx.x * BlockSize * ItemsPerThread;
    rocprim::block_load<Type, BlockSize, ItemsPerThread, LoadMethod> load;
    rocprim::block_store<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.load(device_input + offset, _items, (unsigned int)valid);
    store.store(device_output + offset, _items, (unsigned int)valid);
}

template<
    class Type,
    rocprim::block_load_method LoadMethod,
    rocprim::block_store_method StoreMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class Def
>
__global__
__launch_bounds__(BlockSize)
void load_store_valid_default_kernel(Type* device_input, Type* device_output, size_t valid, Def _default)
{
    Type _items[ItemsPerThread];
    auto offset = blockIdx.x * BlockSize * ItemsPerThread;
    rocprim::block_load<Type, BlockSize, ItemsPerThread, LoadMethod> load;
    rocprim::block_store<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.load(device_input + offset, _items, (unsigned int)valid, _default);
    store.store(device_output + offset, _items);
}

#endif // TEST_BLOCK_LOAD_STORE_KERNELS_HPP_
