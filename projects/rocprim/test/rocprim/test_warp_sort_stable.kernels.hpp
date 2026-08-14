// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_WARP_SORT_STABLE_KERNELS_HPP_
#define TEST_WARP_SORT_STABLE_KERNELS_HPP_

#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include <rocprim/warp/warp_sort_stable.hpp>

template<unsigned int LogicalWarpSize>
__device__ __forceinline__
unsigned int get_warp_id()
{
    return threadIdx.x / LogicalWarpSize;
}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         class T>
__global__
__launch_bounds__(BlockSize)
auto test_hip_warp_sort_stable(T* d_output) ->
    typename std::enable_if<ItemsPerThread == 1, void>::type
{
    const unsigned int i     = threadIdx.x + (blockIdx.x * blockDim.x);
    T                  value = d_output[i];

    using StableSort = rocprim::
        warp_sort_stable<T, BlockSize, LogicalWarpSize, ItemsPerThread, rocprim::empty_type, Algo>;

    constexpr unsigned int            WarpsInBlock = BlockSize / LogicalWarpSize;
    __shared__
    typename StableSort::storage_type storage[WarpsInBlock];

    unsigned int warp_id = get_warp_id<LogicalWarpSize>();

    StableSort().sort(value, storage[warp_id]);

    d_output[i] = value;
}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         class KeyType,
         class ValueType>
__global__
__launch_bounds__(BlockSize)
auto test_hip_sort_stable_key_value_kernel(KeyType* d_output_key, ValueType* d_output_value) ->
    typename std::enable_if<ItemsPerThread == 1, void>::type
{
    unsigned int i     = threadIdx.x + (blockIdx.x * blockDim.x);
    KeyType      key   = d_output_key[i];
    ValueType    value = d_output_value[i];
    using StableSort   = rocprim::
        warp_sort_stable<KeyType, BlockSize, LogicalWarpSize, ItemsPerThread, ValueType, Algo>;
    StableSort().sort(key, value);
    d_output_key[i]   = key;
    d_output_value[i] = value;
}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         class KeyType>
__device__
auto test_hip_warp_sort_stable_impl(KeyType* device_key_output)
    -> std::enable_if_t<(LogicalWarpSize <= ::rocprim::arch::wavefront::max_size())>
{
    if(LogicalWarpSize <= ::rocprim::arch::wavefront::size())
    {
        KeyType keys[ItemsPerThread];
        ::rocprim::block_load_direct_blocked(threadIdx.x,
                                             device_key_output
                                                 + (blockIdx.x * BlockSize * ItemsPerThread),
                                             keys);

        using StableSort = rocprim::warp_sort_stable<KeyType,
                                                     BlockSize,
                                                     LogicalWarpSize,
                                                     ItemsPerThread,
                                                     rocprim::empty_type,
                                                     Algo>;

        constexpr unsigned int            WarpsInBlock = BlockSize / LogicalWarpSize;
        __shared__
        typename StableSort::storage_type storage[WarpsInBlock];

        unsigned int warp_id = get_warp_id<LogicalWarpSize>();

        StableSort().sort(keys, storage[warp_id]);

        ::rocprim::block_store_direct_blocked(threadIdx.x,
                                              device_key_output
                                                  + (blockIdx.x * BlockSize * ItemsPerThread),
                                              keys);
    }
}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         class KeyType>
__device__
auto test_hip_warp_sort_stable_impl(KeyType*)
    -> std::enable_if_t<(LogicalWarpSize > ::rocprim::arch::wavefront::max_size())>
{}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         class KeyType>
__global__
__launch_bounds__(BlockSize)
auto test_hip_warp_sort_stable(KeyType* device_key_output) ->
    typename std::enable_if<(ItemsPerThread != 1), void>::type
{
    test_hip_warp_sort_stable_impl<Algo, ItemsPerThread, BlockSize, LogicalWarpSize, KeyType>(
        device_key_output);
}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         class KeyType,
         class ValueType>
__device__
auto test_hip_sort_stable_key_value_impl(KeyType* device_key_output, ValueType* device_value_output)
    -> std::enable_if_t<(LogicalWarpSize <= ::rocprim::arch::wavefront::max_size())>
{
    if(LogicalWarpSize <= ::rocprim::arch::wavefront::size())
    {
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        KeyType   keys[ItemsPerThread];
        ValueType values[ItemsPerThread];

        ::rocprim::block_load_direct_blocked(threadIdx.x, device_key_output + block_offset, keys);

        ::rocprim::block_load_direct_blocked(threadIdx.x,
                                             device_value_output + block_offset,
                                             values);

        using StableSort = rocprim::
            warp_sort_stable<KeyType, BlockSize, LogicalWarpSize, ItemsPerThread, ValueType, Algo>;

        constexpr unsigned int            WarpsInBlock = BlockSize / LogicalWarpSize;
        __shared__
        typename StableSort::storage_type storage[WarpsInBlock];

        unsigned int warp_id = get_warp_id<LogicalWarpSize>();

        StableSort().sort(keys, values, storage[warp_id]);

        ::rocprim::block_store_direct_blocked(threadIdx.x, device_key_output + block_offset, keys);
        ::rocprim::block_store_direct_blocked(threadIdx.x,
                                              device_value_output + block_offset,
                                              values);
    }
}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         class KeyType,
         class ValueType>
__device__
auto test_hip_sort_stable_key_value_impl(KeyType*, ValueType*)
    -> std::enable_if_t<(LogicalWarpSize > ::rocprim::arch::wavefront::max_size())>
{}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         class KeyType,
         class ValueType>
__global__
__launch_bounds__(BlockSize)
auto test_hip_sort_stable_key_value_kernel(KeyType*   device_key_output,
                                           ValueType* device_value_output) ->
    typename std::enable_if<(ItemsPerThread != 1), void>::type
{
    test_hip_sort_stable_key_value_impl<Algo,
                                        ItemsPerThread,
                                        BlockSize,
                                        LogicalWarpSize,
                                        KeyType,
                                        ValueType>(device_key_output, device_value_output);
}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         typename T>
void run_stable_sort_key_only_case(const std::vector<T>& input,
                                   const std::vector<T>& expected,
                                   size_t                grid_size)
{
    SCOPED_TRACE(testing::Message()
                 << "Algorithm: "
                 << (Algo == rocprim::warp_sort_stable_algorithm::merge_path ? "MergePath"
                                                                             : "Shuffle"));

    common::device_ptr<T> d_output(input);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            test_hip_warp_sort_stable<Algo, ItemsPerThread, BlockSize, LogicalWarpSize, T>),
        dim3(grid_size),
        dim3(BlockSize),
        0,
        0,
        d_output.get());

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output = d_output.load();
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
}

template<rocprim::warp_sort_stable_algorithm Algo,
         unsigned int                        ItemsPerThread,
         unsigned int                        BlockSize,
         unsigned int                        LogicalWarpSize,
         typename KeyT,
         typename ValueT>
void run_stable_sort_key_value_case(const std::vector<KeyT>&   input_keys,
                                    const std::vector<ValueT>& input_values,
                                    const std::vector<KeyT>&   expected_keys,
                                    const std::vector<ValueT>& expected_values,
                                    size_t                     grid_size)
{
    SCOPED_TRACE(testing::Message()
                 << "Algorithm: "
                 << (Algo == rocprim::warp_sort_stable_algorithm::merge_path ? "MergePath"
                                                                             : "Shuffle"));

    common::device_ptr<KeyT>   d_keys(input_keys);
    common::device_ptr<ValueT> d_values(input_values);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(test_hip_sort_stable_key_value_kernel<Algo,
                                                                             ItemsPerThread,
                                                                             BlockSize,
                                                                             LogicalWarpSize,
                                                                             KeyT,
                                                                             ValueT>),
                       dim3(grid_size),
                       dim3(BlockSize),
                       0,
                       0,
                       d_keys.get(),
                       d_values.get());

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<KeyT>   output_keys   = d_keys.load();
    std::vector<ValueT> output_values = d_values.load();

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_keys, expected_keys));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_values, expected_values));
}

#endif // TEST_WARP_SORT_STABLE_KERNELS_HPP_
