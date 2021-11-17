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

#ifndef TEST_BLOCK_SORT_KERNELS_HPP_
#define TEST_BLOCK_SORT_KERNELS_HPP_

template<
    unsigned int BlockSize,
    class key_type
>
__global__
__launch_bounds__(BlockSize)
void sort_key_kernel(key_type * device_key_output)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    key_type key = device_key_output[index];
    rocprim::block_sort<key_type, BlockSize> bsort;
    bsort.sort(key);
    device_key_output[index] = key;
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type
>
__global__
__launch_bounds__(BlockSize)
void sort_keys_kernel(key_type * device_key_output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    key_type keys[ItemsPerThread];
    ::rocprim::block_load_direct_striped<BlockSize>(lid, device_key_output + block_offset, keys);

    ::rocprim::block_sort<key_type, BlockSize, ItemsPerThread> sort;
    sort.sort(keys);

    ::rocprim::block_store_direct_blocked(lid, device_key_output + block_offset, keys);
}

template<class Key, class Value>
struct pair_comparator
{
    using less_key = typename test_utils::select_less_operator<Key>::type;
    using eq_key = typename test_utils::select_equal_to_operator<Key>::type;
    using less_value = typename test_utils::select_less_operator<Value>::type;

    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return (less_key()(lhs.first, rhs.first) || (eq_key()(lhs.first, rhs.first) && less_value()(lhs.second, rhs.second)));
    }
};

template<
    unsigned int BlockSize,
    class key_type,
    class value_type
>
__global__
__launch_bounds__(BlockSize)
void sort_key_value_kernel(key_type * device_key_output, value_type * device_value_output)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    key_type key = device_key_output[index];
    value_type value = device_value_output[index];
    rocprim::block_sort<key_type, BlockSize, 1, value_type> bsort;
    bsort.sort(key, value);
    device_key_output[index] = key;
    device_value_output[index] = value;
}

template<class Key, class Value>
struct key_value_comparator
{
    using greater_key = typename test_utils::select_greater_operator<Key>::type;
    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return greater_key()(lhs.first, rhs.first);
    }
};

template<
    unsigned int BlockSize,
    class key_type,
    class value_type
>
__global__
__launch_bounds__(BlockSize)
void custom_sort_key_value_kernel(key_type * device_key_output, value_type * device_value_output)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    key_type key = device_key_output[index];
    value_type value = device_value_output[index];
    rocprim::block_sort<key_type, BlockSize, 1, value_type> bsort;
    bsort.sort(key, value, rocprim::greater<key_type>());
    device_key_output[index] = key;
    device_value_output[index] = value;
}

template<class T>
inline constexpr bool is_power_of_two(const T x)
{
    static_assert(::rocprim::is_integral<T>::value, "T must be integer type");
    return (x > 0) && ((x & (x - 1)) == 0);
}

#endif // TEST_BLOCK_SORT_KERNELS_HPP_
