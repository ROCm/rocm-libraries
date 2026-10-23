// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_DEVICE_BATCH_MEMCPY_HPP_
#define COMMON_DEVICE_BATCH_MEMCPY_HPP_

#include <rocprim/detail/various.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace common
{

// Used for generating offsets. We generate a permutation map and then derive
// offsets via a sum scan over the sizes in the order of the permutation. This
// allows us to keep the order of buffers we pass to batch_memcpy, but still
// have source and destinations mappings not be the identity function:
//
//  batch_memcpy(
//    [&a0 , &b0 , &c0 , &d0 ], // from (note the order is still just a, b, c, d!)
//    [&a0', &b0', &c0', &d0'], // to   (order is the same as above too!)
//    [3   , 2   , 1   , 2   ]) // size
//
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │b0 │b1 │a0 │a1 │a2 │d0 │d1 │c0 │ buffer x contains buffers a, b, c, d
// └───┴───┴───┴───┴───┴───┴───┴───┘ note that the order of buffers is shuffled!
//  ───┬─── ─────┬───── ───┬─── ───
//     └─────────┼─────────┼───┐
//           ┌───┘     ┌───┘   │ what batch_memcpy does
//           ▼         ▼       ▼
//  ─── ─────────── ─────── ───────
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │c0'│a0'│a1'│a2'│d0'│d1'│b0'│b1'│ buffer y contains buffers a', b', c', d'
// └───┴───┴───┴───┴───┴───┴───┴───┘
template<typename T, typename S, typename RandomGenerator>
std::vector<T> shuffled_exclusive_scan(const std::vector<S>& input, RandomGenerator& rng)
{
    const auto n = input.size();
    assert(n > 0);

    std::vector<T> result(n);
    std::vector<T> permute(n);

    std::iota(permute.begin(), permute.end(), 0);
    std::shuffle(permute.begin(), permute.end(), rng);

    T sum = 0;
    for(size_t i = 0; i < n; ++i)
    {
        result[permute[i]] = sum;
        sum += input[permute[i]];
    }

    return result;
}

template<bool IsMemCpy,
         typename ContainerMemCpy,
         typename ContainerCopy,
         typename byte_offset_type,
         typename std::enable_if<IsMemCpy, int>::type = 0>
void init_input(ContainerMemCpy& h_input_for_memcpy,
                ContainerCopy& /*h_input_for_copy*/,
                std::mt19937_64& rng,
                byte_offset_type total_num_bytes)
{
    std::independent_bits_engine<std::mt19937_64, 64, uint64_t> bits_engine{rng};

    const size_t num_ints = rocprim::detail::ceiling_div(total_num_bytes, sizeof(uint64_t));
    h_input_for_memcpy    = std::vector<unsigned char>(num_ints * sizeof(uint64_t));

    // generate_n for uninitialized memory, pragmatically use placement-new, since there are no
    // uint64_t objects alive yet in the storage.
    std::for_each(
        reinterpret_cast<uint64_t*>(h_input_for_memcpy.data()),
        reinterpret_cast<uint64_t*>(h_input_for_memcpy.data() + num_ints * sizeof(uint64_t)),
        [&bits_engine](uint64_t& elem) { ::new(&elem) uint64_t{bits_engine()}; });
}

template<bool IsMemCpy,
         typename ContainerMemCpy,
         typename ContainerCopy,
         typename byte_offset_type,
         typename std::enable_if<!IsMemCpy, int>::type = 0>
void init_input(ContainerMemCpy& /*h_input_for_memcpy*/,
                ContainerCopy&   h_input_for_copy,
                std::mt19937_64& rng,
                byte_offset_type total_num_bytes)
{
    using value_type = typename ContainerCopy::value_type;

    std::independent_bits_engine<std::mt19937_64, 64, uint64_t> bits_engine{rng};

    const size_t num_ints = rocprim::detail::ceiling_div(total_num_bytes, sizeof(uint64_t));
    const size_t num_of_elements
        = rocprim::detail::ceiling_div(num_ints * sizeof(uint64_t), sizeof(value_type));
    h_input_for_copy = std::vector<value_type>(num_of_elements);

    // generate_n for uninitialized memory, pragmatically use placement-new, since there are no
    // uint64_t objects alive yet in the storage.
    std::for_each(reinterpret_cast<uint64_t*>(h_input_for_copy.data()),
                  reinterpret_cast<uint64_t*>(h_input_for_copy.data()) + num_ints,
                  [&bits_engine](uint64_t& elem) { ::new(&elem) uint64_t{bits_engine()}; });
}

} // namespace common

#endif // COMMON_DEVICE_BATCH_MEMCPY_HPP_
