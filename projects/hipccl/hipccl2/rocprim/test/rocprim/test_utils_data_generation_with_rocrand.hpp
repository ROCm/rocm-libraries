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

#ifndef ROCPRIM_TEST_UTILS_DATA_GENERATION_WITH_ROCRAND_HPP
#define ROCPRIM_TEST_UTILS_DATA_GENERATION_WITH_ROCRAND_HPP

#include "../../common/utils_device_ptr.hpp"
#include "../../common/utils_half.hpp"
#include "common_test_header.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"

#include <rocprim/test_seed.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <rocrand/rocrand.hpp>
#include <rocrand/rocrand_kernel.h>

namespace test_utils_with_rocrand
{

template<typename T, class StateT, typename U, typename V>
inline __device__
auto generate_casting(T* output, StateT& state, U min, V max, unsigned int global_id)
{

    float f_value = (static_cast<float>(rocrand(&state)) / static_cast<float>(UINT_MAX))
                        * (static_cast<float>(max) - static_cast<float>(min))
                    + static_cast<float>(min);

    if constexpr(std::is_same<T, half>::value)
    {
        output[global_id] = static_cast<T>(__float2half_rn(f_value));
    }
    else if constexpr(common::is_custom_type_copyable<T>::value)
    {
        output[global_id].x = static_cast<typename T::first_type>(f_value);
        output[global_id].y = static_cast<typename T::second_type>(f_value);
    }
    else
    {
        output[global_id] = static_cast<T>(f_value);
    }
}

template<typename T, class StateT, typename U, typename V>
__global__
void generate_random_kernel(T*                       output,
                            U                        min,
                            V                        max,
                            const unsigned long long size,
                            const unsigned long long seed   = 0,
                            const unsigned long long offset = 0)
{
    const unsigned int flat_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(flat_id < size)
    {
        StateT             state;
        const unsigned int subsequence = flat_id;
        rocrand_init(seed, subsequence, offset, &state);
        generate_casting(output, state, min, max, flat_id);
    }
}

template<class OutputIter, class U, class V>
inline auto
    generate_random_data_n(OutputIter& it, size_t size, U min, V max, unsigned long long seed_value)
{
    if(size == 0)
        return it.begin() + size;

    using T = typename OutputIter::value_type;

    // Allocate device memory
    common::device_ptr<T> d_random_data(size);

    using state_t = rocrand_state_xorwow;

    constexpr int threadsPerBlock = 512;
    int           blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

    generate_random_kernel<T, state_t, U, V>
        <<<blocksPerGrid, threadsPerBlock>>>(d_random_data.get(), min, max, size, seed_value, 0);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(it.data(), d_random_data.get(), size * sizeof(T), hipMemcpyDeviceToHost));

    return it.begin() + size;
}

template<class T, class U, class V>
std::vector<T> get_random_data(size_t size, U min, V max, unsigned long long seed_value)
{
    std::vector<T> data(size);
    generate_random_data_n(data, size, min, max, seed_value);
    return data;
}

template<class T, class U, class V>
inline auto get_random_value(U min, V max) -> std::enable_if_t<rocprim::is_arithmetic<T>::value, T>
{
    T result;
    generate_random_data_n(&result, 1, min, max);
    return result;
}

} // namespace test_utils_with_rocrand

#endif // ROCPRIM_TEST_UTILS_DATA_GENERATION_ROCRAND_HPP
