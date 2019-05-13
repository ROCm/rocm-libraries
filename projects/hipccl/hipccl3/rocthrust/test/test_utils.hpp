/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        0, 1, 2, 12, 63, 64, 211, 256, 344,
        1024, 2048, 5096, 34567, (1 << 17) - 1220, 1000000, (1 << 20) - 123
    };
    return sizes;
}

template <class T>
inline auto get_random_data(size_t size, T, T) ->
    typename std::enable_if<std::is_same<T, bool>::value, thrust::host_vector<T>>::type
{
    std::random_device          rd;
    std::default_random_engine  gen(rd());
    std::bernoulli_distribution distribution(0.5);
    thrust::host_vector<T>      data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template <class T>
inline auto get_random_data(size_t size, T min, T max) ->
    typename std::enable_if<rocprim::is_integral<T>::value && !std::is_same<T, bool>::value,
                            thrust::host_vector<T>>::type
{
    std::random_device               rd;
    std::default_random_engine       gen(rd());
    std::uniform_int_distribution<T> distribution(min, max);
    thrust::host_vector<T>           data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template <class T>
inline auto get_random_data(size_t size, T min, T max) ->
    typename std::enable_if<rocprim::is_floating_point<T>::value, thrust::host_vector<T>>::type
{
    std::random_device                rd;
    std::default_random_engine        gen(rd());
    std::uniform_real_distribution<T> distribution(min, max);
    thrust::host_vector<T>            data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template <class T>
struct custom_compare_less
{
    __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs < rhs;
    }
}; // end less

struct user_swappable
{
    inline __host__ __device__ user_swappable(bool swapped = false)
        : was_swapped(swapped)
    {
    }

    bool was_swapped;
};

inline __host__ __device__ bool operator==(const user_swappable& x, const user_swappable& y)
{
    return x.was_swapped == y.was_swapped;
}

inline __host__ __device__ void swap(user_swappable& x, user_swappable& y)
{
    x.was_swapped = true;
    y.was_swapped = false;
}

class my_system : public thrust::device_execution_policy<my_system>
{
public:
    my_system(int)
        : correctly_dispatched(false)
        , num_copies(0)
    {
    }

    my_system(const my_system& other)
        : correctly_dispatched(false)
        , num_copies(other.num_copies + 1)
    {
    }

    void validate_dispatch()
    {
        correctly_dispatched = (num_copies == 0);
    }

    bool is_valid()
    {
        return correctly_dispatched;
    }

private:
    bool correctly_dispatched;

    // count the number of copies so that we can validate
    // that dispatch does not introduce any
    unsigned int num_copies;

    // disallow default construction
    my_system();
};

struct my_tag : thrust::device_execution_policy<my_tag>
{
};

template <typename T, unsigned int N>
struct FixedVector
{
    T data[N];

    __host__ __device__ FixedVector()
    {
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
            data[i] = T();
    }

    __host__ __device__ FixedVector(T init)
    {
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
            data[i] = init;
    }

    __host__ __device__ FixedVector operator+(const FixedVector& bs) const
    {
        FixedVector output;
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
            output.data[i] = data[i] + bs.data[i];
        return output;
    }

    __host__ __device__ bool operator<(const FixedVector& bs) const
    {
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
        {
            if(data[i] < bs.data[i])
                return true;
            else if(bs.data[i] < data[i])
                return false;
        }
        return false;
    }

    __host__ __device__ bool operator==(const FixedVector& bs) const
    {
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
        {
            if(!(data[i] == bs.data[i]))
                return false;
        }
        return true;
    }
};
