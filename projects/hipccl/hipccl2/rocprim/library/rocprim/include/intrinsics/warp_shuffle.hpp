// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_
#define ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "detail/config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Checks whether an object of type T can be shuffled in single shuffle operation
template<class T>
struct is_single_shuffleable {
    static const bool value = sizeof(T) <= sizeof(int);
};

} // end namespace detail

constexpr unsigned int warp_size() [[hc]] [[cpu]]
{
    // Using marco allows contexpr, but we may have to
    // change it to hc::__wavesize() for safety
    return __HSA_WAVEFRONT_SIZE__;
    // return hc::__wavesize();
}

template<class T>
inline
auto warp_shuffle_up(T input, const unsigned int delta, const int width = hc::__wavesize()) [[hc]]
    -> typename std::enable_if<detail::is_single_shuffleable<T>::value, T>::type
{
    int * shfl_input = reinterpret_cast<int *>(&input);
    return hc::__shfl_up(*shfl_input, delta, width);
}

template<class T>
inline
auto warp_shuffle_up(T input, const unsigned int delta, const int width = hc::__wavesize()) [[hc]]
    -> typename std::enable_if<!detail::is_single_shuffleable<T>::value, T>::type
{
    constexpr int shfl_values = (sizeof(T) + sizeof(int) - 1) / sizeof(int);
    T output;

    int * shfl_input  = reinterpret_cast<int *>(&input);
    int * shfl_output = reinterpret_cast<int *>(&output);

    #pragma unroll
    for(int i = 0; i < shfl_values; i++)
    {
        shfl_output[i] = hc::__shfl_up(shfl_input[i], delta, width);
    }
    return output;
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_
