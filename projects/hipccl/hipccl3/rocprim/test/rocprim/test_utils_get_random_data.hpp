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

#ifndef ROCPRIM_TEST_UTILS_GET_RANDOM_DATA_HPP
#define ROCPRIM_TEST_UTILS_GET_RANDOM_DATA_HPP

#include "test_utils_data_generation.hpp"
#ifdef WITH_ROCRAND
    #include "test_utils_data_generation_with_rocrand.hpp"
#endif

namespace test_utils
{
template<class T, class U, class V>
std::vector<T> get_random_data_wrapped(size_t size, U min, V max, unsigned long long seed_value)
{
#ifdef WITH_ROCRAND
    return test_utils_with_rocrand::get_random_data<T>(size, min, max, seed_value);
#else
    return test_utils::get_random_data<T>(size, min, max, seed_value);
#endif
}
} // namespace test_utils

#endif // ROCPRIM_TEST_UTILS_GET_RANDOM_DATA_HPP
