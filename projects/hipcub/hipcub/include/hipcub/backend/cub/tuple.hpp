// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_CUB_TUPLE_HPP_
#define HIPCUB_CUB_TUPLE_HPP_

#include "../../config.hpp"

#include <cuda/std/tuple>

BEGIN_HIPCUB_NAMESPACE

template<class... Args>
using tuple = ::cuda::std::tuple<Args...>;

template<size_t I, class T>
using tuple_element = ::cuda::std::tuple_element<I, T>;

template<size_t I, class T>
using tuple_element_t = ::cuda::std::tuple_element_t<I, T>;

template<class T>
using tuple_size = ::cuda::std::tuple_size<T>;

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_TUPLE_HPP_
