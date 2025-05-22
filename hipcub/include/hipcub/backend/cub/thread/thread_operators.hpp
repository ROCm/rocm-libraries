// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_CUB_THREAD_THREAD_OPERATORS_HPP_
#define HIPCUB_CUB_THREAD_THREAD_OPERATORS_HPP_

#include "../../../config.hpp"

#include <cub/thread/thread_operators.cuh> // IWYU pragma: export

#include <cuda/std/__functional/invoke.h>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

template<typename Invokable, typename InputT, typename InitT = InputT>
using accumulator_t = ::cuda::std::__accumulator_t<Invokable, InputT, InitT>;

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_THREAD_THREAD_OPERATORS_HPP_
