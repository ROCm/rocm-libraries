// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_ALL_TRUE_HPP_
#define ROCPRIM_DETAIL_ALL_TRUE_HPP_

#include <type_traits>

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{


// all_of
template<bool... values>
struct all_true : std::true_type
{
};

template<bool... values>
struct all_true<true, values...> : all_true<values...>
{
};

template<bool... values>
struct all_true<false, values...> : std::false_type
{
};

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_ALL_TRUE_HPP_
