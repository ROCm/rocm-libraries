// Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_
#define ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_

#include "../config.hpp"

#include "../type_traits.hpp"

ROCPRIM_PRAGMA_MESSAGE("Internal 'match_result_type.hpp'-header has been depracated. Please "
                       "include 'rocprim/type_traits.hpp' instead!");

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

template<class F, class... ArgTypes>
using invoke_result [[deprecated("Use 'rocprim::invoke_result' instead!")]]
= rocprim::invoke_result<F, ArgTypes...>;

template<class InputType, class BinaryFunction>
using match_result [[deprecated("Use 'rocprim::invoke_result_binary_op' instead!")]]
= rocprim::invoke_result_binary_op<InputType, BinaryFunction>;

template<class InputType, class BinaryFunction>
using match_result_type [[deprecated("Use 'rocprim::invoke_result_binary_op_t' instead!")]]
= rocprim::invoke_result_binary_op_t<InputType, BinaryFunction>;

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_
