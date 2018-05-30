// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

template<class InputType, class OutputType, class BinaryFunction>
struct match_result_type
{
private:
    #ifdef __cpp_lib_is_invocable
    using binary_result_type = typename std::invoke_result<BinaryFunction, InputType, InputType>::type;
    #else
    using binary_result_type = typename std::result_of<BinaryFunction(InputType, InputType)>::type;
    #endif
    // Fixed output_type in case OutputType is void
    using output_type =
        typename std::conditional<
            std::is_void<OutputType>::value, InputType, OutputType
        >::type;
    // output_type is not a valid result_type if we can't covert it to binary_result_type
    static constexpr bool is_output_type_valid =
        std::is_convertible<output_type, binary_result_type>::value;

public:
    using type =
        typename std::conditional<
            is_output_type_valid, output_type, binary_result_type
        >::type;
};

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_

