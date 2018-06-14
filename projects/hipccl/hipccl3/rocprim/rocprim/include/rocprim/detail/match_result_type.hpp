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
#include "../types/tuple.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

// tuple_contains_type::value is false if Tuple is not rocprim::tuple<> or Tuple is
// rocprim::tuple<> class which does not contain element of type T; otherwise it's true.
template<class T, class Tuple>
struct tuple_contains_type : std::false_type {};

template<class T>
struct tuple_contains_type<T, ::rocprim::tuple<>> : std::false_type {};

template<class T, class U, class... Ts>
struct tuple_contains_type<T, ::rocprim::tuple<U, Ts...>> : tuple_contains_type<T, ::rocprim::tuple<Ts...>> {};

template<class T, class... Ts>
struct tuple_contains_type<T, ::rocprim::tuple<T, Ts...>> : std::true_type {};

template<class InputType, class OutputType, class BinaryFunction>
struct match_result_type
{
private:
    #ifdef __cpp_lib_is_invocable
    using binary_result_type = typename std::invoke_result<BinaryFunction, InputType, InputType>::type;
    #else
    using binary_result_type = typename std::result_of<BinaryFunction(InputType, InputType)>::type;
    #endif

    // Fixed output_type in case OutputType is void or is a tuple containing void
    static constexpr bool is_output_type_invalid =
        std::is_void<OutputType>::value || tuple_contains_type<void, OutputType>::value;
    using value_type =
        typename std::conditional<is_output_type_invalid, InputType, OutputType>::type;

    // value_type is not a valid result_type if we can't covert it to binary_result_type
    static constexpr bool is_value_type_valid =
        std::is_convertible<value_type, binary_result_type>::value;

public:
    using type = typename std::conditional<is_value_type_valid, value_type, binary_result_type>::type;
};

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_

