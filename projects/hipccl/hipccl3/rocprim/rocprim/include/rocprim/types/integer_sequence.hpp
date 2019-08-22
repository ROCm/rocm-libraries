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

#ifndef ROCPRIM_TYPES_INTEGER_SEQUENCE_HPP_
#define ROCPRIM_TYPES_INTEGER_SEQUENCE_HPP_

#include <type_traits>

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE
#if defined(__cpp_lib_integer_sequence) && !defined(DOXYGEN_SHOULD_SKIP_THIS)
// For C++14 or newer we just use standard implementation
using std::integer_sequence;
using std::index_sequence;
using std::make_integer_sequence;
using std::make_index_sequence;
using std::index_sequence_for;
#else
/// \brief Compile-time sequence of integers
///
/// Implements std::integer_sequence for C++11. When C++14 is supported
/// it is just an alias for std::integer_sequence.
template<class T, T... Ints>
class integer_sequence
{
    using value_type = T;

    static inline constexpr size_t size() noexcept
    {
        return sizeof...(Ints);
    }
};

template<size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

// DETAILS
namespace detail
{

template<class T, class IntegerSequence>
struct integer_sequence_cat;

template<class T, T... Indices>
struct integer_sequence_cat<T, ::rocprim::integer_sequence<T, Indices...>>
{
    using type = typename ::rocprim::integer_sequence<T, Indices..., sizeof...(Indices)>;
};

template<class T, size_t Count>
struct make_integer_sequence_impl :
    integer_sequence_cat<T, typename make_integer_sequence_impl<T, Count - 1>::type>
{
};

template<class T>
struct make_integer_sequence_impl<T, 0>
{
    using type = ::rocprim::integer_sequence<T>;
};

} // end detail namespace

template<class T, T N>
using make_integer_sequence = typename detail::make_integer_sequence_impl<T, N>::type;

template<size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

template<class... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;
#endif

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_TYPES_INTEGER_SEQUENCE_HPP_
