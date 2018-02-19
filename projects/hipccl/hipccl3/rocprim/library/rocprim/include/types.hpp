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

#ifndef ROCPRIM_TYPES_HPP_
#define ROCPRIM_TYPES_HPP_

#include <type_traits>

// Meta configuration for rocPRIM
#include "config.hpp"

#include "types/double_buffer.hpp"

/// \addtogroup utilsmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Define vector types that will be used by rocPRIM internally.
#define DEFINE_VECTOR_TYPE(base) \
\
struct base##2 \
{ \
    base x, y; \
} __attribute__((aligned(sizeof(base) * 2))); \
\
struct base##4 \
{ \
    base x, y, w, z; \
} __attribute__((aligned(sizeof(base) * 4))); \

DEFINE_VECTOR_TYPE(char);
DEFINE_VECTOR_TYPE(int);
DEFINE_VECTOR_TYPE(short);

// Takes a scalar type T and matches to a vector type based on NumElements.
template <class T, unsigned int NumElements>
struct make_vector_type
{
    using type = void;
};

template <>
struct make_vector_type<char, 1>
{
    using type = char;
};

template <>
struct make_vector_type<int, 1>
{
    using type = int;
};

template <>
struct make_vector_type<short, 1>
{
    using type = short;
};

#define DEFINE_MAKE_VECTOR_TYPE(base, suffix) \
\
template<> \
struct make_vector_type<base, suffix> \
{ \
    using type = base##suffix; \
};

DEFINE_MAKE_VECTOR_TYPE(char, 2);
DEFINE_MAKE_VECTOR_TYPE(char, 4);
DEFINE_MAKE_VECTOR_TYPE(int, 2);
DEFINE_MAKE_VECTOR_TYPE(int, 4);
DEFINE_MAKE_VECTOR_TYPE(short, 2);
DEFINE_MAKE_VECTOR_TYPE(short, 4);

} // end namespace detail

/// \brief Empty type used as a placeholder, usually used to flag that given
/// template parameter should not be used.
struct empty_type
{

};

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule

#endif // ROCPRIM_TYPES_HPP_
