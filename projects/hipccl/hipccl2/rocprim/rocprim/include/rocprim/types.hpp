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

#ifndef ROCPRIM_TYPES_HPP_
#define ROCPRIM_TYPES_HPP_

#include <type_traits>

// Meta configuration for rocPRIM
#include "config.hpp"

#include "types/double_buffer.hpp"
#include "types/integer_sequence.hpp"
#include "types/key_value_pair.hpp"
#include "types/tuple.hpp"

/// \addtogroup utilsmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Define vector types that will be used by rocPRIM internally.
// We don't use HIP vector types because they don't generate correct
// load/store operations, see https://github.com/RadeonOpenCompute/ROCm/issues/341
#define DEFINE_VECTOR_TYPE(name, base) \
\
struct name##2 \
{ \
    typedef base vector_value_type __attribute__((ext_vector_type(2))); \
    union { \
        vector_value_type data; \
        struct { base x, y; }; \
    }; \
} __attribute__((aligned(sizeof(base) * 2))); \
\
struct name##4 \
{ \
    typedef base vector_value_type __attribute__((ext_vector_type(4))); \
    union { \
        vector_value_type data; \
        struct { base x, y, w, z; }; \
    }; \
} __attribute__((aligned(sizeof(base) * 4)));

DEFINE_VECTOR_TYPE(char, char);
DEFINE_VECTOR_TYPE(short, short);
DEFINE_VECTOR_TYPE(int, int);
DEFINE_VECTOR_TYPE(longlong, long long);

// Takes a scalar type T and matches to a vector type based on NumElements.
template <class T, unsigned int NumElements>
struct make_vector_type
{
    using type = void;
};

#define DEFINE_MAKE_VECTOR_N_TYPE(name, base, suffix) \
template<> \
struct make_vector_type<base, suffix> \
{ \
    using type = name##suffix; \
};

#define DEFINE_MAKE_VECTOR_TYPE(name, base) \
\
template <> \
struct make_vector_type<base, 1> \
{ \
    using type = base; \
}; \
DEFINE_MAKE_VECTOR_N_TYPE(name, base, 2) \
DEFINE_MAKE_VECTOR_N_TYPE(name, base, 4)

DEFINE_MAKE_VECTOR_TYPE(char, char);
DEFINE_MAKE_VECTOR_TYPE(short, short);
DEFINE_MAKE_VECTOR_TYPE(int, int);
DEFINE_MAKE_VECTOR_TYPE(longlong, long long);

#undef DEFINE_VECTOR_TYPE
#undef DEFINE_MAKE_VECTOR_TYPE
#undef DEFINE_MAKE_VECTOR_N_TYPE

} // end namespace detail

/// \brief Empty type used as a placeholder, usually used to flag that given
/// template parameter should not be used.
struct empty_type
{

};

/// \brief Half-precision floating point type
using half = ::__half;

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule

#endif // ROCPRIM_TYPES_HPP_
