// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_vector_types.h>

// Meta configuration for rocPRIM
#include "config.hpp"

#include "types/double_buffer.hpp"
#include "types/future_value.hpp"
#include "types/integer_sequence.hpp"
#include "types/key_value_pair.hpp"
#include "types/tuple.hpp"
#include "types/uninitialized_array.hpp"

#include "intrinsics/arch.hpp"

/// \addtogroup utilsmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Takes a scalar type T and matches to a vector type based on NumElements.
template<class T, unsigned int NumElements>
struct make_vector_type
{
    using type = HIP_vector_type<T, NumElements>;
};

} // end namespace detail

/// \brief Empty type used as a placeholder, usually used to flag that given
/// template parameter should not be used.
struct empty_type
{};

/// \brief A decomposer that must be passed to the radix sort algorithms when
/// sorting keys that are arithmetic types.
/// To sort custom types, a custom decomposer should be provided.
struct identity_decomposer
{};

/// \brief 128 bit unsigned integer
using uint128_t = __uint128_t;
/// \brief 128 bit signed integer
using int128_t = __int128_t;

/// \brief Half-precision floating point type
using half = ::__half;
/// \brief bfloat16 floating point type
using bfloat16 = ::hip_bfloat16;

/// \brief The lane_mask_type is an integer that contains one bit per thread.
///
/// When targeting AMDGCN, the total number of bits is equal to the total
/// number of threads in a warp. Used for warp-level operations. When
/// targeting SPIR-V, it assumes 64 threads per warp.
///
/// \note When called on the host, assumes 64-bit wide masks.
///
/// \note When targeting SPIR-V, this type will be 64-bit wide. Extra
/// precaution must be taken as the number of bits in this type is not
/// always the same as the number of lanes.
using lane_mask_type = std::conditional_t<::rocprim::arch::wavefront::get_target()
                                              == ::rocprim::arch::wavefront::target::size32,
                                          unsigned int,
                                          unsigned long long int>;

/// \brief Native half-precision floating point type
using native_half = _Float16;

/// \brief native bfloat16 type
using native_bfloat16 = bfloat16;

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule

#endif // ROCPRIM_TYPES_HPP_
