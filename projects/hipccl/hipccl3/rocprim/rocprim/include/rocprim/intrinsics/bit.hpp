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

#ifndef ROCPRIM_INTRINSICS_BIT_HPP_
#define ROCPRIM_INTRINSICS_BIT_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup intrinsicsmodule
/// @{

/// \brief Returns a single bit at 'i' from 'x'
ROCPRIM_DEVICE inline
int get_bit(int x, int i)
{
    return (x >> i) & 1;
}

/// \brief Bit count
///
/// Returns the number of bit of \p x set.
ROCPRIM_DEVICE inline
unsigned int bit_count(unsigned int x)
{
    return __popc(x);
}

/// \brief Bit count
///
/// Returns the number of bit of \p x set.
ROCPRIM_DEVICE inline
unsigned int bit_count(unsigned long long x)
{
    return __popcll(x);
}

/// @}
// end of group intrinsicsmodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_BIT_HPP_
