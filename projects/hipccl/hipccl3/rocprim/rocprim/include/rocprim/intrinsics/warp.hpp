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

#ifndef ROCPRIM_INTRINSICS_WARP_HPP_
#define ROCPRIM_INTRINSICS_WARP_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup intrinsicsmodule
/// @{

/// Evaluate predicate for all active work-items in the warp and return an integer
/// whose <tt>i</tt>-th bit is set if and only if \p predicate is <tt>true</tt>
/// for the <tt>i</tt>-th thread of the warp and the <tt>i</tt>-th thread is active.
///
/// \param predicate - input to be evaluated for all active lanes
ROCPRIM_DEVICE inline
unsigned long long ballot(int predicate)
{
    #ifdef ROCPRIM_HC_API
        return hc::__ballot(predicate);
    #else // HIP
        return ::__ballot(predicate);
    #endif
}

/// \brief Masked bit count
///
/// For each thread, this function returns the number of active threads which
/// have <tt>i</tt>-th bit of \p x set and come before the current thread.
ROCPRIM_DEVICE inline
unsigned int masked_bit_count(unsigned long long x, unsigned int add = 0)
{
    #ifdef ROCPRIM_HC_API
        int c;
        c = hc::__amdgcn_mbcnt_lo(static_cast<int>(x), add);
        c = hc::__amdgcn_mbcnt_hi(static_cast<int>(x >> 32), c);
        return c;
    #else // HIP
        int c;
        c = ::__mbcnt_lo(static_cast<int>(x), add);
        c = ::__mbcnt_hi(static_cast<int>(x >> 32), c);
        return c;
    #endif
}

namespace detail
{

ROCPRIM_DEVICE inline
int warp_any(int predicate)
{
    #ifdef ROCPRIM_HC_API
        return hc::__any(predicate);
    #else // ROCPRIM_HIP_API
        return ::__any(predicate);
    #endif
}

ROCPRIM_DEVICE inline
int warp_all(int predicate)
{
    #ifdef ROCPRIM_HC_API
        return hc::__all(predicate);
    #else // ROCPRIM_HIP_API
        return ::__all(predicate);
    #endif
}

} // end detail namespace

/// @}
// end of group intrinsicsmodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_WARP_HPP_
