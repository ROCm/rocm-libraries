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

#ifndef ROCPRIM_DEVICE_DETAIL_UINT_FAST_DIV_HPP_
#define ROCPRIM_DEVICE_DETAIL_UINT_FAST_DIV_HPP_

#include "../../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Fast division by unsigned "constant"
// Used for fast division on device by precomputing magic numbers on host,
// hence no division by arbitrary values in kernel code.
// Hacker's Delight, Chapter 10, Integer Division By Constants (http://www.hackersdelight.org/)
// http://www.hackersdelight.org/hdcodetxt/magicu.c.txt
struct uint_fast_div
{
    unsigned int magic; // Magic number
    unsigned int shift; // shift amount
    unsigned int add;   // "add" indicator

    ROCPRIM_HOST_DEVICE inline
    uint_fast_div() = default;

    ROCPRIM_HOST_DEVICE inline
    uint_fast_div(unsigned int d)
    {
        // Must have 1 <= d <= 2**32-1.

        if(d == 1)
        {
            magic = 0;
            shift = 0;
            add = 0;
            return;
        }

        int p;
        unsigned int p32 = 1, q, r, delta;
        add = 0;                // Initialize "add" indicator.
        p = 31;                 // Initialize p.
        q = 0x7FFFFFFF/d;       // Initialize q = (2**p - 1)/d.
        r = 0x7FFFFFFF - q*d;   // Init. r = rem(2**p - 1, d).
        do {
            p = p + 1;
            if(p == 32) p32 = 1;     // Set p32 = 2**(p-32).
            else p32 = 2*p32;
            if(r + 1 >= d - r)
            {
                if(q >= 0x7FFFFFFF) add = 1;
                q = 2*q + 1;
                r = 2*r + 1 - d;
            }
            else
            {
                if(q >= 0x80000000) add = 1;
                q = 2*q;
                r = 2*r + 1;
            }
            delta = d - 1 - r;
        } while (p < 64 && p32 < delta);
        magic = q + 1;         // Magic number and
        shift = p - 32;        // shift amount

        if(add) shift--;
    }
};

ROCPRIM_HOST_DEVICE inline
unsigned int operator/(unsigned int n, const uint_fast_div& divisor)
{
    if(divisor.magic == 0)
    {
        // Special case for 1
        return n;
    }

    // Higher 32-bit of 64-bit multiplication
    unsigned int q = (static_cast<unsigned long long>(divisor.magic) * static_cast<unsigned long long>(n)) >> 32;
    if(divisor.add)
    {
        q = ((n - q) >> 1) + q;
    }
    return q >> divisor.shift;
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_UINT_FAST_DIV_HPP_
