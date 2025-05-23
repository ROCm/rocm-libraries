/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_CUB_MACRO_HPP_
#define HIPCUB_CUB_MACRO_HPP_

#include "../../config.hpp"

#include "cub/util_macro.cuh" // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

#ifndef HIPCUB_MAX
    /// Select maximum(a, b)
    #define HIPCUB_MAX(a, b) CUB_MAX(a, b)
#endif

#ifndef HIPCUB_MIN
    /// Select minimum(a, b)
    #define HIPCUB_MIN(a, b) CUB_MIN(a, b)
#endif

#ifndef HIPCUB_QUOTIENT_FLOOR
    /// Quotient of x/y rounded down to nearest integer
    #define HIPCUB_QUOTIENT_FLOOR(x, y) CUB_QUOTIENT_FLOOR(x, y)
#endif

#ifndef HIPCUB_QUOTIENT_CEILING
    /// Quotient of x/y rounded up to nearest integer
    #define HIPCUB_QUOTIENT_CEILING(x, y) CUB_QUOTIENT_CEILING(x, y)
#endif

#ifndef HIPCUB_ROUND_UP_NEAREST
    /// x rounded up to the nearest multiple of y
    #define HIPCUB_ROUND_UP_NEAREST(x, y) CUB_ROUND_UP_NEAREST(x, y)
#endif

#ifndef HIPCUB_ROUND_DOWN_NEAREST
    /// x rounded down to the nearest multiple of y
    #define HIPCUB_ROUND_DOWN_NEAREST(x, y) CUB_ROUND_DOWN_NEAREST(x, y)
#endif

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_MACRO_HPP_
