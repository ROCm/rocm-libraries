// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once
#if !defined(TWIDDLES_H)
#define TWIDDLES_H

#include "../../../shared/gpubuf.h"
#include "rocfft/rocfft.h"
#include <vector>

static const size_t       LTWD_BASE_DEFAULT       = 8;
static const size_t       LARGE_TWIDDLE_THRESHOLD = 4096;
static const unsigned int TWIDDLES_MAX_RADICES    = 8;

gpubuf twiddles_create(size_t                     N,
                       size_t                     length_limit,
                       rocfft_precision           precision,
                       const hipDeviceProp_t&     deviceProp,
                       size_t                     largeTwdBase,
                       bool                       attach_halfN,
                       const std::vector<size_t>& radices,
                       unsigned int               deviceId);
gpubuf twiddles_create_2D(size_t                     N1,
                          size_t                     N2,
                          rocfft_precision           precision,
                          const hipDeviceProp_t&     deviceProp,
                          bool                       attach_halfN,
                          bool                       attach_halfN2,
                          const std::vector<size_t>& radices1,
                          const std::vector<size_t>& radices2,
                          unsigned int               deviceId);

gpubuf twiddles_pp_create(size_t                 N,
                          rocfft_precision       precision,
                          const hipDeviceProp_t& deviceProp,
                          unsigned int           deviceId);

void twiddle_streams_cleanup();

#endif // defined( TWIDDLES_H )
