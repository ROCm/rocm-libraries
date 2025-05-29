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

#ifndef ROCRAND_KERNEL_H_
#define ROCRAND_KERNEL_H_

// IWYU pragma: begin_exports
#include "rocrand/rocrand_lfsr113.h"
#include "rocrand/rocrand_mrg31k3p.h"
#include "rocrand/rocrand_mrg32k3a.h"
#include "rocrand/rocrand_mtgp32.h"
#include "rocrand/rocrand_philox4x32_10.h"
#include "rocrand/rocrand_scrambled_sobol32.h"
#include "rocrand/rocrand_scrambled_sobol64.h"
#include "rocrand/rocrand_sobol32.h"
#include "rocrand/rocrand_sobol64.h"
#include "rocrand/rocrand_threefry2x32_20.h"
#include "rocrand/rocrand_threefry2x64_20.h"
#include "rocrand/rocrand_threefry4x32_20.h"
#include "rocrand/rocrand_threefry4x64_20.h"
#include "rocrand/rocrand_xorwow.h"

#include "rocrand/rocrand_uniform.h"
#include "rocrand/rocrand_normal.h"
#include "rocrand/rocrand_log_normal.h"
#include "rocrand/rocrand_poisson.h"
#include "rocrand/rocrand_discrete.h"
// IWYU pragma: end_exports

#endif // ROCRAND_KERNEL_H_
