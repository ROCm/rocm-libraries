/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef ROCM_GUARD_ROCM_CONFIG_HPP
#define ROCM_GUARD_ROCM_CONFIG_HPP

namespace rocm {

#if !defined(ROCM_USE_CLANG_TIDY)
#define ROCM_INLINE_NS pocket_version_1
#endif

#ifdef __HIP_DEVICE_COMPILE__
// NOLINTNEXTLINE
#define ROCM_HIP_HOST_DEVICE __host__ __device__
#else
// NOLINTNEXTLINE
#define ROCM_HIP_HOST_DEVICE
#endif

#ifdef MIGRAPHX_WORKAROUND_BROKEN_DEDUCTION_GUIDE
// NOLINTNEXTLINE
#define ROCM_HIP_HOST_DEVICE_DEDUCTION_GUIDE ROCM_HIP_HOST_DEVICE
#else
#define ROCM_HIP_HOST_DEVICE_DEDUCTION_GUIDE
#endif

} // namespace rocm
#endif // ROCM_GUARD_ROCM_CONFIG_HPP
