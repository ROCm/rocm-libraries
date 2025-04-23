/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCBLAS_MACROS_H
#define ROCBLAS_MACROS_H

#ifndef ROCBLAS_NO_DEPRECATED_WARNINGS
#ifndef ROCBLAS_DEPRECATED_MSG
#ifndef _MSC_VER
#define ROCBLAS_DEPRECATED_MSG(MSG) __attribute__((deprecated(#MSG)))
#else
#define ROCBLAS_DEPRECATED_MSG(MSG) __declspec(deprecated(#MSG))
#endif
#endif
#else
#ifndef ROCBLAS_DEPRECATED_MSG
#define ROCBLAS_DEPRECATED_MSG(MSG)
#endif
#endif

/*
   ROCBLAS_VA_OPT_PRAGMA(pragma, ...) creates a _Pragma with stringized pragma
   if the trailing argument list is non-empty.

   __VA_OPT__ support is automatically detected if it's available; otherwise,
   the GCC/Clang ##__VA_ARGS__ extension is used to emulate it.
    ********************************************************************/
#define ROCBLAS_VA_OPT_3RD_ARG(_1, _2, _3, ...) _3
#define ROCBLAS_VA_OPT_SUPPORTED(...) ROCBLAS_VA_OPT_3RD_ARG(__VA_OPT__(, ), 1, 0, )

#if ROCBLAS_VA_OPT_SUPPORTED(?)
#define ROCBLAS_VA_OPT_PRAGMA(pragma, ...) __VA_OPT__(_Pragma(#pragma))

#else // ROCBLAS_VA_OPT_SUPPORTED

#define ROCBLAS_VA_OPT_COUNT_IMPL(X, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define ROCBLAS_VA_OPT_COUNT(...) \
    ROCBLAS_VA_OPT_COUNT_IMPL(, ##__VA_ARGS__, N, N, N, N, N, N, N, N, N, N, 0)
#define ROCBLAS_VA_OPT_PRAGMA_SELECT0(...)
#define ROCBLAS_VA_OPT_PRAGMA_SELECTN(pragma, ...) _Pragma(#pragma)
#define ROCBLAS_VA_OPT_PRAGMA_IMPL2(pragma, count) ROCBLAS_VA_OPT_PRAGMA_SELECT##count(pragma)
#define ROCBLAS_VA_OPT_PRAGMA_IMPL(pragma, count) ROCBLAS_VA_OPT_PRAGMA_IMPL2(pragma, count)
#define ROCBLAS_VA_OPT_PRAGMA(pragma, ...) \
    ROCBLAS_VA_OPT_PRAGMA_IMPL(pragma, ROCBLAS_VA_OPT_COUNT(__VA_ARGS__))

#endif // ROCBLAS_VA_OPT_SUPPORTED

#endif /* ROCBLAS_MACROS_H */
