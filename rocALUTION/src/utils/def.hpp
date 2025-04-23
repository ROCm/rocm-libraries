/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_UTILS_DEF_HPP_
#define ROCALUTION_UTILS_DEF_HPP_

// Uncomment to define verbose level
#define VERBOSE_LEVEL 2

// Uncomment for debug mode
// #define DEBUG_MODE

// Uncomment to disable the assert()s
// #define ASSERT_OFF

// Uncomment to log only on specific MPI rank
// When logging into a file, this will be unset
#define LOG_MPI_RANK 0

// Comment to enable automatic object tracking
#define OBJ_TRACKING_OFF

// ******************
// ******************
// Do not edit below!
// ******************
// ******************

#ifdef ASSERT_OFF
#define assert(a) ;
#else
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif
#endif

#ifdef DEBUG_MODE
#define assert_dbg(a) assert(a)
#else
#define assert_dbg(a) ;
#endif

#define SUPPORT_COMPLEX

#endif // ROCALUTION_UTILS_DEF_HPP_
