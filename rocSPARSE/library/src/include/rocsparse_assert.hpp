/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "debug.h"

#ifndef NDEBUG

#define rocsparse_host_assert(cond, msg)                                                        \
    (void)((cond)                                                                               \
           || (((void)printf("%s:%s:%u: rocSPARSE failed assertion `" #cond "', message: " #msg \
                             "\n",                                                              \
                             __FILE__,                                                          \
                             __FUNCTION__,                                                      \
                             __LINE__),                                                         \
                abort()),                                                                       \
               0))

#define rocsparse_device_assert(cond, msg) rocsparse_host_assert(cond, msg)

#else

#define rocsparse_host_assert(cond, msg)                                          \
    rocsparse_debug_variables.get_debug_force_host_assert()                       \
        ? (void)((cond)                                                           \
                 || (((void)printf("%s:%s:%u: rocSPARSE failed assertion `" #cond \
                                   "', message: " #msg "\n",                      \
                                   __FILE__,                                      \
                                   __FUNCTION__,                                  \
                                   __LINE__),                                     \
                      abort()),                                                   \
                     0))                                                          \
        : (void)0

#define rocsparse_device_assert(cond, msg) ((void)0)

#endif
