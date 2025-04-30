/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-types.h"

namespace rocsparse
{
    //
    // Log a message.
    //
    void message(const char* msg_, const char* function_, const char* file_, int line_);

    //
    // Log a warning message.
    //
    void warning_message(const char* msg_, const char* function_, const char* file_, int line_);

    //
    // Log an error message.
    //
    void error_message(rocsparse_status status_,
                       const char*      msg_,
                       const char*      function_,
                       const char*      file_,
                       int              line_);

#define ROCSPARSE_MESSAGE(MESSAGE__) rocsparse::message(MESSAGE__, __FUNCTION__, __FILE__, __LINE__)
#define ROCSPARSE_WARNING_MESSAGE(MESSAGE__) \
    rocsparse::warning_message(MESSAGE__, __FUNCTION__, __FILE__, __LINE__)
#define ROCSPARSE_ERROR_MESSAGE(STATUS__, MESSAGE__) \
    rocsparse::error_message(STATUS__, MESSAGE__, __FUNCTION__, __FILE__, __LINE__)
}
