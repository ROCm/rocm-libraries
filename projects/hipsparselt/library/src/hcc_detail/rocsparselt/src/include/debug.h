/*! \file */
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
static constexpr const char* s_rocsparselt_debug_str = "//rocsparselt.degug: ";

//
// Trace message..
//
#undef ROCSPARSELT_DEBUG_VERBOSE
#define ROCSPARSELT_DEBUG_VERBOSE(msg__)                                                          \
    do                                                                                            \
    {                                                                                             \
        std::cout << s_rocsparselt_debug_str << std::endl;                                        \
        std::cout << s_rocsparselt_debug_str << "verbose" << std::endl;                           \
        std::cout << s_rocsparselt_debug_str << "  function : '" << __FUNCTION__ << "'"           \
                  << std::endl;                                                                   \
        std::cout << s_rocsparselt_debug_str << "  file     : '" << __FILE__ << "'" << std::endl; \
        std::cout << s_rocsparselt_debug_str << "  line     : " << __LINE__ << std::endl;         \
        std::cout << s_rocsparselt_debug_str << "  message  : " << msg__ << std::endl;            \
        std::cout << s_rocsparselt_debug_str << std::endl;                                        \
    } while(false)

inline rocsparselt_status rocsparselt_return_status_trace(const char*        function,
                                                          const char*        file,
                                                          const int          line,
                                                          rocsparselt_status status)
{
    if(rocsparselt_status_success != status)
    {
        std::cerr << s_rocsparselt_debug_str << std::endl
                  << s_rocsparselt_debug_str << "invalid status" << std::endl
                  << s_rocsparselt_debug_str << "function       : '" << function << "'" << std::endl
                  << s_rocsparselt_debug_str << "line           :  " << line << std::endl
                  << s_rocsparselt_debug_str << "file           : '" << file << "'" << std::endl
                  << s_rocsparselt_debug_str << std::endl;
    }
    return status;
}

#undef ROCSPARSELT_RETURN_STATUS
#define ROCSPARSELT_RETURN_STATUS(token__)  \
    return rocsparselt_return_status_trace( \
        __FUNCTION__, __FILE__, __LINE__, rocsparselt_status_##token__)

#endif // DEBUG_H
