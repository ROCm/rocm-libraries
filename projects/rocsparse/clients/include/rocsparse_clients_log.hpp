/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse.h"
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Teardown must not throw: a destructor that throws while another exception is
// unwinding calls std::terminate. Log the status and keep going.
inline void rocsparse_clients_log_hip_error(const char* call, hipError_t status)
{
    if(status != hipSuccess)
    {
        std::cerr << call << " failed: " << hipGetErrorString(status) << " ("
                  << static_cast<int>(status) << ")" << std::endl;
    }
}

inline void rocsparse_clients_log_rocsparse_error(const char* call, rocsparse_status status)
{
    if(status != rocsparse_status_success)
    {
        std::cerr << call << " failed: status " << static_cast<int>(status) << std::endl;
    }
}

// A failed allocation, copy, or create has to stop the caller. std::runtime_error
// is caught by the gtest harness and by rocsparse-test, and the message names the call.
inline void rocsparse_clients_throw_hip_error(const char* call, hipError_t status)
{
    if(status == hipSuccess)
    {
        return;
    }

    std::ostringstream message;
    message << call << " failed: " << hipGetErrorString(status) << " (" << static_cast<int>(status)
            << ")";
    throw std::runtime_error(message.str());
}
