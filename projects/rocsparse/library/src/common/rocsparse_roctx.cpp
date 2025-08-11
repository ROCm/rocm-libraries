/*! \file */

/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-roctx.h"

#include "rocsparse_control.hpp"
#include "rocsparse_envariables.hpp"
#include <map>
#include <mutex>

#include "rocsparse_roctx.hpp"

static std::mutex s_mutex;

bool rocsparse::roctx_variables_st::get_roctx_enabled() const
{
    return this->roctx_enabled;
}

void rocsparse::roctx_variables_st::set_roctx_enabled(bool value)
{
    if(value != this->roctx_enabled)
    {
        // LCOV_EXCL_START
        s_mutex.lock();
        this->roctx_enabled = value;
        s_mutex.unlock();
        // LCOV_EXCL_STOP
    }
}

extern "C" {

// LCOV_EXCL_START
void rocsparse_enable_roctx()
{
    rocsparse_roctx_variables.set_roctx_enabled(true);
}

void rocsparse_disable_roctx()
{
    rocsparse_roctx_variables.set_roctx_enabled(false);
}

int rocsparse_state_roctx()
{
    return rocsparse_roctx_variables.get_roctx_enabled() ? 1 : 0;
}
// LCOV_EXCL_STOP
}
