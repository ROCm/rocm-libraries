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

#pragma once

#include "rocsparse-types.h"
#include "rocsparse_envariables.hpp"

namespace rocsparse
{
    struct roctx_variables_st
    {
    private:
        bool roctx_enabled;

    public:
        bool get_roctx_enabled() const;
        void set_roctx_enabled(bool value);
    };

    struct roctx_st
    {
    private:
        roctx_variables_st m_var{};

    public:
        static roctx_st& instance()
        {
            static roctx_st self;
            return self;
        }

        static roctx_variables_st& var()
        {
            return instance().m_var;
        }

        ~roctx_st() = default;

    private:
        roctx_st()
        {
            const bool roctx_enabled = ROCSPARSE_ENVARIABLES.get(rocsparse::envariables::ROCTX);
            m_var.set_roctx_enabled(roctx_enabled);
        };
    };

#define rocsparse_roctx_variables rocsparse::roctx_st::instance().var()
}
