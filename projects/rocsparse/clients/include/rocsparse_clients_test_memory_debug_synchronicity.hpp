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
#ifdef GOOGLE_TEST

#include "rocsparse-debugging.h"

#include <map>

namespace rocsparse_clients_test
{

    std::string memory_debug_synchronicity_t2string(int32_t value);

    struct memory_debug_synchronicity_info_t
    {
    protected:
        int32_t                     m_synchronicity_value{};
        uint64_t                    m_ncalls{};
        std::map<int32_t, uint64_t> m_histo_calls{};

    public:
        int32_t  get_synchronicity_value() const;
        uint64_t get_ncalls() const;

        uint64_t get_calls(int32_t) const;
        void     add_call(int32_t);
        memory_debug_synchronicity_info_t(int32_t);
        memory_debug_synchronicity_info_t() = default;
    };
}

#endif
