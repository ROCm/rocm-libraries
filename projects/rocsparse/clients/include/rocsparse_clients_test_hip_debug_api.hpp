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
#include <mutex>

namespace rocsparse_clients_test
{

    std::string hip_debug_api_t2string(int32_t value);

    struct hip_debug_api_history_t
    {
    protected:
        int32_t                     m_api_value{};
        uint64_t                    m_ncalls{};
        std::map<int32_t, uint64_t> m_histo_calls{};
        mutable std::mutex          m_mutex{};

    public:
        int32_t  get_api_value() const;
        uint64_t get_ncalls() const;

        uint64_t get_calls(int32_t) const;
        void     add_call(int32_t);
        hip_debug_api_history_t(int32_t);
        hip_debug_api_history_t() = default;
        // Note:
        // The mutex member is neither copyable nor movable. Provide explicit
        // copy/move operations that synchronize access to the source object
        // and leave the destination's mutex default-constructed.
        hip_debug_api_history_t(const hip_debug_api_history_t& other);
        hip_debug_api_history_t(hip_debug_api_history_t&& other) noexcept;
        hip_debug_api_history_t& operator=(const hip_debug_api_history_t& other);
        hip_debug_api_history_t& operator=(hip_debug_api_history_t&& other) noexcept;
    };
}

#endif
