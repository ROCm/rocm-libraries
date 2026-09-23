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

//
// Only for the testing framework.
//
#ifdef GOOGLE_TEST
#include <cstring>

#include "rocsparse-debugging.h"
#include "rocsparse_clients_test_hip_debug_api.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace rocsparse_clients_test
{
    std::string hip_debug_api_t2string(int32_t value)
    {
        const char* names[] = {"host", "synchronous", "partially_synchronous", "asynchronous"};

        const int32_t bits[] = {rocsparse_hip_debug_api_history_none,
                                rocsparse_hip_debug_api_history_sync,
                                rocsparse_hip_debug_api_history_psync,
                                rocsparse_hip_debug_api_history_async};

        int         count = 0;
        const char* first = nullptr;
        std::string out;
        for(int i = 0; i < 4; ++i)
        {
            if(value & bits[i])
            {
                if(count == 0)
                {
                    first = names[i];
                    out   = names[i];
                }
                else
                {
                    out += "_or_";
                    out += names[i];
                }
                ++count;
            }
        }

        if(count == 0)
        {
            return std::string("unknown");
        }
        if((count == 1) && (value != rocsparse_hip_debug_api_history_none))
        {
            return std::string(first) + "_only";
        }
        return out;
    }

    int32_t hip_debug_api_history_t::get_api_value() const
    {
        std::lock_guard<std::mutex> lock(this->m_mutex);
        return this->m_api_value;
    }

    uint64_t hip_debug_api_history_t::get_ncalls() const
    {
        std::lock_guard<std::mutex> lock(this->m_mutex);
        return this->m_ncalls;
    }

    uint64_t hip_debug_api_history_t::get_calls(const int32_t value) const
    {
        std::lock_guard<std::mutex> lock(this->m_mutex);
        if(auto search = this->m_histo_calls.find(value); search != this->m_histo_calls.end())
            return search->second;
        else
            return 0;
    }

    void hip_debug_api_history_t::add_call(const int32_t value)
    {
        std::lock_guard<std::mutex> lock(this->m_mutex);
        this->m_histo_calls[value] += 1;
        ++this->m_ncalls;
    }

    hip_debug_api_history_t::hip_debug_api_history_t(int32_t s)
        : m_api_value(s)
    {
    }

    hip_debug_api_history_t::hip_debug_api_history_t(const hip_debug_api_history_t& other)
    {
        std::lock_guard<std::mutex> lock(other.m_mutex);
        this->m_api_value   = other.m_api_value;
        this->m_ncalls      = other.m_ncalls;
        this->m_histo_calls = other.m_histo_calls;
    }

    hip_debug_api_history_t::hip_debug_api_history_t(hip_debug_api_history_t&& other) noexcept
    {
        std::lock_guard<std::mutex> lock(other.m_mutex);
        this->m_api_value   = other.m_api_value;
        this->m_ncalls      = other.m_ncalls;
        this->m_histo_calls = std::move(other.m_histo_calls);
    }

    hip_debug_api_history_t&
        hip_debug_api_history_t::operator=(const hip_debug_api_history_t& other)
    {
        if(this == &other)
            return *this;
        std::lock(this->m_mutex, other.m_mutex);
        std::lock_guard<std::mutex> lock_this(this->m_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock_other(other.m_mutex, std::adopt_lock);
        this->m_api_value   = other.m_api_value;
        this->m_ncalls      = other.m_ncalls;
        this->m_histo_calls = other.m_histo_calls;
        return *this;
    }

    hip_debug_api_history_t&
        hip_debug_api_history_t::operator=(hip_debug_api_history_t&& other) noexcept
    {
        if(this == &other)
            return *this;
        std::lock(this->m_mutex, other.m_mutex);
        std::lock_guard<std::mutex> lock_this(this->m_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock_other(other.m_mutex, std::adopt_lock);
        this->m_api_value   = other.m_api_value;
        this->m_ncalls      = other.m_ncalls;
        this->m_histo_calls = std::move(other.m_histo_calls);
        return *this;
    }
}
#endif
