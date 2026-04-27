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
#ifdef GOOGLE_TEST
#include <cstring>

#include "rocsparse-debugging.h"
#include "rocsparse_clients_test_memory_debug_synchronicity.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace rocsparse_clients_test
{
    std::string memory_debug_synchronicity_t2string(int32_t value)
    {
        if(value & rocsparse_memory_debug_synchronicity_host)
        {
            std::string name("host_only");
            if(value & rocsparse_memory_debug_synchronicity_sync)
                name += "_or_synchronous";
            if(value & rocsparse_memory_debug_synchronicity_psync)
                name += "_or_partially_synchronous";
            if(value & rocsparse_memory_debug_synchronicity_async)
                name += "_or_asynchronous";
            return name;
        }
        else if(value & rocsparse_memory_debug_synchronicity_sync)
        {
            std::string name("synchronous_only");
            if(value & rocsparse_memory_debug_synchronicity_psync)
                name += "_or_partially_synchronous";
            if(value & rocsparse_memory_debug_synchronicity_async)
                name += "_or_asynchronous";
            return name;
        }
        else if(value & rocsparse_memory_debug_synchronicity_psync)
        {
            std::string name("partially_synchronous_only");
            if(value & rocsparse_memory_debug_synchronicity_async)
                name += "_or_asynchronous";
            return name;
        }
        else if(value & rocsparse_memory_debug_synchronicity_async)
        {
            return std::string("asynchronous_only");
        }
        else
        {
            return std::string("unknown");
        }
    }

    int32_t memory_debug_synchronicity_info_t::get_synchronicity_value() const
    {
        std::lock_guard<std::mutex> lock(this->m_mutex);
        return this->m_synchronicity_value;
    }

    uint64_t memory_debug_synchronicity_info_t::get_ncalls() const
    {
        std::lock_guard<std::mutex> lock(this->m_mutex);
        return this->m_ncalls;
    }

    uint64_t memory_debug_synchronicity_info_t::get_calls(const int32_t value) const
    {
        std::lock_guard<std::mutex> lock(this->m_mutex);
        if(auto search = this->m_histo_calls.find(value); search != this->m_histo_calls.end())
            return search->second;
        else
            return 0;
    }

    void memory_debug_synchronicity_info_t::add_call(const int32_t value)
    {
        std::lock_guard<std::mutex> lock(this->m_mutex);
        this->m_histo_calls[value] += 1;
        ++this->m_ncalls;
    }

    memory_debug_synchronicity_info_t::memory_debug_synchronicity_info_t(int32_t s)
        : m_synchronicity_value(s)
    {
    }

    memory_debug_synchronicity_info_t::memory_debug_synchronicity_info_t(
        const memory_debug_synchronicity_info_t& other)
    {
        std::lock_guard<std::mutex> lock(other.m_mutex);
        this->m_synchronicity_value = other.m_synchronicity_value;
        this->m_ncalls              = other.m_ncalls;
        this->m_histo_calls         = other.m_histo_calls;
    }

    memory_debug_synchronicity_info_t::memory_debug_synchronicity_info_t(
        memory_debug_synchronicity_info_t&& other) noexcept
    {
        std::lock_guard<std::mutex> lock(other.m_mutex);
        this->m_synchronicity_value = other.m_synchronicity_value;
        this->m_ncalls              = other.m_ncalls;
        this->m_histo_calls         = std::move(other.m_histo_calls);
    }

    memory_debug_synchronicity_info_t&
        memory_debug_synchronicity_info_t::operator=(const memory_debug_synchronicity_info_t& other)
    {
        if(this == &other)
            return *this;
        // Lock both mutexes simultaneously to avoid deadlock from
        // concurrent assignments in the opposite direction.
        std::lock(this->m_mutex, other.m_mutex);
        std::lock_guard<std::mutex> lock_this(this->m_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock_other(other.m_mutex, std::adopt_lock);
        this->m_synchronicity_value = other.m_synchronicity_value;
        this->m_ncalls              = other.m_ncalls;
        this->m_histo_calls         = other.m_histo_calls;
        return *this;
    }

    memory_debug_synchronicity_info_t& memory_debug_synchronicity_info_t::operator=(
        memory_debug_synchronicity_info_t&& other) noexcept
    {
        if(this == &other)
            return *this;
        std::lock(this->m_mutex, other.m_mutex);
        std::lock_guard<std::mutex> lock_this(this->m_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock_other(other.m_mutex, std::adopt_lock);
        this->m_synchronicity_value = other.m_synchronicity_value;
        this->m_ncalls              = other.m_ncalls;
        this->m_histo_calls         = std::move(other.m_histo_calls);
        return *this;
    }
}
#endif
