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
#include <stdexcept>
#include <string>

namespace rocsparse_clients_test
{

    const char* memory_debug_synchronicity_t2string(memory_debug_synchronicity_t sync)
    {
        switch(sync)
        {

        case memory_debug_synchronicity_t::unknown:
        {
            return "unknown";
        }

        case memory_debug_synchronicity_t::synchronous:
        {
            return "synchronous";
        }

        case memory_debug_synchronicity_t::host_or_synchronous:
        {
            return "host_or_synchronous";
        }

        case memory_debug_synchronicity_t::host_or_asynchronous:
        {
            return "host_or_asynchronous";
        }

        case memory_debug_synchronicity_t::host_or_partially_synchronous:
        {
            return "host_or_partially_synchronous";
        }

        case memory_debug_synchronicity_t::partially_synchronous:
        {
            return "partially_synchronous";
        }

        case memory_debug_synchronicity_t::asynchronous:
        {
            return "asynchronous";
        }

        case memory_debug_synchronicity_t::host:
        {
            return "host";
        }
        case memory_debug_synchronicity_t::depends:
        {
            return "depends";
        }
        }
        return "internal_error";
    }

    memory_debug_synchronicity_t memory_debug_synchronicity_info_t::get_sync() const
    {
        return this->m_kind;
    }

    uint64_t memory_debug_synchronicity_info_t::get_ncalls() const
    {
        return this->m_ncalls;
    }

    uint64_t
        memory_debug_synchronicity_info_t::get_calls(const memory_debug_synchronicity_t value) const
    {
        if(auto search = this->m_histo_calls.find(value); search != this->m_histo_calls.end())
            return search->second;
        else
            return 0;
    }

    void memory_debug_synchronicity_info_t::add_call(const memory_debug_synchronicity_t value)
    {
        this->m_histo_calls[value] += 1;
        ++this->m_ncalls;
    }

    memory_debug_synchronicity_info_t::memory_debug_synchronicity_info_t(
        memory_debug_synchronicity_t s)
        : m_kind(s)
    {
    }
}
#endif
