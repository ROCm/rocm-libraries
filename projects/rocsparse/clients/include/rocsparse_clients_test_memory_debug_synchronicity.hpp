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
    enum class memory_debug_synchronicity_t
    {
        unknown = 0, // It corresponds to unknown, or error depending on the context.
        synchronous
        = rocsparse_memory_debug_synchronicity_sync, // After the function returns, the queue of the stream is empty.
        asynchronous
        = rocsparse_memory_debug_synchronicity_async, // After the function queues non-blocking only operation on the stream, there is no guarantee that the stream is empty.
        partially_synchronous
        = rocsparse_memory_debug_synchronicity_psync, // The function has a stream synchronization point, but as opposed to a synchronous function, the last operation queued on the stream is a non-blocking function. There is no guarantee that the stream is empty.
        depends
        = rocsparse_memory_debug_synchronicity_sync | rocsparse_memory_debug_synchronicity_host
          | rocsparse_memory_debug_synchronicity_async
          | rocsparse_memory_debug_synchronicity_psync, // It depends on the input configuration of the routine.
        host = rocsparse_memory_debug_synchronicity_host, // No operation on the stream is queued.
        host_or_synchronous
        = rocsparse_memory_debug_synchronicity_sync | rocsparse_memory_debug_synchronicity_host,
        host_or_asynchronous
        = rocsparse_memory_debug_synchronicity_async | rocsparse_memory_debug_synchronicity_host,
        host_or_partially_synchronous
        = rocsparse_memory_debug_synchronicity_psync | rocsparse_memory_debug_synchronicity_host
    };

    const char* memory_debug_synchronicity_t2string(memory_debug_synchronicity_t sync);

    struct memory_debug_synchronicity_info_t
    {
    protected:
        memory_debug_synchronicity_t                     m_kind{};
        uint64_t                                         m_ncalls{};
        std::map<memory_debug_synchronicity_t, uint64_t> m_histo_calls{};

    public:
        memory_debug_synchronicity_t get_sync() const;
        uint64_t                     get_ncalls() const;
        uint64_t                     get_calls(memory_debug_synchronicity_t) const;
        void                         add_call(memory_debug_synchronicity_t);
        memory_debug_synchronicity_info_t(memory_debug_synchronicity_t);
        memory_debug_synchronicity_info_t() = default;
    };
}

#endif
