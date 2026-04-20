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

#include "rocsparse_debug.hpp"
#include <atomic>
#include <map>

namespace rocsparse
{
    struct memory_debug_t
    {
    public:
        typedef enum func_t_
        {
            hip_malloc,
            hip_free,
            hip_malloc_async,
            hip_free_async,
            hip_memcpy,
            hip_memcpy_async,
            hip_memcpy2d_async,
            hip_memset,
            hip_memset_async,
            hip_stream_synchronize,
            hip_device_synchronize,
            hip_launch_kernel
        } func_t;
        static constexpr int32_t s_func_size = 12;

        struct info_t
        {
        public:
            std::atomic<int64_t> m_hip_stack_count{0};
            std::atomic<int64_t> m_hip_count_calls[s_func_size]{};
            std::atomic<int64_t> m_hip_ncalls{0};
            std::atomic<func_t>  m_last_hip_call{(func_t)-1};
            std::atomic<double>  m_gib{0};

            func_t get_last_hip_call() const;
            void   set_last_hip_call(func_t);

            int64_t get_hip_ncalls() const;
            void    set_hip_ncalls(int64_t);

            int64_t get_hip_ncalls(func_t) const;
            void    set_hip_ncalls(func_t, int64_t);

            void flag_hip_launch_kernel();

            bool   is_hip_memory_stack_clean() const;
            bool   hit_hip_stream_synchronize() const;
            bool   hit_hip_device_synchronize() const;
            bool   hit_hip_synchronize() const;
            void   info() const;
            void   reset();
            void   register_call(func_t f);
            void   add_data_transfer(size_t size_in_bytes);
            double get_data_transfer_in_gib() const;
            info_t()  = default;
            ~info_t() = default;
            hipError_t call_memcpy(void*         target,
                                   const void*   source,
                                   size_t        size_in_bytes,
                                   hipMemcpyKind kind);

            hipError_t call_memcpy_async(void*         target,
                                         const void*   source,
                                         size_t        size_in_bytes,
                                         hipMemcpyKind kind,
                                         hipStream_t   stream);
            hipError_t call_memcpy2D_async(void*         target,
                                           size_t        tpitch,
                                           const void*   source,
                                           size_t        spitch,
                                           size_t        width,
                                           size_t        height,
                                           hipMemcpyKind kind,
                                           hipStream_t   stream);

            hipError_t call_memset(void* target, int value, size_t size_in_bytes);

            hipError_t call_memset_async(void*       target,
                                         int         value,
                                         size_t      size_in_bytes,
                                         hipStream_t stream);

            hipError_t call_device_synchronize();

            hipError_t call_stream_synchronize(hipStream_t stream);

            hipError_t call_malloc_async(void** p_that, size_t size_in_bytes, hipStream_t stream);

            hipError_t call_free_async(void* that, hipStream_t stream);

            hipError_t call_malloc(void** p_that, size_t size_in_bytes);

            hipError_t call_free(void* that);
        };

        static info_t& get_info(hipStream_t);
        static void    reset(hipStream_t);

        ~memory_debug_t();

    private:
        std::map<hipStream_t, info_t> m_stream2info{};
        memory_debug_t() = default;
        static memory_debug_t& instance();
    };

}
