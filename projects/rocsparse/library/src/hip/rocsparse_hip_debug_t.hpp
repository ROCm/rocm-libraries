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

#include "rocsparse-debugging.h"
#include "rocsparse_debug.hpp"

#include <list>
#include <map>
#include <shared_mutex>
#include <thread>
#include <tuple>

namespace rocsparse
{
    namespace hip
    {
        /**
       @brief The structure debug_t takes care of the history of HIP memory management function calls.
    */
        struct debug_t
        {
        public:
            struct code_trace_t
            {
                const char* m_function{};
                const char* m_filename{};
                int32_t     m_line{};
            };

            /**
	 @brief Enumerate HIP memory management functions.
      */
            struct api_t
            {
                typedef enum
                {
                    unknown              = rocsparse_hip_debug_api_unknown,
                    hipMalloc            = rocsparse_hip_debug_api_hipMalloc,
                    hipFree              = rocsparse_hip_debug_api_hipFree,
                    hipMallocAsync       = rocsparse_hip_debug_api_hipMallocAsync,
                    hipFreeAsync         = rocsparse_hip_debug_api_hipFreeAsync,
                    hipMemcpy            = rocsparse_hip_debug_api_hipMemcpy,
                    hipMemcpyAsync       = rocsparse_hip_debug_api_hipMemcpyAsync,
                    hipMemcpy2DAsync     = rocsparse_hip_debug_api_hipMemcpy2DAsync,
                    hipMemset            = rocsparse_hip_debug_api_hipMemset,
                    hipMemsetAsync       = rocsparse_hip_debug_api_hipMemsetAsync,
                    hipStreamSynchronize = rocsparse_hip_debug_api_hipStreamSynchronize,
                    hipDeviceSynchronize = rocsparse_hip_debug_api_hipDeviceSynchronize,
                    hipLaunchKernelGGL   = rocsparse_hip_debug_api_hipLaunchKernelGGL
                } type_t;

                static constexpr int32_t size = 12;
            };

            /**
	 @brief debug_t::api_history_item_t stores the function and the stream.
	 The stream can be null, meaning the operation is applied on the default stream.
      */
            struct api_history_item_t
            {
                api_t::type_t m_func{};
                hipStream_t   m_stream{};
                code_trace_t  m_code_trace{};
            };

            /**
	 @brief history
      */
            struct api_history_t
            {
            private:
                int64_t       m_hip_stack_count{0};
                int64_t       m_hip_count_calls[api_t::size]{};
                int64_t       m_hip_ncalls{0};
                api_t::type_t m_last_hip_call{(api_t::type_t)-1};
                double        m_gib{0};

            public:
                std::list<api_history_item_t> m_api_history{};
                void register_call(api_t::type_t f, hipStream_t, const code_trace_t&);
                const api_history_item_t& get_last() const;
                api_history_item_t&       get_last();

                api_t::type_t get_last_hip_call() const;

                int64_t get_hip_ncalls() const;
                void    set_hip_ncalls(int64_t);

                int64_t get_hip_ncalls(api_t::type_t) const;
                void    set_hip_ncalls(api_t::type_t, int64_t);

                bool is_hip_memory_stack_clean() const;
                bool hit_hip_stream_synchronize() const;
                bool hit_hip_device_synchronize() const;
                bool hit_hip_synchronize() const;
                void info() const;
                void reset();

                void   flag_hip_launch_kernel(hipStream_t stream);
                void   add_data_transfer(size_t size_in_bytes);
                double get_data_transfer_in_gib() const;

                api_history_t()  = default;
                ~api_history_t() = default;
            };

            static api_history_t& get_api_history(rocsparse_handle);
            static void           reset(rocsparse_handle);
            static api_history_t* find_api_history(rocsparse_handle);
            static api_history_t* ask_api_history(rocsparse_handle);
            static api_history_t* find_api_history(hipStream_t);
            static bool           enabled();
            static void           disable();
            static void           enable();
            ~debug_t();
            static void       tag_hipLaunchKernelGGL(hipStream_t                                  stream,
                                                     const rocsparse::hip::debug_t::code_trace_t& trace);
            static hipError_t hipMemcpy(void*               target,
                                        const void*         source,
                                        size_t              size_in_bytes,
                                        hipMemcpyKind       kind,
                                        const code_trace_t& trace);

            static hipError_t hipMemcpyAsync(void*               target,
                                             const void*         source,
                                             size_t              size_in_bytes,
                                             hipMemcpyKind       kind,
                                             hipStream_t         stream,
                                             const code_trace_t& trace);

            static hipError_t hipMemcpy2DAsync(void*         target,
                                               size_t        tpitch,
                                               const void*   source,
                                               size_t        spitch,
                                               size_t        width,
                                               size_t        height,
                                               hipMemcpyKind kind,
                                               hipStream_t   stream,

                                               const code_trace_t& trace);

            static hipError_t hipMemset(void*   target,
                                        int32_t value,
                                        size_t  size_in_bytes,

                                        const code_trace_t& trace);

            static hipError_t hipMemsetAsync(void*       target,
                                             int         value,
                                             size_t      size_in_bytes,
                                             hipStream_t stream,

                                             const code_trace_t& trace);

            static hipError_t hipDeviceSynchronize(const code_trace_t& trace);

            static hipError_t hipStreamSynchronize(hipStream_t stream, const code_trace_t& trace);

            static hipError_t hipMallocAsync(void**              p_that,
                                             size_t              size_in_bytes,
                                             hipStream_t         stream,
                                             const code_trace_t& trace);

            static hipError_t
                hipFreeAsync(void* that, hipStream_t stream, const code_trace_t& trace);

            static hipError_t
                hipMalloc(void** p_that, size_t size_in_bytes, const code_trace_t& trace);

            static hipError_t hipFree(void* that, const code_trace_t& trace);
            static void register_call(api_t::type_t f, hipStream_t, const code_trace_t& trace);

            mutable std::shared_mutex shared_mutex{};
            using key_t = std::tuple<std::thread::id, int32_t>;

            std::map<key_t, api_history_t> m_api_history{};

        private:
            bool m_enable{};
            debug_t() = default;

        public:
            static debug_t& instance();
        };
    }

}

std::ostream& operator<<(std::ostream&, const rocsparse::hip::debug_t::api_t::type_t&);

std::ostream& operator<<(std::ostream&, const rocsparse::hip::debug_t::api_history_item_t&);

std::ostream& operator<<(std::ostream&, const rocsparse::hip::debug_t::code_trace_t&);

std::ostream& operator<<(std::ostream&, const rocsparse::hip::debug_t::api_history_t&);

std::ostream& operator<<(std::ostream&, const rocsparse::hip::debug_t&);
