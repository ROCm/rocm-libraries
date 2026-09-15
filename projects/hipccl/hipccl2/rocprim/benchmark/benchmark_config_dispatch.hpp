// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include <rocprim/device/config_types.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

// Based on Google Benchmark's DoNotOptimize()
template<typename T>
inline void DoNotOptimize(T&& value)
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

enum class stream_kind
{
    default_stream,
    per_thread_stream,
    explicit_stream,
    async_stream
};

__global__
void empty_kernel_device()
{}

struct config_dispatch_benchmark : public primbench::benchmark_interface
{
    config_dispatch_benchmark(std::string_view method) : m_method(method) {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "na")
            .add("algo", "config_dispatch")
            .add("method", m_method)
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;

        if(m_method == "empty_kernel")
        {
            state.set_items(1);
            state.add_reads<char>(1);

            state.run([&] { empty_kernel_device<<<dim3(1), dim3(1), 0, stream>>>(); });

            return;
        }

        hipStream_t local_stream   = 0;
        bool        created_stream = false;

        if(m_method == "default_stream")
        {
            local_stream = 0;
        }
        else if(m_method == "per_thread_stream")
        {
            local_stream = hipStreamPerThread;
        }
        else if(m_method == "explicit_stream")
        {
            HIP_CHECK(hipStreamCreate(&local_stream));
            created_stream = true;
        }
        else if(m_method == "async_stream")
        {
            HIP_CHECK(hipStreamCreateWithFlags(&local_stream, hipStreamNonBlocking));
            created_stream = true;
        }
        else
        {
            std::cerr << "Unknown algo: '" << m_method << "'\n";
            exit(EXIT_FAILURE);
        }

        state.set_items(1);
        state.add_reads<char>(1);

        state.run(
            [&]
            {
                rocprim::detail::target_arch target_arch;
                HIP_CHECK(rocprim::detail::host_target_arch(local_stream, target_arch));
                DoNotOptimize(target_arch);
            });

        if(created_stream)
        {
            HIP_CHECK(hipStreamDestroy(local_stream));
        }
    }

private:
    std::string m_method;
};
