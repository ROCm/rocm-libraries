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

#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/config.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_scan.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

enum class scan_type
{
    inclusive_scan,
    exclusive_scan,
    broadcast
};

template<class Runner, class T, unsigned int VirtualWaveSize, unsigned int Trials>
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void kernel(const T* input, T* output, const T init)
{
    if constexpr(VirtualWaveSize <= rocprim::arch::wavefront::max_size())
    {
        Runner::template run<T, VirtualWaveSize, Trials>(input, output, init);
    }
}

struct inclusive_scan
{
    template<typename T, unsigned int VirtualWaveSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        (void)init;
        const unsigned int i     = blockIdx.x * blockDim.x + threadIdx.x;
        auto               value = input[i];

        using wscan_t = rocprim::warp_scan<T, VirtualWaveSize>;
        __shared__
        typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wscan_t().inclusive_scan(value, value, storage);
        }

        output[i] = value;
    }
};

struct exclusive_scan
{
    template<typename T, unsigned int VirtualWaveSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        const unsigned int i     = blockIdx.x * blockDim.x + threadIdx.x;
        auto               value = input[i];

        using wscan_t = rocprim::warp_scan<T, VirtualWaveSize>;
        __shared__
        typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wscan_t().exclusive_scan(value, value, init, storage);
        }

        output[i] = value;
    }
};

struct broadcast
{
    template<typename T, unsigned int VirtualWaveSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        (void)init;
        const unsigned int i        = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int warp_id  = i / VirtualWaveSize;
        const unsigned int src_lane = warp_id % VirtualWaveSize;
        auto               value    = input[i];

        using wscan_t = rocprim::warp_scan<T, VirtualWaveSize>;
        __shared__
        typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            value = wscan_t().broadcast(value, src_lane, storage);
        }

        output[i] = value;
    }
};

template<typename Subalgo>
std::string get_subalgo_name()
{
    if constexpr(std::is_same_v<Subalgo, inclusive_scan>)
        return "inclusive_scan";
    else if constexpr(std::is_same_v<Subalgo, exclusive_scan>)
        return "exclusive_scan";
    else if constexpr(std::is_same_v<Subalgo, broadcast>)
        return "broadcast";
    else
        static_assert(sizeof(Subalgo) == 0, "Unknown subalgo");
}

template<typename T,
         unsigned int BlockSize,
         unsigned int VirtualWaveSize,
         class Subalgo,
         unsigned int Trials = 100,
         typename Config     = rocprim::default_config>
struct warp_scan_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "warp")
            .add("algo", "warp_scan")
            .add("key_type", primbench::name<T>())
            .add("subalgo", get_subalgo_name<Subalgo>())
            .add("ws", VirtualWaveSize)
            .add("cfg", primbench::json{}.add("bs", BlockSize));
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;

        // Ensure items is a multiple of BlockSize
        size_t items = bytes / sizeof(T);
        items        = BlockSize * ((items + BlockSize - 1) / BlockSize);

        // Allocate and fill memory
        std::vector<T>        input(items, (T)1);
        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(items);

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        state.run(
            [&]
            {
                kernel<Subalgo, T, VirtualWaveSize, Trials>
                    <<<dim3(items / BlockSize), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                              d_output.get(),
                                                                              input[0]);
            });
    }
};
