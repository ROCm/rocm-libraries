// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_warp_exchange.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(T, BS, IT, WS, OP) \
    executor.queue<warp_exchange_benchmark<T, BS, IT, WS, OP>>();

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    CREATE_BENCHMARK(int32_t, 256, 1, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 1, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 32, 32, common::BlockedToStripedOp)

    CREATE_BENCHMARK(int32_t, 256, 1, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int32_t, 256, 1, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int32_t, 256, 32, 32, common::StripedToBlockedOp)

    CREATE_BENCHMARK(int32_t, 256, 1, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 1, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 32, 32, common::BlockedToStripedShuffleOp)

    CREATE_BENCHMARK(int32_t, 256, 1, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 1, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int32_t, 256, 32, 32, common::StripedToBlockedShuffleOp)

    CREATE_BENCHMARK(int32_t, 256, 1, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 1, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 4, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(int32_t, 256, 16, 32, ScatterToStripedOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::BlockedToStripedOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::StripedToBlockedOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::BlockedToStripedShuffleOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::StripedToBlockedShuffleOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, ScatterToStripedOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::BlockedToStripedOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::StripedToBlockedOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::BlockedToStripedShuffleOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::StripedToBlockedShuffleOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, ScatterToStripedOp)

    int hip_device = 0;
    HIP_CHECK(::rocprim::detail::get_device_from_stream(hipStreamDefault, hip_device));

    unsigned int warp_size;
    HIP_CHECK(::rocprim::host_warp_size(hip_device, warp_size));

    if(warp_size >= 64)
    {
        CREATE_BENCHMARK(int32_t, 256, 1, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(int32_t, 256, 4, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(int32_t, 256, 16, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(int32_t, 256, 64, 64, common::BlockedToStripedOp)

        CREATE_BENCHMARK(int32_t, 256, 1, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(int32_t, 256, 4, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(int32_t, 256, 16, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(int32_t, 256, 64, 64, common::StripedToBlockedOp)

        CREATE_BENCHMARK(int32_t, 256, 1, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(int32_t, 256, 4, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(int32_t, 256, 16, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(int32_t, 256, 64, 64, common::BlockedToStripedShuffleOp)

        CREATE_BENCHMARK(int32_t, 256, 1, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(int32_t, 256, 4, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(int32_t, 256, 16, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(int32_t, 256, 64, 64, common::StripedToBlockedShuffleOp)

        CREATE_BENCHMARK(int32_t, 256, 1, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(int32_t, 256, 4, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(int32_t, 256, 16, 64, ScatterToStripedOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::BlockedToStripedOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::StripedToBlockedOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::BlockedToStripedShuffleOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::StripedToBlockedShuffleOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, ScatterToStripedOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::BlockedToStripedOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::StripedToBlockedOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::BlockedToStripedShuffleOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::StripedToBlockedShuffleOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, ScatterToStripedOp)
    }

    executor.run();
}
