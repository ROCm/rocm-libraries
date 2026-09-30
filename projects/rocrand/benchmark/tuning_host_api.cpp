// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "benchmark_host_api.hpp"

#include "primbench.hpp"

#include <vector>

namespace benchmark_tuning
{
// clang-format off
void queue_lfsr113        (primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_mrg31k3p       (primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_mrg32k3a       (primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_mt19937        (primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_mtgp32         (primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_philox4x32_10  (primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_threefry2x32_20(primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_threefry2x64_20(primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_threefry4x32_20(primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_threefry4x64_20(primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
void queue_xorwow         (primbench::executor&, size_t, size_t, bool, const std::vector<double>&);
// clang-format on
} // namespace benchmark_tuning

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 512 * primbench::MiB; // In bytes
    settings.min_gpu_ms_per_batch = 100;
    settings.hot                  = true;

    primbench::executor executor(argc, argv, settings, primbench::flags::sync);

    auto dimensions
        = executor.get<size_t>("dimensions", 1, "Number of dimensions of quasi-random values");

    auto offset = executor.get<size_t>("offset", 0, "Offset of generated pseudo-random values");

    auto poisson_lambdas = executor.get<std::vector<double>>(
        "lambda",
        {10.0},
        "Space-separated list of lambdas of Poisson distribution");

    auto benchmark_host
        = executor.get<bool>("host", false, "Run benchmarks on the host instead of on the device");

    using namespace benchmark_tuning;

    queue_lfsr113(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_mrg31k3p(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_mrg32k3a(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_mt19937(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_mtgp32(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_philox4x32_10(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_threefry2x32_20(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_threefry2x64_20(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_threefry4x32_20(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_threefry4x64_20(executor, dimensions, offset, benchmark_host, poisson_lambdas);
    queue_xorwow(executor, dimensions, offset, benchmark_host, poisson_lambdas);

    executor.run();
}
