// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Throughput benchmark for the rocFFT device-side, header-only FFT library
// (<rocfft/device/fft.hpp>).  Each thread block performs one single-block
// transform; a large batch of independent transforms is launched to saturate
// the GPU.
//
// For these lengths (32..512) the transform is firmly memory-bound: arithmetic
// intensity is ~1.5..2.8 flop/byte (single) -- far below the device roofline
// ridge point -- so the relevant ceiling is global-memory bandwidth, not peak
// FLOPs.  Per transform the kernel moves 2*N*sizeof(complex) bytes (read input
// + write output); we report the achieved effective bandwidth as a percentage
// of (a) an empirically measured device-to-device copy and (b) the theoretical
// peak derived from device properties.  GFLOP/s is reported for reference only.
//
// Notes:
//  - The per-buffer footprint (--mib, default 1024) should stay well above the
//    device's last-level/Infinity Cache so the measured bandwidth reflects HBM
//    traffic rather than cache hits.
//  - The prop-derived "theoretical peak" needs a memory-type-specific
//    multiplier that HIP does not expose (x2 HBM vs ~x16 GDDR6); pass the known
//    value with --peak (e.g. --peak 960) for an accurate %peak column.
//  - Interpreting precision: single precision is memory-bandwidth bound and
//    reaches ~90% of peak across the matrix.  Double precision is bound by the
//    FP64 datapath (and lower occupancy from higher VGPR/LDS usage) on GPUs
//    with reduced-rate FP64 (e.g. consumer RDNA: FP64 at 1:16 of FP32), where
//    it runs several times slower than single at identical byte volume; on
//    full-rate-FP64 parts (CDNA / data-center) it tracks the bandwidth ceiling
//    like single does.
//
// This target deliberately depends only on HIP: it does not link roc::rocfft,
// FFTW, or OpenMP.

#include <rocfft/device/fft.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using rocfft::device::Direction;
using rocfft::device::FFT;
using rocfft::device::Precision;
using rocfft::device::precision_traits;

#define HIP_CHECK(expr)                                 \
    do                                                  \
    {                                                   \
        hipError_t status_ = (expr);                    \
        if(status_ != hipSuccess)                       \
        {                                               \
            std::fprintf(stderr,                        \
                         "HIP error %d at %s:%d: %s\n", \
                         int(status_),                  \
                         __FILE__,                      \
                         __LINE__,                      \
                         hipGetErrorString(status_));   \
            std::exit(EXIT_FAILURE);                    \
        }                                               \
    } while(0)

namespace
{

    // Batched FFT: one transform per thread block, indexed by blockIdx.x.
    template <typename FFTType>
    __global__ void fft_kernel(typename FFTType::complex_type* data)
    {
        constexpr unsigned ept  = FFTType::elements_per_thread;
        constexpr unsigned smem = FFTType::shared_memory_size;

        __shared__ char shared_mem[smem];

        auto* block_data = data + static_cast<size_t>(blockIdx.x) * FFTType::length;

        typename FFTType::complex_type thread_data[ept];
        for(unsigned i = 0; i < ept; ++i)
            thread_data[i] = block_data[threadIdx.x + i * FFTType::workgroup_size];

        FFTType::execute(thread_data, shared_mem, threadIdx.x);

        for(unsigned i = 0; i < ept; ++i)
            block_data[threadIdx.x + i * FFTType::workgroup_size] = thread_data[i];
    }

    // Coalesced grid-stride copy used to probe achievable device-to-device
    // bandwidth (the empirical ceiling).
    template <typename T>
    __global__ void copy_kernel(const T* __restrict__ in, T* __restrict__ out, size_t n)
    {
        for(size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
            i += static_cast<size_t>(gridDim.x) * blockDim.x)
            out[i] = in[i];
    }

    struct BenchOptions
    {
        int    device   = 0;
        int    ntrial   = 20;
        size_t mib      = 1024; // target per-buffer footprint in MiB
        int    cu_count = 0;
        // User-supplied theoretical peak bandwidth in GB/s.  The prop-derived
        // estimate is unreliable (it needs a memory-type-specific multiplier:
        // x2 for HBM, x16 for GDDR6, which HIP does not expose), so allow an
        // override (e.g. --peak 960 for a Radeon RX 7900 XTX).
        double peak_override = 0.0;
    };

    double min_of(const std::vector<float>& v)
    {
        double m = v.empty() ? 0.0 : v.front();
        for(float x : v)
            m = std::min(m, double(x));
        return m;
    }

    // Measure achievable device-to-device copy bandwidth (GB/s) over a buffer of
    // the requested footprint.  Traffic = read + write = 2 * bytes.
    double measure_copy_bandwidth(size_t bytes, int ntrial)
    {
        const size_t n = bytes / sizeof(float4);

        float4* d_in  = nullptr;
        float4* d_out = nullptr;
        HIP_CHECK(hipMalloc(&d_in, n * sizeof(float4)));
        HIP_CHECK(hipMalloc(&d_out, n * sizeof(float4)));
        HIP_CHECK(hipMemset(d_in, 1, n * sizeof(float4)));

        const unsigned block = 256;
        // Enough blocks to cover the buffer; the grid-stride loop saturates the
        // device without an artificial grid cap that would under-fill it.
        const unsigned grid
            = static_cast<unsigned>(std::min<size_t>((n + block - 1) / block, 1u << 22));

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        hipLaunchKernelGGL(copy_kernel<float4>, dim3(grid), dim3(block), 0, 0, d_in, d_out, n);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> times(ntrial);
        for(int t = 0; t < ntrial; ++t)
        {
            HIP_CHECK(hipEventRecord(start));
            hipLaunchKernelGGL(copy_kernel<float4>, dim3(grid), dim3(block), 0, 0, d_in, d_out, n);
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            HIP_CHECK(hipEventElapsedTime(&times[t], start, stop));
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_in));
        HIP_CHECK(hipFree(d_out));

        const double best_ms = min_of(times);
        const double traffic = 2.0 * double(n) * sizeof(float4);
        return traffic / (best_ms * 1e6); // GB/s
    }

    // Benchmark one (length, precision) configuration.
    template <unsigned Length, Precision P>
    void run_config(const BenchOptions& opt, double copy_bw, double peak_bw)
    {
        using FFT_t   = FFT<Length, P, Direction::Forward>;
        using complex = typename FFT_t::complex_type;

        const size_t target_bytes = opt.mib << 20;
        const size_t batch = std::max<size_t>(1, target_bytes / (size_t(Length) * sizeof(complex)));
        const size_t count = batch * Length;
        const size_t bytes = count * sizeof(complex);

        complex* d_data = nullptr;
        HIP_CHECK(hipMalloc(&d_data, bytes));
        HIP_CHECK(hipMemset(d_data, 0, bytes));

        int max_blocks_per_cu = 0;
        HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_cu,
            reinterpret_cast<const void*>(&fft_kernel<FFT_t>),
            FFT_t::workgroup_size,
            0));

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warm-up.
        hipLaunchKernelGGL(
            fft_kernel<FFT_t>, dim3(batch), dim3(FFT_t::workgroup_size), 0, 0, d_data);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> times(opt.ntrial);
        for(int t = 0; t < opt.ntrial; ++t)
        {
            HIP_CHECK(hipEventRecord(start));
            hipLaunchKernelGGL(
                fft_kernel<FFT_t>, dim3(batch), dim3(FFT_t::workgroup_size), 0, 0, d_data);
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            HIP_CHECK(hipEventElapsedTime(&times[t], start, stop));
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_data));

        const double best_ms = min_of(times);
        // Per transform: read N + write N complex elements.
        const double traffic = 2.0 * double(batch) * double(Length) * sizeof(complex);
        const double bw      = traffic / (best_ms * 1e6); // GB/s
        const double flops   = 5.0 * double(Length) * std::log2(double(Length)) * double(batch);
        const double gflops  = flops / (best_ms * 1e6);

        char name[32];
        std::snprintf(
            name, sizeof(name), "%u_%s", Length, P == Precision::Single ? "single" : "double");

        std::printf("%-14s %5u %5d %10zu %10.4f %9.1f %7.1f %7.1f %9.1f\n",
                    name,
                    FFT_t::workgroup_size,
                    max_blocks_per_cu,
                    batch,
                    best_ms,
                    bw,
                    100.0 * bw / copy_bw,
                    100.0 * bw / peak_bw,
                    gflops);
    }

    struct DeviceFFTConfig
    {
        unsigned  length;
        Precision prec;
    };

    template <typename F>
    void dispatch_config(const DeviceFFTConfig& cfg, F f)
    {
#define ROCFFT_DEVICE_BENCH_DISPATCH_CASE(LEN)               \
    case LEN:                                                \
        if(cfg.prec == Precision::Single)                    \
            f.template operator()<LEN, Precision::Single>(); \
        else                                                 \
            f.template operator()<LEN, Precision::Double>(); \
        break

        switch(cfg.length)
        {
            ROCFFT_DEVICE_BENCH_DISPATCH_CASE(32);
            ROCFFT_DEVICE_BENCH_DISPATCH_CASE(64);
            ROCFFT_DEVICE_BENCH_DISPATCH_CASE(128);
            ROCFFT_DEVICE_BENCH_DISPATCH_CASE(256);
            ROCFFT_DEVICE_BENCH_DISPATCH_CASE(512);
        default:
            std::fprintf(stderr, "unsupported length %u\n", cfg.length);
            break;
        }

#undef ROCFFT_DEVICE_BENCH_DISPATCH_CASE
    }

    const DeviceFFTConfig kMatrix[] = {
        {32, Precision::Single},
        {32, Precision::Double},
        {64, Precision::Single},
        {64, Precision::Double},
        {128, Precision::Single},
        {128, Precision::Double},
        {256, Precision::Single},
        {256, Precision::Double},
        {512, Precision::Single},
        {512, Precision::Double},
    };

} // namespace

int main(int argc, char** argv)
{
    BenchOptions opt;

    for(int i = 1; i < argc; ++i)
    {
        const std::string arg  = argv[i];
        auto              next = [&](int& dst) {
            if(i + 1 < argc)
                dst = std::atoi(argv[++i]);
        };
        auto next_sz = [&](size_t& dst) {
            if(i + 1 < argc)
                dst = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        };
        auto next_d = [&](double& dst) {
            if(i + 1 < argc)
                dst = std::strtod(argv[++i], nullptr);
        };
        if(arg == "--device")
            next(opt.device);
        else if(arg == "-N" || arg == "--ntrial")
            next(opt.ntrial);
        else if(arg == "--mib")
            next_sz(opt.mib);
        else if(arg == "--peak")
            next_d(opt.peak_override);
        else if(arg == "--help" || arg == "-h")
        {
            std::printf("Usage: rocfft-device-bench [--device id] [--ntrial N] [--mib MiB] "
                        "[--peak GBps]\n");
            return EXIT_SUCCESS;
        }
    }

    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        std::fprintf(stderr, "No HIP device available\n");
        return EXIT_FAILURE;
    }
    HIP_CHECK(hipSetDevice(opt.device));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, opt.device));
    opt.cu_count = props.multiProcessorCount;

    // Theoretical peak BW.  Prefer a user-supplied value (--peak); the
    // prop-derived estimate assumes a x2 (DDR/HBM) multiplier and is wrong for
    // GDDR6 parts (which need ~x16), so it is only a fallback.
    const double prop_peak_bw
        = 2.0 * double(props.memoryClockRate) * 1e3 * (double(props.memoryBusWidth) / 8.0) / 1e9;
    const bool   peak_known = opt.peak_override > 0.0;
    const double peak_bw    = peak_known ? opt.peak_override : prop_peak_bw;

    const double copy_bw = measure_copy_bandwidth(opt.mib << 20, opt.ntrial);

    std::printf("Device: %s (gfx arch %d.%d), %d WGPs\n",
                props.name,
                props.major,
                props.minor,
                props.multiProcessorCount);
    std::printf("Per-buffer footprint target: %zu MiB, trials: %d\n", opt.mib, opt.ntrial);
    std::printf("Achievable copy bandwidth: %.1f GB/s   Theoretical peak: %.1f GB/s%s\n\n",
                copy_bw,
                peak_bw,
                peak_known ? " (user-specified)" : " (prop-derived, approximate)");

    std::printf("%-14s %5s %5s %10s %10s %9s %7s %7s %9s\n",
                "config",
                "wgs",
                "b/CU",
                "batch",
                "time(ms)",
                "GB/s",
                "%copy",
                "%peak",
                "GFLOP/s");

    for(const auto& cfg : kMatrix)
        dispatch_config(
            cfg, [&]<unsigned L, Precision P>() { run_config<L, P>(opt, copy_bw, peak_bw); });

    return EXIT_SUCCESS;
}
