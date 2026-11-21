// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_permute.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/timer.hpp"
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string_view>

namespace {
using namespace gemm_permute;

void check_blas(rocblas_status status)
{
    if(status != rocblas_status_success)
    {
        std::fprintf(stderr, "rocBLAS: %s\n", rocblas_status_to_string(status));
        std::exit(EXIT_FAILURE);
    }
}

// Dyadic inputs keep every product and partial sum exact in FP32 regardless of reduction order.
__global__ void fill(Bf16* x, int n, uint32_t seed)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        uint32_t u = (uint32_t(i) ^ seed) * 0x9e3779b1u;
        u          = (u ^ (u >> 16)) * 0x85ebca6bu;
        x[i]       = ck_tile::float_to_bf16(float(int(u >> 28) - 8) * 0.25f);
    }
}

// Independent oracle: enumerate destination (b,t,d) and gather H heads from the dense result.
__global__ void permute_reference(const Bf16* dense, Bf16* y, int time, int batch)
{
    const int bt = blockIdx.x, b = bt / time, t = bt % time, d = threadIdx.x;
    const auto* src = dense + ((int64_t{t} * batch + b) * D + d) * H;
    alignas(16) Bf16 heads[H];
    __builtin_memcpy(heads, __builtin_assume_aligned(src, 16), sizeof(heads));
#pragma unroll
    for(int h = 0; h < H; ++h)
        y[(int64_t{h} * time * batch + bt) * D + d] = heads[h];
}

__global__ void compare(const Bf16* got, const Bf16* want, int n, unsigned* first)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if(ck_tile::bit_cast<uint16_t>(got[i]) != ck_tile::bit_cast<uint16_t>(want[i]))
        {
            atomicMin(first, static_cast<unsigned>(i));
            break;
        }
}

Bf16* data(const ck_tile::DeviceMem& mem) { return static_cast<Bf16*>(mem.GetDeviceBuffer()); }

bool run(rocblas_handle blas, hipStream_t stream, int time, int batch, bool benchmark)
{
    const int m = time * batch, n = m * N;
    const size_t bytes = size_t(n) * sizeof(Bf16);
    ck_tile::DeviceMem a(size_t(m) * K * sizeof(Bf16)), w(size_t(K) * N * sizeof(Bf16));
    std::array<uint16_t, D> guard{};
    ck_tile::DeviceMem dense(bytes), ref(bytes), out(bytes + 2 * sizeof(guard)),
        first(sizeof(unsigned));
    Bf16* y = data(out) + guard.size();
    fill<<<512, 256, 0, stream>>>(data(a), m * K, 13);
    HIP_CHECK_ERROR(hipGetLastError());
    fill<<<512, 256, 0, stream>>>(data(w), K * N, 79);
    HIP_CHECK_ERROR(hipGetLastError());

    auto fused   = [&] { HIP_CHECK_ERROR(launch(data(a), data(w), y, time, batch, stream)); };
    auto unfused = [&] {
        const float one = 1, zero = 0;
        constexpr auto bf = rocblas_datatype_bf16_r;
        // rocBLAS uses column-major storage: C^T = W^T A^T needs no input copies.
        check_blas((rocblas_gemm_ex)(blas,
                                     rocblas_operation_none,
                                     rocblas_operation_none,
                                     N,
                                     m,
                                     K,
                                     &one,
                                     data(w),
                                     bf,
                                     N,
                                     data(a),
                                     bf,
                                     K,
                                     &zero,
                                     data(dense),
                                     bf,
                                     N,
                                     data(dense),
                                     bf,
                                     N,
                                     rocblas_datatype_f32_r,
                                     rocblas_gemm_algo_standard,
                                     0,
                                     0));
        permute_reference<<<m, D, 0, stream>>>(data(dense), data(ref), time, batch);
        HIP_CHECK_ERROR(hipGetLastError());
    };
    HIP_CHECK_ERROR(hipMemsetAsync(data(out), 0xff, out.GetBufferSize(), stream));
    HIP_CHECK_ERROR(hipMemsetAsync(data(ref), 0, bytes, stream));
    HIP_CHECK_ERROR(hipMemsetAsync(first.GetDeviceBuffer(), 0xff, sizeof(unsigned), stream));
    unfused();
    fused();
    compare<<<512, 256, 0, stream>>>(
        y, data(ref), n, static_cast<unsigned*>(first.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));
    unsigned mismatch;
    first.FromDevice(&mismatch);
    if(mismatch != UINT32_MAX)
    {
        std::fprintf(stderr, "T=%d B=%d: first BF16 mismatch at Y[%u]\n", time, batch, mismatch);
        return false;
    }
    for(const Bf16* border : {data(out), y + n})
    {
        HIP_CHECK_ERROR(hipMemcpy(guard.data(), border, sizeof(guard), hipMemcpyDeviceToHost));
        if(!std::all_of(guard.begin(), guard.end(), [](uint16_t v) { return v == 0xffff; }))
        {
            std::fprintf(stderr, "T=%d B=%d: output guard overwritten\n", time, batch);
            return false;
        }
    }
    std::printf("T=%d B=%d: %d BF16 elements match bit-for-bit; guards intact\n", time, batch, n);
    if(!benchmark)
        return true;

    // Back-to-back launches keep host enqueue cost (larger for rocBLAS) off the timed span.
    constexpr int Launches = 20;
    ck_tile::gpu_timer timer;
    auto measure = [&](auto&& fn) {
        timer.start(stream);
        for(int i = 0; i < Launches; ++i)
            fn();
        timer.stop(stream);
        return 1000.f * timer.duration() / Launches;
    };
    for(int i = 0; i < 3; ++i)
    {
        fused();
        unfused();
    }
    std::array<float, 11> f{}, u{};
    for(size_t i = 0; i < f.size(); ++i)
    {
        if(i % 2 == 0)
            f[i] = measure(fused);
        u[i] = measure(unfused);
        if(i % 2 != 0)
            f[i] = measure(fused);
    }
    std::sort(f.begin(), f.end());
    std::sort(u.begin(), u.end());
    const auto mid = f.size() / 2;
    std::printf("Median of %zu samples (%d launches each): fused %.2f us; rocBLAS + copy %.2f us\n",
                f.size(),
                Launches,
                f[mid],
                u[mid]);
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    const bool benchmark = argc == 2 && std::string_view(argv[1]) == "--benchmark";
    if(argc != 1 && !benchmark)
    {
        std::fprintf(stderr, "usage: %s [--benchmark]\n", argv[0]);
        return EXIT_FAILURE;
    }
    hipDeviceProp_t prop{};
    int device;
    HIP_CHECK_ERROR(hipGetDevice(&device));
    HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, device));
    const std::string_view arch{prop.gcnArchName};
    if(arch.substr(0, arch.find(':')) != "gfx950")
    {
        std::fprintf(stderr, "requires gfx950; selected device is %s\n", prop.gcnArchName);
        return EXIT_FAILURE;
    }
    hipStream_t stream;
    rocblas_handle blas;
    HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    check_blas(rocblas_create_handle(&blas));
    check_blas(rocblas_set_stream(blas, stream));
    const bool ok = run(blas, stream, 1, 1, false) && run(blas, stream, BM + 3, 5, false) &&
                    (!benchmark || run(blas, stream, MaxT, MaxB, true));
    check_blas(rocblas_destroy_handle(blas));
    HIP_CHECK_ERROR(hipStreamDestroy(stream));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
