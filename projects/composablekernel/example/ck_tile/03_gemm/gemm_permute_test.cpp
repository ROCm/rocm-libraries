// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_permute.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/host/timer.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/epilogue.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string_view>

namespace {
using namespace gemm_permute;

// Unfused baseline: the two existing CK-Tile ops the fused kernel replaces, launched back to back.
// The GEMM op is tuned for this shape and shares the fused kernel's tile, mainloop, epilogue and
// tile order; it writes the dense intermediate C[t*B+b, d*H+h].
using DenseTraits =
    ck_tile::TileGemmUniversalTraits<true, false, false, false, Row, Row, Row, true>;
using DenseProblem = ck_tile::UniversalGemmPipelineProblem<Bf16, Bf16, float, Shape, DenseTraits>;
using GemmOp =
    ck_tile::GemmKernel<Partitioner, ck_tile::GemmPipelineAgBgCrCompV3<DenseProblem>, Epilogue>;
// Best existing CK-Tile op for the permute: ElementWiseKernel as a strided PassThrough copy that
// walks C order [t,b,d,h]; 16 B loads of the 8 heads, BF16 stores coalesced over d per wave.
// 4 warps x 64 lanes x 8 elements x 4 repeats = one 8192-element block tile.
using PermuteOp = ck_tile::ElementWiseKernel<
    ck_tile::ElementWisePipelineProblem<Bf16,
                                        Bf16,
                                        Bf16,
                                        ck_tile::ElementWiseShape<ck_tile::sequence<4>,
                                                                  ck_tile::sequence<8192>,
                                                                  ck_tile::sequence<512>,
                                                                  Bf16>,
                                        ck_tile::element_wise::PassThrough,
                                        true>,
    ck_tile::ElementWiseDefaultPolicy>;

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

bool run(hipStream_t stream, int time, int batch, bool benchmark)
{
    const int m = time * batch, n = m * N;
    const size_t bytes = size_t(n) * sizeof(Bf16);
    ck_tile::DeviceMem a(size_t(m) * K * sizeof(Bf16)), w(size_t(K) * N * sizeof(Bf16));
    std::array<uint16_t, D> guard{};
    ck_tile::DeviceMem dense(bytes), two_op_y(bytes), ref(bytes), out(bytes + 2 * sizeof(guard)),
        first(sizeof(unsigned));
    Bf16* y = data(out) + guard.size();
    fill<<<512, 256, 0, stream>>>(data(a), m * K, 13);
    HIP_CHECK_ERROR(hipGetLastError());
    fill<<<512, 256, 0, stream>>>(data(w), K * N, 79);
    HIP_CHECK_ERROR(hipGetLastError());

    const ck_tile::stream_config config{stream};
    const auto gemm_args = GemmOp::MakeKernelArgs(
        ck_tile::GemmHostArgs(data(a), data(w), data(dense), 1, m, N, K, K, N, N));
    if(!GemmOp::IsSupportedArgument(gemm_args))
    {
        std::fprintf(stderr, "T=%d B=%d: CK-Tile GEMM op rejects the problem\n", time, batch);
        return false;
    }
    // Dense C[t,b,d,h] -> Y[h,b,t,d], as torch.permute(3, 1, 0, 2): lengths in C order, C strides
    // in, Y strides out.
    const auto permute_lens        = ck_tile::make_tuple(time, batch, D, H);
    const auto permute_in_strides  = ck_tile::make_tuple(batch * D * H, D * H, H, 1);
    const auto permute_out_strides = ck_tile::make_tuple(D, time * D, 1, batch * time * D);
    const int permute_grid =
        ck_tile::integer_divide_ceil(n, PermuteOp::Problem::BlockShape::kBlockM);

    // The fused path includes the weight reorder on every call, so both paths take the same inputs.
    ck_tile::DeviceMem wp(size_t(K) * N * sizeof(Bf16));
    auto fused = [&] {
        HIP_CHECK_ERROR(prepermute_weight(data(w), data(wp), stream));
        HIP_CHECK_ERROR(launch(data(a), data(wp), y, time, batch, stream));
    };
    auto gemm_op = [&] {
        ck_tile::launch_kernel(
            config,
            ck_tile::make_kernel<1>(
                GemmOp{}, GemmOp::GridSize(m, N, 1), GemmOp::BlockSize(), 0, gemm_args));
        HIP_CHECK_ERROR(hipGetLastError());
    };
    auto two_ops = [&] {
        gemm_op();
        ck_tile::launch_kernel(config,
                               ck_tile::make_kernel<1>(PermuteOp{},
                                                       permute_grid,
                                                       PermuteOp::BlockSize(),
                                                       0,
                                                       permute_lens,
                                                       permute_in_strides,
                                                       permute_out_strides,
                                                       ck_tile::make_tuple(data(dense)),
                                                       data(two_op_y)));
        HIP_CHECK_ERROR(hipGetLastError());
    };

    // The reference GEMM runs on the null stream, which does not order with `stream`.
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));
    ck_tile::reference_gemm_gpu<Bf16, Bf16, float, Bf16, Row, Row, Row>(
        data(a), data(w), data(dense), m, N, K, K, N, N);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    permute_reference<<<m, D, 0, stream>>>(data(dense), data(ref), time, batch);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipMemsetAsync(data(out), 0xff, out.GetBufferSize(), stream));
    HIP_CHECK_ERROR(hipMemsetAsync(data(two_op_y), 0xff, bytes, stream));
    HIP_CHECK_ERROR(hipMemsetAsync(data(dense), 0xff, bytes, stream));
    fused();
    two_ops();
    for(const auto& [got, path] : {std::pair{y, "fused"}, std::pair{data(two_op_y), "two-op"}})
    {
        HIP_CHECK_ERROR(hipMemsetAsync(first.GetDeviceBuffer(), 0xff, sizeof(unsigned), stream));
        compare<<<512, 256, 0, stream>>>(
            got, data(ref), n, static_cast<unsigned*>(first.GetDeviceBuffer()));
        HIP_CHECK_ERROR(hipGetLastError());
        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        unsigned mismatch;
        first.FromDevice(&mismatch);
        if(mismatch != UINT32_MAX)
        {
            std::fprintf(stderr,
                         "T=%d B=%d: %s first BF16 mismatch at Y[%u]\n",
                         time,
                         batch,
                         path,
                         mismatch);
            return false;
        }
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
    std::printf("T=%d B=%d: fused and two-op outputs (%d BF16) match the reference bit-for-bit; "
                "guards intact\n",
                time,
                batch,
                n);
    if(!benchmark)
        return true;

    // Back-to-back launches keep host enqueue cost off the timed span.
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
        two_ops();
    }
    // Rotate the order so no path always runs first in a sample.
    std::array<std::array<float, 11>, 3> t{};
    for(size_t i = 0; i < t[0].size(); ++i)
        for(size_t j = 0; j < t.size(); ++j)
            switch((i + j) % 3)
            {
            case 0: t[0][i] = measure(fused); break;
            case 1: t[1][i] = measure(two_ops); break;
            default: t[2][i] = measure(gemm_op); break;
            }
    for(auto& v : t)
        std::sort(v.begin(), v.end());
    const auto mid = t[0].size() / 2;
    std::printf("Median of %zu samples (%d launches each): fused %.1f us; CK-Tile GEMM + permute "
                "%.1f us (%.2fx); CK-Tile GEMM alone %.1f us\n",
                t[0].size(),
                Launches,
                t[0][mid],
                t[1][mid],
                t[1][mid] / t[0][mid],
                t[2][mid]);
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
    HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    const bool ok = run(stream, 1, 1, false) && run(stream, 67, 5, false) &&
                    run(stream, BM + 3, 5, false) && (!benchmark || run(stream, MaxT, MaxB, true));
    HIP_CHECK_ERROR(hipStreamDestroy(stream));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
