// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_permute_astationary.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/host/timer.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/epilogue.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <string_view>

namespace {
using namespace gemm_permute;

// Unfused baselines: an existing CK-Tile GEMM op writing the dense intermediate C[t*B+b, d*H+h],
// then the existing permute op, launched back to back. This GEMM op is tuned for this shape and
// shares the library fused kernel's tile, mainloop, epilogue and tile order.
using DenseTraits =
    ck_tile::TileGemmUniversalTraits<true, false, false, false, Row, Row, Row, true>;
using DenseProblem = ck_tile::UniversalGemmPipelineProblem<Bf16, Bf16, float, Shape, DenseTraits>;
using GemmOp =
    ck_tile::GemmKernel<Partitioner, ck_tile::GemmPipelineAgBgCrCompV3<DenseProblem>, Epilogue>;
// Fastest existing CK-Tile GEMM op for this shape: the weight-preshuffle GEMM (WP V2 pipeline, A
// global -> LDS async, PermuteNEpilogue), 128x256x64 tile, 1x4 warps of 16x16x32, 2 blocks per
// CU. N tile 256, 4 N warps and 16x16 MFMA make its weight layout the one of
// astationary::repack_weight with identity column order, which runs on every call.
using WpShape   = ck_tile::TileGemmShape<ck_tile::sequence<128, 256, 64>,
                                         ck_tile::sequence<1, 4, 1>,
                                         ck_tile::sequence<16, 16, 32>>;
using WpTraits  = ck_tile::TileGemmUniversalTraits<true,
                                                   false,
                                                   false,
                                                   true,
                                                   Row,
                                                   ck_tile::tensor_layout::gemm::ColumnMajor,
                                                   Row,
                                                   false,
                                                   false,
                                                   false,
                                                   1,
                                                   true,
                                                   16,
                                                   ck_tile::DataCachePrefetchKind::None,
                                                   ck_tile::DataCachePrefetchKind::None,
                                                   true>;
using WpProblem = ck_tile::UniversalGemmPipelineProblem<Bf16, Bf16, float, WpShape, WpTraits>;
using WpGemmOp  = ck_tile::GemmKernel<
     ck_tile::GemmSpatiallyLocalTilePartitioner<WpShape, 8, 16>,
     ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2<WpProblem>,
     ck_tile::PermuteNEpilogue<ck_tile::PermuteNEpilogueProblem<Bf16,
                                                                Bf16,
                                                                ck_tile::tuple<>,
                                                                float,
                                                                Bf16,
                                                                ck_tile::tuple<>,
                                                                Row,
                                                                ck_tile::element_wise::PassThrough,
                                                                128,
                                                                256,
                                                                1,
                                                                4,
                                                                16,
                                                                16,
                                                                32,
                                                                WpProblem::TransposeC,
                                                                false,
                                                                1>>>;
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
    const size_t bytes = size_t(n) * sizeof(Bf16), w_bytes = size_t(K) * N * sizeof(Bf16);
    // Y of the library fused, A-stationary fused, CompV3 two-op and WP two-op paths, each followed
    // by a guard region; one more guard leads.
    constexpr int Outputs = 4;
    std::array<uint16_t, D> guard{};
    const int64_t stride = n + int64_t(guard.size());
    ck_tile::DeviceMem a(size_t(m) * K * sizeof(Bf16)), w(w_bytes), wp(w_bytes), wf(w_bytes),
        wq(w_bytes);
    ck_tile::DeviceMem dense(bytes), wp_dense(bytes), ref(bytes),
        out(Outputs * stride * sizeof(Bf16) + sizeof(guard)), first(sizeof(unsigned));
    auto y = [&](int i) { return data(out) + guard.size() + i * stride; };
    fill<<<512, 256, 0, stream>>>(data(a), m * K, 13);
    HIP_CHECK_ERROR(hipGetLastError());
    fill<<<512, 256, 0, stream>>>(data(w), K * N, 79);
    HIP_CHECK_ERROR(hipGetLastError());

    const ck_tile::stream_config config{stream};
    const auto gemm_args = GemmOp::MakeKernelArgs(
        ck_tile::GemmHostArgs(data(a), data(w), data(dense), 1, m, N, K, K, N, N));
    const auto wp_args = WpGemmOp::MakeKernelArgs(
        ck_tile::GemmHostArgs(data(a), data(wq), data(wp_dense), 1, m, N, K, K, K, N));
    if(!GemmOp::IsSupportedArgument(gemm_args) || !WpGemmOp::IsSupportedArgument(wp_args))
    {
        std::fprintf(stderr, "T=%d B=%d: a CK-Tile GEMM op rejects the problem\n", time, batch);
        return false;
    }
    // Dense C[t,b,d,h] -> Y[h,b,t,d], as torch.permute(3, 1, 0, 2): lengths in C order, C strides
    // in, Y strides out.
    const auto permute_lens        = ck_tile::make_tuple(time, batch, D, H);
    const auto permute_in_strides  = ck_tile::make_tuple(batch * D * H, D * H, H, 1);
    const auto permute_out_strides = ck_tile::make_tuple(D, time * D, 1, batch * time * D);
    const int permute_grid =
        ck_tile::integer_divide_ceil(n, PermuteOp::Problem::BlockShape::kBlockM);

    // Every path takes the original W: the weight reorders run inside every call.
    auto fused = [&] {
        HIP_CHECK_ERROR(prepermute_weight(data(w), data(wp), stream));
        HIP_CHECK_ERROR(launch(data(a), data(wp), y(0), time, batch, stream));
    };
    auto astationary_fused = [&] {
        HIP_CHECK_ERROR(astationary::repack_weight(data(w), data(wf), stream));
        HIP_CHECK_ERROR(astationary::launch(data(a), data(wf), y(1), time, batch, stream));
    };
    auto gemm_op = [&] {
        ck_tile::launch_kernel(
            config,
            ck_tile::make_kernel<1>(
                GemmOp{}, GemmOp::GridSize(m, N, 1), GemmOp::BlockSize(), 0, gemm_args));
        HIP_CHECK_ERROR(hipGetLastError());
    };
    auto wp_gemm_op = [&] {
        HIP_CHECK_ERROR(astationary::repack_weight(data(w), data(wq), stream, false));
        ck_tile::launch_kernel(
            config,
            ck_tile::make_kernel<2>(
                WpGemmOp{}, WpGemmOp::GridSize(m, N, 1), WpGemmOp::BlockSize(), 0, wp_args));
        HIP_CHECK_ERROR(hipGetLastError());
    };
    auto permute_op = [&](const ck_tile::DeviceMem& src, Bf16* dst) {
        ck_tile::launch_kernel(config,
                               ck_tile::make_kernel<1>(PermuteOp{},
                                                       permute_grid,
                                                       PermuteOp::BlockSize(),
                                                       0,
                                                       permute_lens,
                                                       permute_in_strides,
                                                       permute_out_strides,
                                                       ck_tile::make_tuple(data(src)),
                                                       dst));
        HIP_CHECK_ERROR(hipGetLastError());
    };
    auto two_ops = [&] {
        gemm_op();
        permute_op(dense, y(2));
    };
    auto wp_two_ops = [&] {
        wp_gemm_op();
        permute_op(wp_dense, y(3));
    };

    // The reference GEMM runs on the null stream, which does not order with `stream`.
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));
    ck_tile::reference_gemm_gpu<Bf16, Bf16, float, Bf16, Row, Row, Row>(
        data(a), data(w), data(dense), m, N, K, K, N, N);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    permute_reference<<<m, D, 0, stream>>>(data(dense), data(ref), time, batch);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipMemsetAsync(data(out), 0xff, out.GetBufferSize(), stream));
    HIP_CHECK_ERROR(hipMemsetAsync(data(dense), 0xff, bytes, stream));
    HIP_CHECK_ERROR(hipMemsetAsync(data(wp_dense), 0xff, bytes, stream));
    fused();
    astationary_fused();
    two_ops();
    wp_two_ops();
    constexpr std::array<const char*, Outputs> names{
        "library fused", "A-stationary fused", "CompV3 two-op", "WP two-op"};
    for(int i = 0; i < Outputs; ++i)
    {
        HIP_CHECK_ERROR(hipMemsetAsync(first.GetDeviceBuffer(), 0xff, sizeof(unsigned), stream));
        compare<<<512, 256, 0, stream>>>(
            y(i), data(ref), n, static_cast<unsigned*>(first.GetDeviceBuffer()));
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
                         names[i],
                         mismatch);
            return false;
        }
    }
    for(int i = 0; i <= Outputs; ++i)
    {
        HIP_CHECK_ERROR(
            hipMemcpy(guard.data(), data(out) + i * stride, sizeof(guard), hipMemcpyDeviceToHost));
        if(!std::all_of(guard.begin(), guard.end(), [](uint16_t v) { return v == 0xffff; }))
        {
            std::fprintf(stderr, "T=%d B=%d: output guard overwritten\n", time, batch);
            return false;
        }
    }
    std::printf("T=%d B=%d: both fused and both two-op outputs (%d BF16) match the reference "
                "bit-for-bit; guards intact\n",
                time,
                batch,
                n);
    if(!benchmark)
        return true;

    // Back-to-back launches keep host enqueue cost off the timed span.
    constexpr int Launches = 20;
    ck_tile::gpu_timer timer;
    auto measure = [&](const std::function<void()>& fn) {
        timer.start(stream);
        for(int i = 0; i < Launches; ++i)
            fn();
        timer.stop(stream);
        return 1000.f * timer.duration() / Launches;
    };
    const std::array<std::pair<const char*, std::function<void()>>, 6> paths{
        {{"library fused (GemmEPermuteKernel)", fused},
         {"A-stationary fused", astationary_fused},
         {"WP GEMM + ElementWise", wp_two_ops},
         {"CompV3 GEMM + ElementWise", two_ops},
         {"WP GEMM alone", wp_gemm_op},
         {"CompV3 GEMM alone", gemm_op}}};
    for(int i = 0; i < 3; ++i)
        for(const auto& p : paths)
            p.second();
    // Rotate the order so no path always runs first in a sample.
    std::array<std::array<float, 11>, paths.size()> t{};
    for(size_t i = 0; i < t[0].size(); ++i)
        for(size_t j = 0; j < t.size(); ++j)
        {
            const size_t k = (i + j) % t.size();
            t[k][i]        = measure(paths[k].second);
        }
    std::array<float, paths.size()> median{};
    for(size_t k = 0; k < t.size(); ++k)
    {
        std::sort(t[k].begin(), t[k].end());
        median[k] = t[k][t[k].size() / 2];
    }
    // Paths 2 and 3 are the two-op baselines.
    const float best_two_op = std::min(median[2], median[3]);
    std::printf("Median of %zu samples (%d launches each), weight reorder included in every call; "
                "speedup = best two-op / path:\n",
                t[0].size(),
                Launches);
    for(size_t k = 0; k < t.size(); ++k)
        std::printf(
            "  %-36s %8.1f us  %.2fx\n", paths[k].first, median[k], best_two_op / median[k]);
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
                    run(stream, BM + 3, 5, false) && run(stream, 67, 128, false) &&
                    (!benchmark || run(stream, 768, 512, true));
    HIP_CHECK_ERROR(hipStreamDestroy(stream));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
