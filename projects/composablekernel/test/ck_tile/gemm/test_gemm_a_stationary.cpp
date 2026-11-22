// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GemmAStationaryKernel with the gfx1250 async pipeline: every IsSupportedArgument rule on the
// host, then the permuted (GemmEPermuteHostArgs) and the plain (GemmHostArgs) binding run into a
// sentinel-filled buffer (the E span plus guards) that is compared as a whole with the reference
// GEMM scattered through the same view, so stray writes into gaps or guards fail. Instances: the
// K = 384 trace tile with run-time E strides, the same with stride_M0 = 256 at compile time, and
// K = 128. The device cases are smoke-sized (N <= 768, host reference); the trace shape itself is
// only checked on the host here.

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "test_gemm_pipeline_util.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::long_index_t;
using Bf16 = ck_tile::bf16_t;
using Desc = ck_tile::GemmEPermuteDesc;

// 256-row workgroup tile, 64-column wave tiles, 8 waves as 2 row groups x 4 column groups.
template <index_t K, index_t StrideM0 = 0>
using Kernel = ck_tile::GemmAStationaryKernel<ck_tile::GemmPipelineAStationaryAsync<
    ck_tile::GemmAStationaryProblem<Bf16, Bf16, float, Bf16, 256, K, 64, 8, 2, StrideM0>>>;
using Trace  = Kernel<384>;
using TraceS = Kernel<384, 256>;
using ShortK = Kernel<128>;

constexpr uint16_t Sentinel = 0xffff;
constexpr index_t Guard     = 256; // elements before and after the E span

// The trace layout Y[h, b, t, d] of C[t * B + b, h * D + d]; stride_N0 = 0 means packed.
Desc meta_desc(index_t T, index_t B, index_t H, index_t D, index_t sN0 = 0)
{
    return {T, B, H, D, D, T * D, sN0 ? sN0 : B * T * D};
}

// Y[b, h, t, d]: the second permutation order.
Desc bhtd_desc(index_t T, index_t B, index_t H, index_t D)
{
    return {T, B, H, D, D, H * T * D, T * D};
}

// The E view of a plain GEMM with row stride stride_E, as MakeKernelArgs(GemmHostArgs) builds it.
Desc plain_desc(index_t M, index_t N, index_t stride_E) { return {M, 1, 1, N, stride_E, 0, 0}; }

long_index_t offset(const Desc& p, long_index_t m, long_index_t n)
{
    return m / p.M1 * p.stride_M0 + m % p.M1 * p.stride_M1 + n / p.N1 * p.stride_N0 + n % p.N1;
}

// Host arguments of the permuted binding (plain = false) or of the plain binding (p from
// plain_desc), with the trace's K unless given.
template <typename K_ = Trace>
typename K_::KernelArgs host_kargs(const Desc& p,
                                   bool plain     = false,
                                   index_t K      = K_::Pipeline::kK,
                                   index_t kbatch = 1,
                                   const void* a  = nullptr,
                                   const void* w  = nullptr,
                                   void* e        = nullptr)
{
    if(plain)
        return K_::MakeKernelArgs(
            ck_tile::GemmHostArgs(a, w, e, kbatch, p.M0, p.N1, K, K, 0, p.stride_M0));
    return K_::MakeKernelArgs(ck_tile::GemmEPermuteHostArgs(a, w, e, kbatch, K, K, 0, p));
}

// Runs kernel K_ into a sentinel-filled buffer and compares the whole buffer with the reference
// GEMM scattered through the view p. The weight is repacked on the host (RunOnHost).
template <typename K_>
void run_and_check(const Desc& p, bool plain = false)
{
    using Repack    = ck_tile::GemmAStationaryWeightRepackKernel<typename K_::Pipeline>;
    const index_t K = K_::Pipeline::kK, M = p.M0 * p.M1, N = p.N0 * p.N1;
    ck_tile::HostTensor<Bf16> a({M, K}, {K, 1}), w({K, N}, {N, 1}), w_flat({K * N}, {1}),
        c({M, N}, {N, 1});
    ck_tile::FillUniformDistributionIntegerValue<Bf16>{-5, 5, 11939}(a);
    ck_tile::FillUniformDistributionIntegerValue<Bf16>{-5, 5, 11940}(w);
    Repack::RunOnHost(w.data(), w_flat.data(), N, N);
    ck_tile::reference_gemm<Bf16, Bf16, float, Bf16>(a, w, c);

    const std::size_t total = ck_tile::gemm_e_permute_span(p) + 2 * Guard;
    std::vector<uint16_t> want(total, Sentinel), got(total);
    for(index_t m = 0; m < M; ++m)
        for(index_t n = 0; n < N; ++n)
            want[Guard + offset(p, m, n)] = ck_tile::bit_cast<uint16_t>(c(m, n));

    ck_tile::DeviceMem a_dev(a), w_flat_dev(w_flat), e_dev(total * sizeof(Bf16));
    HIP_CHECK_ERROR(hipMemset(e_dev.GetDeviceBuffer(), 0xff, total * sizeof(Bf16)));
    const auto kargs = host_kargs<K_>(p,
                                      plain,
                                      K,
                                      1,
                                      a_dev.GetDeviceBuffer(),
                                      w_flat_dev.GetDeviceBuffer(),
                                      static_cast<Bf16*>(e_dev.GetDeviceBuffer()) + Guard);
    ASSERT_TRUE(K_::IsSupportedArgument(kargs));
    ck_tile::launch_kernel(ck_tile::stream_config{},
                           ck_tile::make_kernel<K_::kBlockPerCu>(
                               K_{}, K_::GridSize(kargs), K_::BlockSize(), 0, kargs));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    e_dev.FromDevice(got.data());

    std::size_t bad = 0, first = 0;
    for(std::size_t i = 0; i < total; ++i)
        if(got[i] != want[i] && bad++ == 0)
            first = i;
    EXPECT_EQ(bad, 0u) << "first mismatch at buffer element " << first << " (span starts at "
                       << Guard << "): got 0x" << std::hex << got[first] << ", want 0x"
                       << want[first];
}

class GemmAStationary : public ::testing::Test
{
    protected:
    void SetUp() override { CK_TILE_SKIP_UNLESS_GFX125(); }
};

// Host-only rejections, before any launch. Each case breaks one rule of an accepted problem.
TEST_F(GemmAStationary, AcceptsTraceShape)
{
    EXPECT_TRUE(Trace::IsSupportedArgument(host_kargs(meta_desc(768, 512, 8, 256))));
    EXPECT_TRUE(Trace::IsSupportedArgument(host_kargs(plain_desc(393216, 2048, 2048), true)));
    EXPECT_FALSE(Trace::GetName().empty());
}

TEST_F(GemmAStationary, RejectsSplitK)
{
    EXPECT_FALSE(
        Trace::IsSupportedArgument(host_kargs(meta_desc(768, 512, 8, 256), false, 384, 2)));
}

TEST_F(GemmAStationary, RejectsOtherK)
{
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(meta_desc(768, 512, 8, 256), false, 128)));
}

TEST_F(GemmAStationary, RejectsOtherStrideM0)
{
    EXPECT_TRUE(TraceS::IsSupportedArgument(host_kargs<TraceS>(meta_desc(768, 512, 8, 256))));
    EXPECT_FALSE(
        TraceS::IsSupportedArgument(host_kargs<TraceS>(plain_desc(768, 2048, 2048), true)));
}

TEST_F(GemmAStationary, RejectsZeroLengthAndNegativeStride)
{
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(meta_desc(0, 512, 8, 256))));
    auto kargs     = host_kargs(meta_desc(768, 512, 8, 256));
    kargs.stride_A = -384;
    EXPECT_FALSE(Trace::IsSupportedArgument(kargs));
}

// Views whose M or N overflow index_t, set directly (the host-argument types compute M and N).
// N = 2^32 would wrap the device's N; 2^22 tiles of m0 times M1 = 2^10 is 2^32 workgroups.
TEST_F(GemmAStationary, RejectsIndexOverflow)
{
    for(const Desc& p : {Desc{1, 1, 1 << 16, 1 << 16, 0, 0, 1 << 16},
                         Desc{1 << 30, 1 << 10, 1, 256, 1 << 18, 256, 0}})
    {
        auto kargs = host_kargs(meta_desc(768, 512, 8, 256));
        kargs.e    = p;
        EXPECT_FALSE(Trace::IsSupportedArgument(kargs));
    }
}

TEST_F(GemmAStationary, RejectsOverlappingE)
{
    auto p      = meta_desc(768, 512, 8, 256);
    p.stride_M1 = p.stride_M0; // b and t alias
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(p)));
    p           = meta_desc(768, 512, 8, 256);
    p.stride_N0 = 0; // heads alias
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(p)));
    // A plain row stride below N.
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(plain_desc(768, 2048, 1024), true)));
}

TEST_F(GemmAStationary, RejectsNNotTiled)
{
    // N1 is not a multiple of the 64-column wave tile.
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(meta_desc(768, 512, 64, 32))));
    // N is not a multiple of the 4 column groups' 256 columns.
    EXPECT_FALSE(
        Trace::IsSupportedArgument(host_kargs(plain_desc(768, 2048 + 64, 2048 + 64), true)));
}

TEST_F(GemmAStationary, RejectsMisalignment)
{
    alignas(16) static Bf16 buf[16];
    const auto p = meta_desc(768, 512, 8, 256);
    EXPECT_TRUE(Trace::IsSupportedArgument(host_kargs(p, false, 384, 1, buf, buf, buf)));
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(p, false, 384, 1, buf + 1, buf, buf)));
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(p, false, 384, 1, buf, buf + 1, buf)));
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(p, false, 384, 1, buf, buf, buf + 1)));
    auto kargs     = host_kargs(p);
    kargs.stride_A = 388;
    EXPECT_FALSE(Trace::IsSupportedArgument(kargs));
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(plain_desc(768, 2048, 2049), true)));
    // stride_M1, then stride_N0, 8 bytes off; the gap between heads keeps E free of overlap.
    auto q = meta_desc(768, 512, 8, 256, 512 * (768 * 256 + 8));
    EXPECT_TRUE(Trace::IsSupportedArgument(host_kargs(q)));
    q.stride_M1 += 4;
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(q)));
    EXPECT_FALSE(
        Trace::IsSupportedArgument(host_kargs(meta_desc(768, 512, 8, 256, 512 * 768 * 256 + 4))));
}

TEST_F(GemmAStationary, RejectsSpansOf2GB)
{
    // A: the 256 rows of a tile are M1 * stride_A apart, 3.2 GB in all.
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(meta_desc(4, 1 << 14, 8, 256))));
    // E: the 128 rows of a wave tile are stride_E apart, 2 GB in all.
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(plain_desc(768, 256, 1 << 23), true)));
}

// The mainloop compiles to nothing off gfx125, so any other device is rejected.
TEST(GemmAStationaryHost, RejectsOtherDevices)
{
    if(ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "gfx125 device";
    EXPECT_FALSE(Trace::IsSupportedArgument(host_kargs(meta_desc(768, 512, 8, 256))));
}

// The device repack writes exactly the host repack's flat weight.
template <typename K_>
void check_device_repack(index_t N, index_t stride_W)
{
    using Repack    = ck_tile::GemmAStationaryWeightRepackKernel<typename K_::Pipeline>;
    const index_t K = K_::Pipeline::kK;
    ck_tile::HostTensor<Bf16> w({K, N}, {stride_W, 1}), want({K * N}, {1}), got({K * N}, {1});
    ck_tile::FillUniformDistribution<Bf16>{-1.f, 1.f, 11941}(w);
    Repack::RunOnHost(w.data(), want.data(), N, stride_W);
    ck_tile::DeviceMem w_dev(w), w_flat_dev(got.get_element_space_size_in_bytes());
    ck_tile::launch_kernel(
        ck_tile::stream_config{},
        ck_tile::make_kernel<1>(Repack{},
                                Repack::GridSize(N),
                                Repack::BlockSize(),
                                0,
                                static_cast<const Bf16*>(w_dev.GetDeviceBuffer()),
                                static_cast<Bf16*>(w_flat_dev.GetDeviceBuffer()),
                                N,
                                stride_W));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    w_flat_dev.FromDevice(got.data());
    EXPECT_EQ(std::memcmp(got.data(), want.data(), got.get_element_space_size_in_bytes()), 0);
}

TEST_F(GemmAStationary, DeviceRepackMatchesHost)
{
    check_device_repack<Trace>(512, 512 + 64);
    check_device_repack<ShortK>(512, 512);
}

// Permuted store; 2 heads of 256 unless noted, so each column group walks two wave tiles.
TEST_F(GemmAStationary, PackedFullTiles) { run_and_check<Trace>(meta_desc(256, 2, 2, 256)); }

// T = 67: one tail tile whose second row group is empty; 131: one row group partly filled;
// 259: a full tile and a 3-row tail.
TEST_F(GemmAStationary, RaggedTWithTailTiles)
{
    for(const index_t T : {67, 131, 259})
        run_and_check<Trace>(meta_desc(T, 2, 2, 256));
}

TEST_F(GemmAStationary, SecondPermutationOrder) { run_and_check<Trace>(bhtd_desc(259, 2, 2, 256)); }

// Gaps between heads must keep the sentinel.
TEST_F(GemmAStationary, GappedTarget)
{
    run_and_check<Trace>(meta_desc(259, 2, 2, 256, 2 * 259 * 256 + 9 * 8));
}

// A wave tile is one head of 64 columns; a column group step covers 4 heads. 12 heads: three
// wave tiles per column group.
TEST_F(GemmAStationary, NarrowHeads) { run_and_check<Trace>(meta_desc(259, 2, 12, 64)); }

// Plain GEMM.
TEST_F(GemmAStationary, PlainSingleRow) { run_and_check<Trace>(plain_desc(1, 512, 512), true); }

TEST_F(GemmAStationary, PlainRaggedM) { run_and_check<Trace>(plain_desc(769, 512, 512), true); }

TEST_F(GemmAStationary, PlainStrideEPastN)
{
    run_and_check<Trace>(plain_desc(259, 512, 512 + 64), true);
}

// stride_M0 = 256 at compile time: the trace layout with D = 256, and a plain GEMM with N = 256.
TEST_F(GemmAStationary, CompileTimeStrideM0)
{
    run_and_check<TraceS>(meta_desc(259, 2, 2, 256));
    run_and_check<TraceS>(plain_desc(300, 256, 256), true);
}

// K = 128: one A piece, a two-slot weight ring and eight E stores per K step.
TEST_F(GemmAStationary, ShortK)
{
    run_and_check<ShortK>(meta_desc(259, 2, 2, 256));
    run_and_check<ShortK>(plain_desc(769, 512, 512 + 64), true);
}

} // namespace
