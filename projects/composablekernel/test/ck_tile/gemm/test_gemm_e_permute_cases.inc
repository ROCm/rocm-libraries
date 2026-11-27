// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GemmEPermuteKernel cases, shared by the XDL, WMMA and TDM tests. The including file defines
// EPermuteTestConfig: the block, warp and warp tiles, TransposeC, DoubleSmemBuffer, PadN (kPadN),
// the Pipeline<Problem> template and DeviceSupported().

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::long_index_t;
using Bf16 = ck_tile::bf16_t;
using Row  = ck_tile::tensor_layout::gemm::RowMajor;
using Cfg  = EPermuteTestConfig;

using Shape =
    ck_tile::TileGemmShape<ck_tile::sequence<Cfg::M_Tile, Cfg::N_Tile, Cfg::K_Tile>,
                           ck_tile::sequence<Cfg::M_Warp, Cfg::N_Warp, 1>,
                           ck_tile::sequence<Cfg::M_Warp_Tile, Cfg::N_Warp_Tile, Cfg::K_Warp_Tile>>;
using Traits   = ck_tile::TileGemmUniversalTraits<false,
                                                  Cfg::PadN,
                                                  false,
                                                  Cfg::DoubleSmemBuffer,
                                                  Row,
                                                  Row,
                                                  Row,
                                                  Cfg::TransposeC>;
using Problem  = ck_tile::UniversalGemmPipelineProblem<Bf16, Bf16, float, Shape, Traits>;
using Pipeline = Cfg::Pipeline<Problem>;
using Epilogue =
    ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<Bf16,
                                                               Bf16,
                                                               ck_tile::tuple<>,
                                                               float,
                                                               Bf16,
                                                               ck_tile::tuple<>,
                                                               Row,
                                                               ck_tile::element_wise::PassThrough,
                                                               Cfg::M_Tile,
                                                               Cfg::N_Tile,
                                                               Cfg::M_Warp,
                                                               Cfg::N_Warp,
                                                               Cfg::M_Warp_Tile,
                                                               Cfg::N_Warp_Tile,
                                                               Cfg::K_Warp_Tile,
                                                               Problem::TransposeC,
                                                               1,
                                                               false,
                                                               1,
                                                               1,
                                                               Cfg::DoubleSmemBuffer>>;
using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<Shape, 8, 16>;
using Kernel      = ck_tile::GemmEPermuteKernel<Partitioner, Pipeline, Epilogue>;
using Dense       = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;

constexpr uint16_t Sentinel = 0xffff;
constexpr index_t Guard     = 256; // elements before and after the E span

// The Meta layout Y[h, b, t, d] of C[t * B + b, h * D + d]; stride_N0 = 0 means packed.
ck_tile::GemmEPermuteDesc meta_desc(index_t T, index_t B, index_t H, index_t D, index_t sN0 = 0)
{
    return {T, B, H, D, D, T * D, sN0 ? sN0 : B * T * D};
}

long_index_t offset(const ck_tile::GemmEPermuteDesc& p, long_index_t m, long_index_t n)
{
    return m / p.M1 * p.stride_M0 + m % p.M1 * p.stride_M1 + n / p.N1 * p.stride_N0 + n % p.N1;
}

long_index_t span(const ck_tile::GemmEPermuteDesc& p)
{
    return 1 + long_index_t{p.M0 - 1} * p.stride_M0 + long_index_t{p.M1 - 1} * p.stride_M1 +
           long_index_t{p.N0 - 1} * p.stride_N0 + p.N1 - 1;
}

Kernel::KernelArgs host_kargs(const ck_tile::GemmEPermuteDesc& p, index_t K, index_t k_batch = 1)
{
    return Kernel::MakeKernelArgs(
        ck_tile::GemmEPermuteHostArgs(nullptr, nullptr, nullptr, k_batch, K, K, p.N0 * p.N1, p));
}

std::vector<uint16_t> bits(const ck_tile::HostTensor<Bf16>& t)
{
    std::vector<uint16_t> v(t.get_element_space_size());
    std::memcpy(v.data(), t.data(), v.size() * sizeof(uint16_t));
    return v;
}

// Runs the kernel into a sentinel-filled buffer (the E span plus guards) and compares the whole
// buffer with the reference GEMM scattered through the same view, so stray writes into gaps,
// guards or fold-back targets fail. Returns the reference [M, N] result.
std::vector<uint16_t> run_and_check(const ck_tile::GemmEPermuteDesc& p, index_t K)
{
    const index_t M = p.M0 * p.M1, N = p.N0 * p.N1;
    ck_tile::HostTensor<Bf16> a({M, K}, {K, 1}), b({K, N}, {N, 1}), c({M, N}, {N, 1});
    ck_tile::FillUniformDistributionIntegerValue<Bf16>{-5, 5, 11939}(a);
    ck_tile::FillUniformDistributionIntegerValue<Bf16>{-5, 5, 11940}(b);
    ck_tile::reference_gemm<Bf16, Bf16, float, Bf16>(a, b, c);

    const std::size_t total = span(p) + 2 * Guard;
    std::vector<uint16_t> want(total, Sentinel), got(total);
    const auto c_bits = bits(c);
    for(index_t m = 0; m < M; ++m)
        for(index_t n = 0; n < N; ++n)
            want[Guard + offset(p, m, n)] = c_bits[m * N + n];

    ck_tile::DeviceMem a_dev(a), b_dev(b), y_dev(total * sizeof(Bf16));
    HIP_CHECK_ERROR(hipMemset(y_dev.GetDeviceBuffer(), 0xff, total * sizeof(Bf16)));
    const auto kargs = Kernel::MakeKernelArgs(
        ck_tile::GemmEPermuteHostArgs(a_dev.GetDeviceBuffer(),
                                      b_dev.GetDeviceBuffer(),
                                      static_cast<Bf16*>(y_dev.GetDeviceBuffer()) + Guard,
                                      1,
                                      K,
                                      K,
                                      N,
                                      p));
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs));
    ck_tile::launch_kernel(ck_tile::stream_config{},
                           ck_tile::make_kernel<1>(
                               Kernel{}, Kernel::GridSize(M, N, 1), Kernel::BlockSize(), 0, kargs));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    y_dev.FromDevice(got.data());

    std::size_t bad = 0, first = 0;
    for(std::size_t i = 0; i < total; ++i)
        if(got[i] != want[i] && bad++ == 0)
            first = i;
    EXPECT_EQ(bad, 0u) << "first mismatch at buffer element " << first << " (span starts at "
                       << Guard << "): got 0x" << std::hex << got[first] << ", want 0x"
                       << want[first];
    return c_bits;
}

class GemmEPermute : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        if(!Cfg::DeviceSupported())
            GTEST_SKIP() << "tile config not built for " << ck_tile::get_device_name();
    }
};

// Host-only rejections, before any launch.
TEST_F(GemmEPermute, RejectsN1NotMultipleOfVectorSize)
{
    // N = N_Tile and N % VecC == 0, so the base accepts it; only N1 = VecC / 2 is wrong.
    static_assert(Kernel::VecC >= 2);
    const auto kargs =
        host_kargs(meta_desc(4, 128, 2 * Cfg::N_Tile / Kernel::VecC, Kernel::VecC / 2), 64);
    EXPECT_TRUE(Dense::IsSupportedArgument(kargs));
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

TEST_F(GemmEPermute, RejectsSubLengthsNotMatchingMN)
{
    const auto p    = meta_desc(4, 128, 2, 256);
    const auto good = host_kargs(p, 64);
    const auto base = Dense::MakeKernelArgs(ck_tile::GemmHostArgs(
        nullptr, nullptr, nullptr, 1, good.M + 1, good.N, 64, 64, good.N, good.N));
    EXPECT_FALSE(Kernel::IsSupportedArgument(Kernel::KernelArgs{base, p, good.e_desc}));
}

TEST_F(GemmEPermute, RejectsZeroLength)
{
    EXPECT_FALSE(Kernel::IsSupportedArgument(host_kargs(meta_desc(0, 128, 2, 256), 64)));
}

TEST_F(GemmEPermute, RejectsOverlappingStrides)
{
    auto p      = meta_desc(4, 128, 2, 256);
    p.stride_M1 = p.stride_M0; // b and t alias
    EXPECT_FALSE(Kernel::IsSupportedArgument(host_kargs(p, 64)));
}

TEST_F(GemmEPermute, RejectsNonPositiveStride)
{
    auto p      = meta_desc(4, 128, 2, 256);
    p.stride_N0 = 0; // heads alias
    EXPECT_FALSE(Kernel::IsSupportedArgument(host_kargs(p, 64)));
    p.stride_N0 = -1;
    EXPECT_FALSE(Kernel::IsSupportedArgument(host_kargs(p, 64)));
}

// M0 = 1: no valid row uses stride_M0, but the masked rows of the M tail fold to m0 >= 1 and a
// negative stride would wrap their offsets back into E. The base accepts it.
TEST_F(GemmEPermute, RejectsNegativeStrideUnderMaskedTail)
{
    const auto kargs = host_kargs({1, 5, 1, Cfg::N_Tile, -(1 << 30), Cfg::N_Tile, Cfg::N_Tile}, 64);
    EXPECT_TRUE(Dense::IsSupportedArgument(kargs));
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

// Without kPadN, a ragged N would fold the columns past N into E unmasked; with it they are masked.
TEST_F(GemmEPermute, RaggedN)
{
    const auto p = meta_desc(4, 128, 3, Cfg::N_Tile / 2);
    if constexpr(Pipeline::kPadN)
        run_and_check(p, 64);
    else
        EXPECT_FALSE(Kernel::IsSupportedArgument(host_kargs(p, 64)));
}

TEST_F(GemmEPermute, RejectsSpanOf2GB)
{
    // Gapped heads: 256 KB of data spread over more than 2 GB (computed, not allocated). The base
    // accepts the dense footprint, so only the span rule can reject it.
    const auto kargs = host_kargs(meta_desc(4, 128, 2, 256, 1 << 30), 64);
    EXPECT_TRUE(Dense::IsSupportedArgument(kargs));
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
    // B = 1024 at the trace's T, H, D: 3.2 GB.
    EXPECT_FALSE(Kernel::IsSupportedArgument(host_kargs(meta_desc(768, 1024, 8, 256), 384)));
}

// The rows of an M tail fold to m0 >= M0; their masked stores are only dropped while their byte
// offsets stay below 2 GB. The base accepts both, so only that rule can reject them.
TEST_F(GemmEPermute, RejectsMaskedTailFoldingPast2GB)
{
    for(const ck_tile::GemmEPermuteDesc p :
        {// M0 = 1 with a 2 GB stride: rows 5.. of the only tile fold to byte offset 2^31 + x.
         ck_tile::GemmEPermuteDesc{1, 5, 1, Cfg::N_Tile, 1 << 30, Cfg::N_Tile, Cfg::N_Tile},
         // Y[t, h, b, d] spans just below 2 GB; the masked rows fold one stride_M0 past it.
         ck_tile::GemmEPermuteDesc{104857, 5, 8, 256, 8 * 5 * 256, 256, 5 * 256}})
    {
        const auto kargs = host_kargs(p, 64);
        EXPECT_TRUE(Dense::IsSupportedArgument(kargs));
        EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
    }
}

TEST_F(GemmEPermute, RejectsSplitK)
{
    EXPECT_FALSE(Kernel::IsSupportedArgument(host_kargs(meta_desc(4, 128, 2, 256), 64, 2)));
}

TEST_F(GemmEPermute, AcceptsTraceShape)
{
    EXPECT_TRUE(Kernel::IsSupportedArgument(host_kargs(meta_desc(768, 512, 8, 256), 384)));
}

TEST_F(GemmEPermute, PackedTileAligned) { run_and_check(meta_desc(4, 128, 2, 256), 64); }

// The example's ragged shape: tiles straddle t boundaries and the last one passes M, whose rows
// would otherwise fold back into Y.
TEST_F(GemmEPermute, StraddlingTilesAndMTail) { run_and_check(meta_desc(67, 5, 2, 256), 64); }

// Several K tiles (an odd count): a multi-buffered mainloop refills and reuses every LDS buffer
// (the run-ahead ring wraps twice) while the tiles straddle t and the last one passes M.
TEST_F(GemmEPermute, SeveralKTiles) { run_and_check(meta_desc(67, 5, 2, 256), 5 * Cfg::K_Tile); }

TEST_F(GemmEPermute, TileSpansSeveralHeads) { run_and_check(meta_desc(67, 5, 8, 64), 64); }

// Gaps between heads must keep the sentinel.
TEST_F(GemmEPermute, GappedTarget)
{
    run_and_check(meta_desc(67, 5, 2, 256, 67 * 5 * 256 + 9 * Kernel::VecC), 64);
}

// The identity view stores exactly what the stock GemmKernel stores: both are bit-equal to the
// same reference.
TEST_F(GemmEPermute, IdentityMatchesGemmKernel)
{
    const index_t M = 67 * 5, N = 512, K = 64;
    const auto want = run_and_check({1, M, 1, N, M * N, N, N}, K);
    ASSERT_FALSE(HasFailure());

    ck_tile::HostTensor<Bf16> a({M, K}, {K, 1}), b({K, N}, {N, 1});
    ck_tile::FillUniformDistributionIntegerValue<Bf16>{-5, 5, 11939}(a);
    ck_tile::FillUniformDistributionIntegerValue<Bf16>{-5, 5, 11940}(b);
    ck_tile::DeviceMem a_dev(a), b_dev(b), c_dev(std::size_t{M} * N * sizeof(Bf16));
    const auto kargs = Dense::MakeKernelArgs(ck_tile::GemmHostArgs(a_dev.GetDeviceBuffer(),
                                                                   b_dev.GetDeviceBuffer(),
                                                                   c_dev.GetDeviceBuffer(),
                                                                   1,
                                                                   M,
                                                                   N,
                                                                   K,
                                                                   K,
                                                                   N,
                                                                   N));
    ASSERT_TRUE(Dense::IsSupportedArgument(kargs));
    ck_tile::launch_kernel(
        ck_tile::stream_config{},
        ck_tile::make_kernel<1>(Dense{}, Dense::GridSize(M, N, 1), Dense::BlockSize(), 0, kargs));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<uint16_t> got(std::size_t{M} * N);
    c_dev.FromDevice(got.data());
    EXPECT_EQ(got, want);
}

} // namespace
