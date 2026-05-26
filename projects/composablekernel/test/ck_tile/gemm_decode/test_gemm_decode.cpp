// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace ck_tile;

namespace {

struct DecodeShape
{
    index_t M = 1;
    index_t N = 256;
    index_t K = 7168;
};

template <typename T>
void FillRandom(HostTensor<T>& tensor, float min_val, float max_val, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for(index_t i = 0; i < static_cast<index_t>(tensor.get_element_space_size()); ++i)
    {
        tensor.mData[i] = type_convert<T>(dist(gen));
    }
}

// FP32 host reference: C[m, n] = sum_k A[m, k] * B[n, k].
// Accumulation is done as float regardless of input dtype to give the
// device kernel a clean numerical target.
template <typename ADataType, typename BDataType, typename CDataType>
void ReferenceGemm(const HostTensor<ADataType>& a,
                   const HostTensor<BDataType>& b,
                   HostTensor<CDataType>& c,
                   index_t M,
                   index_t N,
                   index_t K)
{
    for(index_t m = 0; m < M; ++m)
    {
        for(index_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for(index_t k = 0; k < K; ++k)
            {
                acc += type_convert<float>(a(m, k)) * type_convert<float>(b(n, k));
            }
            c(m, n) = type_convert<CDataType>(acc);
        }
    }
}

// Adaptive tolerance: the warp-per-scalar accumulation order differs from the
// host's sequential sum, and BF16/FP16 inputs amplify the per-element error
// roughly with sqrt(K) * eps. The constants below are conservative and
// matched to the K=7168 test shapes.
template <typename CDataType>
float AbsoluteTolerance(index_t K)
{
    if constexpr(std::is_same_v<CDataType, bf16_t>)
    {
        return 0.5f * std::sqrt(static_cast<float>(K));
    }
    else if constexpr(std::is_same_v<CDataType, fp16_t>)
    {
        return 0.25f * std::sqrt(static_cast<float>(K));
    }
    else
    {
        return 1e-3f * std::sqrt(static_cast<float>(K));
    }
}

template <typename ADataType, typename BDataType, typename CDataType>
::testing::AssertionResult RunUnscaledCase(const std::string& test_name,
                                           const DecodeShape& shape,
                                           index_t k_batch)
{
    using ComputeDataType = float;
    using Problem         = GemmDecodeProblem<ADataType,
                                              BDataType,
                                              ComputeDataType,
                                              CDataType,
                                              /*XScaleDataType=*/float,
                                              /*WScaleDataType=*/float,
                                              /*XScaleLayout=*/void,
                                              /*WScaleLayout=*/void,
                                              /*kVector=*/8,
                                              /*kUseDot2=*/false,
                                              /*kUsePackedFp32=*/false,
                                              /*kMPerWarp=*/1,
                                              /*kNPerWarp=*/1,
                                              GemmDecodeOutputAxis::SmallM,
                                              /*kHasBias=*/false,
                                              /*kWarpsPerBlock=*/1>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    HostTensor<ADataType> a({shape.M, shape.K});
    HostTensor<BDataType> b({shape.N, shape.K});
    HostTensor<CDataType> c_host({shape.M, shape.N});
    HostTensor<CDataType> c_dev({shape.M, shape.N});

    FillRandom(a, -1.0f, 1.0f, 0xA1u);
    FillRandom(b, -1.0f, 1.0f, 0xB2u);

    ReferenceGemm<ADataType, BDataType, CDataType>(a, b, c_host, shape.M, shape.N, shape.K);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());

    // AtomicAdd split-K accumulates partials, so the destination must start at
    // zero. For k_batch == 1 the kernel does a plain store so the zero-init is
    // not strictly needed, but doing it unconditionally simplifies the test
    // bookkeeping.
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        shape.M,
                                        shape.N,
                                        shape.K,
                                        /*stride_a=*/shape.K,
                                        /*stride_b=*/shape.K,
                                        /*stride_c=*/shape.N,
                                        /*k_batch=*/k_batch);

    if(!Kernel::IsSupportedArgument(kargs))
    {
        return ::testing::AssertionFailure()
               << test_name << ": Kargs unexpectedly rejected by IsSupportedArgument().";
    }

    const stream_config s{nullptr, /*time_kernel=*/false};
    try
    {
        launch_gemm_decode_universal<Kernel>(kargs, s);
    }
    catch(const std::exception& ex)
    {
        return ::testing::AssertionFailure() << test_name << ": launch threw: " << ex.what();
    }

    c_buf.FromDevice(c_dev.mData.data());

    const float atol  = AbsoluteTolerance<CDataType>(shape.K);
    float max_diff    = 0.0f;
    index_t bad_index = -1;
    for(index_t i = 0; i < static_cast<index_t>(c_host.get_element_space_size()); ++i)
    {
        const float h = type_convert<float>(c_host.mData[i]);
        const float d = type_convert<float>(c_dev.mData[i]);
        const float diff = std::abs(h - d);
        if(diff > max_diff)
        {
            max_diff  = diff;
            bad_index = i;
        }
    }

    if(max_diff > atol)
    {
        const float h = type_convert<float>(c_host.mData[bad_index]);
        const float d = type_convert<float>(c_dev.mData[bad_index]);
        return ::testing::AssertionFailure()
               << test_name << " (M=" << shape.M << ", N=" << shape.N << ", K=" << shape.K
               << ", k_batch=" << k_batch << "): mismatch at i=" << bad_index << " host=" << h
               << " dev=" << d << " diff=" << max_diff << " atol=" << atol;
    }
    return ::testing::AssertionSuccess();
}

template <typename ADataType, typename BDataType, typename CDataType>
void RunMatrix(const std::string& dtype_name)
{
    constexpr index_t K = 7168;
    const std::vector<index_t> Ms{1, 2, 4, 8};
    const std::vector<index_t> Ns{512, 4096};
    const std::vector<index_t> KBatches{1, 2, 4};

    for(index_t M : Ms)
    {
        for(index_t N : Ns)
        {
            for(index_t kb : KBatches)
            {
                const DecodeShape shape{M, N, K};
                const std::string name = dtype_name + " M=" + std::to_string(M) + " N=" +
                                         std::to_string(N) + " K=" + std::to_string(K) +
                                         " kb=" + std::to_string(kb);
                EXPECT_TRUE((RunUnscaledCase<ADataType, BDataType, CDataType>(name, shape, kb)));
            }
        }
    }
}

// Build a Kargs with a fixed valid baseline that callers can perturb to
// trigger specific IsSupportedArgument rejections.
template <typename Kernel>
typename Kernel::Kargs MakeBaselineArgs()
{
    static int placeholder = 0;
    typename Kernel::Kargs kargs{};
    kargs.p_a       = &placeholder;
    kargs.p_b       = &placeholder;
    kargs.p_c       = &placeholder;
    kargs.p_x_scale = nullptr;
    kargs.p_w_scale = nullptr;
    kargs.p_bias    = nullptr;
    kargs.M         = 4;
    kargs.N         = 256;
    kargs.K         = 7168;
    kargs.stride_a  = kargs.K;
    kargs.stride_b  = kargs.K;
    kargs.stride_c  = kargs.N;
    kargs.k_batch   = 1;
    return kargs;
}

} // namespace

TEST(GemmDecodeUniversalUnscaled, Bf16Bf16Matrix)
{
    RunMatrix<bf16_t, bf16_t, bf16_t>("BF16/BF16");
}

TEST(GemmDecodeUniversalUnscaled, Fp16Fp16Matrix)
{
    RunMatrix<fp16_t, fp16_t, fp16_t>("FP16/FP16");
}

TEST(GemmDecodeUniversalUnscaled, AtomicAddSplitKVariety)
{
    // Spot-check a single shape across k_batch values to confirm the
    // AtomicAdd epilogue stays consistent. RunMatrix already covers this in
    // bulk; this is a focused regression check.
    const DecodeShape shape{1, 4096, 7168};
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t>("BF16 split=2", shape, 2)));
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t>("BF16 split=4", shape, 4)));
}

TEST(GemmDecodeUniversalNegative, RejectsNullPointers)
{
    using Problem = GemmDecodeProblem<bf16_t, bf16_t, float, bf16_t>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs = MakeBaselineArgs<Kernel>();
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs));

    auto k_a = kargs;
    k_a.p_a  = nullptr;
    EXPECT_FALSE(Kernel::IsSupportedArgument(k_a));

    auto k_b = kargs;
    k_b.p_b  = nullptr;
    EXPECT_FALSE(Kernel::IsSupportedArgument(k_b));

    auto k_c = kargs;
    k_c.p_c  = nullptr;
    EXPECT_FALSE(Kernel::IsSupportedArgument(k_c));
}

TEST(GemmDecodeUniversalNegative, RejectsNonDivisibleK)
{
    using Problem = GemmDecodeProblem<bf16_t, bf16_t, float, bf16_t>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs = MakeBaselineArgs<Kernel>();
    // warp_size * kVector = 64 * 8 = 512; choose K not divisible.
    kargs.K        = 7000;
    kargs.stride_a = kargs.K;
    kargs.stride_b = kargs.K;
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

TEST(GemmDecodeUniversalNegative, RejectsKBatchExceedingTileCount)
{
    using Problem = GemmDecodeProblem<bf16_t, bf16_t, float, bf16_t>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs    = MakeBaselineArgs<Kernel>();
    kargs.K       = 1024; // K / (warp_size * kVector) = 1024 / 512 = 2.
    kargs.k_batch = 4;    // > 2 leaves shards with zero iterations.
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

TEST(GemmDecodeUniversalNegative, RejectsBadStrides)
{
    using Problem = GemmDecodeProblem<bf16_t, bf16_t, float, bf16_t>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs     = MakeBaselineArgs<Kernel>();
    kargs.stride_a = kargs.K - 1; // smaller than K
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));

    kargs          = MakeBaselineArgs<Kernel>();
    kargs.stride_c = kargs.N - 1;
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

TEST(GemmDecodeUniversalNegative, RejectsZeroDimensions)
{
    using Problem = GemmDecodeProblem<bf16_t, bf16_t, float, bf16_t>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs = MakeBaselineArgs<Kernel>();
    kargs.M    = 0;
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));

    kargs   = MakeBaselineArgs<Kernel>();
    kargs.N = 0;
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

TEST(GemmDecodeUniversalNegative, RejectsOddNWithSplitK)
{
    using Problem = GemmDecodeProblem<bf16_t, bf16_t, float, bf16_t>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs    = MakeBaselineArgs<Kernel>();
    kargs.N       = 257;
    kargs.K       = 7168;
    kargs.stride_c = kargs.N;
    kargs.k_batch = 2;
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}
