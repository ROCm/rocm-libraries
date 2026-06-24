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
template <typename CDataType, typename ADataType = CDataType>
float AbsoluteTolerance(index_t K)
{
#ifdef CK_TILE_USE_OCP_FP8
    if constexpr(std::is_same_v<ADataType, fp8_t>)
    {
        // FP8 (E4M3) inputs lose ~3 mantissa bits relative to BF16, so the
        // accumulated noise is roughly 4-8x the BF16 figure. Bias also gets
        // multiplied by sA*sB so we leave headroom.
        if constexpr(std::is_same_v<CDataType, bf16_t>)
        {
            return 1.5f * std::sqrt(static_cast<float>(K));
        }
        else if constexpr(std::is_same_v<CDataType, fp16_t>)
        {
            return 1.0f * std::sqrt(static_cast<float>(K));
        }
    }
#endif
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

// Max-abs comparison of host vs device C, returning a descriptive failure on
// the worst element. Shared by the 2D-bias cases below; mirrors the inline
// compare in RunUnscaledCase / RunFp8PerTensorCase.
template <typename CDataType>
::testing::AssertionResult CompareHostDevice(const std::string&           test_name,
                                             const DecodeShape&           shape,
                                             index_t                      k_batch,
                                             const HostTensor<CDataType>& c_host,
                                             const HostTensor<CDataType>& c_dev,
                                             float                        atol)
{
    float   max_diff  = 0.0f;
    index_t bad_index = -1;
    for(index_t i = 0; i < static_cast<index_t>(c_host.get_element_space_size()); ++i)
    {
        const float h    = type_convert<float>(c_host.mData[i]);
        const float d    = type_convert<float>(c_dev.mData[i]);
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

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          bool    kHasBias        = false,
          bool    kChipletSwizzle = false,
          index_t kChipletNumXcds = 8,
          index_t kChipletChunk   = 8,
          index_t kNPerWarp       = 1,
          index_t kMPerWarp       = 1,
          index_t kVector         = 8,
          index_t kWarpsPerBlock  = 1,
          bool    kStageAInLds    = false,
          bool    kStreamB        = false,
          bool    kPersistent     = false>
::testing::AssertionResult RunUnscaledCase(const std::string& test_name,
                                           const DecodeShape& shape,
                                           index_t            k_batch)
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
                                              kVector,
                                              /*kUseDot2=*/false,
                                              /*kUsePackedFp32=*/false,
                                              kMPerWarp,
                                              kNPerWarp,
                                              GemmDecodeOutputAxis::SmallM,
                                              kHasBias,
                                              kWarpsPerBlock,
                                              /*kBPreshuffle=*/false,
                                              kChipletSwizzle,
                                              kChipletNumXcds,
                                              kChipletChunk,
                                              /*kBias2D=*/false,
                                              /*kStageAInLds=*/kStageAInLds,
                                              /*kStreamB=*/kStreamB,
                                              /*kPersistent=*/kPersistent>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    HostTensor<ADataType> a({shape.M, shape.K});
    HostTensor<BDataType> b({shape.N, shape.K});
    HostTensor<CDataType> bias_host({shape.N});
    HostTensor<CDataType> c_host({shape.M, shape.N});
    HostTensor<CDataType> c_dev({shape.M, shape.N});

    FillRandom(a, -1.0f, 1.0f, 0xA1u);
    FillRandom(b, -1.0f, 1.0f, 0xB2u);
    if constexpr(kHasBias)
    {
        FillRandom(bias_host, -2.0f, 2.0f, 0xC3u);
    }

    ReferenceGemm<ADataType, BDataType, CDataType>(a, b, c_host, shape.M, shape.N, shape.K);
    if constexpr(kHasBias)
    {
        for(index_t m = 0; m < shape.M; ++m)
        {
            for(index_t n = 0; n < shape.N; ++n)
            {
                c_host(m, n) = type_convert<CDataType>(type_convert<float>(c_host(m, n)) +
                                                       type_convert<float>(bias_host(n)));
            }
        }
    }

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c_dev.get_element_space_size_in_bytes());
    DeviceMem bias_buf(bias_host.get_element_space_size_in_bytes());

    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    if constexpr(kHasBias)
    {
        bias_buf.ToDevice(bias_host.mData.data());
    }

    // AtomicAdd split-K accumulates partials, so the destination must start at
    // zero. For k_batch == 1 the kernel does a plain store so the zero-init is
    // not strictly needed, but doing it unconditionally simplifies the test
    // bookkeeping.
    c_buf.SetZero();

    typename Kernel::Kargs kargs;
    if constexpr(kHasBias)
    {
        kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                       b_buf.GetDeviceBuffer(),
                                       c_buf.GetDeviceBuffer(),
                                       /*p_x_scale=*/nullptr,
                                       /*p_w_scale=*/nullptr,
                                       bias_buf.GetDeviceBuffer(),
                                       shape.M,
                                       shape.N,
                                       shape.K,
                                       /*stride_a=*/shape.K,
                                       /*stride_b=*/shape.K,
                                       /*stride_c=*/shape.N,
                                       /*k_batch=*/k_batch);
    }
    else
    {
        kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                       b_buf.GetDeviceBuffer(),
                                       c_buf.GetDeviceBuffer(),
                                       shape.M,
                                       shape.N,
                                       shape.K,
                                       /*stride_a=*/shape.K,
                                       /*stride_b=*/shape.K,
                                       /*stride_c=*/shape.N,
                                       /*k_batch=*/k_batch);
    }

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

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          bool kHasBias = false>
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
                EXPECT_TRUE((RunUnscaledCase<ADataType, BDataType, CDataType, kHasBias>(
                    name, shape, kb)));
            }
        }
    }
}

// 2D modular-broadcast bias (wvSplitK* Bx/By). The bias has logical shape
// [By, Bx] (row-major, flattened to By*Bx); the effective bias for output
// (m, n) is BIAS[(n % Bx) + (m % By) * Bx]. Exercises Problem::kBias2D + the
// Kargs bias_x / bias_y extents in the unscaled epilogue. By == 1 with Bx == N
// must reproduce the flat 1D bias bit-for-bit (validated by passing those
// extents and the same reference).
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          index_t kNPerWarp      = 1,
          index_t kMPerWarp      = 1,
          index_t kWarpsPerBlock = 1>
::testing::AssertionResult RunUnscaledBias2DCase(const std::string& test_name,
                                                 const DecodeShape& shape,
                                                 index_t            Bx,
                                                 index_t            By,
                                                 index_t            k_batch)
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
                                              kMPerWarp,
                                              kNPerWarp,
                                              GemmDecodeOutputAxis::SmallM,
                                              /*kHasBias=*/true,
                                              kWarpsPerBlock,
                                              /*kBPreshuffle=*/false,
                                              /*kChipletSwizzle=*/false,
                                              /*kChipletNumXcds=*/8,
                                              /*kChipletChunkSize=*/8,
                                              /*kBias2D=*/true>;
    using Kernel = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    HostTensor<ADataType> a({shape.M, shape.K});
    HostTensor<BDataType> b({shape.N, shape.K});
    HostTensor<CDataType> bias2d({By * Bx});
    HostTensor<CDataType> c_host({shape.M, shape.N});
    HostTensor<CDataType> c_dev({shape.M, shape.N});

    FillRandom(a, -1.0f, 1.0f, 0xA1u);
    FillRandom(b, -1.0f, 1.0f, 0xB2u);
    FillRandom(bias2d, -2.0f, 2.0f, 0xC3u);

    ReferenceGemm<ADataType, BDataType, CDataType>(a, b, c_host, shape.M, shape.N, shape.K);
    for(index_t m = 0; m < shape.M; ++m)
    {
        for(index_t n = 0; n < shape.N; ++n)
        {
            const index_t idx = (n % Bx) + (m % By) * Bx;
            c_host(m, n)      = type_convert<CDataType>(type_convert<float>(c_host(m, n)) +
                                                   type_convert<float>(bias2d.mData[idx]));
        }
    }

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c_dev.get_element_space_size_in_bytes());
    DeviceMem bias_buf(bias2d.get_element_space_size_in_bytes());

    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    bias_buf.ToDevice(bias2d.mData.data());
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        /*p_x_scale=*/nullptr,
                                        /*p_w_scale=*/nullptr,
                                        bias_buf.GetDeviceBuffer(),
                                        shape.M,
                                        shape.N,
                                        shape.K,
                                        /*stride_a=*/shape.K,
                                        /*stride_b=*/shape.K,
                                        /*stride_c=*/shape.N,
                                        /*k_batch=*/k_batch,
                                        /*bias_x=*/Bx,
                                        /*bias_y=*/By);

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
    return CompareHostDevice<CDataType>(test_name, shape, k_batch, c_host, c_dev,
                                        AbsoluteTolerance<CDataType>(shape.K));
}

#ifdef CK_TILE_USE_OCP_FP8
// PerTensor FP8 reference: dequant inputs to FP32, sum, then multiply by
// sA * sB. Mirrors the kernel's epilogue convention.
template <typename ADataType, typename BDataType, typename CDataType>
void ReferenceGemmPerTensor(const HostTensor<ADataType>& a,
                            const HostTensor<BDataType>& b,
                            float                        sA,
                            float                        sB,
                            HostTensor<CDataType>&       c,
                            index_t                      M,
                            index_t                      N,
                            index_t                      K)
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
            c(m, n) = type_convert<CDataType>(acc * sA * sB);
        }
    }
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          bool    kHasBias       = false,
          index_t kNPerWarp      = 1,
          index_t kMPerWarp      = 1,
          index_t kWarpsPerBlock = 1,
          bool    kStageAInLds   = false,
          bool    kStreamB       = false,
          bool    kPersistent    = false>
::testing::AssertionResult RunFp8PerTensorCase(const std::string& test_name,
                                               const DecodeShape& shape,
                                               index_t            k_batch)
{
    using ComputeDataType = float;
    using Problem         = GemmDecodeProblem<ADataType,
                                              BDataType,
                                              ComputeDataType,
                                              CDataType,
                                              /*XScaleDataType=*/float,
                                              /*WScaleDataType=*/float,
                                              GemmDecodeScaleLayout::PerTensor,
                                              GemmDecodeScaleLayout::PerTensor,
                                              /*kVector=*/16,
                                              /*kUseDot2=*/true,
                                              /*kUsePackedFp32=*/false,
                                              kMPerWarp,
                                              kNPerWarp,
                                              GemmDecodeOutputAxis::SmallM,
                                              kHasBias,
                                              kWarpsPerBlock,
                                              /*kBPreshuffle=*/false,
                                              /*kChipletSwizzle=*/false,
                                              /*kChipletNumXcds=*/8,
                                              /*kChipletChunkSize=*/8,
                                              /*kBias2D=*/false,
                                              /*kStageAInLds=*/kStageAInLds,
                                              /*kStreamB=*/kStreamB,
                                              /*kPersistent=*/kPersistent>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    HostTensor<ADataType> a({shape.M, shape.K});
    HostTensor<BDataType> b({shape.N, shape.K});
    HostTensor<CDataType> bias_host({shape.N});
    HostTensor<CDataType> c_host({shape.M, shape.N});
    HostTensor<CDataType> c_dev({shape.M, shape.N});

    // FP8 has limited dynamic range; keep inputs in [-1, 1].
    FillRandom(a, -1.0f, 1.0f, 0xA1u);
    FillRandom(b, -1.0f, 1.0f, 0xB2u);
    if constexpr(kHasBias)
    {
        FillRandom(bias_host, -1.0f, 1.0f, 0xC3u);
    }

    const float sA = 0.125f; // arbitrary non-trivial scales
    const float sB = 0.0625f;

    ReferenceGemmPerTensor<ADataType, BDataType, CDataType>(a, b, sA, sB, c_host,
                                                            shape.M, shape.N, shape.K);
    if constexpr(kHasBias)
    {
        for(index_t m = 0; m < shape.M; ++m)
        {
            for(index_t n = 0; n < shape.N; ++n)
            {
                c_host(m, n) = type_convert<CDataType>(type_convert<float>(c_host(m, n)) +
                                                       type_convert<float>(bias_host(n)));
            }
        }
    }

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c_dev.get_element_space_size_in_bytes());
    DeviceMem sa_buf(sizeof(float));
    DeviceMem sb_buf(sizeof(float));
    DeviceMem bias_buf(bias_host.get_element_space_size_in_bytes());

    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    sa_buf.ToDevice(&sA);
    sb_buf.ToDevice(&sB);
    if constexpr(kHasBias)
    {
        bias_buf.ToDevice(bias_host.mData.data());
    }
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(
        a_buf.GetDeviceBuffer(),
        b_buf.GetDeviceBuffer(),
        c_buf.GetDeviceBuffer(),
        sa_buf.GetDeviceBuffer(),
        sb_buf.GetDeviceBuffer(),
        kHasBias ? bias_buf.GetDeviceBuffer() : nullptr,
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

    const float atol  = AbsoluteTolerance<CDataType, ADataType>(shape.K);
    float max_diff    = 0.0f;
    index_t bad_index = -1;
    for(index_t i = 0; i < static_cast<index_t>(c_host.get_element_space_size()); ++i)
    {
        const float h    = type_convert<float>(c_host.mData[i]);
        const float d    = type_convert<float>(c_dev.mData[i]);
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

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          bool kHasBias = false>
void RunFp8PerTensorMatrix(const std::string& dtype_name)
{
    constexpr index_t K = 7168;
    const std::vector<index_t> Ms{1, 2, 4, 8};
    const std::vector<index_t> Ns{512, 4096, 8192};
    const std::vector<index_t> KBatches{1, 2};

    for(index_t M : Ms)
    {
        for(index_t N : Ns)
        {
            for(index_t kb : KBatches)
            {
                const DecodeShape shape{M, N, K};
                const std::string name = dtype_name + " M=" + std::to_string(M) +
                                         " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                         " kb=" + std::to_string(kb);
                EXPECT_TRUE((RunFp8PerTensorCase<ADataType, BDataType, CDataType, kHasBias>(
                    name, shape, kb)));
            }
        }
    }
}

// PerToken FP8 reference: per-token (per output row m) X scale sA[m] and a
// single per-tensor W scale sB. c(m,n) = (sum_k a*b) * sA[m] * sB. This is the
// wvSplitKQ per-token activation-quant member of the family.
template <typename ADataType, typename BDataType, typename CDataType>
void ReferenceGemmPerToken(const HostTensor<ADataType>& a,
                           const HostTensor<BDataType>& b,
                           const std::vector<float>&    sA, // [M]
                           float                        sB,
                           HostTensor<CDataType>&       c,
                           index_t                      M,
                           index_t                      N,
                           index_t                      K)
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
            c(m, n) = type_convert<CDataType>(acc * sA[m] * sB);
        }
    }
}

// PerToken FP8 case: X = GemmDecodeScaleLayout::PerToken (an [M] FP32 scale
// vector), W = PerTensor (one FP32 scalar). Mirrors RunFp8PerTensorCase; the
// only kernel-visible difference is the X-scale layout (per-row gather vs
// scalar), so it exercises the per-token epilogue + IsSupportedArgument path.
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          index_t kNPerWarp      = 1,
          index_t kMPerWarp      = 1,
          index_t kWarpsPerBlock = 1,
          bool    kStageAInLds   = false,
          bool    kStreamB       = false,
          bool    kPersistent    = false>
::testing::AssertionResult RunFp8PerTokenCase(const std::string& test_name,
                                              const DecodeShape& shape,
                                              index_t            k_batch)
{
    using ComputeDataType = float;
    using Problem         = GemmDecodeProblem<ADataType,
                                              BDataType,
                                              ComputeDataType,
                                              CDataType,
                                              /*XScaleDataType=*/float,
                                              /*WScaleDataType=*/float,
                                              GemmDecodeScaleLayout::PerToken,
                                              GemmDecodeScaleLayout::PerTensor,
                                              /*kVector=*/16,
                                              /*kUseDot2=*/true,
                                              /*kUsePackedFp32=*/false,
                                              kMPerWarp,
                                              kNPerWarp,
                                              GemmDecodeOutputAxis::SmallM,
                                              /*kHasBias=*/false,
                                              kWarpsPerBlock,
                                              /*kBPreshuffle=*/false,
                                              /*kChipletSwizzle=*/false,
                                              /*kChipletNumXcds=*/8,
                                              /*kChipletChunkSize=*/8,
                                              /*kBias2D=*/false,
                                              /*kStageAInLds=*/kStageAInLds,
                                              /*kStreamB=*/kStreamB,
                                              /*kPersistent=*/kPersistent>;
    using Kernel  = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    HostTensor<ADataType> a({shape.M, shape.K});
    HostTensor<BDataType> b({shape.N, shape.K});
    HostTensor<CDataType> c_host({shape.M, shape.N});
    HostTensor<CDataType> c_dev({shape.M, shape.N});

    FillRandom(a, -1.0f, 1.0f, 0xA1u);
    FillRandom(b, -1.0f, 1.0f, 0xB2u);

    // Distinct per-token X scales so a wrong (e.g. row-0-broadcast) index fails.
    std::vector<float> sA(static_cast<size_t>(shape.M));
    for(index_t m = 0; m < shape.M; ++m)
    {
        sA[static_cast<size_t>(m)] = 0.0625f * static_cast<float>(m + 1);
    }
    const float sB = 0.125f;

    ReferenceGemmPerToken<ADataType, BDataType, CDataType>(a, b, sA, sB, c_host,
                                                           shape.M, shape.N, shape.K);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c_dev.get_element_space_size_in_bytes());
    DeviceMem sa_buf(sA.size() * sizeof(float));
    DeviceMem sb_buf(sizeof(float));

    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    sa_buf.ToDevice(sA.data());
    sb_buf.ToDevice(&sB);
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(
        a_buf.GetDeviceBuffer(),
        b_buf.GetDeviceBuffer(),
        c_buf.GetDeviceBuffer(),
        sa_buf.GetDeviceBuffer(),
        sb_buf.GetDeviceBuffer(),
        /*p_bias=*/nullptr,
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

    const float atol  = AbsoluteTolerance<CDataType, ADataType>(shape.K);
    float max_diff    = 0.0f;
    index_t bad_index = -1;
    for(index_t i = 0; i < static_cast<index_t>(c_host.get_element_space_size()); ++i)
    {
        const float h    = type_convert<float>(c_host.mData[i]);
        const float d    = type_convert<float>(c_dev.mData[i]);
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
void RunFp8PerTokenMatrix(const std::string& dtype_name)
{
    constexpr index_t K = 7168;
    const std::vector<index_t> Ms{1, 2, 4, 8};
    const std::vector<index_t> Ns{512, 4096, 8192};
    const std::vector<index_t> KBatches{1, 2};

    for(index_t M : Ms)
    {
        for(index_t N : Ns)
        {
            for(index_t kb : KBatches)
            {
                const DecodeShape shape{M, N, K};
                const std::string name = dtype_name + " M=" + std::to_string(M) +
                                         " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                         " kb=" + std::to_string(kb);
                EXPECT_TRUE((RunFp8PerTokenCase<ADataType, BDataType, CDataType>(name, shape, kb)));
            }
        }
    }
}

// FP8 per-tensor + 2D modular-broadcast bias: the wvSplitKQ target. Mirrors
// RunFp8PerTensorCase but with Problem::kBias2D and a [By, Bx] bias added in
// the scaled epilogue (after sA*sB) using BIAS[(n % Bx) + (m % By) * Bx].
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          index_t kNPerWarp      = 1,
          index_t kMPerWarp      = 1,
          index_t kWarpsPerBlock = 1>
::testing::AssertionResult RunFp8Bias2DCase(const std::string& test_name,
                                            const DecodeShape& shape,
                                            index_t            Bx,
                                            index_t            By,
                                            index_t            k_batch)
{
    using ComputeDataType = float;
    using Problem         = GemmDecodeProblem<ADataType,
                                              BDataType,
                                              ComputeDataType,
                                              CDataType,
                                              /*XScaleDataType=*/float,
                                              /*WScaleDataType=*/float,
                                              GemmDecodeScaleLayout::PerTensor,
                                              GemmDecodeScaleLayout::PerTensor,
                                              /*kVector=*/16,
                                              /*kUseDot2=*/true,
                                              /*kUsePackedFp32=*/false,
                                              kMPerWarp,
                                              kNPerWarp,
                                              GemmDecodeOutputAxis::SmallM,
                                              /*kHasBias=*/true,
                                              kWarpsPerBlock,
                                              /*kBPreshuffle=*/false,
                                              /*kChipletSwizzle=*/false,
                                              /*kChipletNumXcds=*/8,
                                              /*kChipletChunkSize=*/8,
                                              /*kBias2D=*/true>;
    using Kernel = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    HostTensor<ADataType> a({shape.M, shape.K});
    HostTensor<BDataType> b({shape.N, shape.K});
    HostTensor<CDataType> bias2d({By * Bx});
    HostTensor<CDataType> c_host({shape.M, shape.N});
    HostTensor<CDataType> c_dev({shape.M, shape.N});

    FillRandom(a, -1.0f, 1.0f, 0xA1u);
    FillRandom(b, -1.0f, 1.0f, 0xB2u);
    FillRandom(bias2d, -1.0f, 1.0f, 0xC3u);

    const float sA = 0.125f;
    const float sB = 0.0625f;

    ReferenceGemmPerTensor<ADataType, BDataType, CDataType>(a, b, sA, sB, c_host,
                                                            shape.M, shape.N, shape.K);
    for(index_t m = 0; m < shape.M; ++m)
    {
        for(index_t n = 0; n < shape.N; ++n)
        {
            const index_t idx = (n % Bx) + (m % By) * Bx;
            c_host(m, n)      = type_convert<CDataType>(type_convert<float>(c_host(m, n)) +
                                                   type_convert<float>(bias2d.mData[idx]));
        }
    }

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c_dev.get_element_space_size_in_bytes());
    DeviceMem sa_buf(sizeof(float));
    DeviceMem sb_buf(sizeof(float));
    DeviceMem bias_buf(bias2d.get_element_space_size_in_bytes());

    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    sa_buf.ToDevice(&sA);
    sb_buf.ToDevice(&sB);
    bias_buf.ToDevice(bias2d.mData.data());
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        sa_buf.GetDeviceBuffer(),
                                        sb_buf.GetDeviceBuffer(),
                                        bias_buf.GetDeviceBuffer(),
                                        shape.M,
                                        shape.N,
                                        shape.K,
                                        /*stride_a=*/shape.K,
                                        /*stride_b=*/shape.K,
                                        /*stride_c=*/shape.N,
                                        /*k_batch=*/k_batch,
                                        /*bias_x=*/Bx,
                                        /*bias_y=*/By);

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
    return CompareHostDevice<CDataType>(test_name, shape, k_batch, c_host, c_dev,
                                        AbsoluteTolerance<CDataType, ADataType>(shape.K));
}
#endif // CK_TILE_USE_OCP_FP8

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

TEST(GemmDecodeUniversalUnscaled, Bf16Bf16BiasMatrix)
{
    RunMatrix<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true>("BF16/BF16+bias");
}

TEST(GemmDecodeUniversalUnscaled, Fp16Fp16BiasMatrix)
{
    RunMatrix<fp16_t, fp16_t, fp16_t, /*kHasBias=*/true>("FP16/FP16+bias");
}

// 2D modular-broadcast bias parity (wvSplitK* Bx/By). Validates the kBias2D
// epilogue index BIAS[(n % Bx) + (m % By) * Bx] across: a full feature period
// (Bx = N) with a token period By, feature-axis tiling (Bx < N so n wraps),
// the 1D-equivalent (By = 1, Bx = N), and composition with split-K, the
// multi-warp epilogue site, and M/N register tiling.
TEST(GemmDecodeUniversalUnscaled, Bf16Bf16Bias2D)
{
    // True 2D: full feature period, token period 4 (M=4 -> each row distinct).
    EXPECT_TRUE((RunUnscaledBias2DCase<bf16_t, bf16_t, bf16_t>(
        "BF16 2Dbias Bx=N By=4", DecodeShape{4, 4096, 7168}, /*Bx=*/4096, /*By=*/4, 1)));
    // Feature-axis tiling: Bx < N so (n % Bx) wraps; token period 2.
    EXPECT_TRUE((RunUnscaledBias2DCase<bf16_t, bf16_t, bf16_t>(
        "BF16 2Dbias Bx=512 By=2", DecodeShape{4, 4096, 7168}, /*Bx=*/512, /*By=*/2, 1)));
    // 1D-equivalence: By = 1, Bx = N is the flat per-feature bias.
    EXPECT_TRUE((RunUnscaledBias2DCase<bf16_t, bf16_t, bf16_t>(
        "BF16 2Dbias By=1 (1D-equiv)", DecodeShape{2, 512, 7168}, /*Bx=*/512, /*By=*/1, 1)));
    // Compose with split-K (bias added on the k_id == 0 shard only).
    EXPECT_TRUE((RunUnscaledBias2DCase<bf16_t, bf16_t, bf16_t>(
        "BF16 2Dbias split=2", DecodeShape{4, 4096, 7168}, /*Bx=*/4096, /*By=*/4, 2)));
    // Compose with the multi-warp epilogue site (kWarpsPerBlock > 1, mp=np=1).
    EXPECT_TRUE((RunUnscaledBias2DCase<bf16_t, bf16_t, bf16_t,
                                       /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/4>(
        "BF16 2Dbias WPB=4", DecodeShape{4, 4096, 7168}, /*Bx=*/512, /*By=*/2, 1)));
    // Compose with the M/N register tile (2x2).
    EXPECT_TRUE((RunUnscaledBias2DCase<bf16_t, bf16_t, bf16_t,
                                       /*kNPerWarp=*/2, /*kMPerWarp=*/2>(
        "BF16 2Dbias MxN=2x2", DecodeShape{4, 4096, 7168}, /*Bx=*/512, /*By=*/4, 1)));
}

// XCD-aware workgroup swizzle: the kernel output must be bit-identical
// (within tolerance) to the un-swizzled path, since the swizzle is a
// pure permutation of which workgroup computes which (m, n) scalar.
// We exercise four shapes that span num_n_blocks above and below the
// chunk_size * num_xcds = 64 boundary, plus a multi-row M to stress
// the (m, n_block) flatten/unflatten.
TEST(GemmDecodeUniversalChipletSwizzle, Bf16Bf16Match)
{
    constexpr bool kSwizzle = true;
    const std::vector<DecodeShape> shapes{
        DecodeShape{1, 512,  7168},
        DecodeShape{1, 4096, 7168},
        DecodeShape{1, 8192, 7168},
        DecodeShape{2, 4096, 7168},
    };
    for(const auto& s : shapes)
    {
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t,
                                     /*kHasBias=*/false,
                                     kSwizzle>("BF16 swizzle", s, 1)));
    }
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t,
                                 /*kHasBias=*/false,
                                 kSwizzle>("BF16 swizzle split-K",
                                           DecodeShape{1, 4096, 7168}, 2)));
}

// N-tile register reuse: one warp computes kNPerWarp adjacent output columns
// by loading the shared A row once and reusing it against kNPerWarp B rows.
// The result must match the FP32 reference exactly as for kNPerWarp = 1.
TEST(GemmDecodeUniversalNReuse, Bf16Bf16)
{
    const std::vector<DecodeShape> shapes{
        DecodeShape{1, 512, 7168},
        DecodeShape{1, 4096, 7168},
        DecodeShape{4, 4096, 7168},
    };
    for(const auto& s : shapes)
    {
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/2>(
            "BF16 NPerWarp=2", s, 1)));
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/4>(
            "BF16 NPerWarp=4", s, 1)));
    }
    // N-reuse must compose with split-K (atomic-add) and with bias.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/4>(
        "BF16 NPerWarp=4 split=2", DecodeShape{2, 4096, 7168}, 2)));
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/2>(
        "BF16 NPerWarp=2 bias", DecodeShape{1, 4096, 7168}, 1)));
}

// N-reuse composed with the chiplet swizzle: the grid is now
// ceil(N / kNPerWarp) wide on the n axis, so this also checks the
// flatten/unflatten still lands every (m, n_block).
TEST(GemmDecodeUniversalNReuse, Bf16Bf16WithSwizzle)
{
    const std::vector<DecodeShape> shapes{
        DecodeShape{1, 4096, 7168},
        DecodeShape{4, 8192, 7168},
    };
    for(const auto& s : shapes)
    {
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/true, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/4>(
            "BF16 swizzle+NPerWarp=4", s, 1)));
    }
}

// M-tile register reuse (B-reuse): one warp computes kMPerWarp adjacent output
// rows, loading each B row once and reusing it against the kMPerWarp A rows
// held in registers. The non-divisible M shapes exercise the
// ceil(M / kMPerWarp) tail block: out-of-range A rows are clamped on load and
// masked out in the epilogue, so the result must still match the FP32
// reference exactly.
TEST(GemmDecodeUniversalMReuse, Bf16Bf16)
{
    // kMPerWarp = 2, including M = 3 (one masked tail row).
    const std::vector<DecodeShape> shapes_mp2{
        DecodeShape{2, 4096, 7168},
        DecodeShape{3, 4096, 7168},
        DecodeShape{8, 8192, 7168},
    };
    for(const auto& s : shapes_mp2)
    {
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/2>(
            "BF16 MPerWarp=2", s, 1)));
    }
    // kMPerWarp = 4, including M = 5 and 7 (three / one masked tail rows).
    const std::vector<DecodeShape> shapes_mp4{
        DecodeShape{4, 4096, 7168},
        DecodeShape{5, 4096, 7168},
        DecodeShape{7, 8192, 7168},
    };
    for(const auto& s : shapes_mp4)
    {
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/4>(
            "BF16 MPerWarp=4", s, 1)));
    }

    // Combined M- and N-reuse (2x2 and 4x2 register tiles), with bias.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/2, /*kMPerWarp=*/2>(
        "BF16 MxN=2x2", DecodeShape{3, 4096, 7168}, 1)));
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/2, /*kMPerWarp=*/4>(
        "BF16 MxN=4x2 bias", DecodeShape{6, 4096, 7168}, 1)));

    // B-reuse must compose with split-K (atomic-add) including a tail block.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/4>(
        "BF16 MPerWarp=4 split=2", DecodeShape{6, 4096, 7168}, 2)));
}

// B-reuse composed with the chiplet swizzle: the grid is now
// ceil(M / kMPerWarp) tall on the m axis, so this checks the flatten/unflatten
// still lands every (m_block, n_block) when both axes are register-tiled,
// including a tail block (M = 5, kMPerWarp = 2).
TEST(GemmDecodeUniversalMReuse, Bf16Bf16WithSwizzle)
{
    const std::vector<DecodeShape> shapes{
        DecodeShape{4, 4096, 7168},
        DecodeShape{5, 8192, 7168},
    };
    for(const auto& s : shapes)
    {
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/true, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/2, /*kMPerWarp=*/2>(
            "BF16 swizzle+MxN=2x2", s, 1)));
    }
}

// A5: wide vectorized global loads (kVector = 16, i.e. 32-byte BF16 loads).
// The output tile distribution is <warp_size, kVector>, so K must be divisible
// by warp_size * kVector (= 1024 here); 7168 = 7 * 1024. Exercises the wide
// load on its own and composed with N-/M-reuse and the chiplet swizzle.
TEST(GemmDecodeUniversalWideVector, Bf16Bf16)
{
    // Plain warp-per-scalar (mp = np = 1) at kVector = 16.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16>(
        "BF16 kVector=16", DecodeShape{1, 8192, 7168}, 1)));

    // Wide loads composed with N-reuse (A1).
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/2, /*kMPerWarp=*/1,
                                 /*kVector=*/16>(
        "BF16 kVector=16 np=2", DecodeShape{2, 8192, 7168}, 1)));

    // Wide loads composed with B-reuse (A4) + bias, including a masked tail
    // (M = 5, kMPerWarp = 4).
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/4,
                                 /*kVector=*/16>(
        "BF16 kVector=16 mp=4 bias", DecodeShape{5, 4096, 7168}, 1)));

    // Wide loads + chiplet swizzle on a 2x2 register tile.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/true, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/2, /*kMPerWarp=*/2,
                                 /*kVector=*/16>(
        "BF16 kVector=16 swizzle MxN=2x2", DecodeShape{4, 8192, 7168}, 1)));
}

// Multi-warp occupancy path (design doc §15.F probe): kWarpsPerBlock independent
// warps per workgroup, each owning one output column, sharing the activation
// row via the broadcast distribution. This is a pure scheduling change, so the
// result must match the single-warp FP32 reference exactly. mp=np=1 only;
// N must be divisible by kWarpsPerBlock and k_batch must be 1.
TEST(GemmDecodeUniversalMultiWarp, Bf16Bf16)
{
    const std::vector<DecodeShape> shapes{
        DecodeShape{1, 4096, 7168},
        DecodeShape{1, 7168, 7168},
        DecodeShape{1, 8192, 7168},
        DecodeShape{4, 4096, 7168},
    };
    for(const auto& s : shapes)
    {
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                     /*kVector=*/16, /*kWarpsPerBlock=*/2>("BF16 WPB=2", s, 1)));
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                     /*kVector=*/16, /*kWarpsPerBlock=*/4>("BF16 WPB=4", s, 1)));
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                     /*kVector=*/16, /*kWarpsPerBlock=*/8>("BF16 WPB=8", s, 1)));
    }
    // Multi-warp composed with bias.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/4>(
        "BF16 WPB=4 bias", DecodeShape{1, 8192, 7168}, 1)));
}

// A-in-LDS staging on the multi-warp path (wvSplitK* A-staging / WD-OPT-21):
// the workgroup stages the shared activation row in LDS once and every warp
// streams it from LDS instead of re-reading global. Pure data-source change,
// so the result must match the single-warp FP32 reference exactly. Requires
// kWarpsPerBlock > 1, mp = np = 1, k_batch = 1, and K <= kLdsStageMaxK (8192).
TEST(GemmDecodeUniversalMultiWarp, Bf16Bf16LdsStage)
{
    const std::vector<DecodeShape> shapes{
        DecodeShape{1, 4096, 7168},
        DecodeShape{1, 8192, 7168},
    };
    for(const auto& s : shapes)
    {
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                     /*kVector=*/16, /*kWarpsPerBlock=*/4,
                                     /*kStageAInLds=*/true>("BF16 WPB=4 A-LDS", s, 1)));
        EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                     /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                     /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                     /*kVector=*/16, /*kWarpsPerBlock=*/8,
                                     /*kStageAInLds=*/true>("BF16 WPB=8 A-LDS", s, 1)));
    }
    // A-LDS composed with the multi-warp bias epilogue site.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/4,
                                 /*kStageAInLds=*/true>("BF16 WPB=4 A-LDS bias",
                                                        DecodeShape{1, 4096, 7168}, 1)));
}

// Non-temporal B loads (wvSplitK* cache-bypass). kStreamB only changes the
// coherence hint on the B (weight) buffer loads, so the numerical result must
// be identical to the cacheable path / FP32 reference. Exercise it across the
// single-warp main path, the register-tiled path, split-K, the multi-warp
// path, and composed with A-in-LDS staging and bias.
TEST(GemmDecodeUniversalStreamB, Bf16Bf16)
{
    const DecodeShape shape{4, 8192, 7168};

    // Single-warp main path.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/1,
                                 /*kStageAInLds=*/false, /*kStreamB=*/true>(
        "BF16 stream-B", shape, 1)));
    // Split-K (grid-z atomic reduction) with streamed B.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/1,
                                 /*kStageAInLds=*/false, /*kStreamB=*/true>(
        "BF16 stream-B split=4", shape, 4)));
    // Register tiling (kNPerWarp x kMPerWarp) with streamed B.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/2, /*kMPerWarp=*/2,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/1,
                                 /*kStageAInLds=*/false, /*kStreamB=*/true>(
        "BF16 stream-B MxN=2x2 bias", shape, 1)));
    // Multi-warp + A-in-LDS + bias, all composed with streamed B.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/4,
                                 /*kStageAInLds=*/true, /*kStreamB=*/true>(
        "BF16 stream-B WPB=4 A-LDS bias", DecodeShape{1, 4096, 7168}, 1)));
}

// Persistent fat-WG launch (wvSplitK* "1 WG/CU"). kPersistent caps the grid at
// the CU count and grid-strides each workgroup over the tile space, so it is a
// pure scheduling change: every tile is computed exactly once and the result
// must match the per-tile launch / FP32 reference. Exercise the main path,
// split-K (grid-z folded into the persistent work index), register tiling, the
// chiplet swizzle (its unflatten must use the logical grid, not the CU count),
// and the multi-warp + A-in-LDS + bias fat-WG combo (the wvSplitKQ recipe,
// where the persistent leading LDS barrier guards a_smem reuse across strides).
TEST(GemmDecodeUniversalPersistent, Bf16Bf16)
{
    const DecodeShape shape{4, 8192, 7168};

    // Single-warp main path, grid-strided.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/1,
                                 /*kStageAInLds=*/false, /*kStreamB=*/false,
                                 /*kPersistent=*/true>("BF16 persistent", shape, 1)));
    // Split-K shards enumerated by the same persistent work index.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/1,
                                 /*kStageAInLds=*/false, /*kStreamB=*/false,
                                 /*kPersistent=*/true>("BF16 persistent split=4 bias", shape, 4)));
    // Register tiling (kNPerWarp x kMPerWarp) under the persistent grid.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/2, /*kMPerWarp=*/2,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/1,
                                 /*kStageAInLds=*/false, /*kStreamB=*/false,
                                 /*kPersistent=*/true>("BF16 persistent MxN=2x2", shape, 1)));
    // Persistent composed with the chiplet swizzle (logical-grid unflatten).
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/false,
                                 /*kChipletSwizzle=*/true, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/1,
                                 /*kStageAInLds=*/false, /*kStreamB=*/false,
                                 /*kPersistent=*/true>("BF16 persistent swizzle", shape, 1)));
    // Fat-WG recipe: multi-warp + A-in-LDS + bias + streamed B, all persistent.
    EXPECT_TRUE((RunUnscaledCase<bf16_t, bf16_t, bf16_t, /*kHasBias=*/true,
                                 /*kChipletSwizzle=*/false, /*kChipletNumXcds=*/8,
                                 /*kChipletChunk=*/8, /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                 /*kVector=*/16, /*kWarpsPerBlock=*/4,
                                 /*kStageAInLds=*/true, /*kStreamB=*/true,
                                 /*kPersistent=*/true>(
        "BF16 persistent WPB=4 A-LDS bias stream-B", DecodeShape{1, 4096, 7168}, 1)));
}

#ifdef CK_TILE_USE_OCP_FP8

namespace {

// Blockscale reference: dequantize each FP8 element with its (row, k_block)
// scale and accumulate in FP32. Mirrors the kernel's contract.
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          index_t XScaleBlockN,
          index_t XScaleBlockK,
          index_t WScaleBlockN,
          index_t WScaleBlockK>
void ReferenceGemmBlockscale(const HostTensor<ADataType>& a,
                             const HostTensor<BDataType>& b,
                             const HostTensor<float>&     x_scale, // [M / XScaleBlockN,
                                                                   //  K / XScaleBlockK]
                             const HostTensor<float>&     w_scale, // [N / WScaleBlockN,
                                                                   //  K / WScaleBlockK]
                             HostTensor<CDataType>&       c,
                             index_t                      M,
                             index_t                      N,
                             index_t                      K)
{
    const index_t aqk = K / XScaleBlockK;
    const index_t bqk = K / WScaleBlockK;
    for(index_t m = 0; m < M; ++m)
    {
        for(index_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for(index_t k = 0; k < K; ++k)
            {
                const float xs = x_scale(m / XScaleBlockN, k / XScaleBlockK);
                const float ws = w_scale(n / WScaleBlockN, k / WScaleBlockK);
                const float a_f = type_convert<float>(a(m, k));
                const float b_f = type_convert<float>(b(n, k));
                acc += a_f * b_f * xs * ws;
                (void)aqk;
                (void)bqk;
            }
            c(m, n) = type_convert<CDataType>(acc);
        }
    }
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          index_t XScaleBlockN = 1,
          index_t XScaleBlockK = 128,
          index_t WScaleBlockN = 128,
          index_t WScaleBlockK = 128>
::testing::AssertionResult RunBlockscaleCase(const std::string& test_name,
                                             const DecodeShape& shape,
                                             index_t            k_batch)
{
    using ComputeDataType = float;
    using XLayout = GemmDecodeScaleLayout::Block2D<XScaleBlockN, XScaleBlockK>;
    using WLayout = GemmDecodeScaleLayout::Block2D<WScaleBlockN, WScaleBlockK>;
    using Problem = GemmDecodeProblem<ADataType,
                                      BDataType,
                                      ComputeDataType,
                                      CDataType,
                                      /*XScaleDataType=*/float,
                                      /*WScaleDataType=*/float,
                                      XLayout,
                                      WLayout,
                                      /*kVector=*/16,
                                      /*kUseDot2=*/true,
                                      /*kUsePackedFp32=*/false,
                                      /*kMPerWarp=*/1,
                                      /*kNPerWarp=*/1,
                                      GemmDecodeOutputAxis::SmallM,
                                      /*kHasBias=*/false,
                                      /*kWarpsPerBlock=*/1>;
    using Kernel = GemmDecodeBlockscaleKernel<Problem, GemmDecodePolicy>;

    const index_t aqn = shape.M / XScaleBlockN;
    const index_t aqk = shape.K / XScaleBlockK;
    const index_t bqn = shape.N / WScaleBlockN;
    const index_t bqk = shape.K / WScaleBlockK;

    HostTensor<ADataType> a({shape.M, shape.K});
    HostTensor<BDataType> b({shape.N, shape.K});
    HostTensor<float> x_scale({aqn, aqk});
    HostTensor<float> w_scale({bqn, bqk});
    HostTensor<CDataType> c_host({shape.M, shape.N});
    HostTensor<CDataType> c_dev({shape.M, shape.N});

    FillRandom(a, -1.0f, 1.0f, 0xA1u);
    FillRandom(b, -1.0f, 1.0f, 0xB2u);
    // Scales positive, modest variation so the dequant magnitude stays bounded.
    FillRandom(x_scale, 0.05f, 0.25f, 0xD4u);
    FillRandom(w_scale, 0.05f, 0.25f, 0xE5u);

    ReferenceGemmBlockscale<ADataType,
                            BDataType,
                            CDataType,
                            XScaleBlockN,
                            XScaleBlockK,
                            WScaleBlockN,
                            WScaleBlockK>(a, b, x_scale, w_scale, c_host,
                                          shape.M, shape.N, shape.K);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c_dev.get_element_space_size_in_bytes());
    DeviceMem xs_buf(x_scale.get_element_space_size_in_bytes());
    DeviceMem ws_buf(w_scale.get_element_space_size_in_bytes());

    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    xs_buf.ToDevice(x_scale.mData.data());
    ws_buf.ToDevice(w_scale.mData.data());
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        xs_buf.GetDeviceBuffer(),
                                        ws_buf.GetDeviceBuffer(),
                                        /*p_bias=*/nullptr,
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
        launch_gemm_decode_blockscale<Kernel>(kargs, s);
    }
    catch(const std::exception& ex)
    {
        return ::testing::AssertionFailure() << test_name << ": launch threw: " << ex.what();
    }

    c_buf.FromDevice(c_dev.mData.data());

    const float atol = AbsoluteTolerance<CDataType, ADataType>(shape.K);
    float max_diff = 0.0f;
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
void RunBlockscaleMatrix(const std::string& dtype_name)
{
    const std::vector<index_t> Ms{1, 2, 4};
    const std::vector<index_t> Ns{2048, 4096};
    const std::vector<index_t> Ks{2048, 7168};
    const std::vector<index_t> KBatches{1, 2};

    for(index_t M : Ms)
    {
        for(index_t N : Ns)
        {
            for(index_t K : Ks)
            {
                for(index_t kb : KBatches)
                {
                    const DecodeShape shape{M, N, K};
                    const std::string name = dtype_name + " M=" + std::to_string(M) +
                                             " N=" + std::to_string(N) +
                                             " K=" + std::to_string(K) +
                                             " kb=" + std::to_string(kb);
                    EXPECT_TRUE(
                        (RunBlockscaleCase<ADataType, BDataType, CDataType>(name, shape, kb)));
                }
            }
        }
    }
}

} // namespace

TEST(GemmDecodeUniversalFp8, Fp8Fp8ToBf16PerTensorMatrix)
{
    RunFp8PerTensorMatrix<fp8_t, fp8_t, bf16_t>("FP8/FP8/BF16");
}

TEST(GemmDecodeUniversalFp8, Fp8Fp8ToFp16PerTensorMatrix)
{
    RunFp8PerTensorMatrix<fp8_t, fp8_t, fp16_t>("FP8/FP8/FP16");
}

TEST(GemmDecodeUniversalFp8, Fp8Fp8ToBf16PerTensorBiasMatrix)
{
    RunFp8PerTensorMatrix<fp8_t, fp8_t, bf16_t, /*kHasBias=*/true>("FP8/FP8/BF16+bias");
}

// FP8 per-token activation quant (X = [M] scale vector, W = per-tensor scalar):
// the wvSplitKQ per-token member of the family. Each token row m carries its
// own scale sA[m], applied per-row in the epilogue. Distinct per-row scales
// catch a wrong (broadcast/row-0) index.
TEST(GemmDecodeUniversalFp8, Fp8Fp8ToBf16PerTokenMatrix)
{
    RunFp8PerTokenMatrix<fp8_t, fp8_t, bf16_t>("FP8/FP8/BF16 per-token");
}

TEST(GemmDecodeUniversalFp8, Fp8Fp8ToFp16PerTokenMatrix)
{
    RunFp8PerTokenMatrix<fp8_t, fp8_t, fp16_t>("FP8/FP8/FP16 per-token");
}

// Per-token composed with the register tile and the multi-warp epilogue site,
// so the per-row x_scale[m] gather is exercised on every store path (mp rows,
// np columns, and the kWarpsPerBlock independent-warp path).
TEST(GemmDecodeUniversalFp8, Fp8PerTokenTilesAndMultiWarp)
{
    const DecodeShape s{4, 8192, 7168};
    // 2x2 register tile: per-row scale applied to each of the mp rows.
    EXPECT_TRUE((RunFp8PerTokenCase<fp8_t, fp8_t, bf16_t,
                                    /*kNPerWarp=*/2, /*kMPerWarp=*/2>(
        "per-token mp2/np2", s, 1)));
    // N-reuse only.
    EXPECT_TRUE((RunFp8PerTokenCase<fp8_t, fp8_t, bf16_t, /*kNPerWarp=*/4>(
        "per-token np4", s, 1)));
    // Split-K (atomic-add) per-token: the row scale must fold consistently
    // across shards (it scales the final sum, applied per shard partial).
    EXPECT_TRUE((RunFp8PerTokenCase<fp8_t, fp8_t, bf16_t>("per-token split-K", s, 2)));
    // Multi-warp independent-warp epilogue (mp=np=1, kWarpsPerBlock=4) + A-LDS.
    EXPECT_TRUE((RunFp8PerTokenCase<fp8_t, fp8_t, bf16_t,
                                    /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                    /*kWarpsPerBlock=*/4, /*kStageAInLds=*/true>(
        "per-token WPB=4+A-LDS", DecodeShape{1, 7168, 7168}, 1)));
}

// FP8 per-tensor + 2D modular-broadcast bias (the wvSplitKQ target): the bias
// is added after the sA*sB scale, indexed by BIAS[(n % Bx) + (m % By) * Bx].
TEST(GemmDecodeUniversalFp8, Fp8Bias2D)
{
    EXPECT_TRUE((RunFp8Bias2DCase<fp8_t, fp8_t, bf16_t>(
        "FP8 2Dbias Bx=N By=4", DecodeShape{4, 4096, 7168}, /*Bx=*/4096, /*By=*/4, 1)));
    EXPECT_TRUE((RunFp8Bias2DCase<fp8_t, fp8_t, bf16_t>(
        "FP8 2Dbias Bx=512 By=2", DecodeShape{4, 4096, 7168}, /*Bx=*/512, /*By=*/2, 1)));
    // 1D-equivalence: By = 1, Bx = N.
    EXPECT_TRUE((RunFp8Bias2DCase<fp8_t, fp8_t, bf16_t>(
        "FP8 2Dbias By=1 (1D-equiv)", DecodeShape{2, 8192, 7168}, /*Bx=*/8192, /*By=*/1, 1)));
    // Compose with the M/N register tile (2x2).
    EXPECT_TRUE((RunFp8Bias2DCase<fp8_t, fp8_t, bf16_t, /*kNPerWarp=*/2, /*kMPerWarp=*/2>(
        "FP8 2Dbias MxN=2x2", DecodeShape{4, 4096, 7168}, /*Bx=*/512, /*By=*/4, 1)));
    // Compose with the multi-warp epilogue site (kWarpsPerBlock > 1).
    EXPECT_TRUE((RunFp8Bias2DCase<fp8_t, fp8_t, bf16_t,
                                  /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/4>(
        "FP8 2Dbias WPB=4", DecodeShape{4, 7168, 7168}, /*Bx=*/512, /*By=*/2, 1)));
}

// N-reuse on the dot2 FP8 path: the shared A row is dequantized into BF16x2
// register pairs once and reused against kNPerWarp B rows. Exercises the
// fp8x2_to_bf16x2 a_pairs precompute + per-N b_pair reuse.
TEST(GemmDecodeUniversalFp8, Fp8NReuse)
{
    const std::vector<DecodeShape> shapes{
        DecodeShape{1, 4096, 7168},
        DecodeShape{4, 8192, 7168},
    };
    for(const auto& s : shapes)
    {
        EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                         /*kNPerWarp=*/2>("FP8 NPerWarp=2", s, 1)));
        EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/true,
                                         /*kNPerWarp=*/4>("FP8 NPerWarp=4+bias", s, 1)));
    }
}

// B-reuse on the dot2 FP8 path: each B row is dequantized into BF16x2 register
// pairs once and reused across the kMPerWarp A rows, whose pairs are flat-
// indexed by jm*(kVector/2)+ipair. Includes a tail block (M not divisible by
// kMPerWarp) and a combined 2x2 M/N tile with bias.
TEST(GemmDecodeUniversalFp8, Fp8MReuse)
{
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/2>(
        "FP8 MPerWarp=2", DecodeShape{4, 8192, 7168}, 1)));
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/4>(
        "FP8 MPerWarp=4 tail", DecodeShape{5, 8192, 7168}, 1)));
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/true,
                                     /*kNPerWarp=*/2, /*kMPerWarp=*/2>(
        "FP8 MxN=2x2+bias", DecodeShape{3, 4096, 7168}, 1)));
}

// Multi-warp occupancy path on the dot2 FP8 per-tensor kernel (§15.F probe).
// The shared A row is broadcast to all warps; each warp owns one B row / one
// output column. Pure scheduling change -> must match the PerTensor reference
// exactly. Includes N=7168 (the −23% blemish point) across WPB in {2,4,8}.
TEST(GemmDecodeUniversalFp8, Fp8MultiWarp)
{
    const std::vector<DecodeShape> shapes{
        DecodeShape{1, 7168, 7168},
        DecodeShape{1, 8192, 7168},
        DecodeShape{4, 4096, 7168},
    };
    for(const auto& s : shapes)
    {
        EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                         /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                         /*kWarpsPerBlock=*/2>("FP8 WPB=2", s, 1)));
        EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                         /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                         /*kWarpsPerBlock=*/4>("FP8 WPB=4", s, 1)));
        EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                         /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                         /*kWarpsPerBlock=*/8>("FP8 WPB=8", s, 1)));
    }
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/true,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1,
                                     /*kWarpsPerBlock=*/4>(
        "FP8 WPB=4 bias", DecodeShape{1, 7168, 7168}, 1)));
}

// A-in-LDS staging on the multi-warp FP8 per-tensor path (the wvSplitKQ
// recipe). Stages the shared FP8 A row in LDS once per workgroup; must match
// the PerTensor reference exactly. K <= kLdsStageMaxK (8192).
TEST(GemmDecodeUniversalFp8, Fp8MultiWarpLdsStage)
{
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/4,
                                     /*kStageAInLds=*/true>(
        "FP8 WPB=4 A-LDS", DecodeShape{1, 7168, 7168}, 1)));
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/8,
                                     /*kStageAInLds=*/true>(
        "FP8 WPB=8 A-LDS", DecodeShape{1, 8192, 7168}, 1)));
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/true,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/4,
                                     /*kStageAInLds=*/true>(
        "FP8 WPB=4 A-LDS bias", DecodeShape{1, 7168, 7168}, 1)));
}

// Non-temporal B loads on the FP8 per-tensor path. Pure coherence hint, so the
// dequantized result must match the FP32 reference. Covers the single-warp
// main path, split-K, and the multi-warp + A-LDS + bias composition.
TEST(GemmDecodeUniversalFp8, Fp8StreamB)
{
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/1,
                                     /*kStageAInLds=*/false, /*kStreamB=*/true>(
        "FP8 stream-B", DecodeShape{1, 7168, 7168}, 1)));
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/1,
                                     /*kStageAInLds=*/false, /*kStreamB=*/true>(
        "FP8 stream-B split=4", DecodeShape{1, 7168, 7168}, 4)));
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/true,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/4,
                                     /*kStageAInLds=*/true, /*kStreamB=*/true>(
        "FP8 stream-B WPB=4 A-LDS bias", DecodeShape{1, 7168, 7168}, 1)));
}

// Persistent fat-WG launch on the FP8 per-tensor path. Pure scheduling change,
// so the dequantized result must match the FP32 reference. Covers the main
// path, split-K, and the multi-warp + A-LDS + bias + streamed-B fat-WG combo.
TEST(GemmDecodeUniversalFp8, Fp8Persistent)
{
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/false,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/1,
                                     /*kStageAInLds=*/false, /*kStreamB=*/false,
                                     /*kPersistent=*/true>(
        "FP8 persistent", DecodeShape{1, 7168, 7168}, 1)));
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/true,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/1,
                                     /*kStageAInLds=*/false, /*kStreamB=*/false,
                                     /*kPersistent=*/true>(
        "FP8 persistent split=4 bias", DecodeShape{1, 7168, 7168}, 4)));
    EXPECT_TRUE((RunFp8PerTensorCase<fp8_t, fp8_t, bf16_t, /*kHasBias=*/true,
                                     /*kNPerWarp=*/1, /*kMPerWarp=*/1, /*kWarpsPerBlock=*/4,
                                     /*kStageAInLds=*/true, /*kStreamB=*/true,
                                     /*kPersistent=*/true>(
        "FP8 persistent WPB=4 A-LDS bias stream-B", DecodeShape{1, 7168, 7168}, 1)));
}

TEST(GemmDecodeBlockscaleFp8, Fp8Fp8ToBf16BlockscaleMatrix)
{
    RunBlockscaleMatrix<fp8_t, fp8_t, bf16_t>("FP8/FP8/BF16 1x128/128x128");
}

TEST(GemmDecodeBlockscaleFp8, ScaleLdsBroadcastVsGlobal)
{
    // The kernel falls back to the global-only path when num K-blocks
    // exceeds kMaxScaleBlocks (= 128 -> K > 128 * 128 = 16384). We exercise
    // both:
    //   - K = 7168  (56 K-blocks)   - LDS path
    //   - K = 8192  (64 K-blocks)   - LDS path, larger sweep
    //   - K = 24576 (192 K-blocks)  - global fallback (> kMaxScaleBlocks)
    EXPECT_TRUE((RunBlockscaleCase<fp8_t, fp8_t, bf16_t>("LDS K=7168",
                                                          DecodeShape{1, 4096, 7168}, 1)));
    EXPECT_TRUE((RunBlockscaleCase<fp8_t, fp8_t, bf16_t>("LDS K=8192",
                                                          DecodeShape{2, 2048, 8192}, 2)));
    EXPECT_TRUE((RunBlockscaleCase<fp8_t, fp8_t, bf16_t>("global K=24576",
                                                          DecodeShape{1, 2048, 24576}, 1)));
}
#endif // CK_TILE_USE_OCP_FP8

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
