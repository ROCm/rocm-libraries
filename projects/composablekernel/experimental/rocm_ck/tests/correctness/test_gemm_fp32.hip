// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// FP32 GEMM correctness tests.
//
// OnesTimesZeros:      A = 1, B = 0  =>  C = 0
// IdentityTimesMatrix: A = I, B = f(k,n)  =>  C = B
// RademacherGemm:      A,B random +/-1  =>  C vs CPU reference (exact)
// NormalGemmTriple:     A,B ~ N(0,1)  =>  CK vs GPU references + cross-checks

#include <hip/hip_runtime.h>

#include <rocm_ck/args.hpp>
#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/typed_buffer.hpp>
#include <rocm_ck/verify.hpp>

#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/gemm_dev.hpp>
#endif

#include "gemm_metrics.hpp"
#include "gemm_ref_gpu.hip"
#include "gemm_test_data.hpp"
#include "gemm_test_utils.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

// ============================================================================
// Spec + kernel
// ============================================================================

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype = rocm_ck::DataType::FP32,
                       .ops   = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                           .block_waves = {2, 2, 1},
                           .wave_tile   = {16, 16, 16}},
    rocm_ck::TargetSet::cdna());

__global__ void gemm_kernel(rocm_ck::Args args)
{
#ifdef __HIP_DEVICE_COMPILE__
    rocm_ck::run<kSpec>(args);
#endif
}

// ============================================================================
// Test fixture
// ============================================================================

class GemmFP32 : public ::testing::Test
{
protected:
    void SetUp() override { rocm_ck::test::skipIfNoGpu(); }
};

// ============================================================================
// Test: A = all 1s, B = all 0s => C = all 0s
// ============================================================================

TEST_F(GemmFP32, OnesTimesZeros)
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 32;

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    rocm_ck::test::fillConstant2D(args, kSpec.lhs().args_slot, kSpec.lhs().dtype, 1.0f);
    rocm_ck::test::fillConstant2D(args, kSpec.rhs().args_slot, kSpec.rhs().dtype, 0.0f);
    rocm_ck::test::fillConstant2D(args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> result(M * N);
    buf_c.download(result.data());

    std::vector<float> ref(M * N, 0.0f);

    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}

// ============================================================================
// Test: A = identity, B = sequential pattern => C = B
// ============================================================================

TEST_F(GemmFP32, IdentityTimesMatrix)
{
    constexpr int M    = 128;
    constexpr int N    = 128;
    constexpr int K    = 128; // M == K so A is a square identity
    constexpr int kMod = 8;   // small ints, exact in any FP format

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    rocm_ck::test::fill2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype, rocm_ck::test::IdentityGen{});
    rocm_ck::test::fill2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype, rocm_ck::test::SequentialGen{kMod});
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> result(M * N);
    buf_c.download(result.data());

    auto [c_sm, c_sn] = rocm_ck::layoutStrides(kSpec.output().layout, M, N);
    std::vector<float> ref(M * N);
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
            ref[m * c_sm + n * c_sn] = static_cast<float>((m * N + n) % kMod);

    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}

// ============================================================================
// Test: A,B = Rademacher (+/-1) => C vs CPU reference (exact in FP32)
// ============================================================================

TEST_F(GemmFP32, RademacherGemm)
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    rocm_ck::test::fill2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype, rocm_ck::test::RademacherGen{37});
    rocm_ck::test::fill2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype, rocm_ck::test::RademacherGen{73});
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> host_a(M * K);
    std::vector<float> host_b(K * N);
    buf_a.download(host_a.data());
    buf_b.download(host_b.data());

    std::vector<float> ref(M * N, 0.0f);
    rocm_ck::test::cpuGemm(kSpec, host_a.data(), host_b.data(), ref.data(),
                            M, N, K, args);

    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> result(M * N);
    buf_c.download(result.data());

    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}

// ============================================================================
// Test: A,B ~ N(0,1) => CK vs all GPU reference methods + pairwise cross-checks
//
// Compares the CK kernel result against every reference in kRefMethods
// using aggregate metrics (Frobenius norms, ULP distance),
// then cross-checks all reference pairs against each other.
// ============================================================================

TEST_F(GemmFP32, NormalGemmTriple)
{
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 128;
    constexpr int count = M * N;
    // Standard GEMM error bound: O(K * u) where u = 2^-24 for FP32.
    // We use a generous multiplier because the CK kernel is not an exact
    // implementation of the textbook algorithm — tiling, fused multiply-add,
    // and reduction order all introduce variation.
    constexpr double kRelFrobBound = static_cast<double>(K) * 6.0e-8 * 4.0;

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, count);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    // --- Fill A, B with standard normal N(0, 1) ---
    rocm_ck::test::fill2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype,
        rocm_ck::test::NormalGen{0.0f, 1.0f, 42});
    rocm_ck::test::fill2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype,
        rocm_ck::test::NormalGen{0.0f, 1.0f, 137});
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    // --- CK kernel ---
    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    // CK result stays on device for metrics computation.
    // Copy it to a separate buffer so buf_c can be reused.
    float* dev_ck = nullptr;
    HIP_CHECK(hipMalloc(&dev_ck, count * sizeof(float)));
    HIP_CHECK(hipMemcpy(dev_ck, buf_c.ptr(),
                         count * sizeof(float), hipMemcpyDeviceToDevice));

    // --- GPU references (operate on device float arrays) ---
    static const std::vector<rocm_ck::test::GpuRefMethod> kRefMethods = {
        rocm_ck::test::GpuRefMethod::Multiword,
        rocm_ck::test::GpuRefMethod::FP64,
        rocm_ck::test::GpuRefMethod::Ozakik2,
        rocm_ck::test::GpuRefMethod::Ozakik2Tiled,
    };

    rocm_ck::test::GemmBackend backend;
    auto* dev_a = static_cast<float*>(buf_a.ptr());
    auto* dev_b = static_cast<float*>(buf_b.ptr());

    // Compute all references
    std::vector<float*> ref_bufs(kRefMethods.size());
    for(size_t i = 0; i < kRefMethods.size(); ++i)
    {
        HIP_CHECK(hipMalloc(&ref_bufs[i], count * sizeof(float)));
        rocm_ck::test::gpuGemm(kRefMethods[i], kSpec, ref_bufs[i],
                                dev_a, dev_b, M, N, K, args, backend);
    }

    // --- Metrics: CK vs each reference ---
    for(size_t i = 0; i < kRefMethods.size(); ++i)
    {
        auto label = std::string("CK vs ")
                     + rocm_ck::test::gpuRefMethodName(kRefMethods[i]);
        auto m = rocm_ck::test::computeGemmMetrics(
            dev_ck, ref_bufs[i], count);
        rocm_ck::test::printGemmMetrics(label.c_str(), m);

        EXPECT_LT(m.rel_frob, kRelFrobBound) << label;
    }

    // --- Metrics: pairwise cross-checks (should be very tight) ---
    for(size_t i = 0; i < kRefMethods.size(); ++i)
    {
        for(size_t j = i + 1; j < kRefMethods.size(); ++j)
        {
            auto label = std::string(
                             rocm_ck::test::gpuRefMethodName(kRefMethods[i]))
                         + " vs "
                         + rocm_ck::test::gpuRefMethodName(kRefMethods[j]);
            auto m = rocm_ck::test::computeGemmMetrics(
                ref_bufs[i], ref_bufs[j], count);
            rocm_ck::test::printGemmMetrics(label.c_str(), m);

            EXPECT_LT(m.rel_frob, 1e-6) << label;
            EXPECT_LT(m.max_ulp, 600000) << label;
        }
    }

    // --- Cleanup device buffers ---
    HIP_CHECK(hipFree(dev_ck));
    for(auto* p : ref_bufs)
        HIP_CHECK(hipFree(p));
}
