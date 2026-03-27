// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_mma_pipeline.hpp"
#include <hip/hip_runtime.h>
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include "pipeline_tests_helper.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

using CompilerTargetGfx950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());

TEST(SparseMMATrait, SparseMfmaGfx950Specialization)
{
    // Test fp16 → fp32 sparse MFMA for GFX950 (16x16x32)
    using TestSparseMfma16x16 = amdgcn_mma<fp16_t,
                                           fp16_t,
                                           fp32_t,
                                           16u,
                                           16u,
                                           32u,
                                           DefaultSparseMfmaCtrlFlags,
                                           CompilerTargetGfx950,
                                           MmaOpFamily::SPARSE>;

    static_assert(std::is_same_v<typename TestSparseMfma16x16::OpType, MfmaOp> &&
                      TestSparseMfma16x16::OpFamily == MmaOpFamily::SPARSE,
                  "GFX950 sparse 16x16x32 should have SparseMFMAOp type");

    static_assert(is_mma_op_of_family_v<MmaOpFamily::SPARSE, TestSparseMfma16x16>,
                  "GFX950 sparse 16x16x32 should be detected as Sparse");

    std::cout << "GFX950 sparse MFMA specialization is correct" << std::endl;
}

TEST(SparseMMATrait, MmaOpTraitsIntegration)
{
    // Create a sparse MMA op (16x16x32 fp16 specialization)
    using TestSparseMmma = amdgcn_mma<fp16_t,
                                      fp16_t,
                                      fp32_t,
                                      16u,
                                      16u,
                                      32u,
                                      DefaultSparseMfmaCtrlFlags,
                                      CompilerTargetGfx950,
                                      MmaOpFamily::SPARSE>;

    // Get its traits
    using TestTraits = MmaOpTraits<TestSparseMmma>;

    // Verify trait detection
    static_assert(TestTraits::IsSparse, "Sparse MMA should be detected as sparse");
    static_assert(TestTraits::IsSupported, "Sparse MMA specialization should be supported");
    static_assert(TestTraits::IsMfma, "Sparse MFMA should be detected as MFMA");
    static_assert(!TestTraits::IsWmma, "Sparse MFMA should not be detected as WMMA");

    std::cout << "MmaOpTraits correctly integrates sparse operations" << std::endl;
}

TEST(SparseMMATrait, TestConceptRequirements)
{
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
    using TestSparseMmma = amdgcn_mma<fp16_t,
                                      fp16_t,
                                      fp32_t,
                                      16u,
                                      16u,
                                      32u,
                                      DefaultSparseMfmaCtrlFlags,
                                      CompilerTargetGfx950,
                                      MmaOpFamily::SPARSE>;
    static_assert(MmaOpI<TestSparseMmma>);
#else
    GTEST_SKIP() << "Not compiled with concepts. Skipping test.";
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
}

TEST(SparseMMATrait, DenseVsSparseDistinction)
{
    // Dense MFMA from mfma/mfma_gfx9.hpp
    using DenseMfma = amdgcn_mma<fp16_t,
                                 fp16_t,
                                 fp32_t,
                                 16u,
                                 16u,
                                 32u,
                                 DefaultMfmaCtrlFlags,
                                 CompilerTargetGfx950,
                                 MmaOpFamily::DENSE>;

    // Sparse MFMA on GFX950
    using SparseMfma = amdgcn_mma<fp16_t,
                                  fp16_t,
                                  fp32_t,
                                  16u,
                                  16u,
                                  32u,
                                  DefaultSparseMfmaCtrlFlags,
                                  CompilerTargetGfx950,
                                  MmaOpFamily::SPARSE>;

    // Verify they have different operation types
    static_assert(std::is_same_v<typename DenseMfma::OpType, typename SparseMfma::OpType> &&
                      DenseMfma::OpFamily != SparseMfma::OpFamily,
                  "Dense and Sparse MFMA should have the same OpType tags and different OpFamily");

    // Verify traits correctly identify them
    static_assert(MmaOpTraits<DenseMfma>::IsMfma && MmaOpTraits<DenseMfma>::IsDense &&
                      !MmaOpTraits<DenseMfma>::IsSparse && !MmaOpTraits<DenseMfma>::IsScale &&
                      MmaOpTraits<DenseMfma>::IsSupported,
                  "Dense MFMA should be identified correctly");

    static_assert(MmaOpTraits<SparseMfma>::IsSparse && MmaOpTraits<SparseMfma>::IsMfma &&
                      !MmaOpTraits<SparseMfma>::IsDense && !MmaOpTraits<SparseMfma>::IsScale &&
                      MmaOpTraits<SparseMfma>::IsSupported,
                  "Sparse MFMA should be identified correctly");

    std::cout << "Dense and sparse MMA operations are correctly distinguished" << std::endl;
}

TEST(SparseMMATrait, SparseSelector)
{
    static_for<1, 33, 1>{}([](auto i) {
        using Selected = typename MmaDefaultSelector<fp16_t,
                                                     fp16_t,
                                                     fp32_t,
                                                     static_cast<uint32_t>(i),
                                                     static_cast<uint32_t>(i),
                                                     static_cast<uint32_t>(2 * i),
                                                     CompilerTargetGfx950,
                                                     MmaOpFamily::SPARSE>::SelectedOp;

        static constexpr bool isValid = (i == 16) || (i == 32);
        if constexpr(isValid)
        {
            // Selector should pick a sparse MFMA implementation
            static_assert(MmaOpTraits<Selected>::IsSparse);
            static_assert(MmaOpTraits<Selected>::IsMfma);
            static_assert(MmaOpTraits<Selected>::IsSupported);
            static_assert((std::is_same<typename Selected::OpType, MfmaOp>::value));
        }
        else
        {
            // Selector should pick the unsupported pass through
            static_assert(!MmaOpTraits<Selected>::IsSupported);
        }
    });
}

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK>
__global__ void test_sparse_accum_over_k(void* a, void* b, void* c, void* out)
{
    using Pipeline = SparseMmaPipeline<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK>;

    using AVecType = typename Pipeline::AVecType;
    using BVecType = typename Pipeline::BVecType;
    using CVecType = typename Pipeline::CVecType;

    static constexpr uint32_t kIters = WaveTileK / Pipeline::MmaOp::kK;

    // Initialize the accumulator
    CVecType result = *reinterpret_cast<CVecType*>(c);

    // Accumulate input AxB over WaveTileK/FragK iterations
    for(uint32_t i = 0; i < kIters; ++i)
    {
        result = Pipeline::exec(
            *reinterpret_cast<AVecType*>(a), *reinterpret_cast<BVecType*>(b), result);
    }

    *reinterpret_cast<CVecType*>(out) = result;
}

// Live test on real hardware for sparse selection and execution.
TEST(SparseMMATrait, MmaSelector_Sparse_F16_F16_F32_16x16x32_Real)
{
    MmaPipelineTest<> test;
    const auto should_skip = [](amdgcn_target_id currentArchId) {
        bool isSupportedWmma = (currentArchId >= amdgcn_target_id::GFX1200) &&
                               (currentArchId <= amdgcn_target_id::GFX12_GENERIC);
        bool isSupportedMfma = (currentArchId >= amdgcn_target_id::GFX942) &&
                               (currentArchId <= amdgcn_target_id::GFX950);
        return ((currentArchId == amdgcn_target_id::HOST) || !(isSupportedWmma || isSupportedMfma));
    };
    const std::function<fp32_t(uint32_t)> validator = [](uint32_t waveTileK) {
        return static_cast<fp32_t>(waveTileK) / 2;
    };
    const auto kernel = [](uint32_t waveSize, void* a, void* b, void* c, void* out) {
        test_sparse_accum_over_k<MmaPipelineTest<>::AType,
                                 MmaPipelineTest<>::BType,
                                 MmaPipelineTest<>::CType,
                                 MmaPipelineTest<>::WaveTileM,
                                 MmaPipelineTest<>::WaveTileN,
                                 MmaPipelineTest<>::WaveTileK><<<1, waveSize>>>(a, b, c, out);
    };
    // Initialize A with 2:4 structured sparsity pattern: {1, 0, 1, 0, ...}
    // This ensures the sparse compression transform is actually exercised —
    // a no-op or broken compression would pass zeros through, causing incorrect results.
    const std::function<fp16_t(size_t)> sparseAInit = [](size_t i) -> fp16_t {
        return (i % 2 == 0) ? type_convert<fp16_t>(1) : type_convert<fp16_t>(0);
    };
    test.test_pipeline(should_skip, kernel, validator, sparseAInit);
}

template <uint32_t CompressionRatio, typename Vec>
__global__ void test_sparse_transform(void* a, void* idx)
{
    using ResultT =
        decltype(SparseCompressTransform<CompressionRatio>::exec(*static_cast<Vec*>(a)));
    using FirstT         = std::tuple_element_t<0, ResultT>;
    const auto& [vec, i] = SparseCompressTransform<CompressionRatio>::exec(*static_cast<Vec*>(a));
    *reinterpret_cast<remove_cvref_t<FirstT>*>(a) = vec;
    *reinterpret_cast<int32_t*>(idx)              = i;
}

// 1. Basic correctness: valid divisible sizes
template <int NUM, int RATIO, typename Type>
void sparse_transform_test_case()
{
    static_assert(RATIO == 2, "Extend functionality if other ratio is used.");
    int devCount;
    hipDevice_t dev;
    HIP_CHECK_ERROR(hipGetDevice(&dev));
    HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

    auto currentArchId = hip_device_prop_gcn_arch_name_to_amdgcn_target_id(devProp.gcnArchName);
    bool hasDevice     = static_cast<bool>(devCount > 0);

    // TODO: c++20 add check for arch id
    if(!hasDevice || (currentArchId == amdgcn_target_id::HOST))
    {
        GTEST_SKIP() << "No HIP device found. Skipping test.";
    }

    std::vector<Type> v(NUM);
    for(int i = 0; i < NUM; ++i)
    {
        v[i] = i % 2 == 0 ? i + 1 : 0;
    }

    float* d_v;
    int32_t* d_idx;

    static constexpr auto Size = sizeof(Type) * NUM;
    HIP_CHECK_ERROR(hipMalloc(&d_v, Size));
    HIP_CHECK_ERROR(hipMalloc(&d_idx, sizeof(int32_t)));

    // Copy inputs to device
    HIP_CHECK_ERROR(hipMemcpy(d_v, v.data(), Size, hipMemcpyHostToDevice));

    test_sparse_transform<RATIO, ext_vector_t<Type, NUM>><<<1, 32>>>(d_v, d_idx);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<Type> h_out(NUM / RATIO, static_cast<Type>(0));
    HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_v, Size / RATIO, hipMemcpyDeviceToHost));
    int32_t h_idx;
    HIP_CHECK_ERROR(hipMemcpy(&h_idx, d_idx, sizeof(int32_t), hipMemcpyDeviceToHost));

    EXPECT_NE(h_idx, -1) << "idx should have been written";
    if constexpr(NUM == 8)
    {
        EXPECT_EQ(h_idx, 0b10001000);
    }
    else if constexpr(NUM == 16)
    {
        EXPECT_EQ(h_idx, 0b1000100010001000);
    }
    for(int i = 0; i < NUM / RATIO; ++i)
    {
        EXPECT_EQ(h_out[i], v[i * 2]);
    }

    HIP_CHECK_ERROR(hipFree(d_v));
    HIP_CHECK_ERROR(hipFree(d_idx));
}

TEST(SparseTransformsTest, ValidCompressionRatio)
{
    // TODO: extend those when new sparse builtins are
    // introduced and use different type combinations
    sparse_transform_test_case<8, 2, fp16_t>();
    sparse_transform_test_case<16, 2, fp16_t>();
}
