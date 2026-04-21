// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pipeline_tests_helper.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/scale/scale_mma_pipeline.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/utility/functional.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <type_traits>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

using CompilerTargetGfx950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());

template <typename AType,
          typename BType,
          typename CType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
void ScaleMfmaGfx950Specialization_impl()
{
    using TestScaleMma = amdgcn_mma<AType,
                                    BType,
                                    CType,
                                    WaveTileM,
                                    WaveTileN,
                                    WaveTileK,
                                    DefaultScaleMfmaCtrlFlags,
                                    CompilerTargetGfx950,
                                    MmaOpFamily::SCALE>;

    static_assert(std::is_same_v<typename TestScaleMma::OpType, MfmaOp> &&
                      TestScaleMma::OpFamily == MmaOpFamily::SCALE,
                  "GFX950 scale intrinsic should have ScaleMFMAOp type");

    static_assert(is_mma_op_of_family_v<MmaOpFamily::SCALE, TestScaleMma>,
                  "GFX950 scale intrinsic should be detected as Scale");

    // Get its traits
    using TestTraits = MmaOpTraits<TestScaleMma>;

    // Verify trait detection
    static_assert(TestTraits::IsScale, "Scale MMA should be detected as scale");
    static_assert(TestTraits::IsSupported, "Scale MMA specialization should be supported");
    static_assert(TestTraits::IsMfma, "Scale MFMA should be detected as MFMA");
    static_assert(!TestTraits::IsWmma, "Scale MFMA should not be detected as WMMA");
}

TEST(ScaleMMATrait, ScaleMfmaGfx950Specialization)
{
    // Test fp8 -> fp32 scale MFMA for GFX950 (16x16x128)
    ScaleMfmaGfx950Specialization_impl<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u>();
    // Test bf8 -> fp32 scale MFMA for GFX950 (16x16x128)
    ScaleMfmaGfx950Specialization_impl<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u>();
    // Test fp4 -> fp32 scale MFMA for GFX950 (16x16x128)
    ScaleMfmaGfx950Specialization_impl<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u>();
    // Test fp8 -> fp32 scale MFMA for GFX950 (32x32x64)
    ScaleMfmaGfx950Specialization_impl<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u>();
    // Test bf8 -> fp32 scale MFMA for GFX950 (32x32x64)
    ScaleMfmaGfx950Specialization_impl<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u>();
    // Test fp4 -> fp32 scale MFMA for GFX950 (32x32x64)
    ScaleMfmaGfx950Specialization_impl<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u>();

    std::cout << "GFX950 scale MFMA specialization is correct" << std::endl;
}

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
template <typename AType,
          typename BType,
          typename CType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
void TestConceptRequirements_impl()
{
    using TestScaleMma = amdgcn_mma<AType,
                                    BType,
                                    CType,
                                    WaveTileM,
                                    WaveTileN,
                                    WaveTileK,
                                    DefaultScaleMfmaCtrlFlags,
                                    CompilerTargetGfx950,
                                    MmaOpFamily::SCALE>;
    static_assert(MmaOpI<TestScaleMma>);
}
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

TEST(ScaleMMATrait, TestConceptRequirements)
{
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
    TestConceptRequirements_impl<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u>();
    TestConceptRequirements_impl<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u>();
    TestConceptRequirements_impl<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u>();
    TestConceptRequirements_impl<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u>();
    TestConceptRequirements_impl<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u>();
    TestConceptRequirements_impl<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u>();
#else
    GTEST_SKIP() << "Not compiled with concepts. Skipping test.";
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
}

template <typename AType, typename BType, typename CType>
void ScaleSelector_impl()
{
    static_for<2, 14, 6>{}([](auto k_factor) {
        static_for<1, 33, 1>{}([&](auto i) {
            using Selected                = typename MmaDefaultSelector<AType,
                                                                        BType,
                                                                        CType,
                                                                        static_cast<std::uint32_t>(i),
                                                                        static_cast<std::uint32_t>(i),
                                                                        static_cast<std::uint32_t>(k_factor * i),
                                                                        CompilerTargetGfx950,
                                                                        MmaOpFamily::SCALE>::SelectedOp;
            static constexpr bool isValid = (i == 16 && k_factor == 8) || (i == 32);
            if constexpr(isValid)
            {
                // Selector should pick a scale MFMA implementation
                static_assert(MmaOpTraits<Selected>::IsScale);
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
    });
}

TEST(ScaleMMATrait, ScaleSelector)
{
    ScaleSelector_impl<fp8_t, fp8_t, fp32_t>();
    ScaleSelector_impl<bf8_t, bf8_t, fp32_t>();
    ScaleSelector_impl<pk_fp4_t, pk_fp4_t, fp32_t>();
}

template <typename AType,
          typename BType,
          typename CType,
          typename ScaleAType,
          typename ScaleBType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
struct ScalePipelineKernel
{
    static constexpr int kBlockSize = 64;

    __device__ void
    operator()(const void* a_per_lane, const void* b_per_lane, void* c_per_lane) const
    {
        using CompilerTarget = decltype(get_compiler_target());
        using Pipeline =
            ScaleMmaPipeline<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK, CompilerTarget>;

        using AVecType = typename Pipeline::AVecType;
        using BVecType = typename Pipeline::BVecType;
        using CVecType = typename Pipeline::CVecType;

        const uint32_t lane = threadIdx.x;

        AVecType a;
        BVecType b;
        CVecType c;
        __builtin_memcpy(&a,
                         static_cast<const uint8_t*>(a_per_lane) + lane * sizeof(AVecType),
                         sizeof(AVecType));
        __builtin_memcpy(&b,
                         static_cast<const uint8_t*>(b_per_lane) + lane * sizeof(BVecType),
                         sizeof(BVecType));
        __builtin_memset(&c, 0, sizeof(CVecType));

        if constexpr(MmaOpTraits<typename Pipeline::MmaOp>::IsSupported)
        {
            // scale_a = 126 → 2^(126-127) = 2^-1 = 0.5
            // scale_b = 129 → 2^(129-127) = 2^2  = 4.0
            // Combined scale factor = 0.5 * 4.0 = 2.0
            ScaleAType scale_a = 126;
            ScaleBType scale_b = 129;
            Pipeline::exec(a, b, c, scale_a, scale_b);
            __builtin_memcpy(
                static_cast<uint8_t*>(c_per_lane) + lane * sizeof(CVecType), &c, sizeof(CVecType));
        }
    }
};

template <typename AType,
          typename BType,
          typename CType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
struct ScalePipelineFactory
{
    template <typename Target>
    struct Create
    {
        using type = ScaleMmaPipeline<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK, Target>;
    };
};

template <typename AType,
          typename BType,
          typename CType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
void MmaSelector_Scale_Real_impl()
{
    using ScaleAType = std::uint32_t;
    using ScaleBType = std::uint32_t;

    const auto should_skip = [](amdgcn_target_id currentArchId) {
        bool isSupportedMfma = (currentArchId == amdgcn_target_id::GFX950);
        return ((currentArchId == amdgcn_target_id::HOST) || !isSupportedMfma);
    };

    using Factory = ScalePipelineFactory<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK>;
    using Kernel  = ScalePipelineKernel<AType,
                                        BType,
                                        CType,
                                        ScaleAType,
                                        ScaleBType,
                                        WaveTileM,
                                        WaveTileN,
                                        WaveTileK>;

    // scale_a=126 → 2^-1=0.5, scale_b=129 → 2^2=4.0 → combined = 2.0
    constexpr float reference_scale = 2.0f;

    mma_pipeline_test::
        run_pipeline_matrix_test<Factory::template Create, Kernel, AType, BType, CType>(
            WaveTileM,
            WaveTileN,
            WaveTileK,
            should_skip,
            Kernel{},
            /*isSparse=*/false,
            /*transposeExpected=*/false,
            reference_scale);
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F8_F8_F32_16x16x128_Real)
{
    MmaSelector_Scale_Real_impl<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_BF8_BF8_F32_16x16x128_Real)
{
    MmaSelector_Scale_Real_impl<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F4_F4_F32_16x16x128_Real)
{
    MmaSelector_Scale_Real_impl<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F8_F8_F32_32x32x64_Real)
{
    MmaSelector_Scale_Real_impl<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_BF8_BF8_F32_32x32x64_Real)
{
    MmaSelector_Scale_Real_impl<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F4_F4_F32_32x32x64_Real)
{
    MmaSelector_Scale_Real_impl<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u>();
}
