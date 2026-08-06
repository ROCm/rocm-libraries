// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "DataInitialization.hpp"             // isMXTensor / Problem
#include <roc/host_validation/adapters/tensilelite/DataInitializationHelpers.hpp>
#include <roc/host_validation/adapters/tensilelite/HostValidationBridge.hpp>
#include <roc/host_validation/validation.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
#include <hip/hip_runtime.h>
#include <mxDataGenerator/dataTypeInfo.hpp>
#include <mxDataGenerator/ocp_e2m1_mxfp4.hpp>
#include <mxDataGenerator/ocp_e2m3_mxfp6.hpp>
#include <mxDataGenerator/ocp_e3m2_mxfp6.hpp>
#include <mxDataGenerator/ocp_e4m3_mxfp8.hpp>
#include <mxDataGenerator/ocp_e5m2_mxfp8.hpp>
#endif

using TensileLite::ContractionProblemGemm;
using TensileLite::DataTypeInfo;
using TensileLite::TensorDescriptor;
using TensileLite::Client::isMXProblem;
using TensileLite::Client::isMXTensor;
using TensileLite::Client::initCPUSparseInput;
using TensileLite::Client::InitMode;
using TensileLite::Client::PruneSparseMode;
using TensileLite::Client::toHostValidationScalarType;
using TensileLite::Client::tryHostValidationInitialize;

// Shorthand for the production helper namespace under test (MX builds only).
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
namespace dt = TensileLite::Client::detail;
#endif
namespace
{
    // -----------------------------------------------------------------------
    // Helper: build a ContractionProblemGemm with the requested A/B dtypes.
    // Mirrors tests/MXScalePadding_test.cpp::makeMXProblem so the geometry
    // matches what the real client produces, then enables MX scaling on each
    // side independently. mxBlock==0 means "do NOT call setMXScale*", so the
    // problem's mxBlockA() / mxBlockB() stays 0 and isMXTensor returns
    // false on that side. This is exactly the lever needed to drive every
    // branch of isMXProblem.
    // -----------------------------------------------------------------------
    ContractionProblemGemm makeProblem(rocisa::DataType aType,
                                       rocisa::DataType bType,
                                       int              mxBlockA,
                                       int              mxBlockB,
                                       size_t           M = 128,
                                       size_t           N = 128,
                                       size_t           K = 256,
                                       size_t           batch  = 1,
                                       bool             transA = true,
                                       bool             transB = false)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(
            transA, transB,
            aType, bType,
            rocisa::DataType::BFloat16, rocisa::DataType::BFloat16,
            M, N, K, batch,
            transA ? K : M,                 // lda
            transA ? K * M : M * K,         // strideA
            transB ? N : K,                 // ldb
            transB ? N * K : K * N,         // strideB
            M, M * N,                       // ldc, strideC
            M, M * N,                       // ldd, strideD
            0.0);                           // beta
        if(mxBlockA > 0) problem.setMXScaleA(rocisa::DataType::E8, mxBlockA);
        if(mxBlockB > 0) problem.setMXScaleB(rocisa::DataType::E8, mxBlockB);
        return problem;
    }
} // namespace

TEST(HostValidationDataInitialization, GeneratesStridedProblemDependentPatterns)
{
    TensorDescriptor descriptor("t", rocisa::DataType::Float, {2, 3}, {1, 4});
    std::vector<float> values(descriptor.totalAllocatedElements(), -99.0f);

    ASSERT_TRUE(tryHostValidationInitialize(
        rocisa::DataType::Float, InitMode::SerialIdx, values.data(), descriptor));
    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[1], 1);
    EXPECT_EQ(values[4], 2);
    EXPECT_EQ(values[5], 3);
    EXPECT_EQ(values[8], 4);
    EXPECT_EQ(values[9], 5);
    EXPECT_EQ(values[2], -99);
    EXPECT_EQ(values[3], -99);

    TensorDescriptor identityDescriptor("identity", rocisa::DataType::Float, {3, 4}, {1, 3});
    std::vector<float> identity(identityDescriptor.totalAllocatedElements(), -1.0f);
    ASSERT_TRUE(tryHostValidationInitialize(
        rocisa::DataType::Float, InitMode::Identity, identity.data(), identityDescriptor));
    for(size_t column = 0; column < 4; ++column)
        for(size_t row = 0; row < 3; ++row)
            EXPECT_EQ(identity[row + column * 3], row == column ? 1.0f : 0.0f);
}

TEST(HostValidationDataInitialization, LeavesUnsupportedLegacyModesToFallback)
{
    std::array<float, 4> values{-1, -1, -1, -1};
    EXPECT_FALSE(tryHostValidationInitialize(
        rocisa::DataType::Float, InitMode::RandomNarrow, values.data(), values.size()));
    EXPECT_EQ(values, (std::array<float, 4>{-1, -1, -1, -1}));

    ASSERT_TRUE(tryHostValidationInitialize(
        rocisa::DataType::Float, InitMode::Two, values.data(), values.size()));
    EXPECT_EQ(values, (std::array<float, 4>{2, 2, 2, 2}));

    std::array<float, 4> randomFirst{};
    std::array<float, 4> randomSecond{};
    ASSERT_TRUE(tryHostValidationInitialize(
        rocisa::DataType::Float, InitMode::Random, randomFirst.data(), randomFirst.size()));
    ASSERT_TRUE(tryHostValidationInitialize(
        rocisa::DataType::Float, InitMode::Random, randomSecond.data(), randomSecond.size()));
    EXPECT_NE(randomFirst, randomSecond);
    for(float value : randomFirst)
    {
        EXPECT_GE(value, -100);
        EXPECT_LE(value, 100);
    }
}

TEST(HostValidationStructuredSparsity, TensileAdapterMatchesStandaloneComponent)
{
    auto problem = makeProblem(rocisa::DataType::Int8,
                               rocisa::DataType::Int8,
                               /*mxBlockA=*/0,
                               /*mxBlockB=*/0,
                               /*M=*/2,
                               /*N=*/2,
                               /*K=*/8,
                               /*batch=*/1,
                               /*transA=*/false,
                               /*transB=*/false);
    problem.setSparse(1, 0);

    const TensorDescriptor& denseDescriptor = problem.a();
    const TensorDescriptor& compressedDescriptor = problem.compressed();
    const TensorDescriptor& metadataDescriptor = problem.metadata();
    const size_t sparseAxis = problem.boundIndices()[0].a;

    std::vector<int8_t> original(denseDescriptor.totalAllocatedElements());
    for(size_t index = 0; index < original.size(); ++index)
        original[index] = static_cast<int8_t>(index + 1);

    std::vector<int8_t> legacyPruned = original;
    std::vector<int8_t> legacyCompressed(compressedDescriptor.totalAllocatedElements());
    std::vector<uint8_t> legacyMetadata(metadataDescriptor.totalAllocatedElements());
    initCPUSparseInput(PruneSparseMode::PruneXX00,
                       legacyPruned.data(),
                       legacyCompressed.data(),
                       legacyMetadata.data(),
                       denseDescriptor,
                       compressedDescriptor,
                       metadataDescriptor,
                       sparseAxis,
                       problem.metadataLayout());

    auto layout = [](const TensorDescriptor& descriptor) {
        return roc::host_validation::Layout(
            roc::host_validation::Shape(descriptor.sizes()),
            std::vector<ptrdiff_t>(
                descriptor.strides().begin(), descriptor.strides().end()));
    };
    using namespace roc::host_validation;
    const ScalarType scalarType = toHostValidationScalarType(denseDescriptor.dataType());
    std::vector<int8_t> componentPruned(original.size());
    std::vector<int8_t> componentCompressed(legacyCompressed.size());
    std::vector<uint8_t> componentMetadata(legacyMetadata.size());
    const Shape logicalMetadataShape{denseDescriptor.sizes()[0],
                                     denseDescriptor.sizes()[1] / 8,
                                     denseDescriptor.sizes()[2]};
    const Layout logicalMetadataLayout(
        logicalMetadataShape,
        {static_cast<ptrdiff_t>(metadataDescriptor.strides()[1]),
         static_cast<ptrdiff_t>(metadataDescriptor.strides()[0]),
         static_cast<ptrdiff_t>(metadataDescriptor.strides()[2])});

    StructuredSparsityPattern pattern;
    pattern.axis = sparseAxis;
    pattern.fixedPositions = {0, 1};
    StructuredSparsityProblem componentProblem(
        TensorView(scalarType,
                   layout(denseDescriptor),
                   std::as_bytes(std::span<const int8_t>(original))),
        MutableTensorView(
            scalarType,
            layout(denseDescriptor),
            std::as_writable_bytes(std::span<int8_t>(componentPruned))),
        MutableTensorView(
            scalarType,
            layout(compressedDescriptor),
            std::as_writable_bytes(std::span<int8_t>(componentCompressed))),
        pattern);
    componentProblem.twoOfFourMetadata = MutableTensorView(
        ScalarType::UInt8,
        logicalMetadataLayout,
        std::as_writable_bytes(std::span<uint8_t>(componentMetadata)));
    applyStructuredSparsity(componentProblem);

    EXPECT_EQ(componentPruned, legacyPruned);
    EXPECT_EQ(componentCompressed, legacyCompressed);
    EXPECT_EQ(componentMetadata, legacyMetadata);
}

TEST(HostValidationStructuredSparsity, TensileAdapterCoversModesLayoutsAndSparseSides)
{
    const std::array<PruneSparseMode, 7> modes{
        PruneSparseMode::PruneRandom,
        PruneSparseMode::PruneXX00,
        PruneSparseMode::PruneX0X0,
        PruneSparseMode::Prune0XX0,
        PruneSparseMode::PruneX00X,
        PruneSparseMode::Prune0X0X,
        PruneSparseMode::Prune00XX,
    };
    const std::array<std::array<size_t, 2>, 7> retainedPositionSets{{
        {0, 0},
        {0, 1},
        {0, 2},
        {1, 2},
        {0, 3},
        {1, 3},
        {2, 3},
    }};
    struct SparseCase
    {
        int  sparseSide;
        bool transA;
        bool transB;
    };
    const std::array<SparseCase, 4> sparseCases{{
        {1, false, false},
        {1, true, false},
        {2, false, false},
        {2, false, true},
    }};

    for(const SparseCase& sparseCase : sparseCases)
    {
        for(const int metadataLayout : {0, 1})
        {
            for(const PruneSparseMode mode : modes)
            {
                SCOPED_TRACE(::testing::Message()
                             << "sparseSide=" << sparseCase.sparseSide
                             << " transA=" << sparseCase.transA
                             << " transB=" << sparseCase.transB
                             << " metadataLayout=" << metadataLayout
                             << " mode=" << static_cast<int>(mode));
                auto problem = makeProblem(rocisa::DataType::Int8,
                                           rocisa::DataType::Int8,
                                           0,
                                           0,
                                           8,
                                           8,
                                           16,
                                           2,
                                           sparseCase.transA,
                                           sparseCase.transB);
                problem.setSparse(sparseCase.sparseSide, metadataLayout);

                const TensorDescriptor& denseDescriptor
                    = sparseCase.sparseSide == 1 ? problem.a() : problem.b();
                const TensorDescriptor& compressedDescriptor = problem.compressed();
                const TensorDescriptor& metadataDescriptor = problem.metadata();
                const size_t sparseAxis
                    = sparseCase.sparseSide == 1
                          ? problem.boundIndices()[0].a
                          : problem.boundIndices()[0].b;

                std::vector<int8_t> original(
                    denseDescriptor.totalAllocatedElements());
                for(size_t index = 0; index < original.size(); ++index)
                    original[index]
                        = static_cast<int8_t>(index % 127 + 1);
                std::vector<int8_t> pruned = original;
                std::vector<int8_t> compressed(
                    compressedDescriptor.totalAllocatedElements(),
                    static_cast<int8_t>(-101));
                std::vector<uint8_t> metadata(
                    metadataDescriptor.totalAllocatedElements(),
                    0xff);

                initCPUSparseInput(mode,
                                   pruned.data(),
                                   compressed.data(),
                                   metadata.data(),
                                   denseDescriptor,
                                   compressedDescriptor,
                                   metadataDescriptor,
                                   sparseAxis,
                                   problem.metadataLayout());

                const size_t groupsPerSlice
                    = denseDescriptor.sizes()[sparseAxis] / 4;
                const size_t sliceCount
                    = denseDescriptor.totalLogicalElements()
                      / denseDescriptor.sizes()[sparseAxis];
                std::vector<size_t> denseCoordinates(
                    denseDescriptor.dimensions(), 0);
                std::vector<size_t> compressedCoordinates(
                    compressedDescriptor.dimensions(), 0);
                std::vector<size_t> metadataCoordinates(
                    metadataDescriptor.dimensions(), 0);
                for(size_t slice = 0; slice < sliceCount; ++slice)
                {
                    TensileLite::CoordNumberedExclude(
                        slice,
                        denseCoordinates.begin(),
                        denseCoordinates.end(),
                        denseDescriptor.sizes().begin(),
                        denseDescriptor.sizes().end(),
                        sparseAxis);
                    TensileLite::CoordNumberedExclude(
                        slice,
                        metadataCoordinates.begin(),
                        metadataCoordinates.end(),
                        metadataDescriptor.sizes().begin(),
                        metadataDescriptor.sizes().end(),
                        problem.metadataLayout());
                    compressedCoordinates = denseCoordinates;
                    for(size_t group = 0; group < groupsPerSlice; ++group)
                    {
                        uint32_t selectedMode = static_cast<uint32_t>(mode);
                        if(mode == PruneSparseMode::PruneRandom)
                        {
                            selectedMode = static_cast<uint32_t>(
                                roc::host_validation::tensilelite_adapter::
                                    indexedUniformInteger(
                                        1,
                                        slice * groupsPerSlice + group,
                                        1,
                                        static_cast<int>(
                                            PruneSparseMode::MaxPruneMode)
                                            - 1));
                        }
                        const std::array<size_t, 2>& retained
                            = retainedPositionSets[selectedMode];
                        const uint8_t expectedMetadata = static_cast<uint8_t>(
                            retained[0] | (retained[1] << 2));

                        for(size_t position = 0; position < 4; ++position)
                        {
                            denseCoordinates[sparseAxis]
                                = group * 4 + position;
                            const size_t denseIndex
                                = denseDescriptor.index(denseCoordinates);
                            const bool isRetained
                                = position == retained[0]
                                  || position == retained[1];
                            EXPECT_EQ(pruned[denseIndex],
                                      isRetained ? original[denseIndex] : 0);
                        }
                        for(size_t retainedIndex = 0;
                            retainedIndex < retained.size();
                            ++retainedIndex)
                        {
                            denseCoordinates[sparseAxis]
                                = group * 4 + retained[retainedIndex];
                            compressedCoordinates[sparseAxis]
                                = group * 2 + retainedIndex;
                            EXPECT_EQ(
                                compressed[compressedDescriptor.index(
                                    compressedCoordinates)],
                                original[denseDescriptor.index(
                                    denseCoordinates)]);
                        }

                        metadataCoordinates[problem.metadataLayout()]
                            = group / 2;
                        const size_t metadataIndex
                            = TensileLite::CoordFlattenIndex(
                                metadataCoordinates.begin(),
                                metadataCoordinates.end(),
                                metadataDescriptor.sizes().begin(),
                                metadataDescriptor.sizes().end());
                        const uint8_t observedMetadata = static_cast<uint8_t>(
                            metadata[metadataIndex]
                            >> ((group % 2) * 4));
                        EXPECT_EQ(observedMetadata & 0xfU,
                                  expectedMetadata);
                    }
                }
            }
        }
    }
}

// =============================================================================
//   Section 1 - TensileLite::Client::isMXTensor
//
//       bool isMXTensor(t, mxBlock) {
//           if(mxBlock == 0) return false;            // (a) short-circuit
//           return dt in {Float4, Float6, BFloat6, Float8, BFloat8}; // (b) dtype gate
//       }
// =============================================================================
struct TensorParam
{
    rocisa::DataType dtype;
    size_t           mxBlock;
    bool             expected;
    char const*      name;
};
class IsMXTensorTest : public ::testing::TestWithParam<TensorParam>
{
};
TEST_P(IsMXTensorTest, MatchesContract)
{
    auto const& p = GetParam();
    // 1x1 descriptor is enough; the helper only inspects .dataType().
    TensorDescriptor t("t", p.dtype, {1, 1}, {1, 1});
    EXPECT_EQ(isMXTensor(t, p.mxBlock), p.expected)
        << "case=" << p.name
        << " dtype=" << static_cast<int>(p.dtype)
        << " mxBlock=" << p.mxBlock;
}

INSTANTIATE_TEST_SUITE_P(
    MXFP4OrFP8Coverage,
    IsMXTensorTest,
    ::testing::Values(
        // ----- (a) mxBlock==0 must short-circuit even for MX dtypes --------
        TensorParam{rocisa::DataType::Float4,   0, false, "Float4_block0"},
        TensorParam{rocisa::DataType::Float6,   0, false, "Float6_block0"},
        TensorParam{rocisa::DataType::BFloat6,  0, false, "BFloat6_block0"},
        TensorParam{rocisa::DataType::Float8,   0, false, "Float8_block0"},
        TensorParam{rocisa::DataType::BFloat8,  0, false, "BFloat8_block0"},
        // ----- (b) supported MX dtypes with mxBlock>0 -> true --------------
        TensorParam{rocisa::DataType::Float4,  32, true,  "Float4_block32"},
        TensorParam{rocisa::DataType::Float6,  32, true,  "Float6_block32"},
        TensorParam{rocisa::DataType::BFloat6, 32, true,  "BFloat6_block32"},
        TensorParam{rocisa::DataType::Float8,  32, true,  "Float8_block32"},
        TensorParam{rocisa::DataType::BFloat8, 32, true,  "BFloat8_block32"},
        // ----- (b') unsupported dtypes with mxBlock>0 -> false -------------
        TensorParam{rocisa::DataType::Float,   32, false, "Float_block32"},
        TensorParam{rocisa::DataType::Half,    32, false, "Half_block32"},
        TensorParam{rocisa::DataType::BFloat16,32, false, "BFloat16_block32"},
        TensorParam{rocisa::DataType::Int8,    32, false, "Int8_block32"},
        TensorParam{rocisa::DataType::Int32,   32, false, "Int32_block32"},
        // ----- mxBlock not equal to 32 (any positive value works) ----------
        TensorParam{rocisa::DataType::Float8,    1, true, "Float8_block1"},
        TensorParam{rocisa::DataType::BFloat8, 128, true, "BFloat8_block128"}
    ),
    [](::testing::TestParamInfo<TensorParam> const& info) {
        return std::string(info.param.name);
    }
);

// =============================================================================
//   Section 2 - TensileLite::Client::isMXProblem
//
//   Contract:
//       isMXProblem(P)
//         = isMXTensor(P.a, P.mxBlockA)
//            || isMXTensor(P.b, P.mxBlockB)
// =============================================================================
TEST(IsMXProblem, BothFP4)
{
    auto p = makeProblem(rocisa::DataType::Float4, rocisa::DataType::Float4,
                         /*mxBlockA=*/32, /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, BothFP6)
{
    auto p = makeProblem(rocisa::DataType::Float6, rocisa::DataType::Float6,
                         /*mxBlockA=*/32, /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, BothBFloat6)
{
    auto p = makeProblem(rocisa::DataType::BFloat6, rocisa::DataType::BFloat6,
                         /*mxBlockA=*/32, /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, BothFP8)
{
    auto p = makeProblem(rocisa::DataType::Float8, rocisa::DataType::Float8,
                         /*mxBlockA=*/32, /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, BothBFloat8)
{
    auto p = makeProblem(rocisa::DataType::BFloat8, rocisa::DataType::BFloat8,
                         /*mxBlockA=*/32, /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, MixedFP4AandFP8B)
{
    auto p = makeProblem(rocisa::DataType::Float4, rocisa::DataType::Float8,
                         /*mxBlockA=*/32, /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, MixedBFloat8AandFP4B)
{
    auto p = makeProblem(rocisa::DataType::BFloat8, rocisa::DataType::Float4,
                         /*mxBlockA=*/32, /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, OnlyA_isMX_BIsBF16)
{
    // First disjunct true, second disjunct short-circuits false (mxBlockB=0).
    auto p = makeProblem(rocisa::DataType::Float8, rocisa::DataType::BFloat16,
                         /*mxBlockA=*/32, /*mxBlockB=*/0);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, OnlyB_isMX_AIsBF16)
{
    auto p = makeProblem(rocisa::DataType::BFloat16, rocisa::DataType::Float4,
                         /*mxBlockA=*/0, /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, NeitherIsMX)
{
    auto p = makeProblem(rocisa::DataType::BFloat16, rocisa::DataType::BFloat16,
                         /*mxBlockA=*/0, /*mxBlockB=*/0);
    EXPECT_FALSE(isMXProblem(p));
}
TEST(IsMXProblem, FloatABIsFalse)
{
    auto p = makeProblem(rocisa::DataType::Float, rocisa::DataType::Float,
                         /*mxBlockA=*/0, /*mxBlockB=*/0);
    EXPECT_FALSE(isMXProblem(p));
}

// =============================================================================
//   Section 3 — Byte-stride formula
//
//   For FP8 / BFloat8 the OCP standard packs one element per byte. The
//   DataTypeInfo for these dtypes therefore reports elementSize == 1, and the
//   formula must be the identity on strides[2]. These tests pin BOTH facts:
//   if anyone ever changes elementSize for FP8, or breaks multiplyElementSize,
//   the failure surfaces here instead of as a silent multi-batch FP8 bug.
// =============================================================================
TEST(InitializeMXDataForFP4OrFP8_BatchStrideFormula, FP8_OneBytePerElement)
{
    auto const info = DataTypeInfo::Get(rocisa::DataType::Float8);
    ASSERT_EQ(info.elementSize, 1u)
        << "OCP E4M3 must pack 1 byte per element; if this assertion fires the "
           "patch 3/3 batch-stride formula needs to be revisited.";
    constexpr size_t kStrideElems = 12345; // arbitrary, prime-ish
    size_t const     bytes        = TensileLite::multiplyElementSize(
        kStrideElems, static_cast<float>(info.elementSize));
    EXPECT_EQ(bytes, kStrideElems);
}

TEST(InitializeMXDataForFP4OrFP8_BatchStrideFormula, BFloat8_OneBytePerElement)
{
    auto const info = DataTypeInfo::Get(rocisa::DataType::BFloat8);
    ASSERT_EQ(info.elementSize, 1u) << "OCP E5M2 must pack 1 byte per element.";
    constexpr size_t kStrideElems = 1u << 20; // 1 Mi elements
    size_t const     bytes        = TensileLite::multiplyElementSize(
        kStrideElems, static_cast<float>(info.elementSize));
    EXPECT_EQ(bytes, kStrideElems);
}

// =============================================================================
//   Section 4 — direct calls into TensileLite::Client::detail (MX builds only)
// =============================================================================
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
// -----------------------------------------------------------------------------
// 4.1  detail::hipMxScaleTypeForDataGenerator
// -----------------------------------------------------------------------------
TEST(HipMxScaleTypeForDataGenerator, MapsFloat8ToHIP_R_8F_E4M3)
{
    EXPECT_EQ(dt::hipMxScaleTypeForDataGenerator(rocisa::DataType::Float8),
              HIP_R_8F_E4M3);
}
TEST(HipMxScaleTypeForDataGenerator, MapsE5M3ToHIP_R_8F_E5M3_EXT)
{
    EXPECT_EQ(dt::hipMxScaleTypeForDataGenerator(rocisa::DataType::E5M3),
              static_cast<hipDataType>(HIP_R_8F_E5M3_EXT));
}
TEST(HipMxScaleTypeForDataGenerator, MapsE8AndNoneToHIP_R_8F_UE8M0)
{
    EXPECT_EQ(dt::hipMxScaleTypeForDataGenerator(rocisa::DataType::E8),
              HIP_R_8F_UE8M0);
    EXPECT_EQ(dt::hipMxScaleTypeForDataGenerator(rocisa::DataType::None),
              HIP_R_8F_UE8M0);
}
TEST(HipMxScaleTypeForDataGenerator, ThrowsOnUnsupportedScaleType)
{
    EXPECT_THROW(dt::hipMxScaleTypeForDataGenerator(rocisa::DataType::Float4),
                 std::runtime_error);
    EXPECT_THROW(dt::hipMxScaleTypeForDataGenerator(rocisa::DataType::BFloat8),
                 std::runtime_error);
    EXPECT_THROW(dt::hipMxScaleTypeForDataGenerator(rocisa::DataType::Float),
                 std::runtime_error);
}

// -----------------------------------------------------------------------------
// 4.2  detail::hipMxDataTypeForDataGenerator
// -----------------------------------------------------------------------------
TEST(HipMxDataTypeForDataGenerator, MapsFloat4ToHIP_R_4F_E2M1)
{
    EXPECT_EQ(dt::hipMxDataTypeForDataGenerator(rocisa::DataType::Float4),
              static_cast<hipDataType>(HIP_R_4F_E2M1));
}
TEST(HipMxDataTypeForDataGenerator, MapsFloat8ToHIP_R_8F_E4M3)
{
    EXPECT_EQ(dt::hipMxDataTypeForDataGenerator(rocisa::DataType::Float8),
              HIP_R_8F_E4M3);
}
TEST(HipMxDataTypeForDataGenerator, MapsBFloat8ToHIP_R_8F_E5M2)
{
    EXPECT_EQ(dt::hipMxDataTypeForDataGenerator(rocisa::DataType::BFloat8),
              HIP_R_8F_E5M2);
}
TEST(HipMxDataTypeForDataGenerator, MapsFloat6ToHIP_R_6F_E2M3)
{
    EXPECT_EQ(dt::hipMxDataTypeForDataGenerator(rocisa::DataType::Float6),
              static_cast<hipDataType>(HIP_R_6F_E2M3));
}
TEST(HipMxDataTypeForDataGenerator, MapsBFloat6ToHIP_R_6F_E3M2)
{
    EXPECT_EQ(dt::hipMxDataTypeForDataGenerator(rocisa::DataType::BFloat6),
              static_cast<hipDataType>(HIP_R_6F_E3M2));
}
TEST(HipMxDataTypeForDataGenerator, ThrowsOnUnsupportedDataType)
{
    EXPECT_THROW(dt::hipMxDataTypeForDataGenerator(rocisa::DataType::Float),
                 std::runtime_error);
    EXPECT_THROW(dt::hipMxDataTypeForDataGenerator(rocisa::DataType::Half),
                 std::runtime_error);
    EXPECT_THROW(dt::hipMxDataTypeForDataGenerator(rocisa::DataType::BFloat16),
                 std::runtime_error);
}

// -----------------------------------------------------------------------------
// 4.8  detail::decodeE8M0
// -----------------------------------------------------------------------------
TEST(DecodeE8M0, ZeroByteIsZero)        { EXPECT_EQ(dt::decodeE8M0(0x00), 0.0f); }
TEST(DecodeE8M0, AllOnesByteIsNaN)      { EXPECT_TRUE(std::isnan(dt::decodeE8M0(0xFF))); }
TEST(DecodeE8M0, BiasByteIsOne)         { EXPECT_EQ(dt::decodeE8M0(0x7F), 1.0f); }   // 2^0
TEST(DecodeE8M0, NextAboveBiasIsTwo)    { EXPECT_EQ(dt::decodeE8M0(0x80), 2.0f); }   // 2^1
TEST(DecodeE8M0, NextBelowBiasIsHalf)   { EXPECT_EQ(dt::decodeE8M0(0x7E), 0.5f); }   // 2^-1
TEST(DecodeE8M0, MatchesLdexpFormulaOverWideRange)
{
    for(int byte = 1; byte < 0xFF; ++byte)
    {
        float expected = std::ldexp(1.0f, byte - 127);
        EXPECT_EQ(dt::decodeE8M0(static_cast<uint8_t>(byte)), expected)
            << "byte=" << byte;
    }
}

// -----------------------------------------------------------------------------
// 4.9  detail::decodeMXElement
//      Calls the production dispatcher AND DGen directly with the same inputs;
//      the two must agree for every supported dtype, and the unsupported
//      branch must return NaN.
// -----------------------------------------------------------------------------
TEST(DecodeMXElement, FP4PathMatchesDGenToFloatPacked)
{
    // Two FP4 elements packed in one byte: low nibble 0x2 (= +1.0),
    // high nibble 0x0 (= +0.0). Scale 0x7F = 1.0.
    uint8_t data [1] = {0x02};
    uint8_t scale[1] = {0x7F};
    for(size_t e : {size_t{0}, size_t{1}})
    {
        float prod = dt::decodeMXElement(rocisa::DataType::Float4,
                                         scale, data, 0, e);
        float dgen = DGen::toFloatPacked<DGen::ocp_e2m1_mxfp4>(
            scale, data, 0, e);
        EXPECT_EQ(prod, dgen) << "elem=" << e;
    }
}
TEST(DecodeMXElement, Float8PathMatchesDGenToFloat_E4M3)
{
    uint8_t data [1] = {0x38};   // 2^0 = +1.0 in E4M3
    uint8_t scale[1] = {0x7F};
    float   prod = dt::decodeMXElement(rocisa::DataType::Float8,
                                       scale, data, 0, 0);
    float   dgen = DGen::toFloat<DGen::ocp_e4m3_mxfp8>(scale, data, 0, 0);
    EXPECT_EQ(prod, dgen);
    EXPECT_EQ(prod, 1.0f);
}
TEST(DecodeMXElement, BFloat8PathMatchesDGenToFloat_E5M2)
{
    uint8_t data [1] = {0x3C};   // 2^0 = +1.0 in E5M2
    uint8_t scale[1] = {0x7F};
    float   prod = dt::decodeMXElement(rocisa::DataType::BFloat8,
                                       scale, data, 0, 0);
    float   dgen = DGen::toFloat<DGen::ocp_e5m2_mxfp8>(scale, data, 0, 0);
    EXPECT_EQ(prod, dgen);
    EXPECT_EQ(prod, 1.0f);
}
TEST(DecodeMXElement, Float6PathMatchesDGenToFloatPacked)
{
    uint8_t data [3] = {0x00, 0x00, 0x00};
    uint8_t scale[1] = {0x7F};
    for(size_t e = 0; e < 4; ++e)
    {
        float prod = dt::decodeMXElement(rocisa::DataType::Float6,
                                         scale, data, 0, e);
        float dgen = DGen::toFloatPacked<DGen::ocp_e2m3_mxfp6>(
            scale, data, 0, e);
        EXPECT_EQ(prod, dgen) << "elem=" << e;
    }
}
TEST(DecodeMXElement, BFloat6PathMatchesDGenToFloatPacked)
{
    uint8_t data [3] = {0x00, 0x00, 0x00};
    uint8_t scale[1] = {0x7F};
    for(size_t e = 0; e < 4; ++e)
    {
        float prod = dt::decodeMXElement(rocisa::DataType::BFloat6,
                                         scale, data, 0, e);
        float dgen = DGen::toFloatPacked<DGen::ocp_e3m2_mxfp6>(
            scale, data, 0, e);
        EXPECT_EQ(prod, dgen) << "elem=" << e;
    }
}
TEST(DecodeMXElement, UnsupportedDataTypeReturnsNaN)
{
    uint8_t data [1] = {0x00};
    uint8_t scale[1] = {0x7F};
    EXPECT_TRUE(std::isnan(dt::decodeMXElement(rocisa::DataType::Float,
                                               scale, data, 0, 0)));
    EXPECT_TRUE(std::isnan(dt::decodeMXElement(rocisa::DataType::Half,
                                               scale, data, 0, 0)));
}

#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
