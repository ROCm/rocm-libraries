// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ClientArgsTestUtils.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DataInitializationHelpers.hpp"
#include "ProgramOptions.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/ScaleAlphaVec.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
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
using TensileLite::ContractionSolution;
using TensileLite::DataTypeInfo;
using TensileLite::KernelLanguage;
using TensileLite::PerformanceMetric;
using TensileLite::TensorDescriptor;
using TensileLite::Client::isMXProblem;
using TensileLite::Client::isMXTensor;
using TensileLite::TestUtils::makeMinimalClientArgs;
using TensileLite::TestUtils::setOption;

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
                                       bool             transB = false,
                                       int              mxBlockFreeA = 1,
                                       int              mxBlockFreeB = 1)
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
        if(mxBlockA > 0)
            problem.setMXScaleA(rocisa::DataType::E8, mxBlockA, {},
                                /*padScaleTensorFreeDim=*/true, mxBlockFreeA);
        if(mxBlockB > 0)
            problem.setMXScaleB(rocisa::DataType::E8, mxBlockB, {},
                                /*padScaleTensorFreeDim=*/true, mxBlockFreeB);
        return problem;
    }
} // namespace

TEST(ClientAlphaMode, ScalarSupportDefaultsToScalarAndAllowsRuntimeOverride)
{
    for(int modes : {1, 2, 3, 5, 6, 7})
        for(int runtimeOverride : {-1, 0, 1})
        {
            SCOPED_TRACE(::testing::Message() << "modes=" << modes
                                             << " override=" << runtimeOverride);
            auto args = makeMinimalClientArgs(0);
            setOption(args, "use-scaleAlphaVec", modes);
            if(runtimeOverride >= 0)
                setOption(args, "device-scalar-alpha", runtimeOverride != 0);
            TensileLite::Client::ClientProblemFactory factory(args);
            ASSERT_EQ(factory.problems().size(), 1u);
            auto problem = std::dynamic_pointer_cast<ContractionProblemGemm>(factory.problems()[0]);
            ASSERT_NE(problem, nullptr);
            const bool scalar = runtimeOverride < 0 ? modes >= 5 : runtimeOverride != 0;
            EXPECT_EQ(problem->getParams().deviceScalarAlpha(), scalar);
            EXPECT_EQ(problem->useScaleAlphaVec(),
                      scalar ? TensileLite::ScaleAlphaVecDeviceScalarBit : modes & 3);
            EXPECT_EQ(problem->scaleAlphaVec().totalLogicalElements(), scalar ? 1u : 128u);
        }
}

TEST(ClientAlphaMode, RejectsReservedScalarOnlyAndUnknownSupportModes)
{
    for(int modes : {-1, 4, 8})
    {
        auto args = makeMinimalClientArgs(0);
        setOption(args, "use-scaleAlphaVec", modes);
        EXPECT_THROW(TensileLite::Client::ClientProblemFactory factory(args), std::invalid_argument);
    }
}

// A random alpha or beta is drawn once for the problem and again for the launch, so a
// draw of +-1 here must not restrict the problem to a value-specialized solution: the
// launch would then throw "Alpha value X doesn't match that set in problem".
TEST(ClientScalarRestriction, RandomAlphaAndBetaAreUnrestricted)
{
    auto args = makeMinimalClientArgs(0);
    setOption(args, "init-alpha", TensileLite::Client::InitMode::Random);
    setOption(args, "init-beta", TensileLite::Client::InitMode::Random);
    // Each draw is in [-1000, 1000]; 4000 problems draw +-1 with probability 1 - e^-8.
    for(int i = 0; i < 4000; ++i)
    {
        TensileLite::Client::ClientProblemFactory factory(args);
        auto problem = std::dynamic_pointer_cast<ContractionProblemGemm>(factory.problems()[0]);
        ASSERT_NE(problem, nullptr);
        ASSERT_EQ(problem->alphaRestriction(), TensileLite::ScalarValue::Any) << "problem " << i;
        ASSERT_EQ(problem->betaRestriction(), TensileLite::ScalarValue::Any) << "problem " << i;
    }
}

TEST(ClientScalarRestriction, FixedAlphaAndBetaKeepTheirRestriction)
{
    auto args = makeMinimalClientArgs(0);
    setOption(args, "init-alpha", TensileLite::Client::InitMode::One);
    setOption(args, "init-beta", TensileLite::Client::InitMode::One);
    TensileLite::Client::ClientProblemFactory factory(args);
    auto problem = std::dynamic_pointer_cast<ContractionProblemGemm>(factory.problems()[0]);
    ASSERT_NE(problem, nullptr);
    EXPECT_EQ(problem->alphaRestriction(), TensileLite::ScalarValue::One);
    EXPECT_EQ(problem->betaRestriction(), TensileLite::ScalarValue::One);
}

TEST(ClientProblemFactoryDefaults, DisablesOmittedOptionalOutputsAndScales)
{
    using namespace TensileLite::Client;
    po::variables_map args;
    const std::array  omitted{"use-bias",
                             "bias-source",
                             "use-e",
                             "use-scaleCD",
                             "use-scaleAlphaVec",
                             "use-gradient",
                             "output-amaxD"};
    for(const auto& entry : makeMinimalClientArgs(0))
        if(std::find(omitted.begin(), omitted.end(), entry.first) == omitted.end())
            args.insert(entry);

    // Nonzero storage exposes missing defaults regardless of earlier tests.
    alignas(ClientProblemFactory) std::array<std::byte, sizeof(ClientProblemFactory)> storage;
    storage.fill(std::byte{0xa5});
    auto destroy = [](ClientProblemFactory* factory) { std::destroy_at(factory); };
    std::unique_ptr<ClientProblemFactory, decltype(destroy)> factory(
        std::construct_at(reinterpret_cast<ClientProblemFactory*>(storage.data()), args), destroy);

    ASSERT_EQ(factory->problems().size(), 1u);
    const auto& problem = dynamic_cast<const ContractionProblemGemm&>(*factory->problems().front());
    EXPECT_FALSE(problem.useE());
    EXPECT_FALSE(problem.useGradient());
    EXPECT_FALSE(problem.outputAmaxD());
    EXPECT_EQ(problem.useBias(), 0);
    EXPECT_EQ(problem.biasSrc(), ContractionProblemGemm::TENSOR::D);
    EXPECT_FALSE(problem.useScaleCD());
    EXPECT_EQ(problem.useScaleAlphaVec(), 0);
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
TEST(IsMXProblem, BothFP8Block128)
{
    // 1x128 block scaling: mxBlock only has to be positive, so widening the
    // block must not change MX-problem detection or the scale dtypes.
    auto p = makeProblem(rocisa::DataType::Float8, rocisa::DataType::Float8,
                         /*mxBlockA=*/128, /*mxBlockB=*/128);
    EXPECT_TRUE(isMXProblem(p));
    EXPECT_EQ(p.mxBlockA(), 128);
    EXPECT_EQ(p.mxBlockB(), 128);
}
TEST(IsMXProblem, BothFP8FreeTile128)
{
    // 128x128 block scaling. isMXProblem keys off mxBlock{A,B} only, so adding a
    // free-dimension extent must not change detection either way.
    auto p = makeProblem(rocisa::DataType::Float8, rocisa::DataType::Float8,
                         /*mxBlockA=*/128, /*mxBlockB=*/128,
                         /*M=*/128, /*N=*/128, /*K=*/256, /*batch=*/1,
                         /*transA=*/true, /*transB=*/false,
                         /*mxBlockFreeA=*/128, /*mxBlockFreeB=*/128);
    EXPECT_TRUE(isMXProblem(p));
    EXPECT_EQ(p.mxBlockA(), 128);
    EXPECT_EQ(p.mxBlockB(), 128);
    EXPECT_EQ(p.mxBlockFreeA(), 128);
    EXPECT_EQ(p.mxBlockFreeB(), 128);
}
TEST(IsMXProblem, FreeTileDefaultsToOne)
{
    // Every problem built before MXBlockFree* existed must keep reporting 1,
    // which is the divide-by-nothing 1xmxBlock layout.
    auto p = makeProblem(rocisa::DataType::Float8, rocisa::DataType::Float8,
                         /*mxBlockA=*/128, /*mxBlockB=*/128);
    EXPECT_EQ(p.mxBlockFreeA(), 1);
    EXPECT_EQ(p.mxBlockFreeB(), 1);
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

TEST(IsRandomLike, AcceptsOnlyRandomModes)
{
    using TensileLite::Client::InitMode;
    using TensileLite::Client::IsRandomLike;
    for(auto mode : {InitMode::Random,
                     InitMode::RandomNarrow,
                     InitMode::RandomNegPosLimited,
                     InitMode::UniformLowPrecision})
        EXPECT_TRUE(IsRandomLike(mode)) << mode;
    for(auto mode : {InitMode::Zero,
                     InitMode::One,
                     InitMode::SerialIdx,
                     InitMode::Identity,
                     InitMode::TrigSin,
                     InitMode::TrigIndSin})
        EXPECT_FALSE(IsRandomLike(mode)) << mode;
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

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
TEST(DataInitializationReferenceRecompute, Gfx950RequiresPerSolutionRecomputeForSwizzledMX)
{
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    if(std::string(prop.gcnArchName).find("gfx950") == std::string::npos)
        GTEST_SKIP() << "gfx950-specific recompute behavior is only observable on gfx950";

    auto problem = makeProblem(rocisa::DataType::Float8,
                               rocisa::DataType::Float8,
                               /*mxBlockA=*/32,
                               /*mxBlockB=*/32);
    auto args = makeMinimalClientArgs(/*mxScaleFormat=*/1);
    TensileLite::Client::ClientProblemFactory factory(args);
    TensileLite::Client::DataInitialization   dataInit(args, factory);

    ContractionSolution noSwizzleSolution;
    noSwizzleSolution.problemType.mxScaleFormat = 0;
    ContractionSolution swizzleSolution;
    swizzleSolution.problemType.mxScaleFormat = 1;

    EXPECT_FALSE(dataInit.referenceNeedsPerSolutionRecompute(problem, &noSwizzleSolution));
    EXPECT_TRUE(dataInit.referenceNeedsPerSolutionRecompute(problem, &swizzleSolution));
}

TEST(DataInitializationReferenceRecompute, Gfx1250ReusesPreswizzledMXAcrossSolutions)
{
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    if(std::string(prop.gcnArchName).find("gfx1250") == std::string::npos)
        GTEST_SKIP() << "gfx1250-specific reuse behavior is only observable on gfx1250";

    auto problem = makeProblem(rocisa::DataType::Float8,
                               rocisa::DataType::Float8,
                               /*mxBlockA=*/32,
                               /*mxBlockB=*/32);
    auto args = makeMinimalClientArgs(/*mxScaleFormat=*/1);
    TensileLite::Client::ClientProblemFactory factory(args);
    TensileLite::Client::DataInitialization   dataInit(args, factory);

    ContractionSolution swizzleSolution;
    swizzleSolution.problemType.mxScaleFormat = 1;

    EXPECT_FALSE(dataInit.referenceNeedsPerSolutionRecompute(problem, &swizzleSolution));
}

namespace
{
    // Exposes the state DataInitialization keeps to reuse MX inputs across solutions.
    class MXReuseProbe : public TensileLite::Client::DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;

        bool preswizzledA() const
        {
            return m_mxPreswizzledA;
        }
        TensorDescriptor const& initDescriptor(int tensor) const
        {
            return m_vdata[tensor].pristine.begin()->second.initDescriptor[0];
        }
    };

    bool deviceIsGfx950()
    {
        hipDeviceProp_t prop{};
        return hipGetDeviceProperties(&prop, 0) == hipSuccess
               && std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
    }

    TensileLite::Client::po::variables_map
        makeMXLaunchArgs(std::vector<std::vector<size_t>> const& problemSizes)
    {
        using TensileLite::Client::InitMode;

        auto args = makeMinimalClientArgs(/*mxScaleFormat=*/1);
        setOption(args, "problem-size", problemSizes);
        setOption(args, "mx-a-block", 32);
        setOption(args, "mx-b-block", 32);
        setOption(args, "mx-a-free-block", 1);
        setOption(args, "mx-b-free-block", 1);
        setOption(args, "uniform-summation-order", 0);
        setOption(args, "bias-type-args", std::vector<rocisa::DataType>{rocisa::DataType::Float});
        // Random keeps every scale at 1.0, which makes swizzled and canonical
        // scales identical; RandomNarrow varies them.
        for(auto const* name : {"init-a", "init-b", "init-mx-a", "init-mx-b"})
            setOption(args, name, InitMode::RandomNarrow);
        return args;
    }

    void setMXSolution(ContractionSolution& solution, int mxScaleFormat, int miK)
    {
        solution.problemType.mxScaleFormat     = mxScaleFormat;
        solution.sizeMapping.matrixInstruction = {16, 16, miK, 1};
    }

    std::vector<uint8_t> deviceMXScaleA(std::shared_ptr<TensileLite::ProblemInputs> const& inputs,
                                        ContractionProblemGemm const&                    problem)
    {
        auto const* gemmInputs
            = dynamic_cast<TensileLite::ContractionInputs const*>(inputs.get());
        std::vector<uint8_t> bytes(problem.mxsa().totalAllocatedBytes());
        HIP_CHECK_EXC(
            hipMemcpy(bytes.data(), gemmInputs->mxsa, bytes.size(), hipMemcpyDeviceToHost));
        return bytes;
    }
}

// From the second problem on, preSolution re-initializes the MX inputs before
// prepareGPUInputs runs. With validation off, the non-MX tensors must still be
// initialized for the new problem.
TEST(DataInitializationMXReuse, LaterProblemsInitializeNonMXInputs)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "solution-dependent MX preswizzle only runs on gfx950";

    auto args = makeMXLaunchArgs({{128, 128, 1, 512}, {256, 128, 1, 512}});
    TensileLite::Client::ClientProblemFactory factory(args);
    MXReuseProbe                              dataInit(args, factory);
    ContractionSolution                       solution;
    setMXSolution(solution, /*mxScaleFormat=*/1, /*miK=*/128);

    for(auto const& problem : factory.problems())
    {
        auto const& gemm = dynamic_cast<ContractionProblemGemm const&>(*problem);
        dataInit.preProblem(problem.get());
        dataInit.preSolution(&solution);
        dataInit.prepareGPUInputs(problem.get());
        EXPECT_EQ(dataInit.initDescriptor(ContractionProblemGemm::TENSOR::C), gemm.c());
        dataInit.postSolution();
        dataInit.postProblem();
    }
}

// A NoSwizzle solution uploads the canonical scales. The next HostPreSwizzle
// solution with the same MFMA K must get the swizzled scales back.
TEST(DataInitializationMXReuse, NoSwizzleSolutionDoesNotLeakIntoNextPreswizzle)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "solution-dependent MX preswizzle only runs on gfx950";

    auto args = makeMXLaunchArgs({{128, 128, 1, 512}});
    TensileLite::Client::ClientProblemFactory factory(args);
    MXReuseProbe                              dataInit(args, factory);
    auto*       problem = factory.problems()[0].get();
    auto const& gemm    = dynamic_cast<ContractionProblemGemm const&>(*problem);
    ContractionSolution preswizzle;
    ContractionSolution noSwizzle;
    setMXSolution(preswizzle, /*mxScaleFormat=*/1, /*miK=*/128);
    setMXSolution(noSwizzle, /*mxScaleFormat=*/0, /*miK=*/128);

    auto run = [&](ContractionSolution& solution) {
        dataInit.preSolution(&solution);
        auto bytes = deviceMXScaleA(dataInit.prepareGPUInputs(problem), gemm);
        dataInit.postSolution();
        return bytes;
    };

    dataInit.preProblem(problem);
    // The first prepareGPUInputs builds the swizzled scales; the second
    // solution goes through preSolution's re-init and records its MFMA K.
    auto const swizzled = run(preswizzle);
    EXPECT_EQ(run(preswizzle), swizzled);
    ASSERT_NE(run(noSwizzle), swizzled) << "the layouts must differ for this test to be meaningful";
    EXPECT_EQ(run(preswizzle), swizzled);
    EXPECT_TRUE(dataInit.preswizzledA());
    dataInit.postProblem();
}

namespace
{
    // Exposes the host-side source of each tensor so the staged upload can be checked.
    class SwizzleProbe : public TensileLite::Client::DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;

        const void* cpuSource(int tensor) const
        {
            return m_vdata[tensor].pristine.begin()->second.cpuInput.valid.get();
        }
    };

    std::vector<uint8_t> readDevice(const void* ptr, size_t bytes)
    {
        std::vector<uint8_t> out(bytes);
        HIP_CHECK_EXC(hipMemcpy(out.data(), ptr, bytes, hipMemcpyDeviceToHost));
        return out;
    }
} // namespace

// SwizzleTensorA/B: prepareGPUInputs allocates the padded buffer, stages the source into the
// swizzled layout and uploads it (a second call reuses the cached staging).
TEST(DataInitializationSwizzle, UploadsTheStagedLayoutForAAndB)
{
    using TensileLite::Client::detail::stageSwizzledTensor;
    using TensileLite::Client::detail::swizzleStageGeometry;

    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    const bool isGfx1250 = std::string(prop.gcnArchName).find("gfx1250") != std::string::npos;

    auto args = makeMXLaunchArgs({{48, 40, 1, 100}});
    setOption(args, "mx-a-block", 0);
    setOption(args, "mx-b-block", 0);
    setOption(args, "mx-scale-format", 0);
    // Swizzled operands are K-fastest (TN): A is [K, M], B is [K, N].
    setOption(args, "problem-identifier", std::string("Contraction_l_Alik_Bljk_Cijk_Dijk"));
    setOption(args, "swizzle-tensor-a", true);
    setOption(args, "swizzle-tensor-b", true);
    TensileLite::Client::ClientProblemFactory factory(args);
    SwizzleProbe                              dataInit(args, factory);
    ContractionSolution                       solution;
    auto*                                     problem = factory.problems()[0].get();
    auto const& gemm = dynamic_cast<ContractionProblemGemm const&>(*problem);
    ASSERT_TRUE(gemm.swizzleTensorA());
    ASSERT_TRUE(gemm.swizzleTensorB());

    dataInit.preProblem(problem);
    dataInit.preSolution(&solution);
    for(int pass = 0; pass < 2; ++pass) // the second pass takes the cached-staging path
    {
        auto const inputs = dataInit.prepareGPUInputs(problem);
        auto const* gemmInputs
            = dynamic_cast<TensileLite::ContractionInputs const*>(inputs.get());
        ASSERT_NE(gemmInputs, nullptr);
        struct
        {
            int                     tensor;
            const void*             device;
            TensorDescriptor const& desc;
        } const tensors[] = {{ContractionProblemGemm::TENSOR::A, gemmInputs->a, gemm.a()},
                             {ContractionProblemGemm::TENSOR::B, gemmInputs->b, gemm.b()}};
        for(auto const& t : tensors)
        {
            const auto geom     = swizzleStageGeometry(t.desc, isGfx1250);
            auto       expected = stageSwizzledTensor(t.desc, dataInit.cpuSource(t.tensor), geom);
            auto       got      = readDevice(t.device, expected.getNumBytes());
            EXPECT_EQ(std::memcmp(got.data(), expected.as<void>(), got.size()), 0)
                << "tensor " << t.tensor << " pass " << pass;
        }
    }
    dataInit.postSolution();
    dataInit.postProblem();
}
#endif

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

// -----------------------------------------------------------------------------
// 4.10 detail::naturalMXScaleFromCanonical
//      Must undo restrideMXScaleBufferKFast, so the swizzled GPU scale can be
//      built from the canonical copy instead of regenerating the MX data.
// -----------------------------------------------------------------------------
TEST(NaturalMXScaleFromCanonical, InvertsKFastRestride)
{
    // K = 576 with mxBlock 32: 18 K blocks, padded to 24.
    size_t const         compactFree = 5, compactKBlocks = 18, paddedKBlocks = 24;
    std::vector<uint8_t> natural(compactFree * compactKBlocks);
    std::iota(natural.begin(), natural.end(), uint8_t{1});

    std::vector<uint8_t> canonical(compactFree * paddedKBlocks, 0xAB);
    std::memcpy(canonical.data(), natural.data(), natural.size());
    restrideMXScaleBufferKFast(canonical.data(), compactFree, compactKBlocks, paddedKBlocks, 1);
    ASSERT_NE(std::vector<uint8_t>(canonical.begin(), canonical.begin() + natural.size()), natural);

    EXPECT_EQ(dt::naturalMXScaleFromCanonical(canonical.data(),
                                              natural.size(),
                                              /*kFast=*/true,
                                              compactFree,
                                              compactKBlocks,
                                              paddedKBlocks,
                                              1),
              natural);
}
TEST(NaturalMXScaleFromCanonical, CopiesPrefixWhenNotRestrided)
{
    std::vector<uint8_t> canonical(64);
    std::iota(canonical.begin(), canonical.end(), uint8_t{1});
    std::vector<uint8_t> const prefix(canonical.begin(), canonical.begin() + 40);

    // K-slow: the canonical copy keeps the natural order, padding at the tail.
    EXPECT_EQ(dt::naturalMXScaleFromCanonical(canonical.data(), 40, false, 5, 8, 16, 1), prefix);
    // K-fast without K-block padding: nothing was restrided.
    EXPECT_EQ(dt::naturalMXScaleFromCanonical(canonical.data(), 40, true, 5, 8, 8, 1), prefix);
}

// -----------------------------------------------------------------------------
// 4.11 detail::swizzleNaturalMXScale
//      Swizzling the natural scales must give exactly what generateMXInput
//      emits when asked for the layout, for every operand orientation. The
//      client calls generateMXInput with the data tensor's sizes: {K, MN}
//      when K is along the rows (TN A, N B), {MN, K} otherwise.
// -----------------------------------------------------------------------------
struct SwizzleParam
{
    hipDataType   dataType;
    MXScaleLayout layout;
    bool          isMatrixA;
    bool          transposed;
    size_t        K;
    size_t        MN;
};
class SwizzleNaturalMXScaleTest : public ::testing::TestWithParam<SwizzleParam>
{
};
TEST_P(SwizzleNaturalMXScaleTest, MatchesGenerateMXInputWithLayout)
{
    auto const&       p         = GetParam();
    constexpr int     mxBlock   = 32;
    constexpr uint8_t sentinel  = 0xAB;
    bool const        kIsRows   = (p.isMatrixA && p.transposed) || (!p.isMatrixA && !p.transposed);
    size_t const      rows      = kIsRows ? p.K : p.MN;
    size_t const      cols      = kIsRows ? p.MN : p.K;
    size_t const      dataBytes = p.dataType == static_cast<hipDataType>(HIP_R_4F_E2M1)
                                      ? (rows * cols + 1) / 2
                                      : rows * cols;
    size_t const      naturalBytes = ((rows + mxBlock - 1) / mxBlock) * cols;

    auto generate = [&](MXScaleLayout layout, std::vector<uint8_t>& scale) {
        std::vector<uint8_t> data(dataBytes);
        scale.assign(4 * naturalBytes + 4096, sentinel);
        generateMXInput(p.dataType,
                        HIP_R_8F_UE8M0,
                        data.data(),
                        scale.data(),
                        rows,
                        cols,
                        rows,
                        p.transposed,
                        p.isMatrixA ? mxBlock : 1,
                        p.isMatrixA ? 1 : mxBlock,
                        p.isMatrixA,
                        layout,
                        "Bounded",
                        -1.0f,
                        1.0f);
        return data;
    };

    std::vector<uint8_t> natural, expected;
    auto const           dataNatural  = generate(MXScaleLayout::None, natural);
    auto const           dataSwizzled = generate(p.layout, expected);
    ASSERT_EQ(dataNatural, dataSwizzled);
    natural.resize(naturalBytes);

    auto const swizzled
        = dt::swizzleNaturalMXScale(natural, p.layout, rows, cols, mxBlock, kIsRows);
    ASSERT_LT(swizzled.size(), expected.size());
    EXPECT_EQ(swizzled, std::vector<uint8_t>(expected.begin(), expected.begin() + swizzled.size()));
    EXPECT_EQ(expected[swizzled.size()], sentinel)
        << "generateMXInput wrote more swizzled bytes than swizzleNaturalMXScale returned";
}
INSTANTIATE_TEST_SUITE_P(
    MXScaleLayouts,
    SwizzleNaturalMXScaleTest,
    ::testing::ValuesIn([] {
        std::vector<SwizzleParam> params;
        for(auto dataType : {HIP_R_8F_E4M3, static_cast<hipDataType>(HIP_R_4F_E2M1)})
            for(auto layout : {MXScaleLayout::GFX950, MXScaleLayout::GFX1250})
                for(bool isMatrixA : {true, false})
                    for(bool transposed : {true, false})
                        for(auto [K, MN] : {std::pair<size_t, size_t>{512, 128},
                                            std::pair<size_t, size_t>{576, 96}})
                            params.push_back({dataType, layout, isMatrixA, transposed, K, MN});
        return params;
    }()),
    [](::testing::TestParamInfo<SwizzleParam> const& info) {
        auto const& p = info.param;
        return std::string(p.dataType == HIP_R_8F_E4M3 ? "FP8" : "FP4")
               + (p.layout == MXScaleLayout::GFX950 ? "_gfx950" : "_gfx1250")
               + (p.isMatrixA ? "_A" : "_B") + (p.transposed ? "T" : "N") + "_K"
               + std::to_string(p.K) + "_MN" + std::to_string(p.MN);
    });


// =============================================================================
//   Section 5 - downsampleMXScaleFreeDim
//
//   mxDataGenerator only tiles scales along one axis, so a 2D MX scaling tile
//   (MXBlockFree{A,B} > 1) is produced by generating the 1xmxBlock layout and
//   collapsing every group of mxBlockFree free-dimension entries here. The
//   representative is the group MAXIMUM for UE8M0 scales (byte order == numeric
//   order, so a byte-wise max is exact) and the group's first entry otherwise.
//
//   Two buffer layouts have to work:
//     kFast  (bound dim at index 0, TN A / NT B) -> [free][kBlock]
//     !kFast                                     -> [kBlock][free]
// =============================================================================

// kFast: src is [free][kBlock] with 4 free rows x 2 K-blocks; a 2-row tile
// collapses it to [2][2], taking the max down each column of the pair.
TEST(DownsampleMXScaleFreeDim, KFastTakesGroupMaxPerKBlock)
{
    //            kb0   kb1
    // free 0:     10     1
    // free 1:      3    20
    // free 2:      7     7
    // free 3:      7     8
    std::vector<uint8_t> src = {10, 1, 3, 20, 7, 7, 7, 8};
    std::vector<uint8_t> dst(4, 0xEE);

    dt::downsampleMXScaleFreeDim(dst.data(), src.data(), /*compactFreeDim=*/4,
                                 /*kBlocks=*/2, /*mxBlockFree=*/2, /*elemBytes=*/1,
                                 /*kFast=*/true, /*unsignedExponentScale=*/true);

    EXPECT_EQ(dst[0], 10); // tile 0, kb0: max(10, 3)
    EXPECT_EQ(dst[1], 20); // tile 0, kb1: max(1, 20)
    EXPECT_EQ(dst[2],  7); // tile 1, kb0: max(7, 7)
    EXPECT_EQ(dst[3],  8); // tile 1, kb1: max(7, 8)
}

// !kFast: src is [kBlock][free], so the free axis is the fast one.
TEST(DownsampleMXScaleFreeDim, KSlowTakesGroupMaxPerKBlock)
{
    // kb0: free 0..3 = 10, 3, 7, 7
    // kb1: free 0..3 =  1, 20, 7, 8
    std::vector<uint8_t> src = {10, 3, 7, 7, 1, 20, 7, 8};
    std::vector<uint8_t> dst(4, 0xEE);

    dt::downsampleMXScaleFreeDim(dst.data(), src.data(), /*compactFreeDim=*/4,
                                 /*kBlocks=*/2, /*mxBlockFree=*/2, /*elemBytes=*/1,
                                 /*kFast=*/false, /*unsignedExponentScale=*/true);

    // dst is [kBlock][tiledFree] with tiledFree == 2.
    EXPECT_EQ(dst[0], 10); // kb0, tile 0
    EXPECT_EQ(dst[1],  7); // kb0, tile 1
    EXPECT_EQ(dst[2], 20); // kb1, tile 0
    EXPECT_EQ(dst[3],  8); // kb1, tile 1
}

// A free dimension that is not a multiple of the tile leaves a short last
// group; the max must be taken over only the entries that exist.
TEST(DownsampleMXScaleFreeDim, PartialLastTileStopsAtTheFreeDimension)
{
    // 3 free rows x 1 K-block, tile of 2 -> tiles {0,1} and {2}.
    std::vector<uint8_t> src = {5, 9, 4};
    std::vector<uint8_t> dst(2, 0xEE);

    dt::downsampleMXScaleFreeDim(dst.data(), src.data(), /*compactFreeDim=*/3,
                                 /*kBlocks=*/1, /*mxBlockFree=*/2, /*elemBytes=*/1,
                                 /*kFast=*/true, /*unsignedExponentScale=*/true);

    EXPECT_EQ(dst[0], 9); // max(5, 9)
    EXPECT_EQ(dst[1], 4); // the lone survivor, not a read past the buffer
}

// Without an unsigned-exponent encoding a byte-wise max is meaningless, so the
// group's first entry is copied verbatim.
TEST(DownsampleMXScaleFreeDim, NonExponentScaleKeepsFirstOfEachGroup)
{
    std::vector<uint8_t> src = {5, 9, 4, 1};
    std::vector<uint8_t> dst(2, 0xEE);

    dt::downsampleMXScaleFreeDim(dst.data(), src.data(), /*compactFreeDim=*/4,
                                 /*kBlocks=*/1, /*mxBlockFree=*/2, /*elemBytes=*/1,
                                 /*kFast=*/true, /*unsignedExponentScale=*/false);

    EXPECT_EQ(dst[0], 5);
    EXPECT_EQ(dst[1], 4);
}

// mxBlockFree == 1 is the layout every pre-MXBlockFree config uses: the helper
// must not touch the destination at all, since the caller writes straight into
// the scale buffer in that case.
TEST(DownsampleMXScaleFreeDim, TileExtentOneIsANoOp)
{
    std::vector<uint8_t> src = {5, 9, 4, 1};
    std::vector<uint8_t> dst(4, 0xEE);

    dt::downsampleMXScaleFreeDim(dst.data(), src.data(), 4, 1, /*mxBlockFree=*/1, 1,
                                 true, true);
    for(auto b : dst)
        EXPECT_EQ(b, 0xEE);

    // Empty buffers are equally inert.
    dt::downsampleMXScaleFreeDim(dst.data(), src.data(), /*compactFreeDim=*/0, 1, 2, 1,
                                 true, true);
    dt::downsampleMXScaleFreeDim(dst.data(), src.data(), 4, /*kBlocks=*/0, 2, 1,
                                 true, true);
    for(auto b : dst)
        EXPECT_EQ(b, 0xEE);
}

// Multi-byte scale elements are copied whole. elemBytes != 1 also disables the
// max, since the comparison only inspects the first byte.
TEST(DownsampleMXScaleFreeDim, MultiByteElementsAreCopiedWhole)
{
    // 4 free rows x 1 K-block, 2 bytes per scale, tile of 2.
    std::vector<uint8_t> src = {0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44};
    std::vector<uint8_t> dst(4, 0xEE);

    dt::downsampleMXScaleFreeDim(dst.data(), src.data(), /*compactFreeDim=*/4,
                                 /*kBlocks=*/1, /*mxBlockFree=*/2, /*elemBytes=*/2,
                                 /*kFast=*/true, /*unsignedExponentScale=*/true);

    EXPECT_EQ(dst[0], 0xAA);
    EXPECT_EQ(dst[1], 0xBB);
    EXPECT_EQ(dst[2], 0x11);
    EXPECT_EQ(dst[3], 0x22);
}

// =============================================================================
//   Section 6 - varyMXScaleFreeTiles
//
//   Random data quantizes every block against 0x7F, so the downsampled 2D
//   scales are all equal and a wrong scale row would still validate. The helper
//   offsets each tile by -2..2 so neighbouring free tiles always differ.
// =============================================================================

// kFast: [tiledFree][kBlock]. delta = (g + 2*kb) % 5 - 2.
TEST(VaryMXScaleFreeTiles, KFastNeighbouringTilesDiffer)
{
    std::vector<uint8_t> s(4 * 2, 0x7F); // 4 tiles x 2 K-blocks

    dt::varyMXScaleFreeTiles(s.data(), /*tiledFreeDim=*/4, /*kBlocks=*/2, /*kFast=*/true);

    EXPECT_EQ(s[0], 0x7F - 2); // g0 kb0
    EXPECT_EQ(s[1], 0x7F + 0); // g0 kb1
    EXPECT_EQ(s[2], 0x7F - 1); // g1 kb0
    EXPECT_EQ(s[3], 0x7F + 1); // g1 kb1
    EXPECT_EQ(s[4], 0x7F + 0); // g2 kb0
    EXPECT_EQ(s[6], 0x7F + 1); // g3 kb0
    for(size_t kb = 0; kb < 2; kb++)
        for(size_t g = 0; g + 1 < 4; g++)
            EXPECT_NE(s[g * 2 + kb], s[(g + 1) * 2 + kb]) << "g=" << g << " kb=" << kb;
}

// !kFast: [kBlock][tiledFree] gets the same per-(tile, K-block) offset.
TEST(VaryMXScaleFreeTiles, KSlowMatchesKFastPerTile)
{
    std::vector<uint8_t> fast(3 * 2, 0x7F);
    std::vector<uint8_t> slow(3 * 2, 0x7F);

    dt::varyMXScaleFreeTiles(fast.data(), 3, 2, /*kFast=*/true);
    dt::varyMXScaleFreeTiles(slow.data(), 3, 2, /*kFast=*/false);

    for(size_t g = 0; g < 3; g++)
        for(size_t kb = 0; kb < 2; kb++)
            EXPECT_EQ(fast[g * 2 + kb], slow[kb * 3 + g]) << "g=" << g << " kb=" << kb;
}

// Zero and NaN encodings are left alone; the result never reaches either.
TEST(VaryMXScaleFreeTiles, KeepsZeroNaNAndClampsToFiniteRange)
{
    std::vector<uint8_t> s = {0x00, 0xFF, 0x01, 0xFE};

    // kFast, 4 tiles x 1 K-block: deltas -2, -1, 0, +1.
    dt::varyMXScaleFreeTiles(s.data(), 4, 1, /*kFast=*/true);

    EXPECT_EQ(s[0], 0x00);
    EXPECT_EQ(s[1], 0xFF);
    EXPECT_EQ(s[2], 0x01); // 0x01 + 0
    EXPECT_EQ(s[3], 0xFE); // 0xFE + 1 clamps to 0xFE
}


// ---------------------------------------------------------------------------
//  Swizzled operand (SwizzleTensorA/B): K block per architecture, K padding,
//  allocation size and the host-side staging into [nO, kO, kM, nI, kI].
// ---------------------------------------------------------------------------
namespace
{
    using TensileLite::Client::detail::calculateKforSwizzling;
    using TensileLite::Client::detail::getSwizzledTensorNumAllocatedElements;
    using TensileLite::Client::detail::stageSwizzledTensor;
    using TensileLite::Client::detail::swizzleStageGeometry;
    using TensileLite::Client::detail::swizzledPaddedKUnit;

    constexpr size_t kMi = 16; // swizzled free-dimension block (MiM / MiN)

    // Element (n, k) of a [N, K] row-major operand with leading dimension ld and batch stride bs.
    uint8_t srcByte(size_t b, size_t n, size_t k, size_t ld, size_t bs)
    {
        return uint8_t(1 + (b * bs + n * ld + k) % 251);
    }

    struct SwizzleCase
    {
        rocisa::DataType dtype;
        size_t           bytes; // element bytes (whole bytes only)
        size_t           k, n, batches, ld, bs;
    };

    // Stage a descriptor of dtype with the source filled by srcByte and return the staged bytes.
    std::vector<uint8_t> stageCase(const SwizzleCase& c, bool isGfx1250, size_t& padK)
    {
        // The kernel sees [K, free, batch]: dim0 = K (fastest).
        TensorDescriptor desc("a", c.dtype, {c.k, c.n, c.batches}, {1, c.ld, c.bs});
        const auto       g = swizzleStageGeometry(desc, isGfx1250);
        padK               = g.paddedShape[1];

        std::vector<uint8_t> src((c.batches - 1) * c.bs + (c.n - 1) * c.ld + c.k, 0);
        for(size_t b = 0; b < c.batches; ++b)
            for(size_t n = 0; n < c.n; ++n)
                for(size_t k = 0; k < c.k; ++k)
                    src[b * c.bs + n * c.ld + k] = srcByte(b, n, k, c.ld, c.bs);

        auto                 staged = stageSwizzledTensor(desc, src.data(), g);
        std::vector<uint8_t> out(staged.getNumBytes());
        std::memcpy(out.data(), staged.as<void>(), out.size());
        return out;
    }

    // Independent model of the host layout for 1-byte elements: [nO, kO, kM, nI, kI] with a
    // 16-byte kI block, kM = 2 lane groups (gfx1250 32 B swz granule) over a padK-wide row.
    size_t swizzledOffset(size_t n, size_t k, size_t padK)
    {
        const size_t nO = n / kMi, nI = n % kMi;
        const size_t kBlock = k / 16, kI = k % 16;
        const size_t kO = kBlock / 2, kM = kBlock % 2;
        const size_t KO = padK / 32;
        return (((nO * KO + kO) * 2 + kM) * kMi + nI) * 16 + kI;
    }
} // namespace

TEST(SwizzledPaddedK, Gfx942KeepsItsSwizzleKBlock)
{
    EXPECT_EQ(swizzledPaddedKUnit(64, 1.0f, false), 64u);
    EXPECT_EQ(swizzledPaddedKUnit(32, 2.0f, false), 32u);
    EXPECT_EQ(swizzledPaddedKUnit(128, 0.5f, false), 128u);
}

TEST(SwizzledPaddedK, Gfx1250PadsToSixtyFourBytes)
{
    EXPECT_EQ(swizzledPaddedKUnit(32, 1.0f, true), 64u);  // F8: 32 B swizzle block -> 64 elements
    EXPECT_EQ(swizzledPaddedKUnit(16, 2.0f, true), 32u);  // BF16
    EXPECT_EQ(swizzledPaddedKUnit(64, 0.5f, true), 128u); // F4: byte-packed, 64 B = 128 elements
    EXPECT_EQ(swizzledPaddedKUnit(256, 1.0f, true), 256u) << "a larger block is never narrowed";
}

TEST(SwizzledKBlock, Gfx1250UsesThirtyTwoByteBlocks)
{
    struct Row
    {
        rocisa::DataType dtype;
        size_t           mik942, mik1250;
    };
    const Row rows[] = {{rocisa::DataType::Half, 16, 8},
                        {rocisa::DataType::BFloat16, 16, 8},
                        {rocisa::DataType::Int8, 32, 16},
                        {rocisa::DataType::Float8, 32, 16},
                        {rocisa::DataType::BFloat8, 32, 16},
                        {rocisa::DataType::Float4, 32, 32}};
    for(const auto& r : rows)
    {
        size_t MiK = 0, MiKv = 0, PackK = 0;
        calculateKforSwizzling(r.dtype, MiK, MiKv, PackK, false);
        EXPECT_EQ(MiK, r.mik942) << int(r.dtype);
        MiK = MiKv = PackK = 0;
        calculateKforSwizzling(r.dtype, MiK, MiKv, PackK, true);
        EXPECT_EQ(MiK, r.mik1250) << int(r.dtype);
    }
    size_t MiK = 0, MiKv = 0, PackK = 0;
    EXPECT_THROW(calculateKforSwizzling(rocisa::DataType::None, MiK, MiKv, PackK, true),
                 std::runtime_error);
}

TEST(SwizzledAllocation, SameOnBothArchitecturesForEveryK)
{
    // Different K blocks, same 64-byte K padding: the allocation (and so the layout) must match.
    // (F4 is the exception: gfx942/gfx950 keep their 32-byte padding, see the F4 test below.)
    for(auto dtype : {rocisa::DataType::Float8, rocisa::DataType::BFloat16, rocisa::DataType::Half,
                      rocisa::DataType::Int8})
    {
        for(size_t k = 1; k <= 300; k += 7)
        {
            TensorDescriptor desc("a", dtype, {k, 20, 1});
            size_t           MiK = 0, MiKv = 0, PackK = 0, MiK2 = 0, MiKv2 = 0, PackK2 = 0;
            calculateKforSwizzling(dtype, MiK, MiKv, PackK, false);
            calculateKforSwizzling(dtype, MiK2, MiKv2, PackK2, true);
            EXPECT_EQ(getSwizzledTensorNumAllocatedElements(desc, kMi, MiK, PackK, false),
                      getSwizzledTensorNumAllocatedElements(desc, kMi, MiK2, PackK2, true))
                << int(dtype) << " K=" << k;
        }
    }
}

TEST(SwizzledAllocation, PadsFreeDimAndKAndHonoursTheBatchStride)
{
    size_t MiK = 0, MiKv = 0, PackK = 0;
    calculateKforSwizzling(rocisa::DataType::Float8, MiK, MiKv, PackK, true);
    // K=100 -> 128, N=20 -> 32: one batch is 4096 elements.
    EXPECT_EQ(getSwizzledTensorNumAllocatedElements(
                  TensorDescriptor("a", rocisa::DataType::Float8, {100, 20, 1}), kMi, MiK, PackK, true),
              4096u);
    // Batched, dense: batches follow each other.
    EXPECT_EQ(getSwizzledTensorNumAllocatedElements(
                  TensorDescriptor("a", rocisa::DataType::Float8, {100, 20, 3}, {1, 100, 2000}),
                  kMi, MiK, PackK, true),
              std::max<size_t>(4096 * 3, 2000 * 2 + 4096));
    // A batch stride wider than the padded batch is kept (the kernel steps by it).
    EXPECT_EQ(getSwizzledTensorNumAllocatedElements(
                  TensorDescriptor("a", rocisa::DataType::Float8, {100, 20, 3}, {1, 100, 8192}),
                  kMi, MiK, PackK, true),
              8192u * 2 + 4096);
}

TEST(SwizzledStaging, Gfx1250MatchesTheHostLayoutModel)
{
    for(size_t k : {size_t(32), size_t(33), size_t(64), size_t(100), size_t(200)})
    {
        for(size_t n : {size_t(16), size_t(20), size_t(40)})
        {
            SwizzleCase c{rocisa::DataType::Float8, 1, k, n, 1, k, 0};
            size_t      padK = 0;
            auto        out  = stageCase(c, true, padK);
            ASSERT_EQ(padK % 64, 0u) << "K is padded to 64 bytes";
            ASSERT_GE(padK, k);
            const size_t paddedN = (n + kMi - 1) / kMi * kMi;
            ASSERT_EQ(out.size(), paddedN * padK);
            std::vector<uint8_t> expect(out.size(), 0);
            for(size_t r = 0; r < n; ++r)
                for(size_t kk = 0; kk < k; ++kk)
                    expect[swizzledOffset(r, kk, padK)] = srcByte(0, r, kk, k, 0);
            EXPECT_EQ(out, expect) << "K=" << k << " N=" << n;
        }
    }
}

TEST(SwizzledStaging, IdenticalBytesOnGfx942AndGfx1250)
{
    // F8: gfx942 swizzles in 64-byte K blocks, gfx1250 in 32-byte ones; flattening (kO, kM) gives
    // the same bytes once K is padded the same way.
    for(size_t k : {size_t(40), size_t(64), size_t(150), size_t(256)})
    {
        SwizzleCase c{rocisa::DataType::Float8, 1, k, 24, 1, k, 0};
        size_t      pad942 = 0, pad1250 = 0;
        auto        a = stageCase(c, false, pad942);
        auto        b = stageCase(c, true, pad1250);
        EXPECT_EQ(pad942, pad1250) << "K=" << k;
        EXPECT_EQ(a, b) << "K=" << k;
    }
}

TEST(SwizzledStaging, BatchesSitAtTheSourceBatchStrideAndKeepTheLeadingDimension)
{
    // ld (160) > K (100) and a batch stride wider than one padded batch (128 * 32 = 4096).
    SwizzleCase c{rocisa::DataType::Float8, 1, 100, 20, 3, 160, 5000};
    size_t      padK = 0;
    auto        out  = stageCase(c, true, padK);
    ASSERT_EQ(padK, 128u);
    ASSERT_EQ(out.size(), 2 * 5000u + 32 * 128u);
    for(size_t b = 0; b < c.batches; ++b)
        for(size_t n = 0; n < c.n; ++n)
            for(size_t k = 0; k < c.k; ++k)
                ASSERT_EQ(out[b * c.bs + swizzledOffset(n, k, padK)], srcByte(b, n, k, c.ld, c.bs))
                    << "b=" << b << " n=" << n << " k=" << k;
    // The gap between one batch's padded data and the next batch is zero.
    for(size_t i = 32 * 128; i < 5000; ++i)
        ASSERT_EQ(out[i], 0) << i;
}

TEST(SwizzledStaging, TwoByteElementsUseSixteenByteKBlocks)
{
    // BF16: one 16-byte block is 8 elements; K pads to 32 elements (64 B).
    TensorDescriptor desc("a", rocisa::DataType::BFloat16, {20, 16, 1}); // K=20, N=16
    const auto       g = swizzleStageGeometry(desc, true);
    EXPECT_EQ(g.paddedShape[0], 16u);
    EXPECT_EQ(g.paddedShape[1], 32u);
    EXPECT_EQ(g.totalNative, 16u * 32u);
    EXPECT_FALSE(g.isSubByte);
}

TEST(SwizzledStaging, SubByteElementsStageAsBytes)
{
    // F4 is staged as packed bytes. K=40 elements = 20 bytes: gfx1250 pads K to 64 bytes
    // (128 elements), gfx942/gfx950 keep their 32-byte padding (64 elements).
    TensorDescriptor desc("a", rocisa::DataType::Float4, {40, 16, 1});
    const auto       g1250 = swizzleStageGeometry(desc, true);
    const auto       g942  = swizzleStageGeometry(desc, false);
    EXPECT_TRUE(g1250.isSubByte);
    EXPECT_EQ(g1250.effUnrolled, 20u);
    EXPECT_EQ(g1250.paddedShape[1], 64u);
    EXPECT_EQ(g942.paddedShape[1], 32u);
    EXPECT_EQ(g1250.totalNative, 16u * 128u) << "native elements: 16 rows x 64 B x 2";
    EXPECT_EQ(g942.totalNative, 16u * 64u);

    // With K already a whole 64 bytes the two stage the same bytes.
    TensorDescriptor full("a", rocisa::DataType::Float4, {128, 16, 1});
    const auto       f1250 = swizzleStageGeometry(full, true);
    const auto       f942  = swizzleStageGeometry(full, false);
    EXPECT_EQ(f1250.totalNative, f942.totalNative);
    std::vector<uint8_t> src(16 * 64);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = uint8_t(1 + i % 251);
    auto a = stageSwizzledTensor(full, src.data(), f1250);
    auto b = stageSwizzledTensor(full, src.data(), f942);
    ASSERT_EQ(a.getNumBytes(), b.getNumBytes());
    EXPECT_EQ(std::memcmp(a.as<void>(), b.as<void>(), a.getNumBytes()), 0);
}

TEST(SwizzledStaging, RejectsAStrideThatCannotHoldAPaddedBatch)
{
    // One padded batch is 4096 elements, but the batches are only 2000 apart: the kernel would step
    // into the previous batch's data.
    TensorDescriptor desc("a", rocisa::DataType::Float8, {100, 20, 3}, {1, 100, 2000});
    try
    {
        swizzleStageGeometry(desc, true);
        FAIL() << "expected the batch stride check to throw";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("batch stride 2000 is smaller"), std::string::npos)
            << e.what();
    }
    // A single batch has no stride to violate.
    TensorDescriptor one("a", rocisa::DataType::Float8, {100, 20, 1}, {1, 100, 2000});
    EXPECT_NO_THROW(swizzleStageGeometry(one, true));
}

TEST(SwizzledStaging, RejectsAnUnpackedSubByteLeadingDimension)
{
    TensorDescriptor desc("a", rocisa::DataType::Float4, {64, 16, 1}, {1, 80, 0});
    EXPECT_THROW(swizzleStageGeometry(desc, true), std::runtime_error);
}

#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR

// =============================================================================
//   Section 5 — TensileLite::hip::CopyTensorVoid
// =============================================================================
TEST(CopyTensorVoid, CopiesSource)
{
    TensorDescriptor   desc("t", rocisa::DataType::Float, {3, 2});
    std::vector<float> src(desc.totalAllocatedElements());
    std::iota(src.begin(), src.end(), 1.0f);
    std::vector<float> dst(src.size(), 0.0f);

    TensileLite::hip::CopyTensorVoid(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, src);
}

// =============================================================================
//   Section 8 — DataInitialization::initArrayTrig over a broadcast dimension
// =============================================================================
// A zero-stride batch dimension shares one matrix across batches, so the 3 x 4 x 2 tensor below
// has 12 allocated elements. Trig must fill them once, from batch 0 (sin of the logical index
// 0..11); otherwise batch 1 races to overwrite them with sin(12..23).
TEST(TrigInit, BroadcastDimensionHoldsFirstBatch)
{
    using TensileLite::Client::DataInitialization;
    TensorDescriptor desc("a", rocisa::DataType::Float, {3, 4, 2}, {1, 3, 0});
    ASSERT_EQ(desc.totalAllocatedElements(), 12u);
    for(int rep = 0; rep < 20; rep++)
    {
        std::vector<float> array(12, 42.0f);
        DataInitialization::initArrayTrig<float, false, false>(array.data(), desc);
        for(size_t k = 0; k < array.size(); k++)
            ASSERT_EQ(array[k], static_cast<float>(std::sin(double(k))))
                << "rep " << rep << ", element " << k;
    }
}

// =============================================================================
//   Section 9 — DataInitialization::getValue<T, InitMode::RandomNarrow>
// =============================================================================
// RandomNarrow assembles random sign/significand bits with an exponent in [-100, 0], so every
// value must be finite with magnitude below 2 and a biased exponent in [bias - 100, bias].
// BFloat16 used to pass the assembled bit pattern to BFloat16's value constructor, which treats
// it as a number (0x3f80 became 16256.0f instead of 1.0f).
namespace
{
    template <typename T>
    void expectRandomNarrowRange(int expBits, int sigBits, uint32_t (*bitsOf)(T))
    {
        using TensileLite::Client::DataInitialization;
        using TensileLite::Client::InitMode;
        const int bias = (1 << (expBits - 1)) - 1;
        for(int i = 0; i < 10000; i++)
        {
            T        v    = DataInitialization::getValue<T, InitMode::RandomNarrow>();
            float    f    = static_cast<float>(v);
            uint32_t bits = bitsOf(v);
            int      exp  = (bits >> sigBits) & ((1 << expBits) - 1);
            ASSERT_TRUE(std::isfinite(f)) << "bits=0x" << std::hex << bits;
            ASSERT_LT(std::fabs(f), 2.0f) << "bits=0x" << std::hex << bits;
            ASSERT_GE(exp, bias - 100) << "bits=0x" << std::hex << bits;
            ASSERT_LE(exp, bias) << "bits=0x" << std::hex << bits;
        }
    }
} // namespace

TEST(RandomNarrowInit, FloatStaysInNarrowRange)
{
    expectRandomNarrowRange<float>(8, 23, [](float v) {
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        return u;
    });
}

TEST(RandomNarrowInit, BFloat16StaysInNarrowRange)
{
    expectRandomNarrowRange<TensileLite::BFloat16>(
        8, 7, [](TensileLite::BFloat16 v) { return uint32_t(v.data); });
}
