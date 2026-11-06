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
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include <type_traits>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <hip/hip_runtime.h>
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
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
using TensileLite::Client::DataInitialization;
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
// The CPU reference is recomputed only when DataInitialization::inputEpoch() moves. The epoch is
// published by DataInitialization::preSolution(), so asking for it before that has run for the
// solution must be a loud error (a listener-ordering mistake would otherwise give a stale
// reference silently).
TEST(DataInitializationInputEpoch, PerSolutionEpochRequiresPreSolution)
{
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    auto args = makeMinimalClientArgs(/*mxScaleFormat=*/1);
    TensileLite::Client::ClientProblemFactory factory(args);
    TensileLite::Client::DataInitialization   dataInit(args, factory);

    ContractionSolution solution;
    solution.problemType.mxScaleFormat = 1;
    ContractionSolution otherSolution;

    EXPECT_THROW(dataInit.inputEpoch(&solution), std::logic_error);
    EXPECT_THROW(dataInit.inputEpoch(nullptr), std::logic_error);

    auto const problemEpoch = dataInit.inputEpoch();
    // No problem is registered, so preSolution() has nothing to refresh: the epoch must not move.
    dataInit.preSolution(&solution);
    EXPECT_EQ(dataInit.inputEpoch(&solution), problemEpoch);

    // A different solution that preSolution() has not seen is still rejected.
    EXPECT_THROW(dataInit.inputEpoch(&otherSolution), std::logic_error);
}
#endif

namespace
{
    class UploadCacheProbe : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;

        auto& pristine(size_t tensor, TensorDescriptor const& descriptor)
        {
            return m_vdata[tensor].pristine.at(descriptor.dataType());
        }
        size_t rotatingPoolCount() const
        {
            return m_rm->getRotatingMemory().size();
        }
        void emptyRotatingPools()
        {
            m_rm = std::make_shared<TensileLite::RotatingMemory>(0);
        }
        void invalidateInput(size_t tensor, TensorDescriptor const& descriptor)
        {
            invalidateGPUValid(pristine(tensor, descriptor));
        }
    };

    auto makeFloatUploadArgs()
    {
        auto args = makeMinimalClientArgs(0);
        for(auto const* name : {"type",
                                "a-type",
                                "b-type",
                                "c-type",
                                "d-type",
                                "compute-input-type-A",
                                "compute-input-type-B"})
            setOption(args, name, rocisa::DataType::Float);
        for(auto const* name : {"init-a", "init-b", "init-c"})
            setOption(args, name, TensileLite::Client::InitMode::SerialDim0);
        setOption(args,
                  "problem-size",
                  std::vector<std::vector<size_t>>{{16, 24, 2, 32}, {200, 136, 3, 128}});
        setOption(args, "verify-upload-skip", true);
        setOption(args, "init-beta", TensileLite::Client::InitMode::One);
        return args;
    }
}

// =============================================================================
//   Section 3b - prepareCPUInputs / prepareGPUInputs copy the current problem's extent
//
//   The copies only cover the current problem's tensors, not the run's largest one. With
//   problem-independent data (Random) every GPU call after the first takes the fast path and
//   copies no inputs, so the first copy must still fill the whole buffers: a later, larger
//   problem reads them as they are. With problem-dependent data (SerialDim0) every call copies;
//   its values depend on the coordinate, so a copy that misses part of a problem's extent
//   leaves an earlier problem's values there, which the independent expectation catches.
//   Sizes start below the run's maximum, shrink, then grow to it.
//   Each problem is then reset as before a second solution: the host data has not changed, so
//   the upload to the GPU's pristine copy is skipped, and the debug mode (verify-upload-skip)
//   reads that copy back and throws if it differs from the host data.
// =============================================================================
class DataInitializationCopyExtentTest
    : public ::testing::TestWithParam<TensileLite::Client::InitMode>
{
};

TEST_P(DataInitializationCopyExtentTest, GpuInputsMatchCpuInputsWhenProblemsShrinkAndGrow)
{
    using TensileLite::Client::InitMode;

    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount <= 0)
        GTEST_SKIP() << "No HIP device";

    auto args = makeMinimalClientArgs(/*mxScaleFormat=*/0);
    for(auto const* name : {"type",
                            "a-type",
                            "b-type",
                            "c-type",
                            "d-type",
                            "compute-input-type-A",
                            "compute-input-type-B"})
        setOption(args, name, rocisa::DataType::Float);
    setOption(args,
              "problem-size",
              std::vector<std::vector<size_t>>{
                  {96, 80, 1, 72}, {16, 24, 1, 8}, {200, 136, 2, 120}, {40, 40, 1, 40}});
    for(auto const* name : {"init-a", "init-b", "init-c"})
        setOption(args, name, GetParam());
    setOption(args, "verify-upload-skip", true);

    TensileLite::Client::ClientProblemFactory factory(args);
    TensileLite::Client::DataInitialization   dataInit(args, factory);
    ASSERT_EQ(factory.problems().size(), 4u);

    for(size_t k = 0; k < factory.problems().size(); k++)
    {
        auto const& problem = dynamic_cast<ContractionProblemGemm const&>(*factory.problems()[k]);
        // The client prepares the reference (CPU) inputs before the GPU ones.
        auto cpu = std::dynamic_pointer_cast<TensileLite::ContractionInputs>(
            dataInit.prepareCPUInputs(&problem));
        for(char const* call : {"problem", "solution reset"})
        {
            std::shared_ptr<TensileLite::ContractionInputs> gpu;
            ASSERT_NO_THROW(gpu = std::dynamic_pointer_cast<TensileLite::ContractionInputs>(
                                dataInit.prepareGPUInputs(&problem)))
                << call << " of problem " << k;
            ASSERT_TRUE(cpu && gpu);

            std::pair<void const*, void const*> const inputs[]
                = {{cpu->a, gpu->a}, {cpu->b, gpu->b}, {cpu->c, gpu->c}};
            for(size_t i = 0; i < std::size(inputs); i++)
            {
                auto const&          desc  = problem.tensors()[i];
                size_t               bytes = desc.totalAllocatedBytes();
                std::vector<uint8_t> fromGpu(bytes);
                ASSERT_EQ(hipMemcpy(fromGpu.data(), inputs[i].second, bytes, hipMemcpyDeviceToHost),
                          hipSuccess);
                EXPECT_EQ(std::memcmp(fromGpu.data(), inputs[i].first, bytes), 0)
                    << "tensor " << desc.getName() << " after the " << call << " of problem " << k;
                if(GetParam() == InitMode::SerialDim0)
                {
                    // Packed tensors: every element of the extent is written.
                    std::vector<float> expected(desc.totalAllocatedElements());
                    DataInitialization::initArraySerialDim<float>(expected.data(), 0, desc);
                    EXPECT_EQ(std::memcmp(expected.data(), inputs[i].first, bytes), 0)
                        << "CPU tensor " << desc.getName() << " of problem " << k;
                    EXPECT_EQ(std::memcmp(expected.data(), fromGpu.data(), bytes), 0)
                        << "GPU tensor " << desc.getName() << " after the " << call
                        << " of problem " << k;
                }
            }
        }
    }

    // With problem-dependent data (SerialDim0) every reset takes the slow path, whose uploads
    // are skipped and verified; Random takes the fast path, which uploads nothing.
    testing::internal::CaptureStdout();
    dataInit.finalizeReport();
    std::string const report = testing::internal::GetCapturedStdout();
    if(GetParam() == InitMode::SerialDim0)
        EXPECT_EQ(report.find("Verified 0 "), std::string::npos) << report;
    EXPECT_NE(report.find("skipped GPU input uploads against the host data"), std::string::npos)
        << report;
}

INSTANTIATE_TEST_SUITE_P(InitModes,
                         DataInitializationCopyExtentTest,
                         ::testing::Values(TensileLite::Client::InitMode::Random,
                                           TensileLite::Client::InitMode::SerialDim0),
                         [](::testing::TestParamInfo<TensileLite::Client::InitMode> const& info) {
                             return TensileLite::Client::ToString(info.param);
                         });

// =============================================================================
//   Section 3c - batched-pointer arrays follow the bounds-check padding
//
//   prepareGPUInputs uploads, per tensor, an array with the start pointer of each batch
//   (inputs->batchA..D), and skips the upload when the array was built from the same start
//   pointer and batch geometry. The start pointer includes the bounds-check padding, which
//   depends on the problem and, with GuardPageAll, flips between the two runs of one solution
//   (GuardPageFront: no padding, GuardPageBack: the tensor ends at the guard page) without a new
//   problem, input epoch or buffer. Each problem is run like the client runs two solutions
//   (GuardPageAll: Front, Back, Front, Back). After every call the arrays must hold the pointers
//   of that call's tensors; the debug mode (verify-upload-skip) reads back every skipped array
//   and throws if it is stale. NaN mode keeps its padding for the whole problem, so its later
//   runs skip the upload. GuardPageFront has no padding, so the start pointer stays the same
//   across problems and only the batch geometry tells them apart: the second problem has other
//   batch strides, the last one only another batch count.
// =============================================================================
class DataInitializationBatchedPointersTest
    : public ::testing::TestWithParam<TensileLite::Client::BoundsCheckMode>
{
};

TEST_P(DataInitializationBatchedPointersTest, FollowBoundsCheckRunsAndProblems)
{
    using TensileLite::Client::BoundsCheckMode;

    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount <= 0)
        GTEST_SKIP() << "No HIP device";

    auto args = makeMinimalClientArgs(/*mxScaleFormat=*/0);
    for(auto const* name : {"type",
                            "a-type",
                            "b-type",
                            "c-type",
                            "d-type",
                            "compute-input-type-A",
                            "compute-input-type-B"})
        setOption(args, name, rocisa::DataType::Float);
    // Sizes (i, j, batch, l): the second problem has other batch strides and paddings, the last
    // one the third's strides with more batches.
    setOption(args,
              "problem-size",
              std::vector<std::vector<size_t>>{
                  {64, 48, 3, 40}, {32, 24, 5, 16}, {64, 48, 3, 40}, {64, 48, 5, 40}});
    setOption(args, "bounds-check", GetParam());
    setOption(args, "verify-upload-skip", true);

    TensileLite::Client::ClientProblemFactory factory(args);
    TensileLite::Client::DataInitialization   dataInit(args, factory);
    ASSERT_EQ(factory.problems().size(), 4u);

    TensileLite::Client::TimingEvents const noEvents({}, 0, 0);
    for(size_t k = 0; k < factory.problems().size(); k++)
    {
        auto const& problem = dynamic_cast<ContractionProblemGemm const&>(*factory.problems()[k]);
        ASSERT_EQ(problem.batchIndices().size(), 1u);
        auto const   batchIndex = problem.batchIndices()[0];
        size_t const batchCount = problem.batchSize(0);
        dataInit.prepareCPUInputs(&problem);

        void const* frontA = nullptr;
        for(size_t run = 0; run < 4; run++)
        {
            bool const back = GetParam() == BoundsCheckMode::GuardPageAll && run % 2 == 1;
            std::shared_ptr<TensileLite::ContractionInputs> gpu;
            ASSERT_NO_THROW(gpu = std::dynamic_pointer_cast<TensileLite::ContractionInputs>(
                                dataInit.prepareGPUInputs(&problem)))
                << "run " << run << " of problem " << k;
            ASSERT_TRUE(gpu);
            if(!back)
                frontA = frontA ? frontA : gpu->a;
            // GuardPageAll's two runs use different start pointers, the same in both solutions.
            EXPECT_EQ(gpu->a == frontA, !back) << "run " << run << " of problem " << k;

            struct
            {
                void const*        start;
                void const* const* batch;
                size_t             index;
            } const tensors[] = {{gpu->a, gpu->batchA, batchIndex.a},
                                 {gpu->b, gpu->batchB, batchIndex.b},
                                 {gpu->c, gpu->batchC, batchIndex.c},
                                 {gpu->d, gpu->batchD, batchIndex.d}};
            for(size_t i = 0; i < std::size(tensors); i++)
            {
                auto const&        desc = problem.tensors()[i];
                std::vector<void*> onGpu(batchCount);
                ASSERT_EQ(hipMemcpy(onGpu.data(),
                                    tensors[i].batch,
                                    batchCount * sizeof(void*),
                                    hipMemcpyDeviceToHost),
                          hipSuccess);
                for(size_t b = 0; b < batchCount; b++)
                {
                    auto const* expected = static_cast<uint8_t const*>(tensors[i].start)
                                           + b * desc.strides()[tensors[i].index] * sizeof(float);
                    EXPECT_EQ(onGpu[b], expected) << "tensor " << desc.getName() << " batch " << b
                                                  << ", run " << run << " of problem " << k;
                }
            }

            // End of a run, then of a solution after its second run.
            dataInit.validateWarmups({0}, noEvents, noEvents);
            if(run % 2 == 1)
                dataInit.postSolution();
        }
    }

    testing::internal::CaptureStdout();
    dataInit.finalizeReport();
    std::string const report = testing::internal::GetCapturedStdout();
    if(GetParam() != BoundsCheckMode::GuardPageAll)
        EXPECT_EQ(report.find("Verified 0 skipped GPU batched-pointer uploads"), std::string::npos)
            << report;
    EXPECT_NE(report.find("skipped GPU batched-pointer uploads"), std::string::npos) << report;
}

INSTANTIATE_TEST_SUITE_P(
    BoundsCheckModes,
    DataInitializationBatchedPointersTest,
    ::testing::Values(TensileLite::Client::BoundsCheckMode::NaN,
                      TensileLite::Client::BoundsCheckMode::GuardPageFront,
                      TensileLite::Client::BoundsCheckMode::GuardPageAll),
    [](::testing::TestParamInfo<TensileLite::Client::BoundsCheckMode> const& info) {
        switch(info.param)
        {
        case TensileLite::Client::BoundsCheckMode::NaN:
            return "NaN";
        case TensileLite::Client::BoundsCheckMode::GuardPageFront:
            return "GuardPageFront";
        default:
            return "GuardPageAll";
        }
    });

// The unused destination tail distinguishes extent-limited copies from whole-buffer copies.
// Sparse inputs exercise compressed data/metadata, and both rotating modes use the same uploads.
struct CopyBoundaryParam
{
    TensileLite::Client::BoundsCheckMode bounds;
    int                                  sparse;
    int                                  rotatingMode; // -1 disables rotation
};
class DataInitializationCopyBoundaryTest : public ::testing::TestWithParam<CopyBoundaryParam>
{
};

TEST_P(DataInitializationCopyBoundaryTest, LeavesUnusedCurrentBufferTailsUntouched)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "No HIP device";
    auto args = makeFloatUploadArgs();
    setOption(args, "bounds-check", GetParam().bounds);
    setOption(args, "sparse", GetParam().sparse);
    if(GetParam().sparse)
        for(auto const* name : {"a-type", "b-type", "compute-input-type-A", "compute-input-type-B"})
            setOption(args, name, rocisa::DataType::Half);

    if(GetParam().rotatingMode >= 0)
    {
        setOption(args, "rotating-buffer-size", 2);
        setOption(args, "rotating-buffer-mode", GetParam().rotatingMode);
    }
    TensileLite::Client::ClientProblemFactory factory(args);
    UploadCacheProbe                          init(args, factory);
    auto const& problem = dynamic_cast<ContractionProblemGemm const&>(*factory.problems()[0]);
    using Tensor        = ContractionProblemGemm::TENSOR;
    std::vector<size_t> tensors{Tensor::A, Tensor::B, Tensor::C};
    if(GetParam().sparse)
    {
        tensors.push_back(Tensor::COMPRESSED);
        tensors.push_back(Tensor::METADATA);
    }
    constexpr uint8_t sentinel = 0xa5;
    for(auto i : tensors)
    {
        auto const& desc  = problem.tensors()[i];
        auto&       p     = init.pristine(i, desc);
        size_t      bytes = TensileLite::multiplyElementSize(p.maxElements, desc.elementBytes());
        std::memset(p.cpuInput.current.get(), sentinel, bytes);
        ASSERT_EQ(hipMemset(p.gpuInput.current.get(), sentinel, bytes), hipSuccess);
    }
    init.prepareCPUInputs(problem);
    auto gpuInputs = init.prepareGPUInputs(problem);
    if(GetParam().rotatingMode >= 0)
    {
        auto rotated = init.prepareRotatingGPUOutput(3, &problem, gpuInputs, nullptr);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        size_t const expected
            = GetParam().rotatingMode == 1 ? std::min(size_t{3}, init.rotatingPoolCount()) : 3;
        ASSERT_GE(expected, 2u);
        ASSERT_EQ(rotated.size(), expected);
        for(size_t r = 1; r < rotated.size(); ++r)
        {
            auto const* inputs
                = dynamic_cast<TensileLite::ContractionInputs const*>(rotated[r].get());
            ASSERT_NE(inputs, nullptr);
            void const* sources[] = {inputs->a, inputs->b, inputs->c};
            for(size_t i = 0; i < std::size(sources); ++i)
            {
                auto const& desc = problem.tensors()[i];
                auto&       p    = init.pristine(i, desc);
                EXPECT_NE(sources[i], p.gpuInput.current.get());
                std::vector<uint8_t> bytes(desc.totalAllocatedBytes());
                ASSERT_EQ(hipMemcpy(bytes.data(), sources[i], bytes.size(), hipMemcpyDeviceToHost),
                          hipSuccess);
                EXPECT_EQ(std::memcmp(bytes.data(), p.cpuInput.valid.get(), bytes.size()), 0)
                    << "rotating copy " << r << " tensor " << desc.getName();
            }
        }
    }
    for(auto i : tensors)
    {
        SCOPED_TRACE(problem.tensors()[i].getName());
        auto const& desc   = problem.tensors()[i];
        auto&       p      = init.pristine(i, desc);
        size_t      bytes  = TensileLite::multiplyElementSize(p.maxElements, desc.elementBytes());
        size_t      extent = desc.totalAllocatedBytes();
        ASSERT_LT(extent, bytes);
        auto const* cpu = static_cast<uint8_t const*>(p.cpuInput.current.get());
        EXPECT_EQ(std::memcmp(cpu, p.cpuInput.valid.get(), extent), 0);
        EXPECT_TRUE(
            std::all_of(cpu + extent, cpu + bytes, [](uint8_t v) { return v == sentinel; }));
        std::vector<uint8_t> gpu(bytes);
        ASSERT_EQ(hipMemcpy(gpu.data(), p.gpuInput.current.get(), bytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(std::memcmp(gpu.data(), p.cpuInput.valid.get(), extent), 0);
        EXPECT_TRUE(
            std::all_of(gpu.begin() + extent, gpu.end(), [](uint8_t v) { return v == sentinel; }));
    }
}

INSTANTIATE_TEST_SUITE_P(
    Layouts,
    DataInitializationCopyBoundaryTest,
    ::testing::Values(
        CopyBoundaryParam{TensileLite::Client::BoundsCheckMode::Disable, 0, -1},
        CopyBoundaryParam{TensileLite::Client::BoundsCheckMode::GuardPageFront, 0, -1},
        CopyBoundaryParam{TensileLite::Client::BoundsCheckMode::Disable, 1, -1},
        CopyBoundaryParam{TensileLite::Client::BoundsCheckMode::GuardPageFront, 1, -1},
        CopyBoundaryParam{TensileLite::Client::BoundsCheckMode::Disable, 2, -1},
        CopyBoundaryParam{TensileLite::Client::BoundsCheckMode::GuardPageFront, 2, -1},
        CopyBoundaryParam{TensileLite::Client::BoundsCheckMode::Disable, 0, 0},
        CopyBoundaryParam{TensileLite::Client::BoundsCheckMode::Disable, 0, 1}),
    [](::testing::TestParamInfo<CopyBoundaryParam> const& info) {
        return std::string(info.param.bounds == TensileLite::Client::BoundsCheckMode::Disable
                               ? "Disable"
                               : "GuardFront")
               + "Sparse" + std::to_string(info.param.sparse)
               + (info.param.rotatingMode < 0
                      ? "NoRotation"
                      : "Rotating" + std::to_string(info.param.rotatingMode));
    });

TEST(DataInitializationRotatingTest, EmptyModeOnePoolsKeepOnlyOriginalInputs)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "No HIP device";
    auto args = makeFloatUploadArgs();
    setOption(args, "rotating-buffer-size", 2);
    setOption(args, "rotating-buffer-mode", 1);
    TensileLite::Client::ClientProblemFactory factory(args);
    UploadCacheProbe                          init(args, factory);
    auto const& problem = dynamic_cast<ContractionProblemGemm const&>(*factory.problems()[0]);
    init.prepareCPUInputs(problem);
    auto inputs = init.prepareGPUInputs(problem);
    // Normal allocation creates at least one pool. Exercise the defensive empty-pool
    // path while the initializer retains the original GPU storage.
    init.emptyRotatingPools();
    ASSERT_EQ(init.rotatingPoolCount(), 0u);
    auto rotated = init.prepareRotatingGPUOutput(3, &problem, inputs, nullptr);
    ASSERT_EQ(rotated.size(), 1u);
    EXPECT_EQ(rotated.front(), inputs);
}

class DataInitializationUploadVerificationTest : public ::testing::TestWithParam<bool>
{
};

TEST_P(DataInitializationUploadVerificationTest, RejectsCorruptionAndRepairsAfterInvalidation)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "No HIP device";
    auto                                      args = makeFloatUploadArgs();
    TensileLite::Client::ClientProblemFactory factory(args);
    UploadCacheProbe                          init(args, factory);
    auto const& problem = dynamic_cast<ContractionProblemGemm const&>(*factory.problems()[0]);
    init.prepareCPUInputs(problem);
    init.prepareGPUInputs(problem);
    // Establish a successful verified cache hit before corrupting its contents.
    ASSERT_NO_THROW(init.prepareGPUInputs(problem));
    auto&      p     = init.pristine(ContractionProblemGemm::TENSOR::A, problem.a());
    auto const epoch = init.inputEpoch();
    if(GetParam())
    {
        void* corrupt = nullptr;
        ASSERT_EQ(
            hipMemcpy(p.gpuInput.batch.get(), &corrupt, sizeof(corrupt), hipMemcpyHostToDevice),
            hipSuccess);
    }
    else
    {
        uint8_t corrupt = *static_cast<uint8_t*>(p.cpuInput.valid.get()) ^ 0xff;
        ASSERT_EQ(
            hipMemcpy(p.gpuInput.valid.get(), &corrupt, sizeof(corrupt), hipMemcpyHostToDevice),
            hipSuccess);
    }
    try
    {
        init.prepareGPUInputs(problem);
        FAIL() << "verification accepted corrupted cached contents";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what())
                      .find(GetParam() ? "Stale GPU batched pointers" : "Stale GPU input"),
                  std::string::npos);
    }
    EXPECT_EQ(init.inputEpoch(), epoch);
    if(GetParam())
        p.gpuBatchKey.reset();
    else
        init.invalidateInput(ContractionProblemGemm::TENSOR::A, problem.a());
    ASSERT_NO_THROW(init.prepareGPUInputs(problem)); // repair
    ASSERT_NO_THROW(init.prepareGPUInputs(problem)); // verify repaired bytes
}

INSTANTIATE_TEST_SUITE_P(Buffers,
                         DataInitializationUploadVerificationTest,
                         ::testing::Values(false, true),
                         [](::testing::TestParamInfo<bool> const& info) {
                             return info.param ? "BatchPointers" : "PristineInput";
                         });

TEST(DataInitializationCopyExtent, GroupedGemmsCopyTheWholePristineBuffer)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "No HIP device";
    auto args = makeFloatUploadArgs();
    setOption(args, "grouped-gemm", true);
    TensileLite::Client::ClientProblemFactory factory(args);
    UploadCacheProbe                          init(args, factory);
    ASSERT_EQ(factory.problems().size(), 1u);
    auto const& problem
        = dynamic_cast<TensileLite::ContractionProblemGroupedGemm const&>(*factory.problems()[0]);
    ASSERT_EQ(problem.gemms.size(), 2u);
    init.prepareCPUInputs(problem);
    init.prepareGPUInputs(problem);
    ASSERT_NO_THROW(init.prepareGPUInputs(problem));
    for(size_t i = 0; i < 3; ++i)
    {
        auto const& desc = problem.gemms[0].tensors()[i];
        auto&       p    = init.pristine(i, desc);
        ASSERT_EQ(p.groupedGemmOffsets.size(), 2u);
        size_t bytes = TensileLite::multiplyElementSize(p.maxElements, desc.elementBytes());
        ASSERT_GT(bytes, desc.totalAllocatedBytes());
        EXPECT_EQ(std::memcmp(p.cpuInput.current.get(), p.cpuInput.valid.get(), bytes), 0);
        std::vector<uint8_t> gpu(bytes);
        ASSERT_EQ(hipMemcpy(gpu.data(), p.gpuInput.current.get(), bytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(std::memcmp(gpu.data(), p.cpuInput.valid.get(), bytes), 0);
    }
}

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
namespace
{
    // Exposes the state DataInitialization keeps to reuse MX inputs across solutions.
    class MXReuseProbe : public UploadCacheProbe
    {
    public:
        using UploadCacheProbe::UploadCacheProbe;

        bool preswizzledA() const
        {
            return m_mxPreswizzledA;
        }
        TensorDescriptor const& initDescriptor(int tensor) const
        {
            return m_vdata[tensor].pristine.begin()->second.initDescriptor[0];
        }
        void useGfx950ScaleLayout()
        {
            // Exercise the host layout code on any HIP device; no gfx950 kernel is launched.
            m_mxScaleLayout = MXScaleLayout::GFX950;
        }
        std::vector<uint8_t> canonicalBytes(int tensor, TensorDescriptor const& descriptor) const
        {
            auto const& pristine = m_vdata[tensor].pristine.at(descriptor.dataType());
            auto const* bytes    = static_cast<uint8_t const*>(pristine.cpuInput.valid.get());
            return {bytes, bytes + descriptor.totalAllocatedBytes()};
        }
        void invalidateHostData(int tensor, TensorDescriptor const& descriptor)
        {
            auto& data = m_vdata[tensor];
            data.init  = TensileLite::Client::InitMode::One;
            data.pristine.at(descriptor.dataType()).initDescriptor[0] = TensorDescriptor{};
        }
        void clearSwizzledScaleCache()
        {
            m_mxSwizzledDescriptor.clear();
            // gfx950 keeps its derived scale and upload in separate caches.
            for(int tensor :
                {ContractionProblemGemm::TENSOR::MXSA, ContractionProblemGemm::TENSOR::MXSB})
                for(auto& [type, pristine] : m_vdata[tensor].pristine)
                {
                    pristine.mxSwizzledScale.clear();
                    pristine.gpuMXSwizzledCurrent = false;
                }
        }
        void initializeNonMXB(ContractionProblemGemm const& problem)
        {
            initNonMXSideWithMXGenerator(ContractionProblemGemm::TENSOR::B,
                                         ContractionProblemGemm::TENSOR::MXSB,
                                         problem.b(),
                                         problem.transB(),
                                         false);
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

class DataInitializationCanonicalEpoch : public ::testing::TestWithParam<int>
{
};

TEST_P(DataInitializationCanonicalEpoch, HostRewriteAdvancesButSwizzleAndCacheReuseDoNot)
{
    using Tensor = ContractionProblemGemm::TENSOR;
    auto args    = makeMXLaunchArgs({{128, 128, 1, 512}});
    setOption(args, "mx-data-generator", GetParam());
    TensileLite::Client::ClientProblemFactory factory(args);
    MXReuseProbe                              dataInit(args, factory);
    dataInit.useGfx950ScaleLayout();
    auto*       problem = factory.problems()[0].get();
    auto const& gemm    = dynamic_cast<ContractionProblemGemm const&>(*problem);
    dataInit.preProblem(problem);
    auto const beforeProblem = dataInit.inputEpoch();
    dataInit.prepareCPUInputs(problem);
    EXPECT_GT(dataInit.inputEpoch(), beforeProblem);
    dataInit.prepareGPUInputs(problem); // allocate the GPU buffers required by the swizzle gate
    auto const epoch     = dataInit.inputEpoch();
    auto const hostA     = dataInit.canonicalBytes(Tensor::A, gemm.a());
    auto const hostScale = dataInit.canonicalBytes(Tensor::MXSA, gemm.mxsa());

    ContractionSolution first;
    setMXSolution(first, 1, 128);
    dataInit.preSolution(&first); // first swizzled upload, with canonical inputs already ready
    EXPECT_TRUE(dataInit.preswizzledA());
    EXPECT_EQ(dataInit.inputEpoch(&first), epoch);
    EXPECT_EQ(dataInit.canonicalBytes(Tensor::A, gemm.a()), hostA);
    EXPECT_EQ(dataInit.canonicalBytes(Tensor::MXSA, gemm.mxsa()), hostScale);

    ContractionSolution cached;
    setMXSolution(cached, 1, 64);
    dataInit.preSolution(&cached); // different MI-K, matching canonical and swizzled cache
    EXPECT_EQ(dataInit.inputEpoch(&cached), epoch);

    dataInit.clearSwizzledScaleCache();
    ContractionSolution rebuild;
    setMXSolution(rebuild, 1, 128);
    dataInit.preSolution(&rebuild);
    EXPECT_TRUE(dataInit.preswizzledA());
    EXPECT_EQ(dataInit.inputEpoch(&rebuild), epoch);
    EXPECT_EQ(dataInit.canonicalBytes(Tensor::A, gemm.a()), hostA);
    EXPECT_EQ(dataInit.canonicalBytes(Tensor::MXSA, gemm.mxsa()), hostScale);

    dataInit.invalidateHostData(Tensor::A, gemm.a());
    dataInit.preSolution(&cached); // actual canonical generation, with a different init mode
    EXPECT_GT(dataInit.inputEpoch(&cached), epoch);
    EXPECT_NE(dataInit.canonicalBytes(Tensor::A, gemm.a()), hostA);
}

INSTANTIATE_TEST_SUITE_P(GeneratorModes, DataInitializationCanonicalEpoch, ::testing::Values(0, 1));

TEST(DataInitializationInputEpoch, NonMXGeneratorWriterAdvancesOnlyOnDescriptorMiss)
{
    using Tensor = ContractionProblemGemm::TENSOR;
    auto args    = makeMXLaunchArgs({{128, 128, 1, 512}});
    setOption(args, "mx-b-block", 0);
    setOption(args, "mx-data-generator", 1);
    TensileLite::Client::ClientProblemFactory factory(args);
    MXReuseProbe                              dataInit(args, factory);
    auto const& problem = dynamic_cast<ContractionProblemGemm const&>(*factory.problems()[0]);
    dataInit.prepareCPUInputs(problem);
    auto const before = dataInit.inputEpoch();
    auto const hostB  = dataInit.canonicalBytes(Tensor::B, problem.b());
    dataInit.invalidateHostData(Tensor::B, problem.b());
    dataInit.initializeNonMXB(problem);
    EXPECT_GT(dataInit.inputEpoch(), before);
    EXPECT_NE(dataInit.canonicalBytes(Tensor::B, problem.b()), hostB);
    auto const after = dataInit.inputEpoch();
    dataInit.initializeNonMXB(problem);
    EXPECT_EQ(dataInit.inputEpoch(), after);
}

class DataInitializationMXUploadTest : public ::testing::TestWithParam<int>
{
};

TEST_P(DataInitializationMXUploadTest, CanonicalAndSwizzledTransitionsReplaceWholeScaleBuffers)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "No HIP device";
    auto args = makeMXLaunchArgs({{128, 128, 1, 512}, {256, 256, 1, 512}});
    setOption(args, "mx-data-generator", GetParam());
    setOption(args, "verify-upload-skip", true);
    TensileLite::Client::ClientProblemFactory factory(args);
    MXReuseProbe                              init(args, factory);
    init.useGfx950ScaleLayout(); // host transform only; no architecture-specific kernel is launched
    auto*       problem = factory.problems()[0].get();
    auto const& gemm    = dynamic_cast<ContractionProblemGemm const&>(*problem);
    using Tensor        = ContractionProblemGemm::TENSOR;
    ContractionSolution swizzle, canonical;
    setMXSolution(swizzle, 1, 128);
    setMXSolution(canonical, 0, 128);
    init.preProblem(problem);
    init.prepareCPUInputs(problem);
    init.prepareGPUInputs(problem); // prepare buffers before selecting a solution

    auto const epoch = init.inputEpoch();
    auto       run   = [&](ContractionSolution& solution) {
        for(auto tensor : {Tensor::MXSA, Tensor::MXSB})
        {
            auto const& desc = gemm.tensors()[tensor];
            auto&       p    = init.pristine(tensor, desc);
            size_t bytes = TensileLite::multiplyElementSize(p.maxElements, desc.elementBytes());
            HIP_CHECK_EXC(hipMemset(p.gpuInput.current.get(), 0xa5, bytes));
        }
        init.preSolution(&solution);
        init.prepareGPUInputs(problem);
        init.postSolution();
        std::vector<std::vector<uint8_t>> result;
        for(auto tensor : {Tensor::MXSA, Tensor::MXSB})
        {
            auto const& desc = gemm.tensors()[tensor];
            auto&       p    = init.pristine(tensor, desc);
            size_t bytes = TensileLite::multiplyElementSize(p.maxElements, desc.elementBytes());
            EXPECT_GT(bytes, desc.totalAllocatedBytes());
            std::vector<uint8_t> gpu(bytes);
            HIP_CHECK_EXC(
                hipMemcpy(gpu.data(), p.gpuInput.current.get(), bytes, hipMemcpyDeviceToHost));
            std::vector<uint8_t> pristine(bytes);
            void* selected = solution.problemType.mxScaleFormat == 0 ? p.gpuInput.valid.get()
                                                                     : p.gpuMXSwizzled.get();
            EXPECT_NE(selected, nullptr);
            HIP_CHECK_EXC(hipMemcpy(pristine.data(), selected, bytes, hipMemcpyDeviceToHost));
            EXPECT_EQ(gpu, pristine); // MX copies preserve the whole currently selected layout
            std::vector<uint8_t> canonicalGPU(bytes);
            HIP_CHECK_EXC(hipMemcpy(
                canonicalGPU.data(), p.gpuInput.valid.get(), bytes, hipMemcpyDeviceToHost));
            EXPECT_EQ(std::memcmp(canonicalGPU.data(), p.cpuInput.valid.get(), bytes), 0);
            EXPECT_EQ(p.gpuValidEpoch, epoch);
            EXPECT_EQ(p.gpuValidElements, p.maxElements);
            if(solution.problemType.mxScaleFormat == 0)
            {
                EXPECT_EQ(std::memcmp(gpu.data(), p.cpuInput.valid.get(), bytes), 0);
                EXPECT_EQ(std::memcmp(p.cpuInput.current.get(), p.cpuInput.valid.get(), bytes), 0);
            }
            else
                EXPECT_TRUE(p.gpuMXSwizzledCurrent);
            // Full-buffer equality above covers the unused tail. Compare only the current
            // descriptor's active layout when switching between the two formats.
            gpu.resize(desc.totalAllocatedBytes());
            result.push_back(std::move(gpu));
        }
        EXPECT_EQ(init.inputEpoch(), epoch);
        return result;
    };
    // Establish the canonical upload before selecting the separate swizzled source.
    auto const unpermuted   = run(canonical);
    auto       uploadCounts = [&]() {
        std::vector<std::pair<size_t, size_t>> counts;
        for(auto tensor : {Tensor::MXSA, Tensor::MXSB})
        {
            auto const& p = init.pristine(tensor, gemm.tensors()[tensor]);
            counts.emplace_back(p.gpuValidUploadCount, p.gpuMXSwizzledUploadCount);
        }
        return counts;
    };
    auto const canonicalCounts = uploadCounts();
    auto const swizzled = run(swizzle);
    auto const bothCounts      = uploadCounts();
    for(size_t i = 0; i < bothCounts.size(); ++i)
    {
        EXPECT_EQ(bothCounts[i].first, canonicalCounts[i].first);
        EXPECT_EQ(bothCounts[i].second, canonicalCounts[i].second + 1);
    }
    ASSERT_NE(unpermuted, swizzled) << "scale layouts must differ for the cache test";
    EXPECT_EQ(run(swizzle), swizzled); // same instruction K skips initialization
    EXPECT_EQ(uploadCounts(), bothCounts);
    EXPECT_EQ(run(canonical), unpermuted);
    EXPECT_EQ(uploadCounts(), bothCounts);
    EXPECT_EQ(run(swizzle), swizzled);
    EXPECT_EQ(uploadCounts(), bothCounts);
    EXPECT_EQ(run(canonical), unpermuted);
    EXPECT_EQ(uploadCounts(), bothCounts);
    init.postProblem();
}

TEST_P(DataInitializationMXUploadTest, SameInstructionKRejectsAndRepairsCorruptedSwizzledScale)
{
    auto args = makeMXLaunchArgs({{128, 128, 1, 512}});
    setOption(args, "mx-data-generator", GetParam());
    setOption(args, "verify-upload-skip", true);
    TensileLite::Client::ClientProblemFactory factory(args);
    MXReuseProbe                              init(args, factory);
    init.useGfx950ScaleLayout();
    auto*               problem = factory.problems()[0].get();
    auto const&         gemm    = dynamic_cast<ContractionProblemGemm const&>(*problem);
    ContractionSolution first, next;
    setMXSolution(first, 1, 128);
    setMXSolution(next, 1, 128);
    init.preProblem(problem);
    init.prepareCPUInputs(problem);
    init.prepareGPUInputs(problem);
    init.preSolution(&first);
    init.prepareGPUInputs(problem);
    init.postSolution();
    auto const epoch = init.inputEpoch();
    using Tensor     = ContractionProblemGemm::TENSOR;
    for(auto tensor : {Tensor::MXSA, Tensor::MXSB})
    {
        auto const& desc = gemm.tensors()[tensor];
        auto&       p    = init.pristine(tensor, desc);
        ASSERT_TRUE(p.gpuMXSwizzledCurrent);
        auto const uploads  = p.gpuMXSwizzledUploadCount;
        auto const expected = p.mxSwizzledScale;
        uint8_t    corrupt  = expected[0] ^ 0xff;
        HIP_CHECK_EXC(hipMemcpy(p.gpuMXSwizzled.get(), &corrupt, 1, hipMemcpyHostToDevice));
        init.preSolution(&next); // same K: initializeMXData is skipped
        try
        {
            init.prepareGPUInputs(problem);
            FAIL() << "corrupted swizzled upload was reused";
        }
        catch(std::runtime_error const& error)
        {
            EXPECT_NE(std::string(error.what()).find(desc.getName()), std::string::npos);
            EXPECT_NE(std::string(error.what()).find("GFX950-swizzled scale"), std::string::npos);
        }
        EXPECT_FALSE(p.gpuMXSwizzledCurrent);
        EXPECT_EQ(p.gpuMXSwizzledUploadCount, uploads);
        init.postSolution();
        init.preSolution(&next);
        init.prepareGPUInputs(problem); // invalid tag requires repair, even with the same K
        init.postSolution();
        EXPECT_TRUE(p.gpuMXSwizzledCurrent);
        EXPECT_EQ(p.gpuMXSwizzledUploadCount, uploads + 1);
        std::vector<uint8_t> gpu(expected.size());
        HIP_CHECK_EXC(
            hipMemcpy(gpu.data(), p.gpuMXSwizzled.get(), gpu.size(), hipMemcpyDeviceToHost));
        EXPECT_EQ(gpu, expected);
        init.preSolution(&next);
        EXPECT_NO_THROW(init.prepareGPUInputs(problem)); // repaired cache can be verified again
        init.postSolution();
        EXPECT_EQ(p.gpuMXSwizzledUploadCount, uploads + 1);
        EXPECT_EQ(init.inputEpoch(), epoch);
    }
}

TEST_P(DataInitializationMXUploadTest, PaddedLeadingDimensionStaysWithinCanonicalAllocations)
{
    using Tensor = ContractionProblemGemm::TENSOR;
    // Both a single last batch and earlier batches need bounded writes. The generator
    // emits 2,560 scales per batch, while each canonical slot contains only 2,048.
    for(size_t batches : {1u, 3u})
    {
        SCOPED_TRACE(batches);
        auto args = makeMXLaunchArgs({{256, 256, batches, 256}});
        setOption(args, "a-strides", std::vector<std::vector<size_t>>{{1, 320, 81920}});
        setOption(args, "b-strides", std::vector<std::vector<size_t>>{{1, 320, 81920}});
        setOption(args, "mx-data-generator", GetParam());
        setOption(args, "verify-upload-skip", true);
        TensileLite::Client::ClientProblemFactory factory(args);
        MXReuseProbe                              init(args, factory);
        init.useGfx950ScaleLayout();
        auto*            problem    = factory.problems()[0].get();
        auto const&      gemm       = dynamic_cast<ContractionProblemGemm const&>(*problem);
        constexpr size_t guardBytes = 64;
        for(auto tensor : {Tensor::A, Tensor::B, Tensor::MXSA, Tensor::MXSB})
        {
            auto const& desc = gemm.tensors()[tensor];
            auto&       p    = init.pristine(tensor, desc);
            size_t bytes     = TensileLite::multiplyElementSize(p.maxElements, desc.elementBytes());
            p.cpuInput.valid = std::shared_ptr<void>(malloc(bytes + guardBytes), free);
            ASSERT_NE(p.cpuInput.valid, nullptr);
            std::memset(p.cpuInput.valid.get(), 0xcc, bytes + guardBytes);
        }
        init.preProblem(problem);
        init.prepareCPUInputs(problem);
        for(auto tensor : {Tensor::A, Tensor::B, Tensor::MXSA, Tensor::MXSB})
        {
            auto const& desc = gemm.tensors()[tensor];
            auto&       p    = init.pristine(tensor, desc);
            size_t bytes     = TensileLite::multiplyElementSize(p.maxElements, desc.elementBytes());
            auto*  end       = static_cast<uint8_t*>(p.cpuInput.valid.get()) + bytes;
            EXPECT_EQ(std::vector<uint8_t>(end, end + guardBytes),
                      std::vector<uint8_t>(guardBytes, 0xcc))
                << desc.getName();
        }
        if(GetParam() == 1)
        {
            // The bounded copy must preserve the generated bytes, not merely avoid
            // crossing the allocation boundary. Check all active batches on both sides.
            for(auto tensor : {Tensor::A, Tensor::B})
            {
                bool const  isA   = tensor == Tensor::A;
                auto const& desc  = gemm.tensors()[tensor];
                auto&       p     = init.pristine(tensor, desc);
                auto&       scale = init.pristine(isA ? Tensor::MXSA : Tensor::MXSB,
                                            isA ? gemm.mxsa() : gemm.mxsb());
                std::vector<uint8_t> data(81920), generated(2560);
                generateMXInput(HIP_R_8F_E4M3,
                                HIP_R_8F_UE8M0,
                                data.data(),
                                generated.data(),
                                256,
                                256,
                                320,
                                isA ? gemm.transA() : gemm.transB(),
                                isA ? 32 : 1,
                                isA ? 1 : 32,
                                isA,
                                MXScaleLayout::None,
                                "Bounded",
                                -1.0f,
                                1.0f,
                                "Bounded",
                                /*seedIndex=*/0,
                                false);
                for(size_t b = 0; b < batches; ++b)
                {
                    auto const* actualData
                        = static_cast<uint8_t const*>(p.cpuInput.valid.get()) + b * 81920;
                    EXPECT_EQ(std::memcmp(actualData, data.data(), 81856), 0);
                    auto const* actualScale
                        = static_cast<uint8_t const*>(scale.cpuInput.valid.get()) + b * 2048;
                    EXPECT_EQ(std::memcmp(actualScale, generated.data(), 2048), 0);
                    EXPECT_EQ(std::vector<uint8_t>(scale.mxCompactScale.begin() + b * 2560,
                                                   scale.mxCompactScale.begin() + (b + 1) * 2560),
                              generated);
                }
            }
        }
        // Compare each uploaded batch against an independently sized staging call.
        // Mode 0 uses logical scales, regardless of the data's padded leading dimension.
        init.prepareGPUInputs(problem);
        ContractionSolution swizzle;
        setMXSolution(swizzle, 1, 128);
        ASSERT_NO_THROW(init.preSolution(&swizzle));
        init.prepareGPUInputs(problem);
        for(auto tensor : {Tensor::MXSA, Tensor::MXSB})
        {
            bool const   isA  = tensor == Tensor::MXSA;
            auto const&  desc = gemm.tensors()[tensor];
            auto&        p    = init.pristine(tensor, desc);
            size_t const slot = batches > 1 ? desc.strides()[2] : desc.totalAllocatedBytes();
            std::vector<uint8_t> expected(p.mxSwizzledScale.size(), 0);
            std::vector<uint8_t> data(81920), generated(81920);
            for(size_t b = 0; b < batches; ++b)
            {
                if(GetParam() == 1)
                {
                    generateMXInput(HIP_R_8F_E4M3,
                                    HIP_R_8F_UE8M0,
                                    data.data(),
                                    generated.data(),
                                    256,
                                    256,
                                    320,
                                    isA ? gemm.transA() : gemm.transB(),
                                    isA ? 32 : 1,
                                    isA ? 1 : 32,
                                    isA,
                                    MXScaleLayout::GFX950,
                                    "Bounded",
                                    -1.0f,
                                    1.0f,
                                    "Bounded",
                                    /*seedIndex=*/0,
                                    false);
                    std::copy_n(generated.begin(), slot, expected.begin() + b * slot);
                }
                else
                {
                    auto const* canonical
                        = static_cast<uint8_t const*>(p.cpuInput.valid.get()) + b * slot;
                    std::vector<uint8_t> natural(canonical, canonical + 2048);
                    auto const grid = mxScaleGridGFX950(256,
                                                       256,
                                                       256,
                                                       isA ? gemm.transA() : gemm.transB(),
                                                       isA ? 32 : 1,
                                                       isA ? 1 : 32,
                                                       isA);
                    auto swizzled = swizzleScaleGFX950(natural, grid);
                    std::copy_n(swizzled.begin(), slot, expected.begin() + b * slot);
                }
            }
            EXPECT_EQ(p.mxSwizzledScale, expected);
            std::vector<uint8_t> gpu(expected.size());
            HIP_CHECK_EXC(
                hipMemcpy(gpu.data(), p.gpuInput.current.get(), gpu.size(), hipMemcpyDeviceToHost));
            EXPECT_EQ(gpu, expected);
        }
        init.postSolution();
    }
}

INSTANTIATE_TEST_SUITE_P(GeneratorModes, DataInitializationMXUploadTest, ::testing::Values(0, 1));

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
// 4.11 Natural scale swizzling: shared gfx950 permutation and client gfx1250 helper.
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

    auto const swizzled = p.layout == MXScaleLayout::GFX950
                             ? swizzleScaleGFX950(natural,
                                                  mxScaleGridGFX950(rows,
                                                                    cols,
                                                                    rows,
                                                                    p.transposed,
                                                                    p.isMatrixA ? mxBlock : 1,
                                                                    p.isMatrixA ? 1 : mxBlock,
                                                                    p.isMatrixA))
                             : dt::swizzleNaturalMXScale(natural, p.layout, rows, cols, mxBlock);
    ASSERT_LT(swizzled.size(), expected.size());
    EXPECT_EQ(swizzled, std::vector<uint8_t>(expected.begin(), expected.begin() + swizzled.size()));
    EXPECT_EQ(expected[swizzled.size()], sentinel)
        << "generateMXInput wrote more swizzled bytes than the scale permutation returned";
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

#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR

// =============================================================================
//   Section 7 — TensileLite::hip::CopyTensorVoid
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

// =============================================================================
//   Section 10 - DataInitialization::initArrayTrig(array, tensor)
//
//       Each element gets getTrigValue(idx), idx being its logical (column-major) index, at
//       tensor.index(coordinate of idx) / Packing. Packed descriptors take a flat loop, others
//       a per-coordinate loop; both must write exactly the bytes of the original algorithm
//       (initArrayTrigReference: CoordNumbered + TensorDescriptor::index per element, serial),
//       and leave the stride gaps untouched.
// =============================================================================
namespace
{
    class InitArrayThreadTest : public ::testing::Test
    {
    protected:
        void configureThreads(int requested)
        {
#ifdef _OPENMP
            omp_set_dynamic(0);
            omp_set_num_threads(requested);
            int actual = 0;
#pragma omp parallel
            {
#pragma omp single
                actual = omp_get_num_threads();
            }
            ASSERT_EQ(actual, requested) << "parallel init coverage requires the requested team";
#else
            if(requested > 1)
                GTEST_SKIP() << "parallel init coverage requires an OpenMP-enabled build";
#endif
        }

        void TearDown() override
        {
#ifdef _OPENMP
            omp_set_num_threads(m_previousThreads);
            omp_set_dynamic(m_previousDynamic);
#endif
        }

    private:
#ifdef _OPENMP
        int m_previousThreads = omp_get_max_threads();
        int m_previousDynamic = omp_get_dynamic();
#endif
    };

    template <typename T, bool useCos, bool useAbs>
    void initArrayTrigReference(T* array, TensorDescriptor const& tensor)
    {
        auto const&  sizes = tensor.sizes();
        size_t const count = TensileLite::CoordCount(sizes.begin(), sizes.end());
        for(size_t idx = 0; idx < count; idx += TensileLite::TypeInfo<T>::Packing)
        {
            std::vector<size_t> coord(tensor.dimensions(), 0);
            TensileLite::CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
            array[tensor.index(coord) / TensileLite::TypeInfo<T>::Packing]
                = DataInitialization::getTrigValue<T>(idx, useCos, useAbs);
        }
    }

    struct TrigLayout
    {
        std::vector<size_t> sizes;
        std::vector<size_t> strides; // empty: default (packed) strides
        char const* name;

        friend void PrintTo(TrigLayout const& layout, std::ostream* os)
        {
            *os << layout.name;
        }
    };

    bool supportsPacking(TrigLayout const& layout, size_t packing)
    {
        if(packing == 1)
            return true;
        // Packs must stay within contiguous rows, and outer strides must start whole packs.
        if(layout.sizes[0] % packing != 0)
            return false;
        if(layout.strides.empty())
            return true;
        if(layout.strides[0] != 1)
            return false;
        return std::all_of(layout.strides.begin() + 1, layout.strides.end(),
                           [=](size_t stride) { return stride % packing == 0; });
    }

    template <typename T, bool useCos, bool useAbs>
    void expectTrigMatchesReference(TrigLayout const& layout)
    {
        using Info = TensileLite::TypeInfo<T>;
        if(!supportsPacking(layout, Info::Packing))
            return;

        TensorDescriptor const tensor
            = layout.strides.empty()
                  ? TensorDescriptor("t", Info::Enum, layout.sizes.begin(), layout.sizes.end())
                  : TensorDescriptor("t",
                                     Info::Enum,
                                     layout.sizes.begin(),
                                     layout.sizes.end(),
                                     layout.strides.begin(),
                                     layout.strides.end());
        size_t const bytes = tensor.totalAllocatedElements() / Info::Packing * sizeof(T);

        std::vector<unsigned char> expected(bytes, 0xcc);
        std::vector<unsigned char> actual(bytes, 0xcc);
        initArrayTrigReference<T, useCos, useAbs>(reinterpret_cast<T*>(expected.data()), tensor);
        DataInitialization::initArrayTrig<T, useCos, useAbs>(reinterpret_cast<T*>(actual.data()),
                                                             tensor);

        auto const diff = std::mismatch(expected.begin(), expected.end(), actual.begin());
        EXPECT_TRUE(diff.first == expected.end())
            << "type=" << Info::Name() << " useCos=" << useCos << " useAbs=" << useAbs
            << " tensor=" << tensor << ": first differing byte " << (diff.first - expected.begin())
            << " of " << bytes;
    }

    template <typename T>
    void expectTrigMatchesReferenceAllModes(TrigLayout const& layout)
    {
        expectTrigMatchesReference<T, false, false>(layout);
        expectTrigMatchesReference<T, true, false>(layout);
        expectTrigMatchesReference<T, false, true>(layout);
        expectTrigMatchesReference<T, true, true>(layout);
    }
} // namespace

class InitArrayTrigTest
    : public InitArrayThreadTest,
      public ::testing::WithParamInterface<std::tuple<TrigLayout, int>>
{
    void SetUp() override { configureThreads(std::get<1>(GetParam())); }
};
TEST_P(InitArrayTrigTest, MatchesPerElementAlgorithm)
{
    auto const& layout = std::get<0>(GetParam());
    expectTrigMatchesReferenceAllModes<float>(layout);
    expectTrigMatchesReferenceAllModes<double>(layout);
    expectTrigMatchesReferenceAllModes<TensileLite::Half>(layout);
    expectTrigMatchesReferenceAllModes<TensileLite::BFloat16>(layout);
    expectTrigMatchesReferenceAllModes<TensileLite::Float8>(layout);
    expectTrigMatchesReferenceAllModes<TensileLite::BFloat8_fnuz>(layout);
    expectTrigMatchesReferenceAllModes<std::complex<float>>(layout);
    expectTrigMatchesReferenceAllModes<std::complex<double>>(layout);
#ifndef _WIN32
    expectTrigMatchesReferenceAllModes<TensileLite::Float4x2>(layout);
    expectTrigMatchesReferenceAllModes<TensileLite::Float6x32>(layout);
    expectTrigMatchesReferenceAllModes<TensileLite::BFloat6x32>(layout);
#endif
}

INSTANTIATE_TEST_SUITE_P(
    PackedAndStrided,
    InitArrayTrigTest,
    ::testing::Combine(::testing::Values(
        // ----- packed: default strides (flat loop) -------------------------
        TrigLayout{{4096}, {}, "Packed1D"},
        TrigLayout{{52, 20, 2}, {}, "PackedDefaultStrides"},
        TrigLayout{{64, 40}, {1, 64}, "PackedExplicitStrides"},
        TrigLayout{{1000, 37, 2}, {1, 1000, 37000}, "PackedBatched"},
        // 4800 FP6/BF6 packs cross the 4096-iteration schedule chunk boundary.
        TrigLayout{{256, 600}, {}, "Packed32ElementPacks"},
        // ----- strided (per-coordinate loop) -------------------------------
        TrigLayout{{60, 40}, {1, 64}, "LeadingDimPadding"},
        TrigLayout{{48, 36, 3}, {1, 48, 1800}, "BatchStridePadding"},
        TrigLayout{{1000, 37, 2}, {1, 1004, 40000}, "LeadingDimAndBatchPadding"},
        TrigLayout{{64, 1, 5}, {1, 4, 128}, "SizeOneDimNonDefaultStride"},
        TrigLayout{{40, 64}, {64, 1}, "Transposed"},
        TrigLayout{{256, 600}, {1, 288}, "Padded32ElementPacks"}
    ), ::testing::Values(1, 4)),
    [](::testing::TestParamInfo<std::tuple<TrigLayout, int>> const& info) {
        return std::string(std::get<0>(info.param).name) + "Threads"
               + std::to_string(std::get<1>(info.param));
    }
);

// =============================================================================
//   Section 11 - DataInitialization Serial / Identity init loops
//
//       SerialIdx, SerialDim0/1 and Identity values depend only on the element's logical index,
//       so the parallel loops (dynamic schedule) must write exactly the bytes of a serial run
//       of the same per-element algorithm, and leave the stride gaps untouched. The layouts
//       with 74,000 elements span many schedule chunks.
// =============================================================================
namespace
{
    enum class SerialInit
    {
        SerialIdx,
        SerialDim0,
        SerialDim1,
        Identity
    };

    char const* toString(SerialInit init)
    {
        switch(init)
        {
        case SerialInit::SerialIdx:
            return "SerialIdx";
        case SerialInit::SerialDim0:
            return "SerialDim0";
        case SerialInit::SerialDim1:
            return "SerialDim1";
        case SerialInit::Identity:
            return "Identity";
        }
        return "?";
    }

    template <typename T>
    void initArraySerialReference(T* array, TensorDescriptor const& tensor, SerialInit init)
    {
        using TensileLite::Half;
        auto const&  sizes   = tensor.sizes();
        size_t const count   = TensileLite::CoordCount(sizes.begin(), sizes.end());
        size_t const packing = TensileLite::TypeInfo<T>::Packing;
        // DataInitialization's SerialDim specialization for Half writes the coordinate as raw
        // bits, for every element.
        bool const halfBits = std::is_same_v<T, Half>
                              && (init == SerialInit::SerialDim0 || init == SerialInit::SerialDim1);
        for(size_t idx = 0; idx < count; idx += halfBits ? 1 : packing)
        {
            std::vector<size_t> coord(tensor.dimensions(), 0);
            TensileLite::CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
            size_t const tensorIndex = tensor.index(coord) / packing;
            size_t       value       = 0;
            switch(init)
            {
            case SerialInit::SerialIdx:
                value = idx;
                break;
            case SerialInit::SerialDim0:
                value = coord[0];
                break;
            case SerialInit::SerialDim1:
                value = coord[1];
                break;
            case SerialInit::Identity:
                value = coord[0] == coord[1] ? 1 : 0;
                break;
            }
            if constexpr(std::is_same_v<T, Half>)
            {
                if(halfBits)
                {
                    uint16_t const bits = static_cast<uint16_t>(value);
                    std::memcpy(&array[tensorIndex], &bits, sizeof(bits));
                    continue;
                }
            }
            array[tensorIndex] = DataInitialization::ConvertTo<T>(value);
        }
    }

    template <typename T>
    void runSerialInit(T* array, TensorDescriptor const& tensor, SerialInit init)
    {
        switch(init)
        {
        case SerialInit::SerialIdx:
            DataInitialization::initArraySerialIdx<T>(array, tensor);
            break;
        case SerialInit::SerialDim0:
            DataInitialization::initArraySerialDim<T>(array, 0, tensor);
            break;
        case SerialInit::SerialDim1:
            DataInitialization::initArraySerialDim<T>(array, 1, tensor);
            break;
        case SerialInit::Identity:
            DataInitialization::initArrayIdentity<T>(array, tensor);
            break;
        }
    }

    template <typename T>
    void expectSerialMatchesReference(TrigLayout const& layout, SerialInit init)
    {
        using Info = TensileLite::TypeInfo<T>;
        if(!supportsPacking(layout, Info::Packing))
            return;
        // SerialDim1 and Identity read the coordinate of dimension 1.
        if(init != SerialInit::SerialIdx && init != SerialInit::SerialDim0
           && layout.sizes.size() < 2)
            return;

        TensorDescriptor const tensor
            = layout.strides.empty()
                  ? TensorDescriptor("t", Info::Enum, layout.sizes.begin(), layout.sizes.end())
                  : TensorDescriptor("t",
                                     Info::Enum,
                                     layout.sizes.begin(),
                                     layout.sizes.end(),
                                     layout.strides.begin(),
                                     layout.strides.end());
        size_t const bytes = tensor.totalAllocatedElements() / Info::Packing * sizeof(T);

        std::vector<unsigned char> expected(bytes, 0xcc);
        std::vector<unsigned char> actual(bytes, 0xcc);
        initArraySerialReference<T>(reinterpret_cast<T*>(expected.data()), tensor, init);
        runSerialInit<T>(reinterpret_cast<T*>(actual.data()), tensor, init);

        auto const diff = std::mismatch(expected.begin(), expected.end(), actual.begin());
        EXPECT_TRUE(diff.first == expected.end())
            << "type=" << Info::Name() << " init=" << toString(init) << " tensor=" << tensor
            << ": first differing byte " << (diff.first - expected.begin()) << " of " << bytes;
    }

    // The flat SerialIdx overload (array, elements) writes ConvertTo(i) to pack i.
    template <typename T>
    void expectFlatSerialMatchesReference(size_t elements)
    {
        using Info           = TensileLite::TypeInfo<T>;
        size_t const   packs = elements / Info::Packing;
        // A trailing sentinel keeps zero-pack storage valid and catches stray writes.
        std::vector<T> expected(packs + 1);
        std::vector<T> actual(packs + 1);
        size_t const bytes = expected.size() * sizeof(T);
        std::memset(expected.data(), 0xcc, bytes);
        std::memset(actual.data(), 0xcc, bytes);
        for(size_t i = 0; i < packs; i++)
            expected[i] = DataInitialization::ConvertTo<T>(i);
        DataInitialization::initArrayConvert<T>(actual.data(), elements);
        EXPECT_EQ(std::memcmp(expected.data(), actual.data(), bytes), 0)
            << "type=" << Info::Name() << " elements=" << elements;
    }

    template <typename T>
    void expectSerialMatchesReferenceAllModes(TrigLayout const& layout)
    {
        for(SerialInit init : {SerialInit::SerialIdx,
                               SerialInit::SerialDim0,
                               SerialInit::SerialDim1,
                               SerialInit::Identity})
            expectSerialMatchesReference<T>(layout, init);
    }
} // namespace

class InitArraySerialTest
    : public InitArrayThreadTest,
      public ::testing::WithParamInterface<std::tuple<TrigLayout, int>>
{
    void SetUp() override { configureThreads(std::get<1>(GetParam())); }
};
TEST_P(InitArraySerialTest, MatchesPerElementAlgorithm)
{
    auto const& layout = std::get<0>(GetParam());
    expectSerialMatchesReferenceAllModes<float>(layout);
    expectSerialMatchesReferenceAllModes<double>(layout);
    expectSerialMatchesReferenceAllModes<TensileLite::Half>(layout);
    expectSerialMatchesReferenceAllModes<TensileLite::BFloat16>(layout);
    expectSerialMatchesReferenceAllModes<TensileLite::Float8>(layout);
    expectSerialMatchesReferenceAllModes<TensileLite::BFloat8_fnuz>(layout);
    expectSerialMatchesReferenceAllModes<int32_t>(layout);
    expectSerialMatchesReferenceAllModes<int8_t>(layout);
    expectSerialMatchesReferenceAllModes<std::complex<float>>(layout);
    expectSerialMatchesReferenceAllModes<std::complex<double>>(layout);
#ifndef _WIN32
    expectSerialMatchesReferenceAllModes<TensileLite::Float4x2>(layout);
    expectSerialMatchesReferenceAllModes<TensileLite::Float6x32>(layout);
    expectSerialMatchesReferenceAllModes<TensileLite::BFloat6x32>(layout);
#endif
}

INSTANTIATE_TEST_SUITE_P(
    PackedAndStrided,
    InitArraySerialTest,
    ::testing::Combine(::testing::Values(
        TrigLayout{{4096}, {}, "Packed1D"},
        TrigLayout{{52, 20, 2}, {}, "PackedDefaultStrides"},
        TrigLayout{{1000, 37, 2}, {1, 1000, 37000}, "PackedBatched"},
        TrigLayout{{60, 40}, {1, 64}, "LeadingDimPadding"},
        TrigLayout{{1000, 37, 2}, {1, 1004, 40000}, "LeadingDimAndBatchPadding"},
        TrigLayout{{40, 64}, {64, 1}, "Transposed"},
        TrigLayout{{256, 600}, {}, "Packed32ElementPacks"},
        TrigLayout{{256, 600}, {1, 288}, "Padded32ElementPacks"}
    ), ::testing::Values(1, 4)),
    [](::testing::TestParamInfo<std::tuple<TrigLayout, int>> const& info) {
        return std::string(std::get<0>(info.param).name) + "Threads"
               + std::to_string(std::get<1>(info.param));
    }
);

class InitArrayConvertTest : public InitArrayThreadTest, public ::testing::WithParamInterface<int>
{
    void SetUp() override { configureThreads(GetParam()); }
};

TEST_P(InitArrayConvertTest, MatchesSerialLoop)
{
    for(size_t elements : {size_t(0), size_t(1), size_t(4096), size_t(74000), size_t(153600)})
    {
        expectFlatSerialMatchesReference<float>(elements);
        expectFlatSerialMatchesReference<TensileLite::BFloat16>(elements);
        expectFlatSerialMatchesReference<int32_t>(elements);
#ifndef _WIN32
        expectFlatSerialMatchesReference<TensileLite::Float4x2>(elements);
        expectFlatSerialMatchesReference<TensileLite::Float6x32>(elements);
        expectFlatSerialMatchesReference<TensileLite::BFloat6x32>(elements);
#endif
    }
}

INSTANTIATE_TEST_SUITE_P(ThreadCounts, InitArrayConvertTest, ::testing::Values(1, 4));

class InitArraySeededRandomTest
    : public InitArrayThreadTest, public ::testing::WithParamInterface<int>
{
    void SetUp() override { configureThreads(GetParam()); }
};

TEST_P(InitArraySeededRandomTest, SingleChunkMatchesSeededSerialDraws)
{
#ifndef _WIN32
    using T = TensileLite::Float4x2; // This Random specialization draws from rand(), not random_device.
    using TensileLite::Client::InitMode;
    // The schedule chunk counts packs, so 4096 FP4 packs are 8192 logical elements.
    for(size_t packs : {size_t(1), DataInitialization::initLoopChunk})
        for(unsigned seed : {137u, 987u})
        {
            SCOPED_TRACE(::testing::Message() << "packs=" << packs << " seed=" << seed);
            std::vector<T> expected(packs), actual(packs);
            std::srand(seed);
            for(T& value : expected)
                value = DataInitialization::getValue<T, InitMode::Random>();
            int const nextExpected = std::rand();

            std::srand(seed);
            DataInitialization::initArray<T, InitMode::Random>(
                actual.data(), packs * TensileLite::TypeInfo<T>::Packing);
            EXPECT_EQ(std::memcmp(expected.data(), actual.data(), packs * sizeof(T)), 0);
            EXPECT_EQ(std::rand(), nextExpected) << "the fill must consume the same seeded draws";
        }
#else
    GTEST_SKIP() << "the rand-backed FP4 initialization mode is unavailable on Windows";
#endif
}

INSTANTIATE_TEST_SUITE_P(ThreadCounts, InitArraySeededRandomTest, ::testing::Values(1, 4));
