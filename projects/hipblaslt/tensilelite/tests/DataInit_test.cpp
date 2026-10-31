// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "DataInitialization.hpp"             // isMXTensor / Problem
#include "ClientProblemFactory.hpp"
#include "DataInitializationHelpers.hpp"    // detail::* (MX-only, internally guarded)
#include "ProgramOptions.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
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

// Shorthand for the production helper namespace under test (MX builds only).
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
namespace dt = TensileLite::Client::detail;
#endif
namespace
{
    template <typename T>
    void setOption(TensileLite::Client::po::variables_map& args, std::string const& name, T value)
    {
        args[name].value() = std::move(value);
    }

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

    TensileLite::Client::po::variables_map makeMinimalClientArgs(int mxScaleFormat)
    {
        using namespace TensileLite::Client;

        po::variables_map args;
        setOption(args, "problem-identifier", std::string("Contraction_l_Ailk_Bljk_Cijk_Dijk"));
        setOption(args, "problem-size", std::vector<std::vector<size_t>>{{128, 128, 1, 512}});
        setOption(args, "strided-batched", true);
        setOption(args, "batch-mode", 0);
        setOption(args, "grouped-gemm", false);
        setOption(args, "sparse", 0);
        setOption(args, "high-precision-accumulate", false);
        setOption(args, "kernel-language", KernelLanguage::Any);
        setOption(args, "performance-metric", PerformanceMetric::DeviceEfficiency);
        setOption(args, "deterministic-mode", false);
        setOption(args, "c-equal-d", false);
        setOption(args, "type", rocisa::DataType::Float8);
        setOption(args, "a-type", rocisa::DataType::Float8);
        setOption(args, "b-type", rocisa::DataType::Float8);
        setOption(args, "c-type", rocisa::DataType::Float);
        setOption(args, "d-type", rocisa::DataType::Float);
        setOption(args, "alpha-type", rocisa::DataType::Float);
        setOption(args, "beta-type", rocisa::DataType::Float);
        setOption(args, "compute-input-type-A", rocisa::DataType::Float8);
        setOption(args, "compute-input-type-B", rocisa::DataType::Float8);
        setOption(args, "f32-xdl-math-op", rocisa::DataType::Float);
        setOption(args, "activation-compute-type", rocisa::DataType::Float);
        // Deliberately 0 here (not 32): these only control whether
        // ClientProblemFactory calls setMXScaleA/B on the *internal dummy*
        // problem it builds for sizing purposes (see ClientProblemFactory.cpp),
        // which is unrelated to the `problem` object this test passes directly
        // into the DataInitialization tests. When non-zero here, the
        // DataInitialization ctor's storage-geometry computation for the
        // swizzled MX-scale tensor indexes problem.freeIndicesA()[0] /
        // freeIndicesB()[0] on that dummy problem without a bounds check; for
        // the minimal synthetic problem-size/identifier used here those
        // indices can be empty, which segfaults (only reachable on real
        // hardware, so it stays hidden behind GTEST_SKIP() on any other arch).
        // m_mxScaleFormat is read directly from "mx-scale-format" below,
        // independent of these.
        setOption(args, "mx-a-block", 0);
        setOption(args, "mx-b-block", 0);
        setOption(args, "mx-a-type", rocisa::DataType::E8);
        setOption(args, "mx-b-type", rocisa::DataType::E8);
        setOption(args, "mx-scale-format", mxScaleFormat);
        setOption(args, "fused-gemm-a2a", false);
        setOption(args, "metadata-layout", 0);
        setOption(args, "a-ops", TensileLite::TensorOps{});
        setOption(args, "b-ops", TensileLite::TensorOps{});
        setOption(args, "c-ops", TensileLite::TensorOps{});
        setOption(args, "d-ops", TensileLite::TensorOps{});
        setOption(args, "use-gradient", false);
        setOption(args, "output-amaxD", false);
        setOption(args, "use-scaleAB", std::string());
        setOption(args, "use-scaleCD", false);
        // ClientProblemFactory::m_useBias / m_biasSrc are plain `int` members
        // with NO default member initializer (see ClientProblemFactory.hpp);
        // they're only assigned when args.count("use-bias") /
        // args.count("bias-source") is true. Without these two options,
        // m_useBias holds whatever garbage was left on the stack/heap when
        // ClientProblemFactory was constructed. If that garbage is non-zero,
        // ContractionProblemGemm::setBias() takes a branch that indexes
        // m_tensors[m_biasSrc].sizes()[batchIdx] with batchIdx left at its
        // default of 2 for this non-batched dummy problem (m_batchIndices is
        // empty) -- an out-of-bounds read that segfaults. This is why the
        // crash only showed up when the test ran in isolation (a fresh
        // process, as CI's per-test ctest invocation does): the leftover
        // stack/heap contents differ from running the whole 733-test binary
        // in one process, where prior tests happened to leave zeroed memory
        // behind and masked the bug. Setting these explicitly makes the
        // behavior deterministic regardless of process/memory history.
        setOption(args, "use-bias", 0);
        setOption(args, "bias-source", static_cast<int>(ContractionProblemGemm::TENSOR::D));
        setOption(args, "use-scaleAlphaVec", 0);
        setOption(args, "device-idx", 0);
        setOption(args, "num-elements-to-validate", 0);
        setOption(args, "pristine-on-gpu", true);
        setOption(args, "prune-mode", TensileLite::Client::PruneSparseMode::PruneRandom);
        setOption(args, "rotating-buffer-size", 0);
        setOption(args, "rotating-buffer-mode", 0);
        setOption(args, "bounds-check", TensileLite::Client::BoundsCheckMode::Disable);
        return args;
    }
} // namespace

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

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
// The CPU reference is recomputed only when DataInitialization::inputEpoch() moves. The epoch is
// published by DataInitialization::preSolution(), so asking for it before that has run for the
// solution must be a loud error (a listener-ordering mistake would otherwise give a stale
// reference silently).
TEST(DataInitializationInputEpoch, PerSolutionEpochRequiresPreSolution)
{
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    if(std::string(prop.gcnArchName).find("gfx950") == std::string::npos
       && std::string(prop.gcnArchName).find("gfx1250") == std::string::npos)
        GTEST_SKIP() << "MX data initialization is only exercised on gfx950/gfx1250";

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

// =============================================================================
//   Section 5 - DataInitialization::initArrayTrig(array, tensor)
//
//       Each element gets getTrigValue(idx), idx being its logical (column-major) index, at
//       tensor.index(coordinate of idx) / Packing. Packed descriptors take a flat loop, others
//       a per-coordinate loop; both must write exactly the bytes of the original algorithm
//       (initArrayTrigReference: CoordNumbered + TensorDescriptor::index per element, serial),
//       and leave the stride gaps untouched.
// =============================================================================
namespace
{
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
        // Dimension 0 is contiguous and size[0] and the outer strides are even, so a packed
        // type (Float4x2) maps distinct packs to distinct array slots.
        bool        packableTypes;
        char const* name;

        friend void PrintTo(TrigLayout const& layout, std::ostream* os)
        {
            *os << layout.name;
        }
    };

    template <typename T, bool useCos, bool useAbs>
    void expectTrigMatchesReference(TrigLayout const& layout)
    {
        using Info = TensileLite::TypeInfo<T>;
        if(Info::Packing > 1 && !layout.packableTypes)
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

class InitArrayTrigTest : public ::testing::TestWithParam<TrigLayout>
{
};
TEST_P(InitArrayTrigTest, MatchesPerElementAlgorithm)
{
    auto const& layout = GetParam();
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
#endif
}

INSTANTIATE_TEST_SUITE_P(
    PackedAndStrided,
    InitArrayTrigTest,
    ::testing::Values(
        // ----- packed: default strides (flat loop) -------------------------
        TrigLayout{{4096}, {}, true, "Packed1D"},
        TrigLayout{{52, 20, 2}, {}, true, "PackedDefaultStrides"},
        TrigLayout{{64, 40}, {1, 64}, true, "PackedExplicitStrides"},
        TrigLayout{{1000, 37, 2}, {1, 1000, 37000}, true, "PackedBatched"},
        // ----- strided (per-coordinate loop) -------------------------------
        TrigLayout{{60, 40}, {1, 64}, true, "LeadingDimPadding"},
        TrigLayout{{48, 36, 3}, {1, 48, 1800}, true, "BatchStridePadding"},
        TrigLayout{{1000, 37, 2}, {1, 1004, 40000}, true, "LeadingDimAndBatchPadding"},
        TrigLayout{{64, 1, 5}, {1, 4, 128}, true, "SizeOneDimNonDefaultStride"},
        TrigLayout{{40, 64}, {64, 1}, false, "Transposed"}
    ),
    [](::testing::TestParamInfo<TrigLayout> const& info) {
        return std::string(info.param.name);
    }
);

// =============================================================================
//   Section 6 - DataInitialization Serial / Identity init loops
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
        if(Info::Packing > 1 && !layout.packableTypes)
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
        std::vector<T> expected(packs);
        std::vector<T> actual(packs);
        for(size_t i = 0; i < packs; i++)
            expected[i] = DataInitialization::ConvertTo<T>(i);
        DataInitialization::initArrayConvert<T>(actual.data(), elements);
        EXPECT_EQ(std::memcmp(expected.data(), actual.data(), packs * sizeof(T)), 0)
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

class InitArraySerialTest : public ::testing::TestWithParam<TrigLayout>
{
};
TEST_P(InitArraySerialTest, MatchesPerElementAlgorithm)
{
    auto const& layout = GetParam();
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
#endif
}

INSTANTIATE_TEST_SUITE_P(
    PackedAndStrided,
    InitArraySerialTest,
    ::testing::Values(
        TrigLayout{{4096}, {}, true, "Packed1D"},
        TrigLayout{{52, 20, 2}, {}, true, "PackedDefaultStrides"},
        TrigLayout{{1000, 37, 2}, {1, 1000, 37000}, true, "PackedBatched"},
        TrigLayout{{60, 40}, {1, 64}, true, "LeadingDimPadding"},
        TrigLayout{{1000, 37, 2}, {1, 1004, 40000}, true, "LeadingDimAndBatchPadding"},
        TrigLayout{{40, 64}, {64, 1}, false, "Transposed"}
    ),
    [](::testing::TestParamInfo<TrigLayout> const& info) {
        return std::string(info.param.name);
    }
);

TEST(InitArrayConvertTest, MatchesSerialLoop)
{
    for(size_t elements : {size_t(0), size_t(1), size_t(4096), size_t(74000)})
    {
        expectFlatSerialMatchesReference<float>(elements);
        expectFlatSerialMatchesReference<TensileLite::BFloat16>(elements);
        expectFlatSerialMatchesReference<int32_t>(elements);
#ifndef _WIN32
        expectFlatSerialMatchesReference<TensileLite::Float4x2>(elements);
#endif
    }
}
