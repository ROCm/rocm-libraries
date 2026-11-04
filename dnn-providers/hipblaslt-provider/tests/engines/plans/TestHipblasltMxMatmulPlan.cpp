// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipblasltUtils.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "TestWorkarounds.hpp"
#include "engines/plans/HipblasltMxMatmulPlan.hpp"
#include "engines/plans/MxArchSupport.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using hipdnn_plugin_sdk::archMatches;
using hipdnn_plugin_sdk::ArchMatchMode;

namespace
{

using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_test_sdk::utilities::createValidMxMatmulGraph;

/// Borrowed references to the three node attribute tables of an MX graph.
/// node 0 → dequant A, node 1 → dequant B, node 2 → matmul (emission order of
/// createValidMxMatmulGraph with swapDequantOrder=false).
struct MxNodeAttrs
{
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqA;
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqB;
    const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& matmul;
};

MxNodeAttrs getMxNodeAttrs(const GraphWrapper& graph)
{
    return MxNodeAttrs{
        graph.getNodeWrapper(0)
            .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes>(),
        graph.getNodeWrapper(1)
            .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes>(),
        graph.getNodeWrapper(2)
            .attributesAs<hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes>()};
}

/// M = N = 32, K = 128 graph with K-contiguous A, B and scales (hipBLAS TN).
flatbuffers::FlatBufferBuilder createKContiguousMxGraph(DT xType = DT::FP8_E4M3,
                                                        DT cType = DT::HALF)
{
    return createValidMxMatmulGraph({32, 128},
                                    {128, 1},
                                    {128, 32},
                                    {1, 128},
                                    {32, 32},
                                    {32, 1},
                                    {32, 4},
                                    {4, 32},
                                    xType,
                                    cType,
                                    DT::FP8_E8M0,
                                    DT::FP8_E8M0,
                                    DT::FLOAT,
                                    32,
                                    false,
                                    false,
                                    DT::UNSET,
                                    {4, 1},
                                    {1, 4});
}

bool onGfx950()
{
    return archMatches(
        hipdnn_test_sdk::utilities::currentDeviceArch(), "gfx950", ArchMatchMode::PREFIX);
}

/// Query a single scalar attribute from a hipBLASLt matmul descriptor.
template <typename T>
T getDescAttribute(const HipblasltMatmulDesc& desc, hipblasLtMatmulDescAttributes_t attr)
{
    T value{};
    size_t sizeWritten = 0;
    EXPECT_EQ(hipblasLtMatmulDescGetAttribute(
                  desc.matmulDesc(), attr, static_cast<void*>(&value), sizeof(value), &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(sizeWritten, sizeof(value));
    return value;
}

} // anonymous namespace

// ===========================================================================
// MxMatmulParams (CPU — host-side hipBLASLt layout/descriptor only, no device)
// ===========================================================================

// MxMatmulParams stores operands in hipBLAS's frame: the row-major swap maps our
// B to hipBLAS A and our A to hipBLAS B. So a() traces back to dequant B's X
// tensor, b() to dequant A's X tensor, and c() to the matmul output.
TEST(TestMxMatmulParams, LayoutUidsMatchInputTensors)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams const params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_EQ(params.a().uid(), attrs.deqB.x_tensor_uid()); // hipBLAS A <- our B
    EXPECT_EQ(params.b().uid(), attrs.deqA.x_tensor_uid()); // hipBLAS B <- our A
    EXPECT_EQ(params.c().uid(), attrs.matmul.c_tensor_uid());
}

// Scale UIDs follow the same hipBLAS-frame swap (aScaleUid = our B's scale,
// bScaleUid = our A's scale), and both scale modes must be set to VEC32_UE8M0
// (the OCP MX block-scale mode).
TEST(TestMxMatmulParams, ScaleUidsAndModeWiring)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams const params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_EQ(params.aScaleUid(), attrs.deqB.scale_tensor_uid()); // hipBLAS A_SCALE <- our B
    EXPECT_EQ(params.bScaleUid(), attrs.deqA.scale_tensor_uid()); // hipBLAS B_SCALE <- our A
    EXPECT_NE(params.aScaleUid(), params.bScaleUid());

    EXPECT_EQ(getDescAttribute<hipblasLtMatmulMatrixScale_t>(params.desc(),
                                                             HIPBLASLT_MATMUL_DESC_A_SCALE_MODE),
              HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
    EXPECT_EQ(getDescAttribute<hipblasLtMatmulMatrixScale_t>(params.desc(),
                                                             HIPBLASLT_MATMUL_DESC_B_SCALE_MODE),
              HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
}

// Default MX shape: A is col-major (opA=T), B is row-major (opB=N). Because of
// the row-major BLAS swap, the descriptor is built as
// (transA=getTrans(B), transB=getTrans(A)) → (OP_N, OP_T).
TEST(TestMxMatmulParams, TransposeInferenceDefault)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams const params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSA),
              HIPBLAS_OP_N);
    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSB),
              HIPBLAS_OP_T);
}

// When A is row-major instead of col-major, getTrans(A) flips to OP_N, so the
// descriptor's transB (= getTrans(A)) becomes OP_N. transA (= getTrans(B)) is
// unchanged.
TEST(TestMxMatmulParams, TransposeInferenceRowMajorA)
{
    // Row-major A (strides {128,1}) instead of the default col-major {1,32}.
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {128, 1}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams const params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSA),
              HIPBLAS_OP_N);
    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSB),
              HIPBLAS_OP_N);
}

// When B is col-major instead of row-major, getTrans(B) flips to OP_T, so the
// descriptor's transA (= getTrans(B)) becomes OP_T. transB (= getTrans(A)) is
// unchanged.
TEST(TestMxMatmulParams, TransposeInferenceColMajorB)
{
    // Col-major B (strides {1,128}) instead of the default row-major {32,1}.
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {1, 128}, {32, 32}, {32, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams const params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSA),
              HIPBLAS_OP_T);
    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSB),
              HIPBLAS_OP_T);
}

// K-contiguous A (row-major) and B (column-major) reach hipBLAS as TN, the
// layout frameworks use and the only one with Tensile MX kernels.
TEST(TestMxMatmulParams, TransposeInferenceKContiguous)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {128, 1}, {128, 32}, {1, 128}, {32, 32}, {32, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams const params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSA),
              HIPBLAS_OP_T);
    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSB),
              HIPBLAS_OP_N);
}

// Scale operands are recorded in hipBLAS's frame with their own order and their
// operand's order. Default graph: our A (hipBLAS B) is M-contiguous with a
// K-contiguous (row-major) scale; our B (hipBLAS A) is N-contiguous with an
// N-contiguous (row-major [K/32, N]) scale.
TEST(TestMxMatmulParams, ScaleOperandsDefaultLayout)
{
    auto fb = createValidMxMatmulGraph(
        {48, 128}, {1, 48}, {128, 32}, {32, 1}, {48, 32}, {32, 1}, {48, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams const params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_EQ(params.aScale().uid, attrs.deqB.scale_tensor_uid());
    EXPECT_EQ(params.aScale().mn, 32);
    EXPECT_EQ(params.aScale().kBlocks, 4);
    EXPECT_EQ(params.aScale().operandOrder, MxOrder::MN_CONTIGUOUS);
    EXPECT_EQ(params.aScale().scaleOrder, MxOrder::MN_CONTIGUOUS);

    EXPECT_EQ(params.bScale().uid, attrs.deqA.scale_tensor_uid());
    EXPECT_EQ(params.bScale().mn, 48);
    EXPECT_EQ(params.bScale().kBlocks, 4);
    EXPECT_EQ(params.bScale().operandOrder, MxOrder::MN_CONTIGUOUS);
    EXPECT_EQ(params.bScale().scaleOrder, MxOrder::K_CONTIGUOUS);
}

TEST(TestMxMatmulParams, ScaleOperandsKContiguousLayout)
{
    auto fb = createKContiguousMxGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams const params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    for(const auto* scale : {&params.aScale(), &params.bScale()})
    {
        EXPECT_EQ(scale->operandOrder, MxOrder::K_CONTIGUOUS);
        EXPECT_EQ(scale->scaleOrder, MxOrder::K_CONTIGUOUS);
    }
}

// The gfx950 Tensile path signals its swizzled scales with BLK32_UE8M0_32_8_EXT;
// the gfx1250 one keeps VEC32_UE8M0 (its swizzle is implicit).
TEST(TestMxMatmulParams, ScaleModeFollowsScaleFormat)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {128, 1}, {128, 32}, {1, 128}, {32, 32}, {32, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    for(const auto format :
        {MxScaleFormat::NATURAL, MxScaleFormat::SWIZZLED_GFX950, MxScaleFormat::SWIZZLED_GFX1250})
    {
        MxMatmulParams const params(
            attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap(), format);
        EXPECT_EQ(params.scaleFormat(), format);
        EXPECT_EQ(getDescAttribute<hipblasLtMatmulMatrixScale_t>(
                      params.desc(), HIPBLASLT_MATMUL_DESC_A_SCALE_MODE),
                  scaleModeFor(format))
            << toString(format);
        EXPECT_EQ(getDescAttribute<hipblasLtMatmulMatrixScale_t>(
                      params.desc(), HIPBLASLT_MATMUL_DESC_B_SCALE_MODE),
                  scaleModeFor(format))
            << toString(format);
    }
}

// packedMxOrder classifies the two innermost dims; kAxisFromEnd says which one is K.
TEST(TestMxMatmulParams, PackedMxOrder)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {128, 1}, {128, 32}, {1, 128}, {32, 32}, {32, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);
    const auto& tensorMap = graph.getTensorMap();
    const auto xA = hipblaslt_utils::findTensorAttributes(tensorMap, attrs.deqA.x_tensor_uid());
    const auto xB = hipblaslt_utils::findTensorAttributes(tensorMap, attrs.deqB.x_tensor_uid());
    const auto c = hipblaslt_utils::findTensorAttributes(tensorMap, attrs.matmul.c_tensor_uid());

    EXPECT_EQ(packedMxOrder(xA, 1), MxOrder::K_CONTIGUOUS); // [M, K] row-major
    EXPECT_EQ(packedMxOrder(xB, 2), MxOrder::K_CONTIGUOUS); // [K, N] column-major
    EXPECT_EQ(packedMxOrder(c, 1), MxOrder::K_CONTIGUOUS); // row-major, inner axis
    EXPECT_EQ(packedMxOrder(c, 2), MxOrder::MN_CONTIGUOUS); // row-major, outer axis K
    EXPECT_EQ(packedMxOrder(xA, 3), std::nullopt); // invalid K axis
}

#if WORKAROUND_ISSUE_14286
// The kernel/scale-layout guard reads the MXScaleFormat token of Tensile names.
TEST(TestMxMatmulParams, KernelScaleLayoutFromName)
{
    EXPECT_EQ(mxKernelScaleLayout("Cijk_Alik_Bljk_F8SS_MXAE8B32_MXBE8B32_MXLIBL_MXSFHPS_MO40_X"),
              "HPS");
    EXPECT_EQ(mxKernelScaleLayout("Cijk_Alik_Bljk_F8BS_MXSFIMS"), "IMS");
    EXPECT_EQ(mxKernelScaleLayout("Cijk_Alik_Bljk_F4BS_MXSFNS_WG32"), "NS");
    EXPECT_EQ(mxKernelScaleLayout("Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_MT64x64x256"), "");
    EXPECT_EQ(mxKernelScaleLayout("rr_16x16x128"), "");
}

// The kernel/scale-layout guard: which prepared format each kind of
// selected kernel accepts, and what it then reads. rocRoller reads natural scales;
// Tensile reads what its `_MXSF` token says (HPS: gfx950 32x8, IMS: gfx1250,
// NS: natural); a Tensile name without the token is trusted for the swizzled formats.
TEST(TestMxMatmulParams, KernelScaleFormatGuard)
{
    struct Kernel
    {
        const char* name;
        bool isRocRoller;
        const char* token;
    };
    const std::array<Kernel, 5> kernels = {Kernel{"rocRoller", true, ""},
                                           Kernel{"Tensile HPS", false, "HPS"},
                                           Kernel{"Tensile IMS", false, "IMS"},
                                           Kernel{"Tensile NS", false, "NS"},
                                           Kernel{"Tensile without token", false, ""}};
    using F = MxScaleFormat;
    const std::optional<F> none;
    struct Row
    {
        F prepared;
        std::array<std::optional<F>, 5> reads; // per kernel, in the order above
    };
    const std::array<Row, 3> table
        = {Row{F::NATURAL, {F::NATURAL, none, none, F::NATURAL, none}},
           Row{F::SWIZZLED_GFX950, {none, F::SWIZZLED_GFX950, none, none, F::SWIZZLED_GFX950}},
           Row{F::SWIZZLED_GFX1250,
               {none, none, F::SWIZZLED_GFX1250, F::NATURAL, F::SWIZZLED_GFX1250}}};

    for(const auto& row : table)
    {
        for(size_t k = 0; k < kernels.size(); ++k)
        {
            EXPECT_EQ(kernelScaleFormat(kernels[k].isRocRoller, kernels[k].token, row.prepared),
                      row.reads[k])
                << toString(row.prepared) << " scales, " << kernels[k].name << " kernel";
        }
    }
}
#endif

TEST(TestMxMatmulParams, ScaleFormatForArch)
{
    EXPECT_EQ(mxScaleFormatForArch("gfx1250"), MxScaleFormat::SWIZZLED_GFX1250);
    EXPECT_EQ(mxScaleFormatForArch("gfx1250-strict"), MxScaleFormat::SWIZZLED_GFX1250);
    EXPECT_EQ(mxScaleFormatForArch("gfx950"), MxScaleFormat::NATURAL);
}

TEST(TestMxMatmulParams, PackedMxOrderRejectsPaddedMatrix)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {136, 1}, {128, 32}, {1, 136}, {32, 32}, {32, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);
    const auto& tensorMap = graph.getTensorMap();

    EXPECT_EQ(packedMxOrder(
                  hipblaslt_utils::findTensorAttributes(tensorMap, attrs.deqA.x_tensor_uid()), 1),
              std::nullopt);
    EXPECT_EQ(packedMxOrder(
                  hipblaslt_utils::findTensorAttributes(tensorMap, attrs.deqB.x_tensor_uid()), 2),
              std::nullopt);
}

// ===========================================================================
// MxMatmulPlan (GPU — requires a device and a live hipblasLt handle)
// ===========================================================================

// A device that runs MX and a live hipBLASLt handle.
class MxGpuPlanTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        const auto archName = hipdnn_test_sdk::utilities::currentDeviceArch();
        if(!hipblaslt_plugin_test::isMxSupportedArch(archName))
        {
            GTEST_SKIP() << "MX block-scaled GEMM is not supported on " << archName;
        }
        ASSERT_EQ(hipblasLtCreate(&_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle.hipblasltHandle != nullptr)
        {
            EXPECT_EQ(hipblasLtDestroy(_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
        }
    }

    HipdnnEnginePluginHandle _handle;
};

// The default (M/N-contiguous) test graph, which gfx1250 declines (#13715).
class TestGpuMxMatmulPlan : public MxGpuPlanTestBase
{
protected:
    void SetUp() override
    {
        SKIP_IF_WORKAROUND_ISSUE_13715();
        MxGpuPlanTestBase::SetUp();
    }
};

TEST_F(TestGpuMxMatmulPlan, CreatesPlanWithValidGraph)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_NO_THROW(MxMatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMxMatmulPlan, PlanReturnsValidWorkspaceSize)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());
    MxMatmulPlan const plan(_handle, std::move(params));

    EXPECT_GE(plan.getWorkspaceSize(_handle), 0u);
}

// execute() must reject a null workspace when the plan needs one, as this graph's
// does (our A's K-contiguous scale is transposed into it). The check runs before
// any device-buffer lookup, so no buffers are passed.
TEST_F(TestGpuMxMatmulPlan, ExecuteThrowsOnNullWorkspace)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());
    MxMatmulPlan const plan(_handle, std::move(params));
    ASSERT_GT(plan.getWorkspaceSize(_handle), 0u);

    try
    {
        plan.execute(_handle, nullptr, 0, nullptr);
        ADD_FAILURE() << "execute() accepted a null workspace";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_NE(std::string(e.what()).find("workspace is null"), std::string::npos) << e.what();
    }
}

// K-contiguous operands with K-contiguous scales need no scale preparation on
// rocRoller (gfx950): both scales are handed to hipBLASLt untouched.
TEST_F(TestGpuMxMatmulPlan, KContiguousNaturalScalesArePassedThrough)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "gfx950 (rocRoller) only";
    }
    auto fb = createKContiguousMxGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());
    MxMatmulPlan const plan(_handle, std::move(params));

    EXPECT_TRUE(plan.aScalePrep().isPassThrough());
    EXPECT_TRUE(plan.bScalePrep().isPassThrough());
}

// The default (M/N-contiguous) graph with row-major scales: only our A's
// K-contiguous scale is transposed into its operand's M-contiguous order.
TEST_F(TestGpuMxMatmulPlan, DefaultLayoutTransposesOnlyTheAScale)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());
    MxMatmulPlan const plan(_handle, std::move(params));

    EXPECT_TRUE(plan.aScalePrep().isPassThrough()); // our B's scale
    EXPECT_EQ(plan.bScalePrep().method(), MxScalePrep::Method::TRANSPOSE); // our A's scale
    EXPECT_GE(plan.getWorkspaceSize(_handle), plan.bScalePrep().workspaceSize());
}

// On gfx950 the provider selects natural scales (rocRoller); the Tensile format
// is used only when a caller asks createMxMatmulPlan() for it.
TEST_F(TestGpuMxMatmulPlan, SelectsNaturalScalesOnGfx950)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "gfx950-only check";
    }
    EXPECT_EQ(selectMxScaleFormat(_handle), MxScaleFormat::NATURAL);
}

// hipBLASLt has gfx950 Tensile MX kernels for FP4 x FP4 and E4M3 x E4M3 with
// K-contiguous operands; they read the 32x8 pre-swizzled scales the plan writes.
TEST_F(TestGpuMxMatmulPlan, Gfx950SwizzledFormatUsesTensileKernel)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "gfx950-only check";
    }
    for(const auto xType : {DT::FP8_E4M3, DT::FP4_E2M1})
    {
        auto fb = createKContiguousMxGraph(xType, DT::FLOAT);
        GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
        const auto attrs = getMxNodeAttrs(graph);

        const auto plan = createMxMatmulPlan(_handle,
                                             attrs.deqA,
                                             attrs.deqB,
                                             attrs.matmul,
                                             graph.getTensorMap(),
                                             MxScaleFormat::SWIZZLED_GFX950);

        EXPECT_EQ(plan->scaleFormat(), MxScaleFormat::SWIZZLED_GFX950) << EnumNameDataType(xType);
        EXPECT_FALSE(plan->aScalePrep().isPassThrough()) << EnumNameDataType(xType);
        EXPECT_FALSE(plan->bScalePrep().isPassThrough()) << EnumNameDataType(xType);
    }
}

// Without a usable gfx950 Tensile kernel (FP6, or a layout other than hipBLAS TN)
// the swizzled format falls back to rocRoller with natural scales.
TEST_F(TestGpuMxMatmulPlan, Gfx950SwizzledFormatFallsBackToRocRoller)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "gfx950-only check";
    }
    auto fp6 = createKContiguousMxGraph(DT::FP6_E2M3, DT::FLOAT);
    auto mnContiguous = createValidMxMatmulGraph();
    for(const auto* fb : {&fp6, &mnContiguous})
    {
        GraphWrapper const graph(fb->GetBufferPointer(), fb->GetSize());
        const auto attrs = getMxNodeAttrs(graph);

        const auto plan = createMxMatmulPlan(_handle,
                                             attrs.deqA,
                                             attrs.deqB,
                                             attrs.matmul,
                                             graph.getTensorMap(),
                                             MxScaleFormat::SWIZZLED_GFX950);

        EXPECT_EQ(plan->scaleFormat(), MxScaleFormat::NATURAL);
    }
}

#if WORKAROUND_ISSUE_14286
// Only the gfx950 Tensile format falls back. On gfx950 hipBLASLt runs VEC32 on
// rocRoller, which does not read gfx1250-swizzled scales, so the guard declines.
TEST_F(TestGpuMxMatmulPlan, Gfx1250SwizzledFormatDoesNotFallBack)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "gfx950-only check";
    }
    auto fb = createKContiguousMxGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    EXPECT_THROW(static_cast<void>(createMxMatmulPlan(_handle,
                                                      attrs.deqA,
                                                      attrs.deqB,
                                                      attrs.matmul,
                                                      graph.getTensorMap(),
                                                      MxScaleFormat::SWIZZLED_GFX1250)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
#endif

// ===========================================================================
// MxMatmulPlan::execute (GPU) — scale preparation checked end to end
// ===========================================================================
// The same logical GEMM runs with its scales stored in different orders. A scale
// already in the order hipBLASLt reads is passed through; any other is rewritten
// on the device into a region of the workspace first. The kernel reads the same
// scale values either way, so C must be bitwise equal.

namespace
{

constexpr int64_t EXEC_M = 64;
constexpr int64_t EXEC_N = 96;
constexpr int64_t EXEC_K = 256;
constexpr int64_t EXEC_K_BLOCKS = EXEC_K / VEC32_BLOCK_SIZE;

// Physical order of our A [M, K], our B [K, N] and their scales [M, K/32] and
// [K/32, N]; true means K (or the K-block axis) is contiguous.
struct MxStorage
{
    bool aKContiguous;
    bool bKContiguous;
    bool scaleAKContiguous;
    bool scaleBKContiguous;
};

std::vector<int64_t> rowMajorStrides(int64_t /*rows*/, int64_t cols)
{
    return {cols, 1};
}

std::vector<int64_t> colMajorStrides(int64_t rows, int64_t /*cols*/)
{
    return {1, rows};
}

flatbuffers::FlatBufferBuilder createExecuteGraph(const MxStorage& s)
{
    return createValidMxMatmulGraph(
        {EXEC_M, EXEC_K},
        s.aKContiguous ? rowMajorStrides(EXEC_M, EXEC_K) : colMajorStrides(EXEC_M, EXEC_K),
        {EXEC_K, EXEC_N},
        s.bKContiguous ? colMajorStrides(EXEC_K, EXEC_N) : rowMajorStrides(EXEC_K, EXEC_N),
        {EXEC_M, EXEC_N},
        rowMajorStrides(EXEC_M, EXEC_N),
        {EXEC_M, EXEC_K_BLOCKS},
        {EXEC_K_BLOCKS, EXEC_N},
        DT::FP8_E4M3,
        DT::FLOAT,
        DT::FP8_E8M0,
        DT::FP8_E8M0,
        DT::FLOAT,
        static_cast<int32_t>(VEC32_BLOCK_SIZE),
        false,
        false,
        DT::UNSET,
        s.scaleAKContiguous ? rowMajorStrides(EXEC_M, EXEC_K_BLOCKS)
                            : colMajorStrides(EXEC_M, EXEC_K_BLOCKS),
        s.scaleBKContiguous ? colMajorStrides(EXEC_K_BLOCKS, EXEC_N)
                            : rowMajorStrides(EXEC_K_BLOCKS, EXEC_N));
}

// Logical row-major [rows][cols] bytes laid out row- or column-major.
std::vector<uint8_t>
    inMemoryOrder(const std::vector<uint8_t>& logical, int64_t rows, int64_t cols, bool rowMajor)
{
    if(rowMajor)
    {
        return logical;
    }
    std::vector<uint8_t> out(logical.size());
    for(int64_t r = 0; r < rows; ++r)
    {
        for(int64_t c = 0; c < cols; ++c)
        {
            out[static_cast<size_t>(c * rows + r)] = logical[static_cast<size_t>(r * cols + c)];
        }
    }
    return out;
}

// Logical row-major contents: A [M][K], B [K][N], scale_a [M][K/32], scale_b [K/32][N].
struct MxInputs
{
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    std::vector<uint8_t> scaleA;
    std::vector<uint8_t> scaleB;
};

MxInputs randomInputs()
{
    std::mt19937 rng(20261028);
    // Finite E4M3 values (exponent field <= 12, either sign) and E8M0 scales near 1,
    // so every output element is a finite, nonzero-heavy sum.
    const auto fp8
        = [&rng]() { return static_cast<uint8_t>((rng() % 0x68) | ((rng() & 1u) << 7)); };
    const auto e8m0 = [&rng]() { return static_cast<uint8_t>(122 + rng() % 11); };
    MxInputs in;
    in.a.resize(static_cast<size_t>(EXEC_M * EXEC_K));
    in.b.resize(static_cast<size_t>(EXEC_K * EXEC_N));
    in.scaleA.resize(static_cast<size_t>(EXEC_M * EXEC_K_BLOCKS));
    in.scaleB.resize(static_cast<size_t>(EXEC_K_BLOCKS * EXEC_N));
    for(auto* bytes : {&in.a, &in.b})
    {
        for(auto& value : *bytes)
        {
            value = fp8();
        }
    }
    for(auto* bytes : {&in.scaleA, &in.scaleB})
    {
        for(auto& value : *bytes)
        {
            value = e8m0();
        }
    }
    return in;
}

struct DeviceFree
{
    void operator()(void* ptr) const
    {
        static_cast<void>(hipFree(ptr));
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;

DevicePtr upload(const std::vector<uint8_t>& bytes)
{
    void* ptr = nullptr;
    EXPECT_EQ(hipMalloc(&ptr, bytes.empty() ? 1 : bytes.size()), hipSuccess);
    EXPECT_EQ(hipMemcpy(ptr, bytes.data(), bytes.size(), hipMemcpyHostToDevice), hipSuccess);
    return DevicePtr(ptr);
}

struct MxExecution
{
    MxScaleFormat format = MxScaleFormat::NATURAL;
    size_t workspaceSize = 0;
    bool scaleAPrepared = false; // our A's scale (hipBLAS B_SCALE)
    bool scaleBPrepared = false; // our B's scale (hipBLAS A_SCALE)
    std::vector<uint8_t> c;
};

// `format` defaults to the one the provider selects for the device.
std::unique_ptr<MxMatmulPlan> createExecutePlan(const HipdnnEnginePluginHandle& handle,
                                                const MxStorage& s,
                                                std::optional<MxScaleFormat> format = std::nullopt)
{
    auto fb = createExecuteGraph(s);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);
    return createMxMatmulPlan(handle,
                              attrs.deqA,
                              attrs.deqB,
                              attrs.matmul,
                              graph.getTensorMap(),
                              format.value_or(selectMxScaleFormat(handle)));
}

// `nullWorkspace` passes no workspace, as hipDNN does for a plan that needs none.
MxExecution execute(const HipdnnEnginePluginHandle& handle,
                    const MxStorage& s,
                    const MxInputs& in,
                    bool nullWorkspace = false,
                    std::optional<MxScaleFormat> format = std::nullopt)
{
    auto fb = createExecuteGraph(s);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);
    const auto plan = createExecutePlan(handle, s, format);

    MxExecution result;
    result.format = plan->scaleFormat();
    result.workspaceSize = plan->getWorkspaceSize(handle);
    result.scaleAPrepared = !plan->bScalePrep().isPassThrough();
    result.scaleBPrepared = !plan->aScalePrep().isPassThrough();

    auto a = upload(inMemoryOrder(in.a, EXEC_M, EXEC_K, s.aKContiguous));
    auto b = upload(inMemoryOrder(in.b, EXEC_K, EXEC_N, !s.bKContiguous));
    auto scaleA = upload(inMemoryOrder(in.scaleA, EXEC_M, EXEC_K_BLOCKS, s.scaleAKContiguous));
    auto scaleB = upload(inMemoryOrder(in.scaleB, EXEC_K_BLOCKS, EXEC_N, !s.scaleBKContiguous));
    auto c = upload(std::vector<uint8_t>(static_cast<size_t>(EXEC_M * EXEC_N) * sizeof(float), 0));
    // Garbage in the workspace, so a prepared scale that is not fully written, or is
    // overwritten by the other one, changes C.
    auto workspace = nullWorkspace
                         ? DevicePtr()
                         : upload(std::vector<uint8_t>(plan->getWorkspaceSize(handle), 0xA5));

    const std::array<hipdnnPluginDeviceBuffer_t, 5> buffers
        = {hipdnnPluginDeviceBuffer_t{attrs.deqA.x_tensor_uid(), a.get()},
           hipdnnPluginDeviceBuffer_t{attrs.deqA.scale_tensor_uid(), scaleA.get()},
           hipdnnPluginDeviceBuffer_t{attrs.deqB.x_tensor_uid(), b.get()},
           hipdnnPluginDeviceBuffer_t{attrs.deqB.scale_tensor_uid(), scaleB.get()},
           hipdnnPluginDeviceBuffer_t{attrs.matmul.c_tensor_uid(), c.get()}};
    plan->execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
    EXPECT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);

    result.c.resize(static_cast<size_t>(EXEC_M * EXEC_N) * sizeof(float));
    EXPECT_EQ(hipMemcpy(result.c.data(), c.get(), result.c.size(), hipMemcpyDeviceToHost),
              hipSuccess);
    return result;
}

// C as floats.
std::vector<float> asFloats(const std::vector<uint8_t>& bytes)
{
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
    return values;
}

std::string describe(const MxStorage& s)
{
    const auto order = [](bool kContiguous) { return kContiguous ? "K" : "MN"; };
    return std::string("A ") + order(s.aKContiguous) + ", B " + order(s.bKContiguous) + ", scale_a "
           + order(s.scaleAKContiguous) + ", scale_b " + order(s.scaleBKContiguous);
}

} // namespace

// Unlike TestGpuMxMatmulPlan this also runs on gfx1250, for the layouts hipBLASLt
// has kernels for there.
class TestGpuMxMatmulPlanExecute : public MxGpuPlanTestBase
{
protected:
    // Runs `reference` and each variant on the same inputs and expects the same C.
    // `scalePrepared` gives, per variant, whether scale_a and scale_b are rewritten
    // when the plan uses natural scales (rocRoller).
    void expectSameResult(const MxStorage& reference,
                          const std::vector<std::pair<MxStorage, std::pair<bool, bool>>>& variants)
    {
        const auto inputs = randomInputs();
        const auto expected = execute(_handle, reference, inputs);
        size_t nonZero = 0;
        for(size_t i = 0; i < expected.c.size(); i += sizeof(float))
        {
            float value = 0.f;
            std::memcpy(&value, expected.c.data() + i, sizeof(float));
            nonZero += value != 0.f ? 1 : 0;
        }
        ASSERT_GT(nonZero, expected.c.size() / sizeof(float) / 2)
            << "reference output is mostly zero: " << describe(reference);

        for(const auto& [storage, prepared] : variants)
        {
            const auto actual = execute(_handle, storage, inputs);
            if(actual.format == MxScaleFormat::NATURAL)
            {
                EXPECT_EQ(actual.scaleAPrepared, prepared.first) << describe(storage);
                EXPECT_EQ(actual.scaleBPrepared, prepared.second) << describe(storage);
            }
            EXPECT_TRUE(actual.c == expected.c)
                << describe(storage) << " gives a different C than " << describe(reference);
        }
    }
};

// K-contiguous A and B (hipBLAS TN, every MX arch). With rocRoller a scale in the
// other order is transposed by hipBLASLt; on gfx1250 the relayout kernel swizzles every
// scale from whichever order it has, both in one launch.
TEST_F(TestGpuMxMatmulPlanExecute, KContiguousOperandsAnyScaleOrder)
{
    expectSameResult({true, true, true, true},
                     {{{true, true, false, true}, {true, false}},
                      {{true, true, true, false}, {false, true}},
                      {{true, true, false, false}, {true, true}}});
}

// M/N-contiguous A and B (hipBLAS NT, gfx950 only): a K-contiguous scale, as every
// in-tree NT bundle has for scale_a, is transposed into its operand's order.
TEST_F(TestGpuMxMatmulPlanExecute, MnContiguousOperandsAnyScaleOrder)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "hipBLASLt has M/N-contiguous MX kernels on gfx950 only (#13715)";
    }
    expectSameResult({false, false, false, false},
                     {{{false, false, true, false}, {true, false}},
                      {{false, false, false, true}, {false, true}},
                      {{false, false, true, true}, {true, true}}});
}

// K-contiguous A and N-contiguous B (gfx950 only): the two scales are rewritten
// into different orders, so they are prepared separately.
TEST_F(TestGpuMxMatmulPlanExecute, MixedOperandOrdersPrepareEachScale)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "hipBLASLt has M/N-contiguous MX kernels on gfx950 only (#13715)";
    }
    expectSameResult({true, false, true, false}, {{{true, false, false, true}, {true, true}}});
}

// With every scale already in the order hipBLASLt reads and a kernel that needs no
// workspace (rocRoller without stream-K), the plan needs none, and hipDNN then
// passes no workspace to execute().
TEST_F(TestGpuMxMatmulPlanExecute, RunsWithoutWorkspaceWhenNoneIsNeeded)
{
    const MxStorage passThrough{true, true, true, true};
    const auto inputs = randomInputs();
    const auto withWorkspace = execute(_handle, passThrough, inputs);
    if(withWorkspace.workspaceSize != 0)
    {
        GTEST_SKIP() << "this plan needs " << withWorkspace.workspaceSize << " bytes of workspace";
    }
    const auto withoutWorkspace = execute(_handle, passThrough, inputs, /*nullWorkspace=*/true);
    EXPECT_TRUE(withoutWorkspace.c == withWorkspace.c);
}

// A scale stored in the other order is transposed by hipBLASLt for natural scales and
// rewritten by the provider's relayout kernel for a swizzled format.
TEST_F(TestGpuMxMatmulPlanExecute, PreparesScalesByFormat)
{
    // Both scales stored in the other order, so both are prepared on any MX arch.
    const auto plan = createExecutePlan(_handle, {true, true, false, false});
    const auto expected = plan->scaleFormat() == MxScaleFormat::NATURAL
                              ? MxScalePrep::Method::TRANSPOSE
                              : MxScalePrep::Method::RELAYOUT_KERNEL;
    for(const auto* prep : {&plan->aScalePrep(), &plan->bScalePrep()})
    {
        EXPECT_EQ(prep->method(), expected)
            << toString(plan->scaleFormat()) << ": " << prep->describe();
    }
}

// gfx950's Tensile kernels read 32x8 pre-swizzled scales, which only the relayout kernel
// writes; the provider does not select this format on its own yet. Every scale order
// is swizzled into the same bytes, so C is bitwise equal across them, and it matches
// rocRoller's C on natural scales up to summation order.
TEST_F(TestGpuMxMatmulPlanExecute, Gfx950SwizzledScalesAnyScaleOrder)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "gfx950-only check";
    }
    const MxStorage reference{true, true, true, true};
    const auto inputs = randomInputs();
    const auto swizzled
        = execute(_handle, reference, inputs, false, MxScaleFormat::SWIZZLED_GFX950);
    ASSERT_EQ(swizzled.format, MxScaleFormat::SWIZZLED_GFX950)
        << "hipBLASLt has no gfx950 Tensile kernel for this GEMM";

    const auto natural = asFloats(execute(_handle, reference, inputs).c);
    const auto tensile = asFloats(swizzled.c);
    ASSERT_EQ(tensile.size(), natural.size());
    size_t mismatches = 0;
    for(size_t i = 0; i < natural.size(); ++i)
    {
        if(std::abs(tensile[i] - natural[i]) > 1e-3f * (1.f + std::abs(natural[i])))
        {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "Tensile on swizzled scales differs from rocRoller";

    for(const auto& storage :
        {MxStorage{true, true, false, true}, {true, true, true, false}, {true, true, false, false}})
    {
        const auto actual
            = execute(_handle, storage, inputs, false, MxScaleFormat::SWIZZLED_GFX950);
        EXPECT_EQ(actual.format, MxScaleFormat::SWIZZLED_GFX950) << describe(storage);
        EXPECT_TRUE(actual.c == swizzled.c)
            << describe(storage) << " gives a different C than " << describe(reference);
    }
}
