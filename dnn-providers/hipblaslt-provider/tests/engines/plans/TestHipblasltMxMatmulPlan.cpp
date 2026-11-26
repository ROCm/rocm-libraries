// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <initializer_list>
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
#include "engines/plans/MxTestUtils.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using hipdnn_plugin_sdk::archMatches;
using hipdnn_plugin_sdk::ArchMatchMode;

namespace
{

using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipblaslt_plugin_test::createMxGraphWithLayout;
using hipblaslt_plugin_test::DevicePtr;
using hipblaslt_plugin_test::MxOperandLayout;
using hipblaslt_plugin_test::upload;
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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSA),
              HIPBLAS_OP_T);
    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSB),
              HIPBLAS_OP_T);
}

// K-contiguous A (row-major) and B (column-major) reach hipBLAS as TN, the
// layout frameworks use and the only one with Tensile MX kernels.
TEST(TestMxMatmulParams, TransposeInferenceKContiguous)
{
    auto fb = createMxGraphWithLayout(MxOperandLayout::K_K);
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

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
    auto fb = createMxGraphWithLayout(MxOperandLayout::K_K);
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    for(const auto format :
        {MxScaleFormat::NATURAL, MxScaleFormat::SWIZZLED_GFX950, MxScaleFormat::SWIZZLED_GFX1250})
    {
        const MxMatmulParams params(
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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
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
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());

    EXPECT_NO_THROW(MxMatmulPlan(_handle, std::move(params)));
}

// execute() must reject a null workspace when the plan needs one, as this graph's
// does (our A's K-contiguous scale is transposed into it). The check runs before
// any device-buffer lookup, so no buffers are passed.
TEST_F(TestGpuMxMatmulPlan, ExecuteThrowsOnNullWorkspace)
{
    auto fb = createValidMxMatmulGraph();
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());
    const MxMatmulPlan plan(_handle, std::move(params));
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
    auto fb = createMxGraphWithLayout(MxOperandLayout::K_K);
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());
    const MxMatmulPlan plan(_handle, std::move(params));

    EXPECT_TRUE(plan.aScalePrep().isPassThrough());
    EXPECT_TRUE(plan.bScalePrep().isPassThrough());
}

// The default (M/N-contiguous) graph with row-major scales: only our A's
// K-contiguous scale is transposed into its operand's M-contiguous order.
TEST_F(TestGpuMxMatmulPlan, DefaultLayoutTransposesOnlyTheAScale)
{
    auto fb = createValidMxMatmulGraph();
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());
    const MxMatmulPlan plan(_handle, std::move(params));

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

// A layout other than hipBLAS TN cannot use the gfx950 swizzled format (the relayout
// kernel needs a K-contiguous operand), so the plan falls back to rocRoller with
// natural scales.
TEST_F(TestGpuMxMatmulPlan, Gfx950SwizzledFormatFallsBackToRocRoller)
{
    if(!onGfx950())
    {
        GTEST_SKIP() << "gfx950-only check";
    }
    auto fb = createValidMxMatmulGraph();
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    const auto plan = createMxMatmulPlan(_handle,
                                         attrs.deqA,
                                         attrs.deqB,
                                         attrs.matmul,
                                         graph.getTensorMap(),
                                         MxScaleFormat::SWIZZLED_GFX950);

    EXPECT_EQ(plan->scaleFormat(), MxScaleFormat::NATURAL);
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
    auto fb = createMxGraphWithLayout(MxOperandLayout::K_K);
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
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

// M, N and K of an executed GEMM, and how many leading dims of 1 each tensor has
// (the in-tree bundles use rank-3 and rank-4 tensors with batch 1).
struct MxShape
{
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t leadingOnes = 0;

    int64_t kBlocks() const
    {
        return k / VEC32_BLOCK_SIZE;
    }
};

constexpr MxShape EXEC_SHAPE{64, 96, 256};
// Small M and N with a large K: hipBLASLt asks for a workspace of its own.
constexpr MxShape LARGE_K_SHAPE{16, 16, 8192};
// What C holds before execute(), far above the tolerance of the host reference.
constexpr float INITIAL_C = 1e6f;

// Physical order of our A [M, K], our B [K, N] and their scales [M, K/32] and
// [K/32, N]; true means K (or the K-block axis) is contiguous.
struct MxStorage
{
    bool aKContiguous;
    bool bKContiguous;
    bool scaleAKContiguous;
    bool scaleBKContiguous;
};

// Dims and packed strides of a [rows, cols] matrix behind `leadingOnes` dims of 1.
struct MatrixLayout
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

MatrixLayout matrix(int64_t rows, int64_t cols, bool rowMajor, int64_t leadingOnes)
{
    MatrixLayout layout{std::vector<int64_t>(static_cast<size_t>(leadingOnes), 1),
                        std::vector<int64_t>(static_cast<size_t>(leadingOnes), rows * cols)};
    layout.dims.insert(layout.dims.end(), {rows, cols});
    layout.strides.insert(layout.strides.end(),
                          rowMajor ? std::initializer_list<int64_t>{cols, 1}
                                   : std::initializer_list<int64_t>{1, rows});
    return layout;
}

flatbuffers::FlatBufferBuilder createExecuteGraph(const MxStorage& s, const MxShape& shape)
{
    const auto ones = shape.leadingOnes;
    const auto a = matrix(shape.m, shape.k, s.aKContiguous, ones);
    const auto b = matrix(shape.k, shape.n, !s.bKContiguous, ones);
    const auto c = matrix(shape.m, shape.n, true, ones);
    const auto scaleA = matrix(shape.m, shape.kBlocks(), s.scaleAKContiguous, ones);
    const auto scaleB = matrix(shape.kBlocks(), shape.n, !s.scaleBKContiguous, ones);
    return createValidMxMatmulGraph(a.dims,
                                    a.strides,
                                    b.dims,
                                    b.strides,
                                    c.dims,
                                    c.strides,
                                    scaleA.dims,
                                    scaleB.dims,
                                    DT::FP8_E4M3,
                                    DT::FLOAT,
                                    DT::FP8_E8M0,
                                    DT::FP8_E8M0,
                                    DT::FLOAT,
                                    static_cast<int32_t>(VEC32_BLOCK_SIZE),
                                    false,
                                    false,
                                    DT::UNSET,
                                    scaleA.strides,
                                    scaleB.strides);
}

// Logical row-major [rows][cols] bytes laid out row- or column-major.
std::vector<uint8_t>
    inMemoryOrder(const std::vector<uint8_t>& logical, int64_t rows, int64_t cols, bool rowMajor)
{
    if(rowMajor)
    {
        return logical;
    }
    return hipblaslt_plugin_test::transposeHost(
        logical, static_cast<size_t>(rows), static_cast<size_t>(cols));
}

// Logical row-major contents: A [M][K], B [K][N], scale_a [M][K/32], scale_b [K/32][N].
struct MxInputs
{
    MxShape shape;
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    std::vector<uint8_t> scaleA;
    std::vector<uint8_t> scaleB;
};

MxInputs randomInputs(const MxShape& shape = EXEC_SHAPE)
{
    std::mt19937 rng(20261028);
    // Finite E4M3 values (exponent field <= 12, either sign) and E8M0 scales near 1,
    // so every output element is a finite, nonzero-heavy sum.
    const auto fp8
        = [&rng]() { return static_cast<uint8_t>((rng() % 0x68) | ((rng() & 1u) << 7)); };
    const auto e8m0 = [&rng]() { return static_cast<uint8_t>(122 + rng() % 11); };
    MxInputs in;
    in.shape = shape;
    in.a.resize(static_cast<size_t>(shape.m * shape.k));
    in.b.resize(static_cast<size_t>(shape.k * shape.n));
    in.scaleA.resize(static_cast<size_t>(shape.m * shape.kBlocks()));
    in.scaleB.resize(static_cast<size_t>(shape.kBlocks() * shape.n));
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

struct MxExecution
{
    MxScaleFormat format = MxScaleFormat::NATURAL;
    size_t workspaceSize = 0;
    bool scaleAPrepared = false; // our A's scale (hipBLAS B_SCALE)
    bool scaleBPrepared = false; // our B's scale (hipBLAS A_SCALE)
    size_t preparedScaleBytes = 0; // the two scale regions at the front of the workspace
    std::vector<uint8_t> preparedWhileHeld; // those regions read while the stream was held
    std::vector<uint8_t> c;
};

// Holds a stream back with a host function until release(), so nothing enqueued on it
// after the hold can run before then.
class StreamHold
{
public:
    explicit StreamHold(hipStream_t stream)
        : _stream(stream)
    {
        EXPECT_EQ(hipLaunchHostFunc(stream, &StreamHold::wait, &_released), hipSuccess);
    }
    StreamHold(const StreamHold&) = delete;
    StreamHold& operator=(const StreamHold&) = delete;
    StreamHold(StreamHold&&) = delete;
    StreamHold& operator=(StreamHold&&) = delete;
    // The host function reads _released, so it must finish before the members go,
    // also when execute() threw before release().
    ~StreamHold()
    {
        release();
        EXPECT_EQ(hipStreamSynchronize(_stream), hipSuccess);
    }

    void release()
    {
        if(!_done)
        {
            _promise.set_value();
            _done = true;
        }
    }

private:
    static void wait(void* released)
    {
        static_cast<std::shared_future<void>*>(released)->wait();
    }

    hipStream_t _stream;
    std::promise<void> _promise;
    std::shared_future<void> _released = _promise.get_future().share();
    bool _done = false;
};

// A non-blocking stream, so the default stream never waits for it.
class NonBlockingStream
{
public:
    NonBlockingStream()
    {
        EXPECT_EQ(hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking), hipSuccess);
    }
    NonBlockingStream(const NonBlockingStream&) = delete;
    NonBlockingStream& operator=(const NonBlockingStream&) = delete;
    NonBlockingStream(NonBlockingStream&&) = delete;
    NonBlockingStream& operator=(NonBlockingStream&&) = delete;
    ~NonBlockingStream()
    {
        static_cast<void>(hipStreamDestroy(_stream));
    }

    hipStream_t get() const
    {
        return _stream;
    }

private:
    hipStream_t _stream = nullptr;
};

// `format` defaults to the one the provider selects for the device.
std::unique_ptr<MxMatmulPlan> createExecutePlan(const HipdnnEnginePluginHandle& handle,
                                                const MxStorage& s,
                                                std::optional<MxScaleFormat> format = std::nullopt,
                                                const MxShape& shape = EXEC_SHAPE)
{
    auto fb = createExecuteGraph(s, shape);
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);
    return createMxMatmulPlan(handle,
                              attrs.deqA,
                              attrs.deqB,
                              attrs.matmul,
                              graph.getTensorMap(),
                              format.value_or(selectMxScaleFormat(handle)));
}

// `nullWorkspace` passes no workspace, as hipDNN does for a plan that needs none.
// `holdStream` holds the handle's stream (which must not be the default stream) while
// execute() enqueues its work and reads the prepared-scale regions before releasing it.
MxExecution execute(const HipdnnEnginePluginHandle& handle,
                    const MxStorage& s,
                    const MxInputs& in,
                    bool nullWorkspace = false,
                    std::optional<MxScaleFormat> format = std::nullopt,
                    bool holdStream = false)
{
    const auto m = in.shape.m;
    const auto n = in.shape.n;
    const auto k = in.shape.k;
    const auto kBlocks = in.shape.kBlocks();
    auto fb = createExecuteGraph(s, in.shape);
    const GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);
    const auto plan = createExecutePlan(handle, s, format, in.shape);

    MxExecution result;
    result.format = plan->scaleFormat();
    result.workspaceSize = plan->getWorkspaceSize(handle);
    result.scaleAPrepared = !plan->bScalePrep().isPassThrough();
    result.scaleBPrepared = !plan->aScalePrep().isPassThrough();
    result.preparedScaleBytes
        = plan->aScalePrep().workspaceSize() + plan->bScalePrep().workspaceSize();

    auto a = upload(inMemoryOrder(in.a, m, k, s.aKContiguous));
    auto b = upload(inMemoryOrder(in.b, k, n, !s.bKContiguous));
    auto scaleA = upload(inMemoryOrder(in.scaleA, m, kBlocks, s.scaleAKContiguous));
    auto scaleB = upload(inMemoryOrder(in.scaleB, kBlocks, n, !s.scaleBKContiguous));
    // C starts far from any result, so a matmul that adds to it (beta != 0) changes C.
    std::vector<uint8_t> initialC(static_cast<size_t>(m * n) * sizeof(float));
    const std::vector<float> initialValues(static_cast<size_t>(m * n), INITIAL_C);
    std::memcpy(initialC.data(), initialValues.data(), initialC.size());
    auto c = upload(initialC);
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
    std::optional<StreamHold> hold;
    if(holdStream && handle.getStream() == nullptr)
    {
        ADD_FAILURE() << "holdStream needs a non-default stream: holding the default one deadlocks";
    }
    else if(holdStream)
    {
        hold.emplace(handle.getStream());
    }
    plan->execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
    if(hold)
    {
        result.preparedWhileHeld.resize(result.preparedScaleBytes);
        EXPECT_EQ(hipMemcpy(result.preparedWhileHeld.data(),
                            workspace.get(),
                            result.preparedWhileHeld.size(),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        hold->release();
    }
    EXPECT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);

    result.c.resize(static_cast<size_t>(m * n) * sizeof(float));
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

// OCP FP8 E4M3: bias 7, no infinities; randomInputs() produces no NaN.
double decodeE4M3(uint8_t bits)
{
    const int exponent = (bits >> 3) & 0xF;
    const int mantissa = bits & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return (bits & 0x80) != 0 ? -magnitude : magnitude;
}

// Expects C to equal dequant(A) x dequant(B), each E8M0 scale being 2^(value - 127),
// computed on the host in double. On gfx950 hipBLASLt's C differs from it by up to
// about 1.7e-4 of the sum of the products' magnitudes; the tolerance is 1e-3 of it.
void expectMatchesHostReference(const MxExecution& run, const MxInputs& in, const std::string& name)
{
    const auto m = in.shape.m;
    const auto n = in.shape.n;
    const auto k = in.shape.k;
    const auto kBlocks = in.shape.kBlocks();
    const auto c = asFloats(run.c);
    ASSERT_EQ(c.size(), static_cast<size_t>(m * n)) << name;
    size_t mismatches = 0;
    for(int64_t row = 0; row < m; ++row)
    {
        for(int64_t col = 0; col < n; ++col)
        {
            double sum = 0.0;
            double magnitude = 0.0;
            for(int64_t kk = 0; kk < k; ++kk)
            {
                const auto block = kk / static_cast<int64_t>(VEC32_BLOCK_SIZE);
                const double a
                    = std::ldexp(decodeE4M3(in.a[static_cast<size_t>(row * k + kk)]),
                                 in.scaleA[static_cast<size_t>(row * kBlocks + block)] - 127);
                const double b = std::ldexp(decodeE4M3(in.b[static_cast<size_t>(kk * n + col)]),
                                            in.scaleB[static_cast<size_t>(block * n + col)] - 127);
                sum += a * b;
                magnitude += std::abs(a * b);
            }
            const auto actual = static_cast<double>(c[static_cast<size_t>(row * n + col)]);
            // A NaN in C fails `<=`, so it counts as a mismatch (it would pass a `>` test).
            const bool close = std::abs(actual - sum) <= 1e-3 * magnitude + 1e-6;
            mismatches += close ? 0 : 1;
        }
    }
    EXPECT_EQ(mismatches, 0u) << name << ": C differs from the host reference";
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
        expectMatchesHostReference(expected, inputs, describe(reference));

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

// K-contiguous A and B (hipBLAS TN, every MX arch). With rocRoller the plan transposes
// a scale in the other order (one hipblasLtMatrixTransform); on gfx1250 the relayout
// kernel swizzles every scale from whichever order it has, both in one launch.
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
        GTEST_SKIP() << "the gfx1250 scale format needs K-contiguous operands (#13715)";
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
        GTEST_SKIP() << "the gfx1250 scale format needs K-contiguous operands (#13715)";
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

// The workspace is [A scale prep | B scale prep | hipBLASLt workspace]. With a large K
// hipBLASLt asks for a workspace of its own, so the matmul must use the region after
// the prepared scales: placed over them, or left out, C changes. A format for which
// hipBLASLt asks for no workspace is recorded and passed over; the test skips only
// when that is so for every format.
TEST_F(TestGpuMxMatmulPlanExecute, PreparedScalesAndHipblasltWorkspaceDoNotOverlap)
{
    const auto inputs = randomInputs(LARGE_K_SHAPE);
    std::vector<std::optional<MxScaleFormat>> formats{std::nullopt};
    if(onGfx950())
    {
        formats.emplace_back(MxScaleFormat::SWIZZLED_GFX950);
    }
    std::string withoutWorkspace;
    size_t checked = 0;
    for(const auto& format : formats)
    {
        const auto expected = execute(_handle, {true, true, true, true}, inputs, false, format);
        const auto actual = execute(_handle, {true, true, false, false}, inputs, false, format);
        const auto name = toString(actual.format);
        ASSERT_GT(actual.preparedScaleBytes, 0u) << name;
        if(actual.workspaceSize <= actual.preparedScaleBytes)
        {
            withoutWorkspace += withoutWorkspace.empty() ? "" : ", ";
            withoutWorkspace += name;
            continue;
        }
        ++checked;
        EXPECT_TRUE(actual.c == expected.c)
            << name << ": prepared scales next to a hipBLASLt workspace give a different C";
    }
    if(!withoutWorkspace.empty())
    {
        RecordProperty("formats_without_hipblaslt_workspace", withoutWorkspace);
    }
    if(checked == 0)
    {
        GTEST_SKIP() << "hipBLASLt asks for no workspace for " << LARGE_K_SHAPE.m << "x"
                     << LARGE_K_SHAPE.n << "x" << LARGE_K_SHAPE.k << " (" << withoutWorkspace
                     << ")";
    }
}

// The scale preparation (transpose or relayout) is enqueued on the handle's stream,
// ordered after the work that wrote the scales and before the matmul that reads them.
// With the stream held back, nothing may have been written to the workspace yet.
TEST_F(TestGpuMxMatmulPlanExecute, PreparesScalesOnTheHandleStream)
{
    const MxStorage bothScalesPrepared{true, true, false, false};
    const auto inputs = randomInputs();
    std::vector<std::optional<MxScaleFormat>> formats{std::nullopt};
    if(onGfx950())
    {
        formats.emplace_back(MxScaleFormat::SWIZZLED_GFX950);
    }
    for(const auto& format : formats)
    {
        const auto expected = execute(_handle, bothScalesPrepared, inputs, false, format);

        const NonBlockingStream stream;
        _handle.setStream(stream.get());
        const auto actual = execute(_handle, bothScalesPrepared, inputs, false, format, true);
        _handle.setStream(nullptr);

        const auto name = toString(actual.format);
        ASSERT_GT(actual.preparedScaleBytes, 0u) << name;
        EXPECT_EQ(actual.preparedWhileHeld, std::vector<uint8_t>(actual.preparedScaleBytes, 0xA5))
            << name << ": a scale was prepared before the handle's stream ran";
        EXPECT_TRUE(actual.c == expected.c) << name << ": C differs on a non-default stream";
    }
}

// Tensors of rank 3 and 4 with leading dims of 1, as the in-tree bundles have, are the
// same GEMM: M, N, K and the scale extents come from the last two dims.
TEST_F(TestGpuMxMatmulPlanExecute, LeadingUnitDimsGiveTheSameResult)
{
    std::vector<MxStorage> storages{{true, true, false, false}};
    if(onGfx950())
    {
        storages.push_back({false, false, true, true});
    }
    const auto inputs = randomInputs();
    for(const auto& storage : storages)
    {
        const auto expected = execute(_handle, storage, inputs);
        for(const int64_t leadingOnes : {1, 2})
        {
            auto batched = inputs;
            batched.shape.leadingOnes = leadingOnes;
            EXPECT_TRUE(execute(_handle, storage, batched).c == expected.c)
                << describe(storage) << " with " << leadingOnes << " leading dims of 1";
        }
    }
}

// A scale stored in the other order is transposed by the plan for natural scales and
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
// is swizzled into the same bytes, so C matches the host reference and is bitwise
// equal across them.
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
    if(swizzled.format != MxScaleFormat::SWIZZLED_GFX950)
    {
        GTEST_SKIP() << "hipBLASLt selected no gfx950 Tensile kernel for this GEMM";
    }
    expectMatchesHostReference(swizzled, inputs, describe(reference));

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
