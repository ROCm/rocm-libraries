// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/HipblasltMxMatmulPlan.hpp"
#include "engines/plans/MxArchSupport.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

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

// ===========================================================================
// MxMatmulPlan (GPU — requires a device and a live hipblasLt handle)
// ===========================================================================

class TestGpuMxMatmulPlan : public ::testing::Test
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

// execute() must reject a null workspace: MX GEMM always reports a non-zero
// workspace size (the reserved scale-transpose region), so a null pointer is a
// caller contract violation. The check runs before any device-buffer access, so
// no real buffers are needed here.
TEST_F(TestGpuMxMatmulPlan, ExecuteThrowsOnNullWorkspace)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    const auto attrs = getMxNodeAttrs(graph);

    MxMatmulParams params(attrs.deqA, attrs.deqB, attrs.matmul, graph.getTensorMap());
    MxMatmulPlan const plan(_handle, std::move(params));

    EXPECT_THROW(plan.execute(_handle, nullptr, 0, nullptr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
