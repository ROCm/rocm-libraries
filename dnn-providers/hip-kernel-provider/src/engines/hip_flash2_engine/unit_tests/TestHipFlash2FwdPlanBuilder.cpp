// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for HipFlash2FwdPlanBuilder.

#include <gtest/gtest.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/hip_flash2_engine/HipFlash2FwdPlanBuilder.hpp"

namespace hip_flash2_engine {
namespace {

class TestHipFlash2FwdPlanBuilder : public ::testing::Test {
   protected:
    Handle _handle;
    HipFlash2FwdPlanBuilder _builder;
};

// ── isApplicable: valid cases ─────────────────────────────────────────────────

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsFP16MHACausal) {
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if (arch != "gfx942" && arch != "gfx950") GTEST_SKIP();

    const std::vector<int64_t> dims{1, 32, 4096, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims, strides, dims, strides, dims, strides, dims, strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        /*causal=*/true);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsFP16MHANonCausal) {
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if (arch != "gfx942" && arch != "gfx950") GTEST_SKIP();

    const std::vector<int64_t> dims{1, 32, 2048, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims, strides, dims, strides, dims, strides, dims, strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        /*causal=*/false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsFP16HeadDim64) {
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if (arch != "gfx942" && arch != "gfx950") GTEST_SKIP();

    const std::vector<int64_t> dims{1, 32, 2048, 64};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims, strides, dims, strides, dims, strides, dims, strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// GQA: different Q heads vs KV heads
TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsFP16GQA) {
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if (arch != "gfx942" && arch != "gfx950") GTEST_SKIP();

    const std::vector<int64_t> qDims{1, 32, 4096, 128};  // 32 query heads
    const std::vector<int64_t> kvDims{1, 8, 4096, 128};  // 8 KV heads (GQA ratio=4)
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(qDims);
    const auto kvStrides = hipdnn_data_sdk::utilities::generateStrides(kvDims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims, qStrides, kvDims, kvStrides, kvDims, kvStrides, qDims, qStrides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        /*causal=*/true);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// ── isApplicable: rejection cases ────────────────────────────────────────────

TEST_F(TestHipFlash2FwdPlanBuilder, RejectsBF16) {
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> dims{1, 32, 2048, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims, strides, dims, strides, dims, strides, dims, strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipFlash2FwdPlanBuilder, RejectsUnsupportedHeadDim256) {
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if (arch != "gfx942" && arch != "gfx950") GTEST_SKIP();

    const std::vector<int64_t> dims{1, 32, 2048, 256};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims, strides, dims, strides, dims, strides, dims, strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipFlash2FwdPlanBuilder, RejectsShortSequenceDecodeLength) {
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if (arch != "gfx942" && arch != "gfx950") GTEST_SKIP();

    // seq_q=1 means decode — should use batched GEMM, not Flash2
    const std::vector<int64_t> qDims{1, 32, 1, 128};
    const std::vector<int64_t> kvDims{1, 32, 2048, 128};
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(qDims);
    const auto kvStrides = hipdnn_data_sdk::utilities::generateStrides(kvDims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims, qStrides, kvDims, kvStrides, kvDims, kvStrides, qDims, qStrides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

}  // namespace
}  // namespace hip_flash2_engine
