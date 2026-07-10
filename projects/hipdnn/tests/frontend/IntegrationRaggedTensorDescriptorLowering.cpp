// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/PointwiseConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;

namespace
{

// UID for the auxiliary ragged-offset tensor; distinct from the primary
// pointwise tensor UIDs.
constexpr int64_t K_RAGGED_OFFSET_UID = 1399;

// End-to-end lowering of ragged tensors (RFC 0014): builds a frontend graph,
// lowers it through the REAL backend via
// build_operation_graph_via_descriptors(), retrieves the serialized binary
// graph, and deserializes the flatbuffer schema to assert the per-tensor
// ragged-offset link was actually written into the wire image.
class IntegrationRaggedTensorDescriptorLowering : public IntegrationTestFixture
{
protected:
    // Builds a minimal unary pointwise (RELU) graph. When @p withRaggedOffset is
    // true, the input tensor is given a ragged-offset aux tensor.
    static std::shared_ptr<TestableGraphLowering> makePointwiseGraph(bool withRaggedOffset)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("RaggedLoweringGraph")
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto in0 = std::make_shared<TensorAttributes>();
        in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
        in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

        if(withRaggedOffset)
        {
            auto raggedOffset = std::make_shared<TensorAttributes>();
            raggedOffset->set_uid(K_RAGGED_OFFSET_UID)
                .set_name("RaggedOffset")
                .set_data_type(DataType::INT64)
                .set_dim({2, 1, 1, 1})
                .set_stride({1, 1, 1, 1});
            in0->set_ragged_offset(raggedOffset);
        }

        PointwiseAttributes pwAttrs;
        pwAttrs.set_name("relu_op");
        pwAttrs.set_mode(PointwiseMode::RELU_FWD);

        auto out0 = graph->pointwise(in0, pwAttrs);
        out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

        return graph;
    }
};

} // namespace

// A graph whose input carries a ragged offset must serialize with the
// ragged-offset link recorded on that tensor in the flatbuffer schema.
TEST_F(IntegrationRaggedTensorDescriptorLowering, RaggedOffsetSetsTensorLinkInSchema)
{
    auto graph = makePointwiseGraph(/*withRaggedOffset=*/true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // Sanity: lowering produced a well-formed graph (in0, out0).
    ASSERT_EQ(graphT.tensors.size(), 2u);

    // The primary input tensor must carry the ragged-offset link (by UID) in the
    // serialized schema; this is the sole source of truth for ragged-ness.
    const auto in0It = std::find_if(graphT.tensors.begin(),
                                    graphT.tensors.end(),
                                    [](const auto& t) { return t->uid == K_PW_TENSOR_IN0_UID; });
    ASSERT_NE(in0It, graphT.tensors.end()) << "input tensor IN0 missing from serialized graph";
    ASSERT_TRUE((*in0It)->ragged_offset_tensor_uid.has_value())
        << "IN0 must carry ragged_offset_tensor_uid in the serialized schema.";
    EXPECT_EQ((*in0It)->ragged_offset_tensor_uid.value(), K_RAGGED_OFFSET_UID)
        << "IN0's ragged_offset_tensor_uid must point at the ragged-offset aux tensor.";
}

// A graph with no ragged offsets must serialize with no ragged-offset link on
// any tensor, so ragged-ness derives as false and engine plugins are not
// spuriously gated on ragged-tensor support.
TEST_F(IntegrationRaggedTensorDescriptorLowering, NonRaggedGraphHasNoTensorLinkInSchema)
{
    auto graph = makePointwiseGraph(/*withRaggedOffset=*/false);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    ASSERT_EQ(graphT.tensors.size(), 2u);

    // No tensor may carry a ragged-offset link in a non-ragged graph.
    for(const auto& t : graphT.tensors)
    {
        EXPECT_FALSE(t->ragged_offset_tensor_uid.has_value())
            << "tensor uid " << t->uid
            << " unexpectedly carries ragged_offset_tensor_uid in a non-ragged graph.";
    }
}
