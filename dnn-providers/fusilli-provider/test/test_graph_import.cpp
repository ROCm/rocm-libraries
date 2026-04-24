// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "graph_import.h"
#include "utils.h"

#include <fusilli.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <optional>
#include <string>
#include <vector>

TEST(TestGraphImport, ConvertHipDnnToFusilli) {
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto halfDt, hipDnnDataTypeToFusilliDataType(
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF));
  EXPECT_EQ(halfDt, fusilli::DataType::Half);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto bfloat16Dt,
      hipDnnDataTypeToFusilliDataType(
          hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16));
  EXPECT_EQ(bfloat16Dt, fusilli::DataType::BFloat16);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto floatDt, hipDnnDataTypeToFusilliDataType(
                        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT));
  EXPECT_EQ(floatDt, fusilli::DataType::Float);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto doubleDt,
      hipDnnDataTypeToFusilliDataType(
          hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE));
  EXPECT_EQ(doubleDt, fusilli::DataType::Double);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto uint8Dt, hipDnnDataTypeToFusilliDataType(
                        hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8));
  EXPECT_EQ(uint8Dt, fusilli::DataType::Uint8);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto int8Dt, hipDnnDataTypeToFusilliDataType(
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT8));
  EXPECT_EQ(int8Dt, fusilli::DataType::Int8);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto int32Dt, hipDnnDataTypeToFusilliDataType(
                        hipdnn_flatbuffers_sdk::data_objects::DataType::INT32));
  EXPECT_EQ(int32Dt, fusilli::DataType::Int32);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto int4Dt, hipDnnDataTypeToFusilliDataType(
                       hipdnn_flatbuffers_sdk::data_objects::DataType::INT4));
  EXPECT_EQ(int4Dt, fusilli::DataType::Int4);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto unsetDt, hipDnnDataTypeToFusilliDataType(
                        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET));
  EXPECT_EQ(unsetDt, fusilli::DataType::NotSet);

  auto invalidResult = hipDnnDataTypeToFusilliDataType(
      static_cast<hipdnn_flatbuffers_sdk::data_objects::DataType>(42));
  EXPECT_TRUE(isError(invalidResult));
}

// Build a hipDNN frontend custom op graph and serialize to flatbuffer.
// The customOpId parameter controls the custom_op_id field.
static std::vector<uint8_t>
buildCustomOpGraph(const std::string &customOpId = "fusilli.my_add") {
  using namespace hipdnn_frontend;

  graph::Graph graph;
  graph.set_name("custom_add_import_test")
      .set_io_data_type(DataType::FLOAT)
      .set_compute_data_type(DataType::FLOAT)
      .set_intermediate_data_type(DataType::FLOAT);

  // Input tensors.
  auto in0 = std::make_shared<graph::TensorAttributes>();
  in0->set_uid(0)
      .set_name("in0")
      .set_data_type(DataType::FLOAT)
      .set_dim({4})
      .set_stride({1});
  auto in1 = std::make_shared<graph::TensorAttributes>();
  in1->set_uid(1)
      .set_name("in1")
      .set_data_type(DataType::FLOAT)
      .set_dim({1})
      .set_stride({1});

  // Opaque data: MLIR add template stored directly as bytes (no JSON).
  std::string mlir = R"(
  func.func private @{FUNC_NAME}(%arg0: {IN0_TYPE},
                                   %arg1: {IN1_TYPE})
                                   -> {OUT0_TYPE} {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : {IN0_TYPE},
          {IN1_TYPE},
          !torch.int
        -> {OUT0_TYPE}
    return %0 : {OUT0_TYPE}
  }
)";
  std::vector<uint8_t> opaqueData(mlir.begin(), mlir.end());

  graph::CustomOpAttributes customAttr;
  customAttr.set_name("my_add")
      .set_custom_op_id(customOpId)
      .set_data(opaqueData);

  auto outputs = graph.custom_op({in0, in1}, 1, customAttr);
  outputs[0]
      ->set_uid(2)
      .set_name("out0")
      .set_data_type(DataType::FLOAT)
      .set_dim({4})
      .set_stride({1})
      .set_output(true);

  auto result = graph.validate();
  if (result.is_bad()) {
    throw std::runtime_error("Graph validation failed: " +
                             result.get_message());
  }

  auto [serializedGraph, serErr] = graph.to_binary();
  if (serErr.is_bad()) {
    throw std::runtime_error("Graph serialization failed: " +
                             serErr.get_message());
  }
  return serializedGraph;
}

TEST(TestGraphImport, ImportCustomOpGraph) {
  auto flatbufferGraph = buildCustomOpGraph();

  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = flatbufferGraph.data();
  opGraph.size = flatbufferGraph.size();

  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctx, importGraph(&opGraph));

  // Should have 3 IO tensors tracked: in0 (uid=0), in1 (uid=1), out (uid=2).
  EXPECT_EQ(ctx.uidToFusilliTensorAttr.size(), 3);
  ASSERT_TRUE(ctx.uidToFusilliTensorAttr.contains(0));
  ASSERT_TRUE(ctx.uidToFusilliTensorAttr.contains(1));
  ASSERT_TRUE(ctx.uidToFusilliTensorAttr.contains(2));

  // Check tensor properties.
  const std::vector<int64_t> in0ExpectedDim = {4};
  const std::vector<int64_t> in1ExpectedDim = {1};
  const std::vector<int64_t> out0ExpectedDim = {4};
  const std::vector<int64_t> expectedStride = {1};

  auto in0 = ctx.uidToFusilliTensorAttr.at(0);
  EXPECT_EQ(in0->getDim(), in0ExpectedDim);
  EXPECT_EQ(in0->getStride(), expectedStride);
  EXPECT_EQ(in0->getDataType(), fusilli::DataType::Float);
  EXPECT_FALSE(in0->isVirtual());

  auto in1 = ctx.uidToFusilliTensorAttr.at(1);
  EXPECT_EQ(in1->getDim(), in1ExpectedDim);
  EXPECT_EQ(in1->getStride(), expectedStride);
  EXPECT_EQ(in1->getDataType(), fusilli::DataType::Float);
  EXPECT_FALSE(in1->isVirtual());

  auto out = ctx.uidToFusilliTensorAttr.at(2);
  EXPECT_EQ(out->getDim(), out0ExpectedDim);
  EXPECT_EQ(out->getStride(), expectedStride);
  EXPECT_EQ(out->getDataType(), fusilli::DataType::Float);
  EXPECT_FALSE(out->isVirtual());

  // Graph properties.
  EXPECT_EQ(ctx.graph.context.getIODataType(), fusilli::DataType::Float);
  EXPECT_EQ(ctx.graph.context.getComputeDataType(), fusilli::DataType::Float);
}

TEST(TestGraphImport, RejectCustomOpWithoutFusilliPrefix) {
  // Build a graph with a custom_op_id that doesn't start with "fusilli."
  auto flatbufferGraph = buildCustomOpGraph("other_plugin.my_add");

  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = flatbufferGraph.data();
  opGraph.size = flatbufferGraph.size();

  auto result = importGraph(&opGraph);
  EXPECT_TRUE(isError(result));
}

// Build a minimal valid ConvFProp graph FlatBuffer directly via the
// hipdnn_flatbuffers_sdk builders. Needed (rather than the hipdnn_frontend
// wrapper) so the graph-name field can be explicitly null — that path
// exercises the hash-only fallback branch of importGraph().
//   name:  nullopt leaves the FlatBuffer name field unset (null pointer on
//          the reader side); a string sets it.
//   xDim:  lets callers vary graph structure so the FlatBuffer hash changes
//          between two otherwise-identical graphs.
static flatbuffers::FlatBufferBuilder
buildMinimalConvGraph(std::optional<std::string> name = std::nullopt,
                      const std::vector<int64_t> &xDim = {1, 1, 2, 2}) {
  namespace sdk = hipdnn_flatbuffers_sdk::data_objects;

  flatbuffers::FlatBufferBuilder builder;

  std::vector<int64_t> xStride(xDim.size(), 1);
  for (int i = static_cast<int>(xDim.size()) - 2; i >= 0; --i)
    xStride[i] = xStride[i + 1] * xDim[i + 1];

  std::vector<int64_t> wDim(xDim.size(), 1);
  std::vector<int64_t> wStride(xDim.size(), 1);
  std::vector<int64_t> yDim = xDim;
  std::vector<int64_t> yStride = xStride;

  std::vector<flatbuffers::Offset<sdk::TensorAttributes>> tensors;
  tensors.push_back(sdk::CreateTensorAttributesDirect(
      builder, /*uid=*/1, "x", sdk::DataType::FLOAT, &xStride, &xDim));
  tensors.push_back(sdk::CreateTensorAttributesDirect(
      builder, /*uid=*/2, "w", sdk::DataType::FLOAT, &wStride, &wDim));
  tensors.push_back(sdk::CreateTensorAttributesDirect(
      builder, /*uid=*/3, "y", sdk::DataType::FLOAT, &yStride, &yDim));

  std::vector<int64_t> padding = {0, 0};
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> dilation = {1, 1};
  auto convAttr = sdk::CreateConvolutionFwdAttributesDirect(
      builder, /*x_uid=*/1, /*w_uid=*/2, /*y_uid=*/3, &padding, &padding,
      &stride, &dilation, sdk::ConvMode::CROSS_CORRELATION);

  std::vector<flatbuffers::Offset<sdk::Node>> nodes;
  nodes.push_back(sdk::CreateNodeDirect(
      builder, "conv", sdk::DataType::FLOAT,
      sdk::NodeAttributes::ConvolutionFwdAttributes, convAttr.Union()));

  auto graph = sdk::CreateGraphDirect(
      builder, name.has_value() ? name->c_str() : nullptr, sdk::DataType::FLOAT,
      sdk::DataType::FLOAT, sdk::DataType::FLOAT, &tensors, &nodes);
  builder.Finish(graph);
  return builder;
}

TEST(TestGraphImport, UnnamedGraphGetsHashBasedName) {
  auto builder = buildMinimalConvGraph(/*name=*/std::nullopt);

  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctx, importGraph(&opGraph));
  const std::string &name = ctx.graph.getName();
  EXPECT_TRUE(name.starts_with("hipdnn_"));
  EXPECT_EQ(name.size(), 7u + 16u); // "hipdnn_" + 16 hex digits
}

TEST(TestGraphImport, NamedGraphAppendsHash) {
  auto builder = buildMinimalConvGraph(/*name=*/std::string("my_graph"));

  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctx, importGraph(&opGraph));
  const std::string &name = ctx.graph.getName();
  EXPECT_TRUE(name.starts_with("my_graph_"));
  EXPECT_EQ(name.size(), 9u + 16u); // "my_graph_" + 16 hex digits
}

TEST(TestGraphImport, DuplicateNamesWithDifferentStructuresGetDifferentKeys) {
  // Two graphs sharing a user-supplied name but differing in input shape must
  // produce distinct fusilli graph names — this is the regression guard for
  // the parallel-test cache-dir race.
  auto builderA =
      buildMinimalConvGraph(std::string("shared"), /*xDim=*/{1, 1, 2, 2});
  auto builderB =
      buildMinimalConvGraph(std::string("shared"), /*xDim=*/{1, 1, 4, 4});

  hipdnnPluginConstData_t opGraphA = {builderA.GetBufferPointer(),
                                      builderA.GetSize()};
  hipdnnPluginConstData_t opGraphB = {builderB.GetBufferPointer(),
                                      builderB.GetSize()};

  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctxA, importGraph(&opGraphA));
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctxB, importGraph(&opGraphB));

  EXPECT_TRUE(ctxA.graph.getName().starts_with("shared_"));
  EXPECT_TRUE(ctxB.graph.getName().starts_with("shared_"));
  EXPECT_NE(ctxA.graph.getName(), ctxB.graph.getName());
}
