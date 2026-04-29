// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "graph_import.h"
#include "utils.h"

#include <fusilli.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
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

// Build a hipDNN frontend ConvFProp graph and serialize to flatbuffer. When
// name is nullopt, set_name is skipped. The produced FlatBuffer will
// deserialize name to null on the reader side — exercising the null-name code
// path.
static std::vector<uint8_t>
buildConvFwdGraph(std::optional<std::string> name = std::nullopt) {
  using namespace hipdnn_frontend;

  graph::Graph graph;
  if (name.has_value()) {
    graph.set_name(*name);
  }
  graph.set_io_data_type(DataType::FLOAT)
      .set_compute_data_type(DataType::FLOAT)
      .set_intermediate_data_type(DataType::FLOAT);

  auto x = std::make_shared<graph::TensorAttributes>();
  x->set_uid(1)
      .set_name("x")
      .set_data_type(DataType::FLOAT)
      .set_dim({1, 1, 2, 2})
      .set_stride({4, 4, 2, 1});
  auto w = std::make_shared<graph::TensorAttributes>();
  w->set_uid(2)
      .set_name("w")
      .set_data_type(DataType::FLOAT)
      .set_dim({1, 1, 1, 1})
      .set_stride({1, 1, 1, 1});

  graph::ConvFpropAttributes convAttrs;
  convAttrs.set_pre_padding({0, 0})
      .set_post_padding({0, 0})
      .set_stride({1, 1})
      .set_dilation({1, 1});

  auto y = graph.conv_fprop(x, w, convAttrs);
  y->set_uid(3)
      .set_name("y")
      .set_data_type(DataType::FLOAT)
      .set_dim({1, 1, 2, 2})
      .set_stride({4, 4, 2, 1})
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

TEST(TestGraphImport, UnnamedGraphGetsNonceBasedName) {
  auto fb = buildConvFwdGraph(/*name=*/std::nullopt);

  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = fb.data();
  opGraph.size = fb.size();

  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctx, importGraph(&opGraph));
  const std::string &name = ctx.graph.getName();
  EXPECT_TRUE(name.starts_with("hipdnn_"));
  EXPECT_EQ(name.size(), 7u + 16u); // "hipdnn_" + 16 hex digits
}

TEST(TestGraphImport, NamedGraphAppendsNonce) {
  auto fb = buildConvFwdGraph(/*name=*/std::string("my_graph"));

  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = fb.data();
  opGraph.size = fb.size();

  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctx, importGraph(&opGraph));
  const std::string &name = ctx.graph.getName();
  EXPECT_TRUE(name.starts_with("my_graph_"));
  EXPECT_EQ(name.size(), 9u + 16u); // "my_graph_" + 16 hex digits
}

TEST(TestGraphImport, EachInstanceGetsDistinctName) {
  // Two graphs built from byte-identical input must still produce distinct
  // fusilli graph names – disambiguation comes through per-instance nonce. This
  // ensures parallel compile of graphs with the same name can't race on
  // filesystem resources (the fusilli cache is based off graph name).
  auto fbA = buildConvFwdGraph(std::string("shared"));
  auto fbB = buildConvFwdGraph(std::string("shared"));

  hipdnnPluginConstData_t opGraphA = {fbA.data(), fbA.size()};
  hipdnnPluginConstData_t opGraphB = {fbB.data(), fbB.size()};

  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctxA, importGraph(&opGraphA));
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctxB, importGraph(&opGraphB));

  EXPECT_TRUE(ctxA.graph.getName().starts_with("shared_"));
  EXPECT_TRUE(ctxB.graph.getName().starts_with("shared_"));
  EXPECT_NE(ctxA.graph.getName(), ctxB.graph.getName());
}
