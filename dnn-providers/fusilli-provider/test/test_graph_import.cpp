// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "graph_import.h"
#include "utils.h"

#include <fusilli.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>

namespace {

// Build a minimal valid conv graph with no name field set (nullptr).
flatbuffers::FlatBufferBuilder buildUnnamedValidGraph() {
  namespace sdk = hipdnn_data_sdk::data_objects;

  flatbuffers::FlatBufferBuilder builder;

  std::vector<int64_t> xDims = {1, 1, 2, 2};
  std::vector<int64_t> xStrides = {4, 4, 2, 1};
  std::vector<int64_t> wDims = {1, 1, 1, 1};
  std::vector<int64_t> wStrides = {1, 1, 1, 1};
  std::vector<int64_t> yDims = {1, 1, 2, 2};
  std::vector<int64_t> yStrides = {4, 4, 2, 1};

  std::vector<flatbuffers::Offset<sdk::TensorAttributes>> tensors;
  tensors.push_back(sdk::CreateTensorAttributesDirect(
      builder, 1, "x", sdk::DataType::FLOAT, &xStrides, &xDims));
  tensors.push_back(sdk::CreateTensorAttributesDirect(
      builder, 2, "w", sdk::DataType::FLOAT, &wStrides, &wDims));
  tensors.push_back(sdk::CreateTensorAttributesDirect(
      builder, 3, "y", sdk::DataType::FLOAT, &yStrides, &yDims));

  std::vector<int64_t> padding = {0, 0};
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> dilation = {1, 1};
  auto convAttr = sdk::CreateConvolutionFwdAttributesDirect(
      builder, 1, 2, 3, &padding, &padding, &stride, &dilation,
      sdk::ConvMode::CROSS_CORRELATION);

  std::vector<flatbuffers::Offset<sdk::Node>> nodes;
  nodes.push_back(sdk::CreateNodeDirect(
      builder, "conv", sdk::DataType::FLOAT,
      sdk::NodeAttributes::ConvolutionFwdAttributes, convAttr.Union()));

  auto graph =
      sdk::CreateGraphDirect(builder, nullptr, // no name
                             sdk::DataType::FLOAT, sdk::DataType::FLOAT,
                             sdk::DataType::FLOAT, &tensors, &nodes);
  builder.Finish(graph);
  return builder;
}

} // namespace

TEST(TestGraphImport, ConvertHipDnnToFusilli) {
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto halfDt, hipDnnDataTypeToFusilliDataType(
                       hipdnn_data_sdk::data_objects::DataType::HALF));
  EXPECT_EQ(halfDt, fusilli::DataType::Half);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto bfloat16Dt, hipDnnDataTypeToFusilliDataType(
                           hipdnn_data_sdk::data_objects::DataType::BFLOAT16));
  EXPECT_EQ(bfloat16Dt, fusilli::DataType::BFloat16);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto floatDt, hipDnnDataTypeToFusilliDataType(
                        hipdnn_data_sdk::data_objects::DataType::FLOAT));
  EXPECT_EQ(floatDt, fusilli::DataType::Float);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto doubleDt, hipDnnDataTypeToFusilliDataType(
                         hipdnn_data_sdk::data_objects::DataType::DOUBLE));
  EXPECT_EQ(doubleDt, fusilli::DataType::Double);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto uint8Dt, hipDnnDataTypeToFusilliDataType(
                        hipdnn_data_sdk::data_objects::DataType::UINT8));
  EXPECT_EQ(uint8Dt, fusilli::DataType::Uint8);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto int8Dt, hipDnnDataTypeToFusilliDataType(
                       hipdnn_data_sdk::data_objects::DataType::INT8));
  EXPECT_EQ(int8Dt, fusilli::DataType::Int8);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto int32Dt, hipDnnDataTypeToFusilliDataType(
                        hipdnn_data_sdk::data_objects::DataType::INT32));
  EXPECT_EQ(int32Dt, fusilli::DataType::Int32);
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(
      auto unsetDt, hipDnnDataTypeToFusilliDataType(
                        hipdnn_data_sdk::data_objects::DataType::UNSET));
  EXPECT_EQ(unsetDt, fusilli::DataType::NotSet);

  auto invalidResult = hipDnnDataTypeToFusilliDataType(
      static_cast<hipdnn_data_sdk::data_objects::DataType>(42));
  EXPECT_TRUE(isError(invalidResult));
}

TEST(TestGraphImport, UnnamedGraphGetsHashBasedName) {
  auto builder = buildUnnamedValidGraph();

  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(auto ctx, importGraph(&opGraph));
  // Graph name should be a hash fallback, not empty or null.
  std::string name = ctx.graph.getName();
  EXPECT_TRUE(name.starts_with("hipdnn_"));
  EXPECT_EQ(name.size(), 7 + 16); // "hipdnn_" + 16 hex digits
}
