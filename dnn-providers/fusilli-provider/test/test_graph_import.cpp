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
