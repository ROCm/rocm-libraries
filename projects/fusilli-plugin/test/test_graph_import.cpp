// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "graph_import.h"
#include "utils.h"

#include <fusilli.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>

TEST(TestGraphImport, ConvertHipDnnToFusilli) {
  EXPECT_EQ(FUSILLI_PLUGIN_EXPECT_UNWRAP(hipDnnDataTypeToFusilliDataType(
                hipdnn_sdk::data_objects::DataType::HALF)),
            fusilli::DataType::Half);
  EXPECT_EQ(FUSILLI_PLUGIN_EXPECT_UNWRAP(hipDnnDataTypeToFusilliDataType(
                hipdnn_sdk::data_objects::DataType::BFLOAT16)),
            fusilli::DataType::BFloat16);
  EXPECT_EQ(FUSILLI_PLUGIN_EXPECT_UNWRAP(hipDnnDataTypeToFusilliDataType(
                hipdnn_sdk::data_objects::DataType::FLOAT)),
            fusilli::DataType::Float);
  EXPECT_EQ(FUSILLI_PLUGIN_EXPECT_UNWRAP(hipDnnDataTypeToFusilliDataType(
                hipdnn_sdk::data_objects::DataType::DOUBLE)),
            fusilli::DataType::Double);
  EXPECT_EQ(FUSILLI_PLUGIN_EXPECT_UNWRAP(hipDnnDataTypeToFusilliDataType(
                hipdnn_sdk::data_objects::DataType::UINT8)),
            fusilli::DataType::Uint8);
  EXPECT_EQ(FUSILLI_PLUGIN_EXPECT_UNWRAP(hipDnnDataTypeToFusilliDataType(
                hipdnn_sdk::data_objects::DataType::INT32)),
            fusilli::DataType::Int32);
  EXPECT_EQ(FUSILLI_PLUGIN_EXPECT_UNWRAP(hipDnnDataTypeToFusilliDataType(
                hipdnn_sdk::data_objects::DataType::UNSET)),
            fusilli::DataType::NotSet);

  auto invalidResult = hipDnnDataTypeToFusilliDataType(
      static_cast<hipdnn_sdk::data_objects::DataType>(42));
  EXPECT_TRUE(isError(invalidResult));
}
