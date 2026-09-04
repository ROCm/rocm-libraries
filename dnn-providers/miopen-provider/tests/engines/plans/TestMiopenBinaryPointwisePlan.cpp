// Copyright © Advanced Micro Devices, Inc., or its affiliates.
  // SPDX-License-Identifier:  MIT

  #include "engines/plans/MiopenBinaryPointwisePlan.hpp"
  #include <gtest/gtest.h>
  #include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
  #include <hipdnn_plugin_sdk/PluginException.hpp>
  #include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

  using namespace miopen_plugin;
  using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

  TEST(TestMiopenBinaryPointwisePlan, InitializesFromValidGraphForAdd)
  {
      GTEST_SKIP() << "Requires createValidBinaryPointwiseGraph in FlatbufferGraphTestUtils.hpp";
  }

  TEST(TestMiopenBinaryPointwisePlan, InitializesFromValidGraphForSub)
  {
      GTEST_SKIP() << "Requires createValidBinaryPointwiseGraph in FlatbufferGraphTestUtils.hpp";
  }

  TEST(TestMiopenBinaryPointwisePlan, InitializesFromValidGraphForMul)
  {
      GTEST_SKIP() << "Requires createValidBinaryPointwiseGraph in FlatbufferGraphTestUtils.hpp";
  }

  TEST(TestMiopenBinaryPointwisePlan, ThrowsWhenSecondOperandTensorUidMissing)
  {
      GTEST_SKIP() << "Requires createValidBinaryPointwiseGraph in FlatbufferGraphTestUtils.hpp";
  }

  TEST(TestMiopenBinaryPointwisePlan, GetWorkspaceSizeIsAlwaysZero)
  {
      GTEST_SKIP() << "Requires createValidBinaryPointwiseGraph in FlatbufferGraphTestUtils.hpp";
  }