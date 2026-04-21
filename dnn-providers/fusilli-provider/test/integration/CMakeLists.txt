# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Threads REQUIRED)

# Integration tests
add_fusilli_plugin_test(
  NAME fusilli_plugin_plugin_load_test
  SRCS plugin_load.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
    FUSILLI_PLUGIN_TARGET="${FUSILLI_PLUGIN_TARGET}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_conv_fprop_parameterized_full_test
  SRCS convolution/conv_fprop_parameterized_full.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_conv_fprop_with_relu_and_bias_test
  SRCS convolution/simple_conv_fprop_with_relu_and_bias.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_conv_fprop_parameterized_stream_device_test
  SRCS convolution/simple_conv_fprop_parameterized_stream_device.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_matmul_test
  SRCS matmul/simple_matmul.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_matmul_parameterized_test
  SRCS matmul/matmul_parameterized.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_batched_matmul_parameterized_test
  SRCS matmul/batched_matmul_parameterized.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_batched_matmul_int4_fp16_test
  SRCS matmul/simple_batched_matmul_int4_fp16.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_scaled_matmul_accumulate_test
  SRCS matmul/simple_scaled_matmul_accumulate.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_pointwise_test
  SRCS pointwise/simple_pointwise.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_conv_wgrad_test
  SRCS convolution/simple_conv_wgrad.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_conv_dgrad_test
  SRCS convolution/simple_conv_dgrad.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

if(NOT HIPDNN_ENABLE_SDPA)
  message(WARNING
          "hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
          "Skipping Fusilli SDPA integration test. "
          "Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable it.")
else()
  add_fusilli_plugin_test(
    NAME fusilli_plugin_simple_sdpa_test
    SRCS sdpa/simple_sdpa.cpp
    DEPS
      GTest::gtest_main
      hip::host
      hipdnn_frontend
      hipdnn_plugin_sdk
      hipdnn_data_sdk
      hipdnn_test_sdk
      Threads::Threads
    COMPILE_DEFS
      FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
  )
endif()

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_reduction_test
  SRCS reduction/simple_reduction.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_custom_op_test
  SRCS custom_op/simple_custom_op.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_simple_rmsnorm_test
  SRCS rmsnorm/simple_rmsnorm.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
)
