# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx11|gfx12")
  set(EXAMPLE_DEPTHWISE_CONV_COMPILE_OPTIONS)
  list(APPEND EXAMPLE_DEPTHWISE_CONV_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

  add_executable(tile_example_depthwise_conv_fwd depthwise_conv_fwd.cpp)
  target_compile_options(tile_example_depthwise_conv_fwd PRIVATE ${EXAMPLE_DEPTHWISE_CONV_COMPILE_OPTIONS})
endif()
