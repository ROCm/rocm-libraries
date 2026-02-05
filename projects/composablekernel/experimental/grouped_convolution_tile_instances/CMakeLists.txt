# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
  # Generate instances using python script (empty to just generate empty instance list)
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/grouped_convolution_forward_tile_ndhwgc_fp32.inc)
    find_package(Python3 COMPONENTS Interpreter Development)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_instances.py --mode=tests
      RESULT_VARIABLE ret
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error
    )

    if(NOT ret EQUAL 0)
      message(FATAL_ERROR "Failed to generate instances. Return code: ${ret}\nOutput: ${output}\nError: ${error}")
    endif()
  endif()

  # Find cpp files and create lib for instances
  file(GLOB_RECURSE GROUPED_CONV_FWD_TILE "instances/*.cpp")
  add_instance_library(device_grouped_conv_fwd_tile_instances ${GROUPED_CONV_FWD_TILE})
  target_include_directories(device_grouped_conv_fwd_tile_instances PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/builder/test/utils")
endif()
