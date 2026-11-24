# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Correctness check and timed benchmark of a pooler (GEMM + broadcast bias) and unpooler (GEMM)
# on the run-ahead TDM pipeline. TDM exists only on gfx125x, so the target is built for those
# archs only. One TU per A/B layout and epilogue so the instances compile in parallel.
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_POOLER_UNPOOLER tile_example_gemm_pooler_unpooler_bench)
  set(_pooler_sources
      gemm_pooler_unpooler_bench.cpp
      instances_rcr_tdm.cpp
      instances_rcr_tdmstaged.cpp
      instances_rrr_tdm.cpp
      instances_rrr_tdmstaged.cpp
      instances_crr_tdm.cpp
      instances_crr_tdmstaged.cpp)
  set_source_files_properties(${_pooler_sources} PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_POOLER_UNPOOLER} ${_pooler_sources})
  set(_pooler_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _pooler_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_POOLER_UNPOOLER} PROPERTY HIP_ARCHITECTURES ${_pooler_archs})
  # Re-enable the post-RA machine scheduler that the top-level CMakeLists turns off
  # (-enable-post-misched=0; the later option wins), as example 54 does. Without it the LDS
  # fragment reads of tiles with 6 or 11 M iterations per wave drain before almost every WMMA.
  target_compile_options(${EXAMPLE_POOLER_UNPOOLER} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()
