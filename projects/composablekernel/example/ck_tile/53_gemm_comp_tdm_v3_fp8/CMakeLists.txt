# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Timed benchmark of the comp_tdm_v3 fp8 GEMM pipeline. TDM and the split workgroup barrier
# exist only on gfx125x, so the target is built for those archs only.
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_COMP_TDM_V3_FP8 tile_example_gemm_comp_tdm_v3_fp8_bench)
  set_source_files_properties(gemm_comp_tdm_v3_fp8_bench.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_COMP_TDM_V3_FP8} gemm_comp_tdm_v3_fp8_bench.cpp)
  set(_comp_tdm_v3_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _comp_tdm_v3_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_COMP_TDM_V3_FP8} PROPERTY HIP_ARCHITECTURES ${_comp_tdm_v3_archs})
  set(EXAMPLE_COMP_TDM_V3_FP8_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)
  if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_COMP_TDM_V3_FP8_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
  endif()
  target_compile_options(${EXAMPLE_COMP_TDM_V3_FP8} PRIVATE ${EXAMPLE_COMP_TDM_V3_FP8_COMPILE_OPTIONS})
endif()
