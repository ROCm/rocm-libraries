# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# MX GEMM on the comp_tdm_scheduled pipeline: named instances, correctness check and timing
# (gfx125x only).
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_COMP_TDM_SCHEDULED_MX tile_example_gemm_comp_tdm_scheduled_mx_bench)
  set_source_files_properties(gemm_comp_tdm_scheduled_mx_bench.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_COMP_TDM_SCHEDULED_MX} gemm_comp_tdm_scheduled_mx_bench.cpp)
  set(_comp_tdm_scheduled_mx_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _comp_tdm_scheduled_mx_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_COMP_TDM_SCHEDULED_MX} PROPERTY HIP_ARCHITECTURES ${_comp_tdm_scheduled_mx_archs})
  # rocKE's comgr options: its schedule is in the source, so the machine schedulers stay off.
  target_compile_options(${EXAMPLE_COMP_TDM_SCHEDULED_MX} PRIVATE
      "SHELL:-mllvm -amdgpu-expert-scheduling-mode" "SHELL:-mllvm -enable-misched=false"
      "SHELL:-mllvm -enable-post-misched=false"
      "SHELL:-mllvm -greedy-regclass-priority-trumps-globalness")
  if(CK_USE_OCP_FP8)
    target_compile_options(${EXAMPLE_COMP_TDM_SCHEDULED_MX} PRIVATE -DCK_TILE_USE_OCP_FP8)
  endif()
endif()
