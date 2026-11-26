# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Two chained GEMMs (the target model's pooler and unpooler at decode M) in one persistent
# ck_tile::Megakernel launch. The phases run comp_tdm_v3, which exists only on gfx125x.
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_GEMM_CHAIN tile_example_megakernel_gemm_chain)
  set_source_files_properties(megakernel_gemm_chain.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_GEMM_CHAIN} megakernel_gemm_chain.cpp)
  set(_megakernel_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_GEMM_CHAIN} PROPERTY HIP_ARCHITECTURES ${_megakernel_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_GEMM_CHAIN} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()
