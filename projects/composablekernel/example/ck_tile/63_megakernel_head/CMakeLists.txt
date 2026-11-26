# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The target model's head (final RMSNorm, lm_head GEMM with fused softcap + argmax, DFlash2
# accept) in one persistent ck_tile::Megakernel launch. The GEMM runs comp_tdm_v3, which exists
# only on gfx125x.
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_HEAD tile_example_megakernel_head)
  set_source_files_properties(megakernel_head.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_HEAD} megakernel_head.cpp)
  set(_megakernel_head_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_head_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_HEAD} PROPERTY HIP_ARCHITECTURES ${_megakernel_head_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_HEAD} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()
