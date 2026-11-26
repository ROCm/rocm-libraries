# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The target model's MoE experts (MXFP4 weights, MX fp8 activations: gate/up + SwiGLU + requant,
# down, post-expert norm + weighted combine) as three phases of one persistent ck_tile::Megakernel
# launch. The GEMV uses v_wmma_scale_f32_16x16x128_f8f6f4, which exists only on gfx125x.
# Not wired into the parent CMakeLists.txt yet: add_subdirectory(62_megakernel_moe) to build it.
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_MOE tile_example_megakernel_moe)
  set_source_files_properties(megakernel_moe.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_MOE} megakernel_moe.cpp)
  set(_megakernel_moe_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_moe_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_MOE} PROPERTY HIP_ARCHITECTURES ${_megakernel_moe_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_MOE} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()
