# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Weight-streaming bf16 GEMM at decode M (ck_tile::SmallMGemmPhase) in a persistent
# ck_tile::Megakernel launch: one GEMM, or the target model's pooler -> unpooler chain with the
# cross-phase weight prefetch on and off. Uses gfx1250 async global->LDS loads and bf16 WMMA.
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_SMALL_M_GEMM tile_example_megakernel_small_m_gemm)
  set_source_files_properties(megakernel_small_m_gemm.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_SMALL_M_GEMM} megakernel_small_m_gemm.cpp)
  set(_megakernel_small_m_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_small_m_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_SMALL_M_GEMM} PROPERTY HIP_ARCHITECTURES ${_megakernel_small_m_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_SMALL_M_GEMM} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()
