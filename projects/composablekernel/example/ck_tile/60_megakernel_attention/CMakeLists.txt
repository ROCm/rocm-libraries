# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Split-KV decode / verify attention of the target model (hdim 64, GQA 8, paged bf16 / fp8 KV) as
# a ck_tile::Megakernel phase, plus a host-only check of its work decomposition.
# Not wired into example/ck_tile/CMakeLists.txt yet: add_subdirectory(60_megakernel_attention).
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_ATTENTION tile_example_megakernel_attention)
  set_source_files_properties(megakernel_attention.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_ATTENTION} megakernel_attention.cpp)
  set(_megakernel_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_ATTENTION} PROPERTY HIP_ARCHITECTURES ${_megakernel_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_ATTENTION} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")

  # Host-only: makes no HIP calls (the ck_tile headers still need the HIP language).
  set(EXAMPLE_MEGAKERNEL_ATTENTION_CPU tile_example_megakernel_attention_cpu_check)
  set_source_files_properties(megakernel_attention_cpu_check.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_ATTENTION_CPU} megakernel_attention_cpu_check.cpp)
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_ATTENTION_CPU} PROPERTY HIP_ARCHITECTURES ${_megakernel_archs})
endif()
