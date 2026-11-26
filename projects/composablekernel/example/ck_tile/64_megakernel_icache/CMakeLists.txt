# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Example 58's GEMM chain megakernel launched through an entry point that warms the kernel's own
# code first (ck_tile/ops/megakernel/icache.hpp). The phases run comp_tdm_v3, which exists only on
# gfx125x.
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_ICACHE tile_example_megakernel_icache)
  set_source_files_properties(megakernel_icache.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_ICACHE} megakernel_icache.cpp)
  set(_megakernel_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_ICACHE} PROPERTY HIP_ARCHITECTURES ${_megakernel_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_ICACHE} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()
