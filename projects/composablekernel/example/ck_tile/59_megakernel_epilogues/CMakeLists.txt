# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The target model's fused epilogues (qkv row ops, attention gate, residual norms, expert activation
# quantisation) as phases of one ck_tile::Megakernel launch, after a comp_tdm_v3 qkv GEMM, which
# exists only on gfx125x. The host-only self-check of the CPU reference builds everywhere.
add_executable(tile_example_megakernel_epilogues_ref_selfcheck epilogue_ref_selfcheck.cpp)

if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_EPILOGUES tile_example_megakernel_epilogues)
  set_source_files_properties(megakernel_epilogues.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_EPILOGUES} megakernel_epilogues.cpp)
  set(_megakernel_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_EPILOGUES} PROPERTY HIP_ARCHITECTURES ${_megakernel_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_EPILOGUES} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()
