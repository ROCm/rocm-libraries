# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The target model's MoE router (fp32 split-K logits GEMM + sigmoid / bias / top-8 / route table)
# as two phases of one persistent ck_tile::Megakernel launch. The GEMM uses the fp32 WMMA of gfx125x.
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_ROUTER tile_example_megakernel_router)
  set_source_files_properties(megakernel_router.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_ROUTER} megakernel_router.cpp)
  set(_megakernel_router_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_router_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_ROUTER} PROPERTY HIP_ARCHITECTURES ${_megakernel_router_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_ROUTER} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()

# Host-only checks of the reference and of an fp32 emulation of the finalize phase (no GPU).
add_executable(tile_example_megakernel_router_reference_test router_reference_test.cpp)
set_source_files_properties(router_reference_test.cpp PROPERTIES LANGUAGE CXX)
