# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Cross-GPU megakernel phases (fence-free Lamport all-reduce / all-gather over XGMI peer memory,
# include/ck_tile/ops/megakernel/megakernel_xgpu.hpp) in persistent ck_tile::Megakernel launches,
# one process driving every visible GPU (or one GPU as a self loop).
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_MEGAKERNEL_XGPU tile_example_megakernel_xgpu)
  set_source_files_properties(megakernel_xgpu.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_MEGAKERNEL_XGPU} megakernel_xgpu.cpp)
  set(_megakernel_xgpu_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _megakernel_xgpu_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_MEGAKERNEL_XGPU} PROPERTY HIP_ARCHITECTURES ${_megakernel_xgpu_archs})
  target_compile_options(${EXAMPLE_MEGAKERNEL_XGPU} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
endif()

# Host-only: the exchange protocol (the header's packet code) driven by CPU threads as ranks and
# workgroups, with torn 16-byte packets, replayed launches and sequence wrap-around. No GPU.
add_executable(tile_example_megakernel_xgpu_cpu_check xgpu_cpu_check.cpp)
