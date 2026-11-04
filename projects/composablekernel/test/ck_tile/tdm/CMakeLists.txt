# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx125")
   add_gtest_executable(test_tdm_basic tdm_basic.cpp)
   target_link_libraries(test_tdm_basic PRIVATE utility)
   # The cluster cases call the cluster overload of make_kernel. That overload is declared only
   # when compiling for gfx125, so build for the gfx125 GPU targets only.
   ck_tile_restrict_archs(test_tdm_basic "^gfx125")

   add_gtest_executable(test_tdm_clip_zero_fill tdm_clip_zero_fill.cpp)

   add_gtest_executable(test_tdm_descriptor_stride tdm_descriptor_stride.cpp)
   target_link_libraries(test_tdm_descriptor_stride PRIVATE utility)

   add_gtest_executable(test_tdm_descriptor_element_size tdm_descriptor_element_size.cpp)

   # Host-only: which kernels must be launched as a cluster, the cluster overload of make_kernel
   # rejecting a single-workgroup cluster for them, and how TDMConfig takes its workgroup mask.
   # Nothing is launched. That overload is declared only when compiling for gfx125, so build for
   # the gfx125 GPU targets only.
   add_gtest_executable(test_tdm_cluster_launch_guard tdm_cluster_launch_guard.cpp)
   ck_tile_restrict_archs(test_tdm_cluster_launch_guard "^gfx125")
endif()
