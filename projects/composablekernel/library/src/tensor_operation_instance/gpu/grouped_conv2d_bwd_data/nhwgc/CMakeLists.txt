# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NHWGC layout instances for grouped_conv2d_bwd_data
set(GROUPED_CONV2D_BWD_DATA_NHWGC
   # wmma
   wmma/device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instance.cpp
   wmma/device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   wmma/device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instance.cpp
   wmma/device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_i8_instance.cpp
   wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_bf16_16_16_instance.cpp
   wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_f16_16_16_instance.cpp
   wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   # xdl
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_bf16_16_16_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_bf16_optimized_loads_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f16_16_16_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f16_optimized_loads_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_16_16_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_optimized_loads_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_16_16_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_optimized_loads_instance.cpp
   # xdl_v3
   xdl/device_grouped_conv2d_bwd_data_xdl_v3_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_v3_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_v3_nhwgc_gkyxc_nhwgk_bf16_large_tensors_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_v3_nhwgc_gkyxc_nhwgk_f16_large_tensors_instance.cpp
   xdl/device_grouped_conv2d_bwd_data_xdl_v3_nhwgc_gkyxc_nhwgk_f32_large_tensors_instance.cpp
)

# WORKAROUND (ROCm 7.0.2 toolchain): clang segfaults in the AMDGPU
# "SI Form memory clauses" backend pass when compiling the f32 and f32_tf32
# optimized_loads grouped conv bwd-data instances for gfx908 at -O2 or higher.
# gfx908 is the only affected target; gfx90a/942/950 and all gfx11xx/12xx
# compile clean. Building just these files at -O1 avoids the crash. The guard
# keeps full optimization for every configuration that does not target gfx908.
if(GPU_TARGETS MATCHES "gfx908")
    set_source_files_properties(
        xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_optimized_loads_instance.cpp
        xdl/device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_optimized_loads_instance.cpp
        PROPERTIES COMPILE_OPTIONS "-O1")
endif()

add_instance_library(device_grouped_conv2d_bwd_data_nhwgc_instance ${GROUPED_CONV2D_BWD_DATA_NHWGC})
