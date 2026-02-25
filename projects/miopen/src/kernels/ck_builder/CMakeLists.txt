# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(cmake/ShardInstantiation.cmake)

# Grouped Conv2D Forward - DL (4 files)
set(GROUPED_CONV2D_FWD_DL_SOURCES
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
)

# Grouped Conv2D Forward - WMMA Base (16 files)
set(GROUPED_CONV2D_FWD_WMMA_SOURCES
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_oddc_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_oddc_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_oddc_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_oddc_instance.cpp
)

# Grouped Conv2D Forward - XDL Comp (15 files)
set(GROUPED_CONV2D_FWD_XDL_COMP_SOURCES
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_2x_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_part2_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_2x_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_part2_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_2x_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_part2_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_comp_instance.cpp
)

# Grouped Conv2D Forward - XDL Large Tensor (5 files)
set(GROUPED_CONV2D_FWD_XDL_LARGE_TENSOR_SOURCES
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_int8_instance.cpp
)

# Grouped Conv2D Forward - XDL Merged Groups (8 files)
set(GROUPED_CONV2D_FWD_XDL_MERGED_GROUPS_SOURCES
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_int8_instance.cpp
)

# Grouped Conv2D Forward - WMMA CShuffleV3 (10 files)
set(GROUPED_CONV2D_FWD_WMMA_CSHUFFLEV3_SOURCES
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part1.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part2.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part3.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part4.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part1.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part2.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part3.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part4.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_f16_instance.cpp
)

# Grouped Conv2D Forward - WMMA Large Tensor (2 files)
set(GROUPED_CONV2D_FWD_WMMA_LARGE_TENSOR_SOURCES
    factories/grouped_conv2d_fwd/wmma/large_tensor/device_grouped_conv2d_fwd_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/wmma/large_tensor/device_grouped_conv2d_fwd_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
)

# Grouped Conv2D Forward - XDL Base (bf16/f16 sharded separately)
set(GROUPED_CONV2D_FWD_XDL_SOURCES
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_instance.cpp
)

# Grouped Conv2D Forward - XDL MEM (14 files, int8 inter/intra sharded separately)
set(GROUPED_CONV2D_FWD_XDL_MEM_SOURCES
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_direct_load_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_direct_load_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_intra_instance.cpp
)

# Sharded instantiations for large instance files
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

generate_sharded_instantiations(
    INSTANCES_NAME device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_instances
    TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_instance.in
    NUM_SHARDS 16
    SRC_LIST GROUPED_CONV2D_FWD_XDL_SHARDED_SOURCES
    OUTPUT_DIR ${GENERATED_DIR}/xdl
)

generate_sharded_instantiations(
    INSTANCES_NAME device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instances
    TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instance.in
    NUM_SHARDS 16
    SRC_LIST GROUPED_CONV2D_FWD_XDL_SHARDED_SOURCES
    OUTPUT_DIR ${GENERATED_DIR}/xdl
)

generate_sharded_instantiations(
    INSTANCES_NAME device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_instances
    TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_instance.in
    NUM_SHARDS 16
    SRC_LIST GROUPED_CONV2D_FWD_XDL_SHARDED_SOURCES
    OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

generate_sharded_instantiations(
    INSTANCES_NAME device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_inter_instances
    TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_inter_instance.in
    NUM_SHARDS 21
    SRC_LIST GROUPED_CONV2D_FWD_XDL_SHARDED_SOURCES
    OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

generate_sharded_instantiations(
    INSTANCES_NAME device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_intra_instances
    TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_intra_instance.in
    NUM_SHARDS 21
    SRC_LIST GROUPED_CONV2D_FWD_XDL_SHARDED_SOURCES
    OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

add_library(ck_builder STATIC
    static_tests/grouped_conv_2d_fwd_xdl_f32.cpp
    static_tests/grouped_conv_2d_fwd_xdl_v3_f32.cpp
    static_tests/grouped_conv_2d_fwd_xdl_large_tensor_f32.cpp
    static_tests/grouped_conv_2d_fwd_wmma_f32.cpp
    static_tests/grouped_conv_2d_fwd_dl_f32.cpp
    static_tests/grouped_conv_fwd_xdl_test.cpp

    # Grouped conv2d fwd sources
    ${GROUPED_CONV2D_FWD_DL_SOURCES}
    ${GROUPED_CONV2D_FWD_WMMA_SOURCES}
    ${GROUPED_CONV2D_FWD_WMMA_CSHUFFLEV3_SOURCES}
    ${GROUPED_CONV2D_FWD_WMMA_LARGE_TENSOR_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_COMP_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_LARGE_TENSOR_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_MEM_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_MERGED_GROUPS_SOURCES}

    # Sharded sources (generated at configure time)
    ${GROUPED_CONV2D_FWD_XDL_SHARDED_SOURCES}
)
set_target_properties(ck_builder PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ck_builder PRIVATE -DCK_EXPERIMENTAL_BUILDER)
target_include_directories(ck_builder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(composable_kernel 1.0.0 COMPONENTS device_conv_operations)
target_link_libraries(ck_builder composable_kernel::device_conv_operations)

# CK template instantiations can produce fold expressions exceeding the default bracket-depth limit of 256
target_compile_options(ck_builder PRIVATE $<$<CXX_COMPILER_ID:Clang>:-fbracket-depth=2048>)

# This static library contains a large amount of GPU code that will benefit from offload-compress if it's available
check_cxx_compiler_flag("--offload-compress -x hip" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
if(CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    target_compile_options(ck_builder PRIVATE --offload-compress)
endif()