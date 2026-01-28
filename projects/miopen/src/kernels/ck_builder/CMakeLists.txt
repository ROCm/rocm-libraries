# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(ck_builder STATIC 
    factories/grouped_conv_2d_fwd_multiple_abd_f32.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_merged_groups.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_standard.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_16x16.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_comp.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_large_tensor.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_mem_intra.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_mem_inter.cpp
)
set_target_properties(ck_builder PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ck_builder PRIVATE -DCK_EXPERIMENTAL_BUILDER)
target_include_directories(ck_builder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(composable_kernel 1.0.0 COMPONENTS device_conv_operations)
target_link_libraries(ck_builder composable_kernel::device_conv_operations)
