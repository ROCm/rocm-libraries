# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Define all source files
set(CK_BUILDER_SOURCES
    factories/grouped_conv_2d_fwd_multiple_abd_f32.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_merged_groups.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_standard.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_16x16.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_comp.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_mem_intra.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f32_mem_inter.cpp
    
    factories/grouped_conv_2d_fwd_multiple_abd_f16.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f16_merged_groups.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f16_standard.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f16_16x16.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f16_comp.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f16_comp_2x.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f16_comp_part2.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f16_mem_intra.cpp
    factories/grouped_conv_2d_fwd_multiple_abd_f16_mem_inter.cpp
)

# Create individual object libraries for each source file, this allows the build system to build them in parallel
set(CK_BUILDER_OBJECT_LIBS)
foreach(source_file ${CK_BUILDER_SOURCES})
    # Extract a unique name from the source file path
    get_filename_component(obj_name ${source_file} NAME_WE)
    set(obj_target "ck_builder_obj_${obj_name}")
    
    # Create object library for this source file
    add_library(${obj_target} OBJECT ${source_file})
    target_compile_definitions(${obj_target} PRIVATE -DCK_EXPERIMENTAL_BUILDER)
    target_include_directories(${obj_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(${obj_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    
    # Find and link composable_kernel
    find_package(composable_kernel 1.0.0 COMPONENTS device_conv_operations)
    target_link_libraries(${obj_target} composable_kernel::device_conv_operations)
    
    # Add to list of object libraries
    list(APPEND CK_BUILDER_OBJECT_LIBS $<TARGET_OBJECTS:${obj_target}>)
endforeach()

# Create the final static library from all object files
add_library(ck_builder STATIC ${CK_BUILDER_OBJECT_LIBS})
target_compile_definitions(ck_builder PRIVATE -DCK_EXPERIMENTAL_BUILDER)
target_include_directories(ck_builder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(ck_builder PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(composable_kernel 1.0.0 COMPONENTS device_conv_operations)
target_link_libraries(ck_builder composable_kernel::device_conv_operations)
