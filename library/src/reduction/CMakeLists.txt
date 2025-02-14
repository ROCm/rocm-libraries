###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

# Don't link to consume ck instances, however will customize and create our own.
# Make the ck includes visible so we can build instances.
get_target_property(composable_kernel_INCLUDES composable_kernel::device_reduction_operations INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(composable_kernel_INCLUDES composable_kernel::device_other_operations INTERFACE_INCLUDE_DIRECTORIES)
set(HIPTENSOR_REDUCTION_SOURCES
   ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_reduction.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_cpu_reference.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_cpu_reference_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_1_1_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_2_1_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_2_2_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_1_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_2_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_3_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_1_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_2_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_3_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_4_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_1_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_2_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_3_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_4_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_5_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_1_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_2_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_3_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_4_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_5_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_6_f16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_1_1_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_2_1_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_2_2_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_1_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_2_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_3_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_1_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_2_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_3_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_4_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_1_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_2_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_3_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_4_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_5_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_1_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_2_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_3_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_4_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_5_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_6_bf16_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_1_1_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_2_1_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_2_2_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_1_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_2_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_3_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_1_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_2_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_3_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_4_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_1_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_2_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_3_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_4_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_5_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_1_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_2_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_3_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_4_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_5_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_6_f32_f32_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_1_1_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_2_1_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_2_2_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_1_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_2_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_3_3_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_1_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_2_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_3_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_4_4_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_1_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_2_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_3_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_4_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_5_5_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_1_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_2_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_3_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_4_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_5_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_6_6_f64_f64_instance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/reduction_solution_registry.cpp
)

add_hiptensor_component(hiptensor_reduction ${HIPTENSOR_REDUCTION_SOURCES})
target_include_directories(hiptensor_reduction PRIVATE ${composable_kernel_INCLUDES})
