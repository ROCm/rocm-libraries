###############################################################################
#
# MIT License
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(HIPTENSOR_ELEMENTWISE_SOURCES
   ${CMAKE_CURRENT_SOURCE_DIR}/device/hiptensor_elementwise_scale_instances.hpp
   ${CMAKE_CURRENT_SOURCE_DIR}/device/instance_params.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_elementwise_permute.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_elementwise_binary.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_elementwise_trinary.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank2_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank2_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank2_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank3_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank3_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank3_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank4_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank4_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank4_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank5_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank5_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank5_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank6_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank6_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_binary_solution_rank6_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank2_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank2_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank2_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank3_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank3_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank3_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank4_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank4_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank4_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank5_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank5_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank5_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank6_double_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank6_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_trinary_solution_rank6_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_cpu_reference_rank2_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_cpu_reference_rank3_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_cpu_reference_rank4_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_cpu_reference_rank5_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_cpu_reference_rank6_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank2_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank2_float_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank2_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank2_half_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank3_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank3_float_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank3_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank3_half_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank4_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank4_float_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank4_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank4_half_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank5_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank5_float_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank5_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank5_half_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank6_float_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank6_float_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank6_half_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/instances/elementwise_permute_solution_rank6_half_noop_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_cpu_reference.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_cpu_reference_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_instance_selection.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_solution.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_solution_instances.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_solution_registry.cpp
)

add_hiptensor_component(hiptensor_elementwise ${HIPTENSOR_ELEMENTWISE_SOURCES})
