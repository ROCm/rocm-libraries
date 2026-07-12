###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################

if (NOT ROCWMMA_SAMPLES_INCLUDE_DIR)
  set(ROCWMMA_SAMPLES_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
set(ROCWMMA_COMMUNITY_SAMPLES_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Custom target to build all community samples
add_custom_target(rocwmma_community_samples)

# Function to add community sample with dependency tracking
# Community samples use a separate install component to distinguish them from official samples
function(add_community_sample TARGET_NAME SOURCE_FILE)
  list(APPEND SOURCE_FILE ${ARGN})
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
  target_link_libraries(${TARGET_NAME} ${ROCWMMA_OPENMP_LIBRARY})
  if(UNIX)
    set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${ROCWMMA_OPENMP_INSTALL_RPATH}")
    set_target_properties(${TARGET_NAME} PROPERTIES BUILD_RPATH "${ROCWMMA_OPENMP_BUILD_RPATH}")
    target_link_libraries(${TARGET_NAME} "-fno-rtlib-add-rpath")
  endif()
  target_link_libraries(${TARGET_NAME} rocwmma hiprtc::hiprtc)
  target_include_directories(${TARGET_NAME} PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}
                             ${ROCWMMA_SAMPLES_INCLUDE_DIR}
                             ${ROCWMMA_COMMUNITY_SAMPLES_INCLUDE_DIR})
  add_dependencies(rocwmma_community_samples ${TARGET_NAME})

  # Add support to build the target's assembly files
  if(ROCWMMA_BUILD_ASSEMBLY)
    foreach(file_name ${SOURCE_FILE})
      file(RELATIVE_PATH relative_file_path "${CMAKE_CURRENT_SOURCE_DIR}" "${file_name}")
      string(REPLACE "../" "__/" relative_file_string "${relative_file_path}")
      add_custom_command(TARGET ${TARGET_NAME}
                         POST_BUILD
                         COMMAND make ARGS ${relative_file_string}.s
                         COMMAND ${CMAKE_COMMAND} -E copy
                           "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET_NAME}.dir/${relative_file_string}.s"
                           "${CMAKE_CURRENT_BINARY_DIR}/assembly/${relative_file_string}.s"
                          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
  endif()

  # Install to separate component to distinguish from official samples
  rocm_install_targets(
    TARGETS ${TARGET_NAME}
    COMPONENT community-samples
  )
endfunction()

# Community samples will be added here
# Use template.cpp as a starting point for new samples
#
# Example:
# add_community_sample(example_name ${CMAKE_CURRENT_SOURCE_DIR}/example_name.cpp)
#
# Note: template.cpp is not built as a sample - it's only a reference template

add_community_sample(simple_gemm_swiglu ${CMAKE_CURRENT_SOURCE_DIR}/simple_gemm_swiglu.cpp)
add_community_sample(simple_fusion_gemm ${CMAKE_CURRENT_SOURCE_DIR}/simple_fusion_gemm.cpp)
add_community_sample(simple_gemm_act ${CMAKE_CURRENT_SOURCE_DIR}/simple_gemm_act.cpp)
add_community_sample(simple_lora_adapter_fusion ${CMAKE_CURRENT_SOURCE_DIR}/simple_lora_adapter_fusion.cpp)
