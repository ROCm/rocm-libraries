# MIT License
#
# Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Create meta targets that build all examples for a single configuration:
foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)
  set(config_meta_target ${config_prefix}.examples)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

## cub_add_example
#
# Add an example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for post-processing target information per-backend.
# example_name: The name of the example minus "<config_prefix>.example." For
#   instance, examples/vector.cu will be "vector", and examples/cuda/copy.cu
#   would be "cuda.copy".
# example_src: The source file that implements the example.
# cub_target: The reference cub target with configuration information.
#
function(add_hipcub_example EXAMPLE_NAME EXAMPLE_SOURCES)
  list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
  if(USE_HIPCXX)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE HIP)
  endif()

  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
  target_include_directories(${EXAMPLE_TARGET} SYSTEM BEFORE
    PUBLIC
      $<BUILD_INTERFACE:${GEXAMPLE_INCLUDE_DIRS}>
  )
  target_link_libraries(${EXAMPLE_TARGET}
    PRIVATE
      ${GEXAMPLE_BOTH_LIBRARIES}
      hipcub
  )

  if(HIP_COMPILER STREQUAL "nvcc")
    set_property(TARGET ${EXAMPLE_TARGET} PROPERTY CUDA_STANDARD 17)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
        hipcub_cub
    )
  endif()

  set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example/hipcub"
  )
  add_test(
    NAME ${EXAMPLE_NAME}
    COMMAND ${EXAMPLE_TARGET}
  )
  if (WIN32 AND NOT DEFINED DLLS_COPIED)
    set(DLLS_COPIED "YES")
    set(DLLS_COPIED ${DLLS_COPIED} PARENT_SCOPE)
    # for now adding in all .dll as dependency chain is not cmake based on win32
    file( GLOB third_party_dlls
    LIST_DIRECTORIES ON
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    )
    foreach( file_i ${third_party_dlls})
      add_custom_command( TARGET ${EXAMPLE_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/examples )
    endforeach( file_i )
  endif()   
endfunction()

add_subdirectory(block)
add_subdirectory(device)
