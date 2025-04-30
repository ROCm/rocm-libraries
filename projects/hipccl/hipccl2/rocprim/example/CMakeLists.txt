# MIT License
#
# Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

function(add_rocprim_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)

  if(USE_HIPCXX)
    set_source_files_properties(${EXAMPLE_SOURCE} PROPERTIES LANGUAGES HIP)
  endif()

  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

  target_link_libraries(${EXAMPLE_TARGET}
    PRIVATE
    rocprim
  )
  if(USE_HIPCXX)
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
      $<IF:$<LINK_LANGUAGE:HIP>,hip::host,hip::device>
    )
  else()
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
      hip::device)
  endif()
  set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
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
      add_custom_command( TARGET ${EXAMPLE_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/example )
    endforeach( file_i )
  endif()
endfunction()

# ****************************************************************************
# rocPRIM examples
# ****************************************************************************

add_rocprim_example(example_temporary_storage.cpp)
add_rocprim_example(example_type_traits_interface.cpp)
