# ########################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc.
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
# ########################################################################

set(ROCSPARSEIO_SOURCES
  ../common/rocsparseio.cpp
#  ../common/rocsparse_importer_matrixmarket.cpp
)

set(ROCSPARSEIO_TOOLS_SOURCES
  rocsparseio-info
  rocsparseio-convert
)

foreach(app ${ROCSPARSEIO_TOOLS_SOURCES})
  add_executable(${app} ${app}.cpp ${ROCSPARSEIO_SOURCES})

  # Internal common header
  target_include_directories(${app} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../common)

  # Target link libraries
  target_link_libraries(${app} PRIVATE roc::rocsparse hip::host ${EXTRA_LIBS})

  # Add OpenMP if available
  if(OPENMP_FOUND)
    if (NOT WIN32)
        target_link_libraries(${app} PRIVATE OpenMP::OpenMP_CXX -Wl,-rpath=${HIP_CLANG_ROOT}/lib)
    else()
        target_link_libraries(${app} PRIVATE libomp)
    endif()
  endif()

  # Set tools output directory
  set_target_properties(${app} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

  # Install
  if(WIN32)
    rocm_install(PROGRAMS $<TARGET_FILE:${app}> DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT clients)
  else()
    rocm_install(PROGRAMS $<TARGET_FILE:${app}> DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocsparse COMPONENT clients)
  endif()

endforeach()
