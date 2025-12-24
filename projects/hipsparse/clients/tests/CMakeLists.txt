# ########################################################################
# Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

find_package(GTest REQUIRED)


#
# Client matrices.
#
if(NOT EXISTS "${CMAKE_MATRICES_DIR}")
  #
  # Download.
  #
  set(CMAKE_MATRICES_DIR ${PROJECT_BINARY_DIR}/matrices CACHE STRING "Matrices directory.")
  file(MAKE_DIRECTORY ${CMAKE_MATRICES_DIR})

  if(NOT TARGET hipsparse)
    set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/../deps/convert.cpp CACHE STRING "Convert tool mtx2csr.")
    include(${CMAKE_SOURCE_DIR}/../cmake/ClientMatrices.cmake)
  else()
   set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/deps/convert.cpp CACHE STRING "Convert tool mtx2csr.")
    include(${CMAKE_SOURCE_DIR}/cmake/ClientMatrices.cmake)
  endif()

else()
  #
  # Copy.
  #
  if(NOT CMAKE_MATRICES_DIR STREQUAL "${PROJECT_BINARY_DIR}/matrices")
    message("Copy matrix files from ${CMAKE_MATRICES_DIR} to ${PROJECT_BINARY_DIR}/matrices")

    file(COPY ${CMAKE_MATRICES_DIR}/ DESTINATION ${PROJECT_BINARY_DIR}/matrices)
  
    if(STATUS AND NOT STATUS EQUAL 0)
      message(FATAL_ERROR "Failed to copy matrix .bin files, aborting.")
    endif()
  endif()

endif()

set(HIPSPARSE_CLIENTS_COMMON
  ../common/arg_check.cpp
  ../common/unit.cpp
  ../common/utility.cpp
  ../common/hipsparse_parse_data.cpp
  ../common/hipsparse_template_specialization.cpp
)

# Include sources
if(NOT USE_CUDA)
  include(rocm/CMakeLists.txt)
else()
  if(CUDA_VERSION VERSION_EQUAL "12.8")
    include(cuda/12.8/CMakeLists.txt)
    message("Building hipsparse-test executable using CUDA_VERSION ${CUDA_VERSION}")
  else()
    message("ERROR: Testing on CUDA_VERSION ${CUDA_VERSION} not supported.")
  endif()
endif()

add_executable(hipsparse-test hipsparse_gtest_main.cpp ${HIPSPARSE_TEST_SOURCES} ${HIPSPARSE_CLIENTS_COMMON})

# Set GOOGLE_TEST definition
target_compile_definitions(hipsparse-test PRIVATE GOOGLE_TEST)

# Target compile options
target_compile_options(hipsparse-test PRIVATE -Wno-unused-command-line-argument -Wall)

# Internal common header
target_include_directories(hipsparse-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
target_include_directories(hipsparse-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

if (BUILD_CLIENTS_ONLY)
  target_include_directories(hipsparse-test PRIVATE ${hipsparse_INCLUDE_DIRS}/hipsparse)
endif()

# Target link libraries
target_link_libraries(hipsparse-test PRIVATE GTest::gtest roc::hipsparse)

# Link against stdc++fs for std::experimental::filesystem support
if(NOT WIN32)
  target_link_libraries(hipsparse-test PRIVATE stdc++fs)
endif()

# Add OpenMP if available
if(OPENMP_FOUND AND THREADS_FOUND)
  target_link_libraries(hipsparse-test PRIVATE OpenMP::OpenMP_CXX ${OpenMP_CXX_FLAGS})
endif()

if(NOT USE_CUDA)
  target_link_libraries(hipsparse-test PRIVATE hip::host)
else()
  target_compile_definitions(hipsparse-test PRIVATE __HIP_PLATFORM_NVIDIA__)
  target_include_directories(hipsparse-test PRIVATE ${HIP_INCLUDE_DIRS})
  target_link_libraries(hipsparse-test PRIVATE ${CUDA_LIBRARIES})
endif()

set_target_properties(hipsparse-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS hipsparse-test COMPONENT tests)

# Prepare testing data
set(HIPSPARSE_TEST_DATA "${PROJECT_BINARY_DIR}/staging/hipsparse_test.data")
add_custom_command(OUTPUT "${HIPSPARSE_TEST_DATA}"
                   COMMAND ${Python3_EXECUTABLE} ../common/hipsparse_gentest.py -m ${PROJECT_BINARY_DIR}/matrices -I ../include ${HIPSPARSE_TEST_YAML_INCLUDE} -o "${HIPSPARSE_TEST_DATA}"
                   DEPENDS ../common/hipsparse_gentest.py ${HIPSPARSE_TEST_YAML_INCLUDE} ../include/hipsparse_common.yaml ${HIPSPARSE_TEST_YAMLS}
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(hipsparse-test-data
  DEPENDS "${HIPSPARSE_TEST_DATA}" )

add_dependencies(hipsparse-test  hipsparse-test-data hipsparse-common)

rocm_install(FILES ${HIPSPARSE_TEST_DATA} DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/test" COMPONENT tests)

if (WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    $ENV{HIP_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/hipinfo.exe
    ${ROCSPARSE_PATH}/bin/rocsparse.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET hipsparse-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )

endif()

add_test(hipsparse-test hipsparse-test)
