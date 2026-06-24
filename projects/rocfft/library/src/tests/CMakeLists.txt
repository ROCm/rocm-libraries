# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

add_executable( rocfft-internal-test
  function_pool_test.cpp
)

target_include_directories( rocfft-internal-test PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/device>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/shared>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

set_target_properties( rocfft-internal-test PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# Hold the test to the same warning bar as the rest of the library.
target_compile_options( rocfft-internal-test PRIVATE ${WARNING_FLAGS} )

# Link rocfft-function-pool directly (not roc::rocfft, which would
# duplicate the function_pool_data singleton).
target_link_libraries( rocfft-internal-test PRIVATE
  rocfft-function-pool
  hip::host
)

# Use a pre-installed GTest if available, otherwise download and build
# it (gives BUILD_GTEST=ON support and a PIC build that links cleanly).
if( NOT BUILD_GTEST )
  find_package( GTest QUIET )
endif()
if( GTest_FOUND )
  target_link_libraries( rocfft-internal-test PRIVATE GTest::gtest GTest::gtest_main )
else()
  include( ${PROJECT_SOURCE_DIR}/cmake/build-gtest.cmake )
  add_dependencies( rocfft-internal-test gtest )
  target_include_directories( rocfft-internal-test PRIVATE ${GTEST_INCLUDE_DIRS} )
  target_link_libraries( rocfft-internal-test PRIVATE ${GTEST_LIBRARIES} )
endif()

include( GoogleTest )
gtest_discover_tests( rocfft-internal-test )

# Optionally run the tests at build time, failing the build on test
# failure.  run_internal_tests.cmake applies the filter and also fails
# if the filter selects nothing (gtest exits 0 on an empty selection).
if( ROCFFT_RUN_INTERNAL_TESTS )
  add_custom_command( TARGET rocfft-internal-test POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      -DTEST_EXE=$<TARGET_FILE:rocfft-internal-test>
      -DTEST_FILTER=${ROCFFT_INTERNAL_TESTS_FILTER}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/run_internal_tests.cmake
    COMMENT "Running internal tests..."
    VERBATIM )
endif()
