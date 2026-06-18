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

# Internal unit tests link rocFFT object libraries directly so they can
# exercise non-exported functionality (e.g. function_pool) without going
# through the public API.  GTest must be pre-installed; if it is not
# found, the suite is skipped rather than downloaded.
find_package( GTest QUIET )
if( NOT GTest_FOUND )
  message( WARNING
    "GTest not found - skipping rocfft-internal-test "
    "(ROCFFT_BUILD_INTERNAL_TESTS requires a pre-installed GTest)." )
  return()
endif()

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

# Link the function-pool object library directly for the generated
# function_pool_data() definition.  Do NOT also link roc::rocfft - it
# already absorbs rocfft-function-pool, which would duplicate the
# function_pool_data singleton.
target_link_libraries( rocfft-internal-test PRIVATE
  rocfft-function-pool
  hip::host
  GTest::gtest
  GTest::gtest_main
)

# Register with ctest.  gtest_discover_tests adds one ctest entry per
# TEST, so e.g. "ctest -R function_pool" works.
include( GoogleTest )
gtest_discover_tests( rocfft-internal-test )

# Opt-in: also run the tests automatically when the executable is built,
# so a test failure fails the build.  ROCFFT_INTERNAL_TESTS_FILTER (if set)
# restricts which tests run via GoogleTest's --gtest_filter.
if( ROCFFT_RUN_INTERNAL_TESTS )
  set( internal_tests_run_args "" )
  if( ROCFFT_INTERNAL_TESTS_FILTER )
    set( internal_tests_run_args --gtest_filter=${ROCFFT_INTERNAL_TESTS_FILTER} )
  endif()
  add_custom_command( TARGET rocfft-internal-test POST_BUILD
    COMMAND rocfft-internal-test ${internal_tests_run_args}
    COMMENT "Running internal tests..."
    VERBATIM )
endif()
