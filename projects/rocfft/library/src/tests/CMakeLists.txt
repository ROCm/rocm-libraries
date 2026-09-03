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

include( ../../../cmake/sqlite.cmake )

# The planning core is compiled into the rocfft shared library rather than an
# object library, so build those sources here to reach them.
set( rocfft_planning_source
  ../assignment_policy.cpp
  ../auxiliary.cpp
  ../callback_map.cpp
  ../chirp.cpp
  ../data_layout.cpp
  ../enum_printer.cpp
  ../exec_info.cpp
  ../fuse_shim.cpp
  ../kargs.cpp
  ../node_factory.cpp
  ../plan.cpp
  ../rccl_wrapper.cpp
  ../repo.cpp
  ../rocfft_stub.cpp
  ../rtc_exports.cpp
  ../transform.cpp
  ../tree_node.cpp
  ../tree_node_1D.cpp
  ../tree_node_2D.cpp
  ../tree_node_3D.cpp
  ../tree_node_bluestein.cpp
  ../tree_node_real.cpp
  ../tuning_kernel_tuner.cpp
  ../tuning_plan_tuner.cpp
  ../twiddles.cpp
)

add_executable( rocfft-internal-test
  data_layout_test.cpp
  function_map_key_test.cpp
  function_pool_test.cpp
  kernel_name_test.cpp
  plan_test.cpp
  repo_test.cpp
  rtc_test.cpp
  solution_map_test.cpp
  ${rocfft_planning_source}
)

# ROCFFT_BUILD_INTERNAL_TEST picks the powX stubs out of rocfft_stub.cpp.
target_compile_definitions( rocfft-internal-test PRIVATE
  ROCFFT_BUILD_INTERNAL_TEST
  ROCFFT_TEST_SOLUTION_MAP_DIR="${ROCFFT_SOLUTION_MAP_DIR}"
)

target_include_directories( rocfft-internal-test PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/device>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/device/generator>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/src/device>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/shared>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

set_target_properties( rocfft-internal-test PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_compile_options( rocfft-internal-test PRIVATE ${WARNING_FLAGS} )

# Link tests directly, not roc::*, to avoid duplication.
target_link_libraries( rocfft-internal-test PRIVATE
  rocfft-function-pool
  rocfft-rtc-cache
  rocfft-rtc-common
  rocfft-rtc-compile
  rocfft-rtc-gen
  rocfft-rtc-launch
  rocfft-rtc-subprocess
  rocfft-solution-map
  rocfft-tuning-helper
  generator
  hip::device
  ${ROCFFT_SQLITE_LIB}
)

# Use installed GTest, or build from source (BUILD_GTEST=ON forces this).
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
gtest_discover_tests( rocfft-internal-test
  TEST_FILTER ${ROCFFT_INTERNAL_TESTS_FILTER}
)

add_executable( rtc_helper_crash rtc_helper_crash.cpp )
add_dependencies( rocfft-internal-test rtc_helper_crash )

# Run at build time; --no-tests=error fails if the filter matched nothing.
if( ROCFFT_RUN_INTERNAL_TESTS )
  add_custom_command( TARGET rocfft-internal-test POST_BUILD
    COMMAND ${CMAKE_CTEST_COMMAND} --no-tests=error --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running internal tests..."
    VERBATIM )
endif()
