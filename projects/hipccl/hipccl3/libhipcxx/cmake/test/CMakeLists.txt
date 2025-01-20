# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # There's a bug that prevents build-and-test from working on MSVC.
  # See NVIDIA/nvbench#43.
  return()
endif()

set(cmake_opts
  -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  -D "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
)

# Temporary installation prefix for tests against installed project:
set(tmp_install_prefix "${CMAKE_CURRENT_BINARY_DIR}/test_install")

# Add a build-and-test CTest.
# - full_test_name_var will be set to the full name of the test.
# - subdir is the relative path to the test project directory.
# - test_id is used to generate a unique name for this test, allowing the
#   subdir to be reused.
# - Any additional args will be passed to the project configure step.
function(libhipcxx_add_compile_test full_test_name_var subdir test_id)
  set(test_name libhipcxx.test.cmake.${subdir}.${test_id})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${subdir}/${test_id}")
  add_test(NAME ${test_name}
    COMMAND "${CMAKE_CTEST_COMMAND}"
      --build-and-test "${src_dir}" "${build_dir}"
      --build-generator "${CMAKE_GENERATOR}"
      --build-options
        ${cmake_opts}
        ${ARGN}
      --test-command "${CMAKE_CTEST_COMMAND}" --output-on-failure
  )
  set(${full_test_name_var} ${test_name} PARENT_SCOPE)
endfunction()

################################################################################
# Test against source dir

libhipcxx_add_compile_test(test_name
  test_export
  source_tree
  -D "libhipcxx_DIR=${libhipcxx_SOURCE_DIR}/lib/cmake/libhipcxx/"
  -D TEST_TYPE=SOURCE_TREE
)

################################################################################
# Test against install tree

libhipcxx_add_compile_test(test_name
  test_export
  install_tree
  -D "libhipcxx_DIR=${tmp_install_prefix}/lib/cmake/libhipcxx/"
  -D TEST_TYPE=INSTALL_TREE
)
set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED install_tree)

################################################################################
# Install tree fixtures
add_test(NAME libhipcxx.test.cmake.install_tree.install
  COMMAND "${CMAKE_COMMAND}"
    --install "${libhipcxx_BINARY_DIR}"
    --prefix "${tmp_install_prefix}"
)
set_tests_properties(libhipcxx.test.cmake.install_tree.install PROPERTIES
  FIXTURES_SETUP install_tree
)

add_test(NAME libhipcxx.test.cmake.install_tree.cleanup
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${tmp_install_prefix}"
)
set_tests_properties(libhipcxx.test.cmake.install_tree.cleanup PROPERTIES
  FIXTURES_CLEANUP install_tree
)
