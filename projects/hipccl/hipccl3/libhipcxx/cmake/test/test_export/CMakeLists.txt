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

cmake_minimum_required(VERSION 3.20.1)
project(libcudacxxTestExport LANGUAGES CXX)

message(STATUS "libhipcxx_DIR=${libhipcxx_DIR}")
find_package(hip REQUIRED)
find_package(libhipcxx REQUIRED)

add_executable(version_check version_check.cxx)
target_link_libraries(version_check PRIVATE libhipcxx::libhipcxx hip::host)
enable_testing()
add_test(NAME version_check COMMAND "$<TARGET_FILE:version_check>")
set_property(TEST version_check PROPERTY
  PASS_REGULAR_EXPRESSION
  "${libcudacxx_VERSION_MAJOR}\.${libcudacxx_VERSION_MINOR}\.${libcudacxx_VERSION_PATCH}"
)

# Install tree exporting of version_check_objs
install(TARGETS version_check EXPORT test-targets)
install(EXPORT test-targets FILE test-targets.cmake DESTINATION "lib/cmake/libcudacxx-test/")

# Build tree exporting of version_check_objs
export(EXPORT test-targets FILE "${CMAKE_CURRENT_BINARY_DIR}/libcudacxx-test/test-targets.cmake")
