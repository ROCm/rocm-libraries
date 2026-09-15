# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.21)

project(libhipcxx-examples LANGUAGES CXX HIP)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(hip)
find_package(libhipcxx)

add_executable(hash_map concurrent_hash_table.hip)
target_compile_features(hash_map PRIVATE cxx_std_14)
target_link_libraries(hash_map PRIVATE libhipcxx::libhipcxx)

#This example requires C++20
#add_executable(chrono_sysclock_workaround chrono_sysclock_workaround.hip)
#target_compile_features(chrono_sysclock_workaround PRIVATE cxx_std_20)
#target_compile_options(chrono_sysclock_workaround PRIVATE -D_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP)
#target_link_libraries(chrono_sysclock_workaround PRIVATE libhipcxx::libhipcxx)
