# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# MIT License
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

set(TENSILELITE_CLIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/Tensile/Source/client")
set(_CMAKE_CURRENT_SOURCE_DIR ${TENSILELITE_CLIENT_DIR})
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
add_executable(tensilelite-client "${_CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(tensilelite-client
    PRIVATE
        hip::device
        hip::host
        tensilelite::tensilelite-host
        rocisa::rocisa-cpp
        Boost::program_options
        Boost::filesystem
        OpenMP::OpenMP_CXX
)
if(rocm_smi_FOUND)
    target_link_libraries(tensilelite-client PRIVATE rocm_smi64)
endif()
target_include_directories(tensilelite-client PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(tensilelite-client PRIVATE TENSILE_DEFAULT_SERIALIZATION)

set_target_properties(tensilelite-client
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(tensilelite-client PRIVATE)
endif()

add_subdirectory(include)
add_subdirectory(src)

