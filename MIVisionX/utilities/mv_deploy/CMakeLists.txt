# MIT License
#
# Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.10)

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()

project(mv_compile)

include_directories(${CMAKE_INSTALL_PREFIX}/include
                    ${PROJECT_SOURCE_DIR}
                )

add_executable(mv_compile mv_compile.cpp)

# install MIVisionX executables -- {ROCM_PATH}/bin
install(TARGETS mv_compile DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mf16c")
target_link_libraries(mv_compile ${CMAKE_DL_LIBS} stdc++fs)

# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES mvdeploy_api.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
install(FILES mv_extras_postproc.h DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/mivisionx/model_compiler/mv_deploy COMPONENT dev)
install(FILES mv_extras_postproc.cpp DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/mivisionx/model_compiler/mv_deploy COMPONENT dev)
