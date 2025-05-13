# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.25.0)
project(device-library VERSION 0.0.1 LANGUAGES CXX ASM)

set(TENSILELITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite)

list(JOIN GPU_TARGETS "$<SEMICOLON>" TENSILE_GPU_TARGETS_SEMI_ESCAPED)

set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--architecture=${TENSILE_GPU_TARGETS_SEMI_ESCAPED}")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--cxx-compiler=${CMAKE_CXX_COMPILER}")
if(HIPBLASLT_ENABLE_ASAN)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--address-sanitizer")
endif()
if(HIPBLASLT_DEVICE_JOBS)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--jobs=${HIPBLASLT_DEVICE_JOBS}")
endif()
if(HIPBLASLT_KEEP_TMP)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--keep-build-tmp")
endif()
if(NOT HIPBLASLT_LIBLOGIC_PATH)
    set(HIPBLASLT_LIBLOGIC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../library")
endif()

set(output_dir "${hipblaslt_BINARY_DIR}/Tensile")
set(output_stamp "${CMAKE_CURRENT_BINARY_DIR}/Tensile.stamp")
set(TENSILE_CREATE_LIBRARY_COMMAND
    ${HIPBLASLT_PYTHON_COMMAND} -m Tensile.TensileCreateLibrary
    ${TENSILE_BUILD_OPTS}
    ${HIPBLASLT_LIBLOGIC_PATH}
    "${output_dir}"
    HIP
)

add_custom_command(
    OUTPUT "${output_stamp}"
    COMMENT "Building device libraries to ${output_dir} ..."
    COMMAND ${TENSILE_CREATE_LIBRARY_COMMAND}
    COMMAND ${CMAKE_COMMAND} -E touch "${output_stamp}"
    DEPENDS  ${HIPBLASLT_PYTHON_DEPS}
    # Because the command can contain special characters
    VERBATIM
    # Because this can be very long running and difficult to debug deadlocks
    # without streaming.
    USES_TERMINAL
)

block(SCOPE_FOR VARIABLES)
    list(JOIN TENSILE_CREATE_LIBRARY_COMMAND " " FORMATTED_TCL)
    message(VERBOSE "Device lib build command: ${FORMATTED_TCL}")
endblock()

add_custom_target(tensilelite-device-libraries ALL
    DEPENDS "${output_stamp}"
)

add_subdirectory(extops) # can't add this given current architecture of Tensile
add_subdirectory(matrix-transform)
