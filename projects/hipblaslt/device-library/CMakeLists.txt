# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.0)
project(device-library VERSION 0.0.1 LANGUAGES CXX ASM)

set(HIPBLASLT_TENSILE_LIBPATH "${hipblaslt_BINARY_DIR}/Tensile" CACHE STRING "Path to output the device gemm libraries.")
file(MAKE_DIRECTORY "${HIPBLASLT_TENSILE_LIBPATH}")

list(JOIN GPU_TARGETS "$<SEMICOLON>" TENSILE_GPU_TARGETS_SEMI_ESCAPED)

set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--architecture=${TENSILE_GPU_TARGETS_SEMI_ESCAPED}")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--cxx-compiler=${CMAKE_CXX_COMPILER}")
if(HIPBLASLT_ENABLE_ASAN)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--address-sanitizer")
endif()
if(HIPBLASLT_DEVICE_JOBS)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--jobs=${HIPBLASLT_DEVICE_JOBS}")
endif()
if(HIPBLASLT_KEEP_BUILD_TMP)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--keep-build-tmp")
endif()
if(HIPBLASLT_ASM_DEBUG)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--asm-debug")
endif()
if(HIPBLASLT_LIBRARY_FORMAT)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--library-format=${HIPBLASLT_LIBRARY_FORMAT}")
endif()
if(HIPBLASLT_LOGIC_FITLER)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--library-format=${HIPBLASLT_LOGIC_FITLER}")
endif()
if(HIPBLASLT_NO_COMPRESS)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--no-compress")
endif()
if(HIPBLASLT_EXPERIMENTAL)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--experimental")
endif()

if(NOT HIPBLASLT_LIBLOGIC_PATH)
    set(HIPBLASLT_LIBLOGIC_PATH "${hipblaslt_SOURCE_DIR}/library")
endif()

set(output_stamp "${CMAKE_CURRENT_BINARY_DIR}/Tensile.stamp")
set(TENSILE_CREATE_LIBRARY_COMMAND
    ${HIPBLASLT_PYTHON_COMMAND} -m Tensile.TensileCreateLibrary
    ${TENSILE_BUILD_OPTS}
    ${HIPBLASLT_LIBLOGIC_PATH}
    "${HIPBLASLT_TENSILE_LIBPATH}"
    HIP
)

add_custom_command(
    OUTPUT "${output_stamp}"
    COMMENT "Building device libraries to ${HIPBLASLT_TENSILE_LIBPATH} ..."
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

add_subdirectory(extops)
add_subdirectory(matrix-transform)
