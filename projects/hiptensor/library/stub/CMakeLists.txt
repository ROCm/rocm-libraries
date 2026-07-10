###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ###############################################################################

# Host-only stub of the hipTensor public API.
#
# The stub serves two purposes:
#
#   1. On architectures with no usable GPU target (HIPTENSOR_DISABLE_DEVICE=ON),
#      it IS the installed libhiptensor: the device kernels and their Composable
#      Kernel dependency cannot be built, but the package, headers, and a
#      well-defined runtime error stay available for downstream consumers.
#
#   2. In every build (device or stub), it backs a build-time completeness check
#      (hiptensor_add_stub_api_coverage_check below) that guarantees the stub
#      implements the entire public API regardless of GPU_TARGETS.
#
# The stub sources are:
#   * generated: every public API function returning hiptensorStatus_t, emitted
#     from the public header by generate_api_stub.py (--mode stub) so it can
#     never drift out of sync with the API.
#   * hiptensor_stub_special.cpp: the few informational helpers with real
#     behavior (non-status return types), hand-written.

include(GenerateExportHeader)

# These functions are invoked from the parent (root) scope after this directory
# is added, where CMAKE_CURRENT_* would point at the project root. Capture this
# directory now so the helpers reference the stub sources regardless of caller.
set(HIPTENSOR_STUB_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "hipTensor stub source dir")
set(HIPTENSOR_STUB_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "hipTensor stub binary dir")

# Generate the stub implementation source (status-returning functions) from the
# public header, into ${out_var}. Runs the generator now so the file exists at
# configure time, and registers a build-time custom command so header edits
# refresh it on rebuild.
function(_hiptensor_generate_stub_source out_var)
    set(_generated "${HIPTENSOR_STUB_BINARY_DIR}/hiptensor_stub_generated.cpp")
    set(${out_var} "${_generated}" PARENT_SCOPE)

    # Both the stub library and the coverage check use the generated source;
    # register the generation rule only once to avoid a duplicate OUTPUT rule.
    get_property(_already GLOBAL PROPERTY HIPTENSOR_STUB_SOURCE_GENERATED)
    if(_already)
        return()
    endif()
    set_property(GLOBAL PROPERTY HIPTENSOR_STUB_SOURCE_GENERATED TRUE)

    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(_header "${PROJECT_SOURCE_DIR}/library/include/hiptensor/hiptensor.h")
    set(_generator "${HIPTENSOR_STUB_DIR}/generate_api_stub.py")

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${_generator}
                --mode stub --header ${_header} --output ${_generated}
        RESULT_VARIABLE _gen_result
    )
    if(NOT _gen_result EQUAL 0)
        message(FATAL_ERROR "Failed to generate hipTensor stub source")
    endif()
    add_custom_command(
        OUTPUT ${_generated}
        COMMAND ${Python3_EXECUTABLE} ${_generator}
                --mode stub --header ${_header} --output ${_generated}
        DEPENDS ${_generator} ${_header}
        COMMENT "Generating hipTensor stub source"
        VERBATIM
    )
endfunction()

# Build the host-only stub shared library used as libhiptensor when device
# support is disabled.
function(create_hiptensor_stub_library)
    include(GenerateExportHeader)
    _hiptensor_generate_stub_source(_stub_generated)

    add_library(hiptensor SHARED
        ${_stub_generated}
        ${HIPTENSOR_STUB_DIR}/hiptensor_stub_special.cpp)
    add_library(hiptensor::hiptensor ALIAS hiptensor)

    target_compile_options(hiptensor PRIVATE ${CLANG_DRIVER_MODE})
    target_link_options(hiptensor PRIVATE ${CLANG_DRIVER_MODE})
    target_compile_definitions(hiptensor PRIVATE hiptensor_EXPORTS)
    target_include_directories(hiptensor
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    # hiptensor.h includes "internal/hiptensor-export.h" relative to the
    # build/include/hiptensor directory, matching the real library build.
    target_include_directories(hiptensor PRIVATE ${PROJECT_BINARY_DIR}/include/hiptensor)
    target_link_libraries(hiptensor PUBLIC hip::host)
    set_target_properties(hiptensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

    set(hiptensor_SOVERSION 0.1)
    rocm_set_soversion(hiptensor ${hiptensor_SOVERSION})

    # hiptensor.h includes an export header normally produced by the device
    # build. Generate it here so the installed headers remain self-contained.
    generate_export_header(hiptensor
        EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hiptensor/internal/hiptensor-export.h)

    rocm_install_targets(
        TARGETS hiptensor
        INCLUDE
          ${PROJECT_SOURCE_DIR}/library/include
          ${PROJECT_BINARY_DIR}/include
    )
endfunction()

# Add a build-time check that the stub implements the entire public API.
#
# A coverage translation unit is generated from the public header (one address
# reference per HIPTENSOR_EXPORT function) and linked together with the stub
# sources into a tiny, non-installed host executable. If the stub is missing any
# public API function, that reference is an unresolved symbol and the link --
# hence the build -- fails. Because both the coverage test and the generated stub
# are refreshed from the header, a newly added public API is covered
# automatically. This is independent of GPU_TARGETS and of the device build, so
# it also guards the stub in device builds where the stub is not otherwise
# compiled.
#
# Self-contained: it compiles the stub sources itself and writes its own export
# header, so it has no ordering dependency on the device/stub library.
function(hiptensor_add_stub_api_coverage_check)
    if(TARGET hiptensor_stub_api_coverage)
        return()
    endif()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(_header "${PROJECT_SOURCE_DIR}/library/include/hiptensor/hiptensor.h")
    set(_generator "${HIPTENSOR_STUB_DIR}/generate_api_stub.py")
    set(_generated "${HIPTENSOR_STUB_BINARY_DIR}/hiptensor_stub_api_coverage.cpp")
    set(_private_inc "${HIPTENSOR_STUB_BINARY_DIR}/stub_coverage_include")

    # hiptensor.h unconditionally includes "internal/hiptensor-export.h", which
    # is normally produced by generate_export_header() for a library target.
    # This check links an executable, so write a minimal export header directly:
    # the symbol-visibility macros only need to be defined for a host compile.
    file(WRITE "${_private_inc}/hiptensor/internal/hiptensor-export.h"
"#ifndef HIPTENSOR_EXPORT_H
#define HIPTENSOR_EXPORT_H
#define HIPTENSOR_EXPORT
#define HIPTENSOR_NO_EXPORT
#endif // HIPTENSOR_EXPORT_H
")

    # Generate the coverage source now (so it exists at configure) and again at
    # build time when the header changes.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${_generator} --header ${_header} --output ${_generated}
        RESULT_VARIABLE _gen_result
    )
    if(NOT _gen_result EQUAL 0)
        message(FATAL_ERROR "Failed to generate hipTensor stub API coverage check")
    endif()
    add_custom_command(
        OUTPUT ${_generated}
        COMMAND ${Python3_EXECUTABLE} ${_generator} --header ${_header} --output ${_generated}
        DEPENDS ${_generator} ${_header}
        COMMENT "Generating hipTensor stub API coverage check"
        VERBATIM
    )

    # Link the coverage test against the same stub sources the library uses, so a
    # missing implementation fails the link.
    _hiptensor_generate_stub_source(_stub_generated)
    add_executable(hiptensor_stub_api_coverage
        ${_generated}
        ${_stub_generated}
        ${HIPTENSOR_STUB_DIR}/hiptensor_stub_special.cpp)
    # Exclude from the default build set; depended on explicitly below so it
    # still builds with the library but does not get installed or packaged.
    set_target_properties(hiptensor_stub_api_coverage PROPERTIES EXCLUDE_FROM_ALL TRUE)
    target_compile_options(hiptensor_stub_api_coverage PRIVATE ${CLANG_DRIVER_MODE})
    target_link_options(hiptensor_stub_api_coverage PRIVATE ${CLANG_DRIVER_MODE})
    target_compile_definitions(hiptensor_stub_api_coverage PRIVATE hiptensor_EXPORTS)
    target_include_directories(hiptensor_stub_api_coverage PRIVATE
        ${PROJECT_SOURCE_DIR}/library/include
        ${_private_inc}
        ${_private_inc}/hiptensor)
    target_link_libraries(hiptensor_stub_api_coverage PRIVATE hip::host)

    # Tie the check into the default build so it runs on every build of the
    # library, turning a missing stub implementation into a build failure.
    if(TARGET hiptensor)
        add_dependencies(hiptensor hiptensor_stub_api_coverage)
    endif()
endfunction()
