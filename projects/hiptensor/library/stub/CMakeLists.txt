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

# Builds a host-only stub of the hipTensor shared library. This is used instead
# of the real library (library/src) when hipTensor is configured for an
# architecture set that resolves to no GPU targets (HIPTENSOR_ENABLE_DEVICE=OFF),
# where the device kernels and their Composable Kernel dependency cannot be
# built. The stub implements the full public API but reports
# HIPTENSOR_STATUS_NOT_SUPPORTED for every entry point, so the package and its
# headers stay available for downstream find_package(hiptensor) while giving
# callers a well-defined runtime error.

# Create stub shared library target
function(create_hiptensor_stub_library)
    add_library(hiptensor SHARED ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_stub.cpp)

    # Create alias for users
    add_library(hiptensor::hiptensor ALIAS hiptensor)

    # Add driver mode to both cxx and link flags
    target_compile_options(hiptensor PRIVATE ${CLANG_DRIVER_MODE})
    target_link_options(hiptensor PRIVATE ${CLANG_DRIVER_MODE})

    # Add export definitions
    target_compile_definitions(hiptensor PRIVATE hiptensor_EXPORTS)

    target_include_directories(hiptensor
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    # hiptensor.h includes "internal/hiptensor-export.h" relative to the
    # build/include/hiptensor directory, matching the real library build.
    target_include_directories(hiptensor PRIVATE
        ${PROJECT_BINARY_DIR}/include/hiptensor
    )

    # Only the host runtime is required for the stub (no device code).
    target_link_libraries(hiptensor PUBLIC hip::host)
    set_target_properties(hiptensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Set version information
    set(hiptensor_SOVERSION 0.1)
    rocm_set_soversion(hiptensor ${hiptensor_SOVERSION})
endfunction()

create_hiptensor_stub_library()

# hiptensor.h includes an export header that is normally produced by
# generate_export_header() during the device build. Generate it here too so the
# installed headers remain self-contained.
include(GenerateExportHeader)
generate_export_header(hiptensor EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hiptensor/internal/hiptensor-export.h)

rocm_install_targets(
    TARGETS hiptensor
    INCLUDE
      ${PROJECT_SOURCE_DIR}/library/include
      ${PROJECT_BINARY_DIR}/include
)
