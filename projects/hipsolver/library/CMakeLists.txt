# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ABI version for the shared library
set(hipsolver_SOVERSION 1.0)

# Configure version header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/hipsolver-version.h.in"
    "${PROJECT_BINARY_DIR}/include/hipsolver/internal/hipsolver-version.h"
)

if(HIPSOLVER_ENABLE_FORTRAN)
    set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/include/hipsolver/internal")
endif()

add_subdirectory(src)
