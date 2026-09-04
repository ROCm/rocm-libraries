#[[
Copyright © 2019-2026 Advanced Micro Devices, Inc. or its affiliates.
SPDX-License-Identifier: MIT
]]

# FindROCFFT.cmake - Locate rocFFT library for RPP audio FFT acceleration

find_path(ROCFFT_INCLUDE_DIR
    NAMES rocfft.h
    PATHS
        ${ROCM_PATH}/include
        /opt/rocm/include
        /opt/rocm-*/include
        ${ROCM_PATH}/lib/python*/site-packages/_rocm_sdk_devel/include
        /usr/local/include
        /usr/include
    PATH_SUFFIXES rocfft ""
)

find_library(ROCFFT_LIBRARY
    NAMES rocfft
    PATHS
        ${ROCM_PATH}/lib
        /opt/rocm/lib
        /opt/rocm-*/lib
        ${ROCM_PATH}/lib/rocm_sysdeps/lib
        /usr/local/lib
        /usr/lib
        /usr/local/lib64
        /usr/lib64
    PATH_SUFFIXES lib64 lib
)

# Mark the variables as advanced
mark_as_advanced(ROCFFT_INCLUDE_DIR ROCFFT_LIBRARY)

# Check if we found the library and headers
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCFFT
    REQUIRED_VARS ROCFFT_LIBRARY ROCFFT_INCLUDE_DIR
)

if(ROCFFT_FOUND)
    set(ROCFFT_LIBRARIES ${ROCFFT_LIBRARY})
    set(ROCFFT_INCLUDE_DIRS ${ROCFFT_INCLUDE_DIR})
endif()

# Create imported target for modern CMake
if(ROCFFT_FOUND AND NOT TARGET rocfft::rocfft)
    add_library(rocfft::rocfft SHARED IMPORTED)
    set_target_properties(rocfft::rocfft PROPERTIES
        IMPORTED_LOCATION "${ROCFFT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ROCFFT_INCLUDE_DIRS}"
    )
endif()
