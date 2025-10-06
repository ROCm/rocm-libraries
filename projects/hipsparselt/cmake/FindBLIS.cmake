# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# FindBLIS.cmake - Find BLIS library
#
# This module defines:
#   BLIS_FOUND - System has BLIS
#   BLIS_INCLUDE_DIRS - The BLIS include directories
#   BLIS_LIBRARIES - The libraries needed to use BLIS
#   BLIS::blis - Imported target for BLIS

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_BLIS QUIET blis)
endif()

find_path(BLIS_INCLUDE_DIR
    NAMES blis.h
    HINTS ${PC_BLIS_INCLUDE_DIRS}
    PATHS
        /usr/include
        /usr/local/include
        /opt/blis/include
)

find_library(BLIS_LIBRARY
    NAMES blis
    HINTS ${PC_BLIS_LIBRARY_DIRS}
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/blis/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLIS
    FOUND_VAR BLIS_FOUND
    REQUIRED_VARS BLIS_LIBRARY BLIS_INCLUDE_DIR
)

if(BLIS_FOUND)
    set(BLIS_LIBRARIES ${BLIS_LIBRARY})
    set(BLIS_INCLUDE_DIRS ${BLIS_INCLUDE_DIR})

    if(NOT TARGET BLIS::blis)
        add_library(BLIS::blis UNKNOWN IMPORTED)
        set_target_properties(BLIS::blis PROPERTIES
            IMPORTED_LOCATION "${BLIS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(BLIS_INCLUDE_DIR BLIS_LIBRARY)
