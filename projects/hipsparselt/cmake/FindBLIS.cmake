# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# FindBLIS.cmake - Find BLIS library
#
# This module defines: BLIS_FOUND - System has BLIS BLIS_INCLUDE_DIRS - The BLIS include directories
# BLIS_LIBRARIES - The libraries needed to use BLIS BLIS::blis - Imported target for BLIS

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_BLIS QUIET blis)
endif()

find_path(
    BLIS_INCLUDE_DIR
    NAMES blis.h
    HINTS ${PC_BLIS_INCLUDE_DIRS}
    PATHS /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/include_ILP64
          /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/include_ILP64
          /opt/AMD/aocl/aocl-linux-aocc-4.0/include_ILP64
          /opt/blis/include
          /usr/local/include
          /usr/include
)

# Save current CMAKE_FIND_LIBRARY_SUFFIXES and set to find only static libraries
set(_blis_orig_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib)
else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
endif()

find_library(
    BLIS_LIBRARY
    NAMES blis-mt blis
    HINTS ${PC_BLIS_LIBRARY_DIRS}
    PATHS /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64
          /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64
          /opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64 /opt/blis/lib /usr/local/lib /usr/lib
)

# Restore original CMAKE_FIND_LIBRARY_SUFFIXES
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_blis_orig_CMAKE_FIND_LIBRARY_SUFFIXES})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    BLIS FOUND_VAR BLIS_FOUND REQUIRED_VARS BLIS_LIBRARY BLIS_INCLUDE_DIR
)

if(BLIS_FOUND)
    set(BLIS_LIBRARIES ${BLIS_LIBRARY})
    set(BLIS_INCLUDE_DIRS ${BLIS_INCLUDE_DIR})

    if(NOT TARGET BLIS::blis)
        add_library(BLIS::blis STATIC IMPORTED)
        set_target_properties(
            BLIS::blis PROPERTIES IMPORTED_LOCATION "${BLIS_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES
                                                                      "${BLIS_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(BLIS_INCLUDE_DIR BLIS_LIBRARY)
