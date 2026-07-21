# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FindPackageHandleStandardArgs)

set(BLIS_PATH_4_2_0 "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc")
set(BLIS_PATH_4_1_0 "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc")
set(BLIS_PATH_4_0   "/opt/AMD/aocl/aocl-linux-aocc-4.0")

find_path(BLIS_INCLUDE_DIR
    NAMES blis.h
    PATHS
        ${BLIS_ROOT}
        ENV BLIS_ROOT
        "${BLIS_PATH_4_2_0}/include_ILP64"
        "${BLIS_PATH_4_1_0}/include_ILP64"
        "${BLIS_PATH_4_0}/include_ILP64"
        "${CMAKE_CURRENT_BINARY_DIR}/../deps/blis/include/blis"
        "${CMAKE_CURRENT_BINARY_DIR}/../deps/amd-blis/include/ILP64"
        "${PROJECT_BINARY_DIR}/deps/blis/include/blis"
        "${PROJECT_BINARY_DIR}/deps/amd-blis/include/ILP64"
        "/usr/local/include/blis"
)

find_library(BLIS_LIB
    NAMES libblis-mt.a libblis.a
    PATHS
        ${BLIS_ROOT}
        ENV BLIS_ROOT
        "${BLIS_PATH_4_2_0}/lib_ILP64"
        "${BLIS_PATH_4_1_0}/lib_ILP64"
        "${BLIS_PATH_4_0}/lib_ILP64"
        "${CMAKE_CURRENT_BINARY_DIR}/../deps/blis/lib"
        "${CMAKE_CURRENT_BINARY_DIR}/../deps/amd-blis/lib/ILP64"
        "${PROJECT_BINARY_DIR}/deps/blis/lib"
        "${PROJECT_BINARY_DIR}/deps/amd-blis/lib/ILP64"
        "/usr/local/lib"
)

find_package_handle_standard_args(BLIS
    REQUIRED_VARS BLIS_LIB BLIS_INCLUDE_DIR
)

if(BLIS_FOUND)
    set(BLIS_LIBRARIES ${BLIS_LIB})
    set(BLIS_INCLUDE_DIRS ${BLIS_INCLUDE_DIR})

    # Create an imported target for BLIS
    if(NOT TARGET BLIS::BLIS)
        add_library(BLIS::BLIS UNKNOWN IMPORTED)
        set_target_properties(BLIS::BLIS PROPERTIES
            IMPORTED_LOCATION "${BLIS_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
        )
    endif()
endif()

message(STATUS "Found BLIS: ${BLIS_LIB}")
message(STATUS "Found BLIS: ${BLIS_INCLUDE_DIR}")
