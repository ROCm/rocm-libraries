# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FindPackageHandleStandardArgs)

set(BLIS_PATH_4_2_0 "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc")
set(BLIS_PATH_4_1_0 "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc")
set(BLIS_PATH_4_0   "/opt/AMD/aocl/aocl-linux-aocc-4.0")

# # NOTE: Keeping this code for reference of previous implementation
# # Only the first three paths are used in the current find_path/find_library calls below
# if(EXISTS                "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a" )
#     set( BLIS_LIB         /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a )
#     set( BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/include_ILP64/ )
# elseif(EXISTS            "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a" )
#     set( BLIS_LIB         /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a )
#     set( BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/include_ILP64/ )
# elseif(EXISTS            "/opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a" )
#     set( BLIS_LIB         /opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a )
#     set( BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-aocc-4.0/include_ILP64/ )
# elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../deps/amd-blis/lib/ILP64/libblis-mt.a") # 4.0 and 4.1.0
#     set( BLIS_LIB ${CMAKE_CURRENT_BINARY_DIR}/../deps/amd-blis/lib/ILP64/libblis-mt.a )
#     set( BLIS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../deps/amd-blis/include/ILP64 )
# elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../deps/blis/lib/libblis.a")
#     set( BLIS_LIB ${CMAKE_CURRENT_BINARY_DIR}/../deps/blis/lib/libblis.a )
#     set( BLIS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../deps/blis/include/blis )
# elseif(EXISTS      "/usr/local/lib/libblis.a")
#     set( BLIS_LIB /usr/local/lib/libblis.a )
#     set( BLIS_INCLUDE_DIR /usr/local/include/blis )
# else()
#     message(FATAL_ERROR "BLIS lib not found.")
# endif()

find_path(BLIS_INCLUDE_DIR
    NAMES blis.h
    PATHS
        ${BLIS_ROOT}
        ENV BLIS_ROOT
        ${BLIS_PATH_4_2_0}/include_ILP64
        ${BLIS_PATH_4_1_0}/include_ILP64
        ${BLIS_PATH_4_0}/include_ILP64
)

find_library(BLIS_LIB
    NAMES libblis-mt.a libblis.a blis-mt blis
    PATHS
        ${BLIS_ROOT}
        ENV BLIS_ROOT
        ${BLIS_PATH_4_2_0}/lib_ILP64
        ${BLIS_PATH_4_1_0}/lib_ILP64
        ${BLIS_PATH_4_0}/lib_ILP64
    NO_DEFAULT_PATH
)

# If static library wasn't found, try shared libraries
if(NOT BLIS_LIB)
    find_library(BLIS_LIB
        NAMES blis-mt blis
        PATHS
            ${BLIS_ROOT}
            ENV BLIS_ROOT
            ${BLIS_PATH_4_2_0}/lib_ILP64
            ${BLIS_PATH_4_1_0}/lib_ILP64
            ${BLIS_PATH_4_0}/lib_ILP64
    )
endif()

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
