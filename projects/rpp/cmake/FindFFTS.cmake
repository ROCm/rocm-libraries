#[[
Copyright © 2019-2025 Advanced Micro Devices, Inc. or its affiliates.
SPDX-License-Identifier: MIT
]]

find_path(FFTS_INCLUDE_DIR
    NAMES ffts/ffts.h ffts.h
    PATHS
        /usr/local/include
        /usr/include
        /opt/local/include
        /opt/include
    PATH_SUFFIXES ffts
)

find_library(FFTS_LIBRARY
    NAMES ffts
    PATHS
        /usr/local/lib
        /usr/lib
        /usr/local/lib64
        /usr/lib64
        /opt/local/lib
        /opt/lib
)

# Mark the variables as advanced
mark_as_advanced(FFTS_INCLUDE_DIR FFTS_LIBRARY)

# Check if we found the library and headers
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTS
    REQUIRED_VARS FFTS_LIBRARY FFTS_INCLUDE_DIR
)

if(FFTS_FOUND)
    set(FFTS_LIBRARIES ${FFTS_LIBRARY})
    set(FFTS_INCLUDE_DIRS ${FFTS_INCLUDE_DIR})
endif()

# Create imported target
if(FFTS_FOUND AND NOT TARGET FFTS::FFTS)
    add_library(FFTS::FFTS SHARED IMPORTED)
    set_target_properties(FFTS::FFTS PROPERTIES
        IMPORTED_LOCATION "${FFTS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTS_INCLUDE_DIRS}"
    )
endif()
