# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED COMPONENT_SOURCE_DIR)
    message(FATAL_ERROR "COMPONENT_SOURCE_DIR is required.")
endif()

set(component_files
    "${COMPONENT_SOURCE_DIR}/CMakeLists.txt"
    "${COMPONENT_SOURCE_DIR}/cmake/ROCMxLayoutTransformsConfig.cmake.in"
    "${COMPONENT_SOURCE_DIR}/include/roc/mx_layout_transforms/pre_swizzle.hpp"
)
set(forbidden_patterns
    "#[ \t]*include[ \t]*[<\"]hip[/<\"]"
    "hip::"
    "#[ \t]*include[ \t]*[<\"][^>\"]*[Bb][Ll][Aa][Ss]"
    "find_package\\([ \t]*BLAS"
    "[Bb][Ll][Aa][Ss]::"
    "host[_-]validation"
    "mxDataGenerator"
    "namespace[ \t]+DGen"
)

foreach(component_file IN LISTS component_files)
    file(READ "${component_file}" contents)
    foreach(pattern IN LISTS forbidden_patterns)
        if(contents MATCHES "${pattern}")
            message(
                FATAL_ERROR
                "Forbidden dependency pattern '${pattern}' found in ${component_file}"
            )
        endif()
    endforeach()
endforeach()
