# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED HOST_VALIDATION_SOURCE_DIR)
    message(FATAL_ERROR "HOST_VALIDATION_SOURCE_DIR is required.")
endif()

if(EXISTS "${HOST_VALIDATION_SOURCE_DIR}/adapters")
    message(
        FATAL_ERROR
        "Product adapter directory must not exist inside the host-validation component."
    )
endif()

file(
    GLOB_RECURSE
    component_sources
    "${HOST_VALIDATION_SOURCE_DIR}/include/*.h"
    "${HOST_VALIDATION_SOURCE_DIR}/include/*.hpp"
    "${HOST_VALIDATION_SOURCE_DIR}/src/*.c"
    "${HOST_VALIDATION_SOURCE_DIR}/src/*.cpp"
    "${HOST_VALIDATION_SOURCE_DIR}/python/*.cmake"
    "${HOST_VALIDATION_SOURCE_DIR}/python/src/*.cpp"
    "${HOST_VALIDATION_SOURCE_DIR}/python/tests/*.py"
    "${HOST_VALIDATION_SOURCE_DIR}/tests/*.cpp"
)
list(
    APPEND
    component_sources
    "${HOST_VALIDATION_SOURCE_DIR}/CMakeLists.txt"
    "${HOST_VALIDATION_SOURCE_DIR}/python/CMakeLists.txt"
)

set(forbidden_patterns
    "hipblas"
    "Tensile"
    "rocisa"
    "#[ \t]*include[ \t]*[<\"]hip[/<\"]"
    "hip::"
    "HIP_[A-Za-z0-9_]*"
)

foreach(source IN LISTS component_sources)
    file(READ "${source}" contents)
    foreach(pattern IN LISTS forbidden_patterns)
        if(contents MATCHES "${pattern}")
            message(
                FATAL_ERROR
                "Product dependency '${pattern}' found in host-validation component file: ${source}"
            )
        endif()
    endforeach()
endforeach()
