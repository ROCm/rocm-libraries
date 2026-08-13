# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED HOST_VALIDATION_SOURCE_DIR)
    message(FATAL_ERROR "HOST_VALIDATION_SOURCE_DIR is required.")
endif()

set(layout_header
    "${HOST_VALIDATION_SOURCE_DIR}/include/roc/host_validation/amd_gpu_layout/mx.hpp"
)
set(forbidden_layout_patterns
    "#[ \t]*include[ \t]*[<\"]hip[/<\"]"
    "hip::"
    "#[ \t]*include[ \t]*[<\"][^>\"]*[Bb][Ll][Aa][Ss]"
    "find_package\\([ \t]*BLAS"
    "[Bb][Ll][Aa][Ss]::"
    "mxDataGenerator"
    "namespace[ \t]+DGen"
    "hipblas"
    "Tensile"
    "rocisa"
    "rocRoller"
    "rocroller"
    "#[ \t]*include[ \t]*[<\"]roc/host_validation/"
)

file(READ "${layout_header}" layout_contents)
foreach(pattern IN LISTS forbidden_layout_patterns)
    if(layout_contents MATCHES "${pattern}")
        message(
            FATAL_ERROR
            "Forbidden dependency pattern '${pattern}' found in ${layout_header}"
        )
    endif()
endforeach()

file(
    GLOB_RECURSE
    generic_sources
    "${HOST_VALIDATION_SOURCE_DIR}/include/roc/host_validation/*.hpp"
    "${HOST_VALIDATION_SOURCE_DIR}/src/*.cpp"
    "${HOST_VALIDATION_SOURCE_DIR}/src/*.hpp"
)
list(FILTER generic_sources EXCLUDE REGEX "/amd_gpu_layout/")
foreach(generic_source IN LISTS generic_sources)
    file(READ "${generic_source}" contents)
    if(contents MATCHES "amd_gpu_layout")
        message(
            FATAL_ERROR
            "Generic host-validation source depends on AMD GPU layout code: ${generic_source}"
        )
    endif()
endforeach()
