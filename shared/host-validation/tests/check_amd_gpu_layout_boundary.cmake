# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED HOST_VALIDATION_SOURCE_DIR)
    message(FATAL_ERROR "HOST_VALIDATION_SOURCE_DIR is required.")
endif()

set(layout_header
    "${HOST_VALIDATION_SOURCE_DIR}/include/roc/host_validation/amd_gpu_layout/mx.hpp"
)
set(layout_source
    "${HOST_VALIDATION_SOURCE_DIR}/src/amd_gpu_layout/mx.cpp"
)
set(component_cmake "${HOST_VALIDATION_SOURCE_DIR}/CMakeLists.txt")
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
    "#[ \t]*include[ \t]*[<\"]omp\\.h[>\"]"
    "#[ \t]*pragma[ \t]+omp"
    "_OPENMP"
    "omp_[A-Za-z0-9_]+"
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

if(NOT EXISTS "${layout_source}")
    message(FATAL_ERROR
        "AMD GPU layout implementation source is missing: ${layout_source}"
    )
endif()
file(READ "${layout_source}" layout_source_contents)
if(NOT layout_source_contents MATCHES
   "#[ \t]*include[ \t]*[<\"]omp\\.h[>\"]")
    message(FATAL_ERROR
        "OpenMP implementation is not isolated in ${layout_source}"
    )
endif()

file(READ "${component_cmake}" component_cmake_contents)
if(NOT component_cmake_contents MATCHES
   "add_library\\([ \t\r\n]*host-validation-amd-gpu-layout[ \t\r\n]+STATIC")
    message(FATAL_ERROR
        "AMD GPU layout must be a compiled STATIC component target."
    )
endif()
if(NOT component_cmake_contents MATCHES
   "target_link_libraries\\([ \t\r\n]*host-validation-amd-gpu-layout[ \t\r\n]+PRIVATE[ \t\r\n]+OpenMP::OpenMP_CXX")
    message(FATAL_ERROR
        "AMD GPU layout must keep OpenMP private to the compiled target."
    )
endif()

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
