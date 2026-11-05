# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED HOSTNUMERICS_SOURCE_DIR)
    message(FATAL_ERROR "HOSTNUMERICS_SOURCE_DIR is required.")
endif()

file(GLOB layout_headers
    "${HOSTNUMERICS_SOURCE_DIR}/include/roc/hostnumerics/amd_gpu_layout/*.hpp"
)
if(NOT layout_headers)
    message(FATAL_ERROR "No AMD GPU layout public headers were found.")
endif()
set(component_cmake "${HOSTNUMERICS_SOURCE_DIR}/CMakeLists.txt")
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
    "#[ \t]*include[ \t]*[<\"]roc/hostnumerics/"
    "#[ \t]*include[ \t]*[<\"]omp\\.h[>\"]"
    "#[ \t]*pragma[ \t]+omp"
    "_OPENMP"
    "omp_[A-Za-z0-9_]+"
    "DimensionShufflePlan"
    "GFX[0-9]+ScalePlan"
    "ParallelChunkFunction"
    "parallelForChunks"
    "operationThreadCount"
    "checkedMultiply"
    "computeStrides"
    "computeShuffledStrides"
    "shuffleDims"
    "preSwizzleScalesGFX950PaddedSize"
    "preSwizzleScalesGFX1250PaddedSize"
    "std::function"
    "void[ \t]*\\*"
)

foreach(layout_header IN LISTS layout_headers)
    file(READ "${layout_header}" layout_contents)
    foreach(pattern IN LISTS forbidden_layout_patterns)
        if(layout_contents MATCHES "${pattern}")
            message(
                FATAL_ERROR
                "Forbidden dependency pattern '${pattern}' found in ${layout_header}"
            )
        endif()
    endforeach()
endforeach()

file(READ "${component_cmake}" component_cmake_contents)
if(NOT component_cmake_contents MATCHES
   "add_library\\([ \t\r\n]*hostnumerics-amd-gpu-layout[ \t\r\n]+STATIC")
    message(FATAL_ERROR
        "AMD GPU layout must be a compiled STATIC component target."
    )
endif()

file(
    GLOB_RECURSE
    generic_sources
    "${HOSTNUMERICS_SOURCE_DIR}/include/roc/hostnumerics/*.hpp"
    "${HOSTNUMERICS_SOURCE_DIR}/src/*.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/src/*.hpp"
)
list(FILTER generic_sources EXCLUDE REGEX "/amd_gpu_layout/")
foreach(generic_source IN LISTS generic_sources)
    file(READ "${generic_source}" contents)
    if(contents MATCHES "amd_gpu_layout")
        message(
            FATAL_ERROR
            "Generic HostNumerics source depends on AMD GPU layout code: ${generic_source}"
        )
    endif()
endforeach()
