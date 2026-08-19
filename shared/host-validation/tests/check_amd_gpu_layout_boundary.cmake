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
set(layout_threading
    "${HOST_VALIDATION_SOURCE_DIR}/src/amd_gpu_layout/mx_threading.hpp"
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
if(NOT EXISTS "${layout_threading}")
    message(FATAL_ERROR
        "AMD GPU layout private threading header is missing: ${layout_threading}"
    )
endif()
file(READ "${layout_source}" layout_source_contents)
file(READ "${layout_threading}" layout_threading_contents)
set(layout_implementation_contents
    "${layout_source_contents}\n${layout_threading_contents}"
)
if(NOT layout_implementation_contents MATCHES
   "#[ \t]*include[ \t]*[<\"]omp\\.h[>\"]")
    message(FATAL_ERROR
        "OpenMP implementation is missing from the private AMD GPU layout sources."
    )
endif()
if(NOT layout_implementation_contents MATCHES
   "#[ \t]*pragma[ \t]+omp")
    message(FATAL_ERROR
        "OpenMP work sharing is missing from the private AMD GPU layout sources."
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
