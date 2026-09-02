# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED HOST_NUMERICS_SOURCE_DIR)
    message(FATAL_ERROR "HOST_NUMERICS_SOURCE_DIR is required.")
endif()

if(EXISTS "${HOST_NUMERICS_SOURCE_DIR}/adapters")
    message(
        FATAL_ERROR
        "Product adapter directory must not exist inside the host-numerics component."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/reference_gemm.hpp"
)
    message(
        FATAL_ERROR
        "Reference GEMM implementation must remain behind the compiled component boundary."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/reference_epilogue.hpp"
)
    message(
        FATAL_ERROR
        "Reference epilogue implementation must remain behind the compiled component boundary."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/reference_reduction.hpp"
)
    message(
        FATAL_ERROR
        "Reference reduction implementation must remain behind the compiled component boundary."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/linear_combination.hpp"
)
    message(
        FATAL_ERROR
        "Linear-combination implementation must remain behind the compiled component boundary."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/data_generation.hpp"
)
    message(
        FATAL_ERROR
        "Option-driven data generation must remain behind the compiled component boundary."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/structured_sparsity.hpp"
)
    message(
        FATAL_ERROR
        "Structured sparsity implementation must remain behind the compiled component boundary."
    )
endif()

foreach(_private_header
        reference_common
        reference_softmax
        reference_layer_norm)
    if(EXISTS
       "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/${_private_header}.hpp"
    )
        message(
            FATAL_ERROR
            "${_private_header} implementation must remain behind the compiled component boundary."
        )
    endif()
endforeach()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/tensor_views.hpp"
)
    message(
        FATAL_ERROR
        "Legacy rank-specific tensor views must not reappear inside detail."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/matrix_view.hpp"
)
    message(
        FATAL_ERROR
        "Legacy rank-specific matrix views must not duplicate the tensor-view API."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/comparison.hpp"
)
    message(
        FATAL_ERROR
        "The unused comparison forwarding header must not be installed."
    )
endif()

if(EXISTS
   "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/detail/comparison_impl.hpp"
)
    message(
        FATAL_ERROR
        "Comparison implementation must remain behind the compiled component boundary."
    )
endif()

file(
    READ
    "${HOST_NUMERICS_SOURCE_DIR}/include/roc/host_numerics/comparison.hpp"
    comparison_header
)
foreach(forbidden IN ITEMS "typed_comparison" "detail/comparison" "template <")
    string(FIND "${comparison_header}" "${forbidden}" position)
    if(NOT position EQUAL -1)
        message(
            FATAL_ERROR
            "Canonical comparison facade exposes typed implementation text: ${forbidden}"
        )
    endif()
endforeach()

file(
    GLOB_RECURSE
    component_sources
    "${HOST_NUMERICS_SOURCE_DIR}/include/*.h"
    "${HOST_NUMERICS_SOURCE_DIR}/include/*.hpp"
    "${HOST_NUMERICS_SOURCE_DIR}/src/*.c"
    "${HOST_NUMERICS_SOURCE_DIR}/src/*.cpp"
    "${HOST_NUMERICS_SOURCE_DIR}/src/*.h"
    "${HOST_NUMERICS_SOURCE_DIR}/src/*.hpp"
    "${HOST_NUMERICS_SOURCE_DIR}/python/*.cmake"
    "${HOST_NUMERICS_SOURCE_DIR}/python/src/*.cpp"
    "${HOST_NUMERICS_SOURCE_DIR}/python/tests/*.py"
    "${HOST_NUMERICS_SOURCE_DIR}/tests/*.cpp"
)
list(
    APPEND
    component_sources
    "${HOST_NUMERICS_SOURCE_DIR}/CMakeLists.txt"
    "${HOST_NUMERICS_SOURCE_DIR}/python/CMakeLists.txt"
)

set(forbidden_patterns
    "hipblas"
    "Tensile"
    "rocisa"
    "rocRoller"
    "rocroller"
    "mxDataGenerator"
    "#[ \t]*include[ \t]*[<\"]hip[/<\"]"
    "hip::"
    "HIP_[A-Za-z0-9_]*"
    "TensorView"
    "MutableTensorView"
    "TypedTensorView"
    "fromExternal"
)

foreach(source IN LISTS component_sources)
    file(READ "${source}" contents)
    foreach(pattern IN LISTS forbidden_patterns)
        if(contents MATCHES "${pattern}")
            message(
                FATAL_ERROR
                "Product dependency '${pattern}' found in host-numerics component file: ${source}"
            )
        endif()
    endforeach()
endforeach()
