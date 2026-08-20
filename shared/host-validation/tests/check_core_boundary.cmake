# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED CORE_INCLUDE_DIR)
    message(FATAL_ERROR "CORE_INCLUDE_DIR is required")
endif()

set(core_headers
    "${CORE_INCLUDE_DIR}/index_order.hpp"
    "${CORE_INCLUDE_DIR}/scalar.hpp"
    "${CORE_INCLUDE_DIR}/scalar_codec.hpp"
    "${CORE_INCLUDE_DIR}/tensor.hpp"
)
set(core_contents)
foreach(core_header IN LISTS core_headers)
    if(NOT EXISTS "${core_header}")
        message(FATAL_ERROR "Core header is missing: ${core_header}")
    endif()
    file(READ "${core_header}" header_contents)
    string(APPEND core_contents "\n${header_contents}")
endforeach()

foreach(
    forbidden
    IN ITEMS
        GEMM
        AMDGPU
        amd_gpu_layout
        HIP
        hipBLASLt
        TensileLite
        rocisa
        BLAS
        GTest
        TensorView
        MutableTensorView
        TypedTensorView
        fromExternal
)
    string(FIND "${core_contents}" "${forbidden}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR
            "Core headers contain forbidden product/operation term: ${forbidden}")
    endif()
endforeach()

file(READ "${CORE_INCLUDE_DIR}/scalar.hpp" scalar_contents)
file(READ "${CORE_INCLUDE_DIR}/scalar_codec.hpp" codec_contents)
file(READ "${CORE_INCLUDE_DIR}/tensor.hpp" tensor_contents)

string(FIND
    "${scalar_contents}"
    "#include <roc/host_validation/scalar_codec.hpp>"
    scalar_codec_include_position
)
if(scalar_codec_include_position EQUAL -1)
    message(FATAL_ERROR "scalar.hpp does not include its codec template definitions.")
endif()

foreach(required_scalar_declaration
        "enum class ScalarCategory"
        "enum class ScalarType"
        "struct ScalarTypeInfo"
        "struct NativeScalarType"
        "visitScalarType"
        "class Scalar")
    string(FIND "${scalar_contents}" "${required_scalar_declaration}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR
            "scalar.hpp is missing scalar API declaration: ${required_scalar_declaration}")
    endif()
endforeach()

foreach(forbidden_scalar_declaration
        "class Shape"
        "class Layout"
        "class TensorStorage"
        "class Tensor {")
    string(FIND "${scalar_contents}" "${forbidden_scalar_declaration}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR
            "scalar.hpp contains tensor API declaration: ${forbidden_scalar_declaration}")
    endif()
endforeach()

string(FIND
    "${tensor_contents}"
    "#include <roc/host_validation/scalar.hpp>"
    tensor_scalar_include_position
)
if(tensor_scalar_include_position EQUAL -1)
    message(FATAL_ERROR "tensor.hpp does not preserve scalar API compatibility through scalar.hpp.")
endif()

foreach(forbidden_tensor_declaration
        "enum class ScalarCategory"
        "enum class ScalarType"
        "struct ScalarTypeInfo"
        "struct NativeScalarType"
        "class Scalar {")
    string(FIND "${tensor_contents}" "${forbidden_tensor_declaration}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR
            "tensor.hpp duplicates scalar API declaration: ${forbidden_tensor_declaration}")
    endif()
endforeach()

string(FIND
    "${codec_contents}"
    "namespace roc::host_validation::detail"
    codec_detail_namespace_position
)
if(codec_detail_namespace_position EQUAL -1)
    message(FATAL_ERROR "The scalar codec implementation is not confined to the detail namespace.")
endif()

foreach(forbidden_codec_declaration
        "class Scalar"
        "class Shape"
        "class Layout"
        "class Tensor")
    string(FIND "${codec_contents}" "${forbidden_codec_declaration}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR
            "The scalar codec declares public object type: ${forbidden_codec_declaration}")
    endif()
endforeach()
