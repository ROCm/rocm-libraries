# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED CORE_INCLUDE_DIR)
    message(FATAL_ERROR "CORE_INCLUDE_DIR is required")
endif()

set(core_headers
    "${CORE_INCLUDE_DIR}/index_order.hpp"
    "${CORE_INCLUDE_DIR}/layout.hpp"
    "${CORE_INCLUDE_DIR}/scalar.hpp"
    "${CORE_INCLUDE_DIR}/scalar_binary_float.hpp"
    "${CORE_INCLUDE_DIR}/scalar_codec.hpp"
    "${CORE_INCLUDE_DIR}/scalar_conversion.hpp"
    "${CORE_INCLUDE_DIR}/scalar_storage.hpp"
    "${CORE_INCLUDE_DIR}/scalar_type.hpp"
    "${CORE_INCLUDE_DIR}/shape.hpp"
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
file(READ "${CORE_INCLUDE_DIR}/scalar_type.hpp" scalar_type_contents)
string(APPEND scalar_contents "\n${scalar_type_contents}")
file(READ "${CORE_INCLUDE_DIR}/scalar_codec.hpp" codec_contents)
foreach(codec_header
        scalar_binary_float.hpp
        scalar_conversion.hpp
        scalar_storage.hpp)
    file(READ "${CORE_INCLUDE_DIR}/${codec_header}" codec_header_contents)
    string(APPEND codec_contents "\n${codec_header_contents}")
endforeach()
file(READ "${CORE_INCLUDE_DIR}/tensor.hpp" tensor_contents)
file(READ
    "${CORE_INCLUDE_DIR}/tensor_transformations.hpp"
    tensor_transformations_contents
)

string(FIND
    "${scalar_contents}"
    "#include <roc/host_numerics/scalar_storage.hpp>"
    scalar_storage_include_position
)
if(scalar_storage_include_position EQUAL -1)
    message(FATAL_ERROR "scalar.hpp does not include its storage template definitions.")
endif()
string(FIND
    "${scalar_contents}"
    "#include <roc/host_numerics/scalar_codec.hpp>"
    scalar_codec_include_position
)
if(NOT scalar_codec_include_position EQUAL -1)
    message(FATAL_ERROR "scalar.hpp must not include its compatibility wrapper.")
endif()
string(FIND
    "${codec_contents}"
    "#include <roc/host_numerics/scalar.hpp>"
    scalar_compatibility_include_position
)
if(scalar_compatibility_include_position EQUAL -1)
    message(FATAL_ERROR "scalar_codec.hpp does not preserve scalar.hpp compatibility.")
endif()

foreach(required_scalar_declaration
        "enum class ScalarCategory"
        "enum class ScalarType"
        "struct ScalarTypeInfo"
        "struct NativeScalarType"
        "visitScalarType")
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
        "class Tensor {"
        "class Scalar {")
    string(FIND "${scalar_contents}" "${forbidden_scalar_declaration}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR
            "scalar.hpp contains tensor API declaration: ${forbidden_scalar_declaration}")
    endif()
endforeach()

string(FIND
    "${tensor_contents}"
    "#include <roc/host_numerics/scalar.hpp>"
    tensor_scalar_include_position
)
if(tensor_scalar_include_position EQUAL -1)
    message(FATAL_ERROR "tensor.hpp does not preserve scalar API compatibility through scalar.hpp.")
endif()
string(FIND
    "${tensor_contents}"
    "#include <roc/host_numerics/tensor_transformations.hpp>"
    tensor_transformations_include_position
)
if(NOT tensor_transformations_include_position EQUAL -1)
    message(FATAL_ERROR "tensor.hpp must not include its compatibility wrapper.")
endif()
string(FIND
    "${tensor_transformations_contents}"
    "#include <roc/host_numerics/tensor.hpp>"
    tensor_compatibility_include_position
)
if(tensor_compatibility_include_position EQUAL -1)
    message(FATAL_ERROR
        "tensor_transformations.hpp does not preserve tensor.hpp compatibility.")
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
    "namespace roc::host_numerics::detail"
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
