# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT DEFINED CORE_HEADER)
    message(FATAL_ERROR "CORE_HEADER is required")
endif()

file(READ "${CORE_HEADER}" core_contents)

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
            "Tensor core header contains forbidden product/operation term: ${forbidden}")
    endif()
endforeach()
