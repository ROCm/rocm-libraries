# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocblaslt)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_ostream.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext-op.cpp"
)

if(HIPBLASLT_ENABLE_USER_KERNELS)
    target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_USER_KERNELS)
    target_sources(hipblaslt
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-user-kernel.cpp"
    )
endif()
