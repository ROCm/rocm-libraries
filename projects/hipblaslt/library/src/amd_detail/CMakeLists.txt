# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocblaslt)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_ostream.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext-partitioned.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext-op.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-experimental.cpp"
)

if(HIPBLASLT_ENABLE_LOCAL_KERNELS)
    target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_LOCAL_KERNELS)
    if(HIPBLASLT_ENABLE_LOCAL_KERNEL_TEST)
        target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_LOCAL_KERNEL_TEST)
    endif()
    target_sources(hipblaslt
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-local-kernel.cpp"
    )
endif()

# hipblaslt-experimental-registry.inc, the generated feature list for
# hipblasLtExperimentalGetFeatureRevision (see library/experimental/features.cmake).
target_include_directories(hipblaslt PRIVATE "${PROJECT_BINARY_DIR}/library/src")
