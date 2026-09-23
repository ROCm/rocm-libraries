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

# Build-time opt-in implementation of the public experimental JIT extension.
if(HIPBLASLT_ENABLE_JIT_GEMM)
    find_package(Boost CONFIG REQUIRED)
    target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_JIT_GEMM)
    target_sources(hipblaslt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-gemm.cpp")
    target_link_libraries(hipblaslt PRIVATE Boost::headers)
    add_executable(hipblaslt-jit-gemm
        "${PROJECT_SOURCE_DIR}/clients/samples/29_hipblaslt_jit_gemm/sample_hipblaslt_jit_gemm.cpp")
    target_link_libraries(hipblaslt-jit-gemm PRIVATE roc::hipblaslt hip::device)
    target_compile_features(hipblaslt-jit-gemm PRIVATE cxx_std_17)
    set_target_properties(hipblaslt-jit-gemm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
endif()
