# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(hipblaslt-rocroller OBJECT)
target_sources(hipblaslt-rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/parameter_selection.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime_args_selection.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocroller_host.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/solution_selection.cpp"
)
target_compile_features(hipblaslt-rocroller PRIVATE cxx_std_20)
set_target_properties(hipblaslt-rocroller PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hipblaslt-rocroller PRIVATE roc::origami-headers)
if(TARGET roc::rocroller-interface)
    target_link_libraries(hipblaslt-rocroller PRIVATE roc::rocroller-interface)
else()
    target_link_libraries(hipblaslt-rocroller PRIVATE roc::rocroller_interface)
endif()
target_link_libraries(hipblaslt-rocroller PRIVATE hip::host roc::hipblas-common)
target_compile_definitions(hipblaslt-rocroller PRIVATE HIPBLASLT_USE_ROCROLLER)
target_include_directories(hipblaslt-rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${PROJECT_BINARY_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/tensilelite/include"
)
target_link_libraries(hipblaslt PRIVATE hipblaslt-rocroller)
