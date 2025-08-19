# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Debug.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_auxiliary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_mat.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/status.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensile_host.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/UserDrivenTuningParser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_transform.cpp"
)
target_include_directories(hipblaslt PRIVATE "${PROJECT_SOURCE_DIR}/device-library")

if(HIPBLASLT_ENABLE_ROCROLLER)
    add_library(hipblaslt-rocroller OBJECT)
    target_sources(hipblaslt-rocroller
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/rocroller_host.cpp"
    )
    target_compile_features(hipblaslt-rocroller PRIVATE cxx_std_20)
    set_target_properties(hipblaslt-rocroller PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(hipblaslt-rocroller PRIVATE origami::origami)
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
            "${PROJECT_BINARY_DIR}/library/include"
            "${PROJECT_SOURCE_DIR}/library/include"
            "${PROJECT_SOURCE_DIR}/tensilelite/include"
    )
    target_link_libraries(hipblaslt PRIVATE hipblaslt-rocroller)
endif()
