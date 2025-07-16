# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${ROCBLASLT_LIB_DIR}/src")

target_sources(hipblaslt
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/Debug.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/handle.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/status.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_mat.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_transform.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_transform.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UserDrivenTuningParser.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/tensile_host.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_auxiliary.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
)

if(HIPBLASLT_ENABLE_ROCROLLER)
    add_library(hipblaslt-rocroller OBJECT)
    target_sources(hipblaslt-rocroller
        PRIVATE
            "${_CMAKE_CURRENT_SOURCE_DIR}/rocroller_host.cpp"
    )
    target_compile_features(hipblaslt-rocroller PRIVATE cxx_std_20)
    set_target_properties(hipblaslt-rocroller PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(hipblaslt-rocroller PRIVATE hip::host roc::hipblas-common roc::rocroller)
    target_compile_definitions(hipblaslt-rocroller PRIVATE HIPBLASLT_USE_ROCROLLER)
    target_include_directories(hipblaslt-rocroller PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/include")
    #target_include_directories(hipblaslt-rocroller PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/../include")
    #target_include_directories(hipblaslt-rocroller PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
    #target_include_directories(hipblaslt-rocroller PRIVATE "${PROJECT_BINARY_DIR}/library/include")
    target_link_libraries(hipblaslt PRIVATE hipblaslt-rocroller)
endif()
