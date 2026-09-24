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

# The tuning cache's file format and store. A library of its own so the
# host-only tests can link it without the rest of hipBLASLt or a device.
add_library(hipblaslt-tuning-store STATIC)
target_sources(hipblaslt-tuning-store
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/TuningCacheStore.cpp"
)
set_target_properties(hipblaslt-tuning-store
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(hipblaslt-tuning-store PUBLIC cxx_std_17)
target_compile_definitions(hipblaslt-tuning-store PRIVATE ROCM_USE_FLOAT16)
target_include_directories(hipblaslt-tuning-store
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        "${PROJECT_SOURCE_DIR}/library/src/amd_detail/include"
        "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include"
        "${PROJECT_BINARY_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/library/include"
)
target_link_libraries(hipblaslt-tuning-store
    PUBLIC
        roc::tensilelite-host
    PRIVATE
        roc::${hipblas_target}
        hip::host
)
if(HIPBLASLT_ENABLE_COVERAGE)
    target_compile_options(hipblaslt-tuning-store PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt-tuning-store PUBLIC -fprofile-instr-generate)
endif()
target_link_libraries(hipblaslt PRIVATE hipblaslt-tuning-store)

if(NOT BUILD_SHARED_LIBS)
    install(TARGETS hipblaslt-tuning-store
        EXPORT hipblaslt-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel)
endif()

if(HIPBLASLT_ENABLE_ROCROLLER)
    add_subdirectory(rocroller)
endif()
