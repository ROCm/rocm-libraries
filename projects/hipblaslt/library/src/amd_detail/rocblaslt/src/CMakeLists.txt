# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT



target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Debug.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_adaptive_sm_controller.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_auxiliary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_grouped_matmul.cpp"
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
# hipblaslt itself is compiled as HIP. The rocRoller ranker includes C++20
# headers, so it is a separate C++ library and hipblaslt only calls it.
if(WIN32)
    # No Windows GPU target has checked-in rocRoller kernels, so the tiles this
    # ranker returns would have nothing to run. Build one that returns none.
    add_library(hipblaslt-rocroller-selection STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/rocroller_selected_windows.cpp")
else()
    add_library(hipblaslt-rocroller-selection STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/rocroller_selected.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocroller/solution_selection.cpp")
endif()
target_compile_features(hipblaslt-rocroller-selection PRIVATE cxx_std_20)
set_target_properties(hipblaslt-rocroller-selection PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hipblaslt-rocroller-selection
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocroller/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${PROJECT_SOURCE_DIR}/library/include"
        "${PROJECT_BINARY_DIR}/library/include")
# No rocRoller include directory, and so no libfmt either: the ranker describes a
# kernel in the types hipBLASLt already uses for one -- hipDataType and the
# scaling format on RocblasltContractionProblem -- and names nothing of
# rocRoller's. That is what lets a checkout without shared/rocroller build this
# target.
target_compile_definitions(hipblaslt-rocroller-selection PRIVATE
    __HIP_PLATFORM_AMD__)
target_link_libraries(hipblaslt-rocroller-selection PRIVATE
    roc::origami-headers
    hip::host
    roc::hipblas-common)
target_link_libraries(hipblaslt PRIVATE hipblaslt-rocroller-selection roc::origami)

# The tuning file's row format and store. A library of its own so the
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
# TensileLite's headers, not tensilelite-host itself: the store calls nothing in
# it, and linking it would bring the solution library and HIP's device runtime.
target_include_directories(hipblaslt-tuning-store
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:$<TARGET_PROPERTY:tensilelite-host,INTERFACE_INCLUDE_DIRECTORIES>>
    PRIVATE
        "${PROJECT_SOURCE_DIR}/library/src/amd_detail/include"
        "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include"
        "${PROJECT_BINARY_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/library/include"
)
target_link_libraries(hipblaslt-tuning-store
    PUBLIC
        $<BUILD_INTERFACE:rocisa::rocisa-cpp>
    PRIVATE
        roc::${hipblas_target}
        hip::host
)
if(HIPBLASLT_ENABLE_COVERAGE)
    target_compile_options(hipblaslt-tuning-store PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt-tuning-store PUBLIC -fprofile-instr-generate)
endif()
if(HIPBLASLT_ENABLE_ASAN OR HIPBLASLT_ENABLE_TSAN)
    hipblaslt_target_configure_sanitizers(hipblaslt-tuning-store PUBLIC)
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
