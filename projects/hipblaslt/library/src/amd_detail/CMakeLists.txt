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

if(HIPBLASLT_ENABLE_JIT)
    find_package(ZLIB REQUIRED)
    target_link_libraries(hipblaslt PRIVATE ZLIB::ZLIB)
    target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_JIT)
    target_sources(hipblaslt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-tensilelite.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-tensilelite-predictor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-process.cpp")
endif()

if(HIPBLASLT_BUILD_TESTING)
    if(HIPBLASLT_ENABLE_JIT)
        add_executable(hipblaslt-jit-api-test
            "${PROJECT_SOURCE_DIR}/clients/tests/jit/public_gemm_test.cpp")
        target_link_libraries(hipblaslt-jit-api-test PRIVATE roc::hipblaslt hip::device)
        target_compile_features(hipblaslt-jit-api-test PRIVATE cxx_std_17)
        set_target_properties(hipblaslt-jit-api-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    else()
        add_executable(hipblaslt-jit-disabled-test
            "${PROJECT_SOURCE_DIR}/clients/tests/jit/disabled_api_test.cpp")
        target_link_libraries(hipblaslt-jit-disabled-test PRIVATE roc::hipblaslt hip::host)
        target_compile_features(hipblaslt-jit-disabled-test PRIVATE cxx_std_17)
        set_target_properties(hipblaslt-jit-disabled-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    endif()
endif()

# Exercise generator subprocess handling independently of GPU execution.
if(HIPBLASLT_BUILD_TESTING)
    find_package(Threads REQUIRED)
    add_executable(hipblaslt-jit-process-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/provider_process_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-process.cpp")
    target_include_directories(hipblaslt-jit-process-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(hipblaslt-jit-process-test PRIVATE cxx_std_17)
    target_link_libraries(hipblaslt-jit-process-test PRIVATE Threads::Threads)
    set_target_properties(hipblaslt-jit-process-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
endif()

if(HIPBLASLT_BUILD_TESTING)
    find_package(ZLIB REQUIRED)
    add_executable(hipblaslt-jit-artifacts-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/tensilelite_artifacts_test.cpp")
    target_include_directories(hipblaslt-jit-artifacts-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(hipblaslt-jit-artifacts-test PRIVATE cxx_std_17)
    target_link_libraries(hipblaslt-jit-artifacts-test PRIVATE ZLIB::ZLIB)
    set_target_properties(hipblaslt-jit-artifacts-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
endif()

if(HIPBLASLT_ENABLE_JIT AND HIPBLASLT_BUILD_TESTING)
    add_executable(hipblaslt-jit-backend-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/alternate_backend_test.cpp")
    target_include_directories(hipblaslt-jit-backend-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt/include")
    target_link_libraries(hipblaslt-jit-backend-test PRIVATE roc::hipblaslt hip::device)
    target_compile_features(hipblaslt-jit-backend-test PRIVATE cxx_std_17)

    add_executable(hipblaslt-jit-generic-api-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/public_gemm_test.cpp")
    target_compile_definitions(hipblaslt-jit-generic-api-test PRIVATE HIPBLASLT_TEST_GENERIC_JIT)
    target_compile_features(hipblaslt-jit-generic-api-test PRIVATE cxx_std_17)
    target_link_libraries(hipblaslt-jit-generic-api-test PRIVATE roc::hipblaslt hip::device)
    set_target_properties(hipblaslt-jit-backend-test hipblaslt-jit-generic-api-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
endif()
