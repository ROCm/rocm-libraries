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
