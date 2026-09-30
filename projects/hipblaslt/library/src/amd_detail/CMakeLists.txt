# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocblaslt)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_ostream.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext-op.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-mode.cpp"
)

if(HIPBLASLT_ENABLE_JIT)
    if(NOT TARGET _rocisa)
        message(FATAL_ERROR "HIPBLASLT_ENABLE_JIT requires the local _rocisa target")
    endif()
    find_package(ZLIB REQUIRED)
    find_package(amd_comgr CONFIG REQUIRED)
    target_link_libraries(hipblaslt PRIVATE ZLIB::ZLIB amd_comgr)
    # The generator HIPBLASLT_JIT runs by default; each path has an environment override.
    target_compile_definitions(hipblaslt PRIVATE
        HIPBLASLT_ENABLE_JIT
        HIPBLASLT_JIT_DEFAULT_PYTHON="${Python_EXECUTABLE}"
        HIPBLASLT_JIT_DEFAULT_TENSILE_SOURCE="${PROJECT_SOURCE_DIR}/tensilelite"
        HIPBLASLT_JIT_DEFAULT_PYTHONPATH="$<TARGET_FILE_DIR:_rocisa>/.."
        HIPBLASLT_JIT_DEFAULT_CXX="${CMAKE_CXX_COMPILER}")
    add_dependencies(hipblaslt _rocisa)
    target_sources(hipblaslt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-code-object.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-fs.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-msgpack.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-library.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-builder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-tensilelite.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-tensilelite-defaults.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-origami-predictor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-problem-type.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-component.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-loader.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-mock-backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-process.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-report.cpp")
endif()

if(HIPBLASLT_BUILD_TESTING)
    if(HIPBLASLT_ENABLE_JIT)
        add_executable(hipblaslt-jit-api-test
            "${PROJECT_SOURCE_DIR}/clients/tests/jit/public_gemm_test.cpp")
        add_executable(hipblaslt-jit-direct-gemm-test
            "${PROJECT_SOURCE_DIR}/clients/tests/jit/direct_gemm_test.cpp")
        add_executable(hipblaslt-jit-generic-gemm-test
            "${PROJECT_SOURCE_DIR}/clients/tests/jit/generic_gemm_test.cpp")
        foreach(test_target hipblaslt-jit-api-test hipblaslt-jit-direct-gemm-test
                            hipblaslt-jit-generic-gemm-test)
            target_include_directories(${test_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
            target_link_libraries(${test_target} PRIVATE roc::hipblaslt hip::device)
            target_compile_features(${test_target} PRIVATE cxx_std_17)
            set_target_properties(${test_target} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
        endforeach()
    else()
        add_executable(hipblaslt-jit-disabled-test
            "${PROJECT_SOURCE_DIR}/clients/tests/jit/disabled_api_test.cpp")
        target_link_libraries(hipblaslt-jit-disabled-test PRIVATE roc::hipblaslt hip::host)
        target_compile_features(hipblaslt-jit-disabled-test PRIVATE cxx_std_17)
        set_target_properties(hipblaslt-jit-disabled-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    endif()
endif()

# Public API only, so it also runs against builds without JIT.
if(HIPBLASLT_BUILD_TESTING)
    find_package(Threads REQUIRED)
    add_executable(hipblaslt-jit-heuristic-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/heuristic_test.cpp")
    target_link_libraries(hipblaslt-jit-heuristic-test
        PRIVATE roc::hipblaslt hip::device Threads::Threads)
    target_compile_features(hipblaslt-jit-heuristic-test PRIVATE cxx_std_17)
    set_target_properties(hipblaslt-jit-heuristic-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
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
    add_executable(hipblaslt-jit-mock-backend-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/mock_backend_test.cpp")
    target_include_directories(hipblaslt-jit-mock-backend-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt/include")
    target_link_libraries(hipblaslt-jit-mock-backend-test PRIVATE roc::hipblaslt hip::device)
    target_compile_features(hipblaslt-jit-mock-backend-test PRIVATE cxx_std_17)

    # Compiles the Jit stages directly; runs without a GPU.
    add_executable(hipblaslt-jit-component-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/jit_component_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-component.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-tensilelite-defaults.cpp")
    target_include_directories(hipblaslt-jit-component-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(hipblaslt-jit-component-test PRIVATE
        roc::hipblaslt hip::host Threads::Threads)
    target_compile_features(hipblaslt-jit-component-test PRIVATE cxx_std_17)

    # Compiles the comgr builder directly; runs without a GPU unless --gpu is given.
    add_executable(hipblaslt-jit-code-object-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/code_object_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-code-object.cpp")
    target_include_directories(hipblaslt-jit-code-object-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt/src/include")
    target_link_libraries(hipblaslt-jit-code-object-test PRIVATE amd_comgr hip::host Threads::Threads)
    target_compile_features(hipblaslt-jit-code-object-test PRIVATE cxx_std_17)

    add_executable(hipblaslt-jit-generic-api-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/public_gemm_test.cpp")
    target_include_directories(hipblaslt-jit-generic-api-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(hipblaslt-jit-generic-api-test PRIVATE HIPBLASLT_TEST_GENERIC_JIT)
    target_compile_features(hipblaslt-jit-generic-api-test PRIVATE cxx_std_17)
    target_link_libraries(hipblaslt-jit-generic-api-test PRIVATE roc::hipblaslt hip::device)
    set_target_properties(hipblaslt-jit-mock-backend-test hipblaslt-jit-component-test
                          hipblaslt-jit-code-object-test hipblaslt-jit-generic-api-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")

    # Compiles the JIT solution library directly; runs without a GPU.
    if(NOT HIPBLASLT_ENABLE_YAML)
        add_executable(hipblaslt-jit-library-test
            "${PROJECT_SOURCE_DIR}/clients/tests/jit/jit_library_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-code-object.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-fs.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-library.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-msgpack.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-problem-type.cpp")
        target_include_directories(hipblaslt-jit-library-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt/src/include")
        target_compile_definitions(hipblaslt-jit-library-test PRIVATE HIPBLASLT_JIT_LIBRARY_TESTING)
        target_link_libraries(hipblaslt-jit-library-test PRIVATE
            roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} ZLIB::ZLIB amd_comgr
            hip::device Threads::Threads)
        target_compile_features(hipblaslt-jit-library-test PRIVATE cxx_std_17)
        set_target_properties(hipblaslt-jit-library-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    endif()
endif()
