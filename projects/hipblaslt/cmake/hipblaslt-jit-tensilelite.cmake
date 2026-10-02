# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The TensileLite generator backend: HIPBLASLT_JIT runs TensileLite Python at
# run time. Included by the HIPBLASLT_ENABLE_JIT block of
# library/src/amd_detail/CMakeLists.txt.
option(HIPBLASLT_JIT_TENSILELITE "Serve HIPBLASLT_JIT with the TensileLite generator." ON)
if(NOT HIPBLASLT_JIT_TENSILELITE)
    return()
endif()
if(NOT TARGET _rocisa)
    message(FATAL_ERROR "HIPBLASLT_JIT_TENSILELITE requires the local _rocisa target")
endif()

set(hipblaslt_jit_tensilelite_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_tensilelite_tests "${PROJECT_SOURCE_DIR}/clients/tests/jit")

# The tools the generator runs by default; each path has an environment override.
target_compile_definitions(hipblaslt PRIVATE
    HIPBLASLT_JIT_DEFAULT_PYTHON="${Python_EXECUTABLE}"
    HIPBLASLT_JIT_DEFAULT_TENSILE_SOURCE="${PROJECT_SOURCE_DIR}/tensilelite"
    HIPBLASLT_JIT_DEFAULT_PYTHONPATH="$<TARGET_FILE_DIR:_rocisa>/.."
    HIPBLASLT_JIT_DEFAULT_CXX="${CMAKE_CXX_COMPILER}")
add_dependencies(hipblaslt _rocisa)
target_sources(hipblaslt PRIVATE
    "${hipblaslt_jit_tensilelite_source}/hipblaslt-jit-tensilelite.cpp"
    "${hipblaslt_jit_tensilelite_source}/hipblaslt-jit-gemm.cpp"
    "${hipblaslt_jit_tensilelite_source}/hipblaslt-jit-process.cpp")
set(hipblaslt_jit_process_backend hipblaslt-jit-tensilelite-backend.cpp)

if(HIPBLASLT_BUILD_TESTING)
    find_package(Threads REQUIRED)
    add_executable(hipblaslt-jit-api-test
        "${hipblaslt_jit_tensilelite_tests}/public_gemm_test.cpp")
    add_executable(hipblaslt-jit-generic-api-test
        "${hipblaslt_jit_tensilelite_tests}/public_gemm_test.cpp")
    target_compile_definitions(hipblaslt-jit-generic-api-test PRIVATE HIPBLASLT_TEST_GENERIC_JIT)
    add_executable(hipblaslt-jit-direct-gemm-test
        "${hipblaslt_jit_tensilelite_tests}/direct_gemm_test.cpp")
    add_executable(hipblaslt-jit-generic-gemm-test
        "${hipblaslt_jit_tensilelite_tests}/generic_gemm_test.cpp")
    foreach(test_target hipblaslt-jit-api-test hipblaslt-jit-generic-api-test
                        hipblaslt-jit-direct-gemm-test hipblaslt-jit-generic-gemm-test)
        target_include_directories(${test_target} PRIVATE "${hipblaslt_jit_tensilelite_source}")
        target_link_libraries(${test_target} PRIVATE roc::hipblaslt hip::device)
    endforeach()

    # Exercises generator subprocess handling independently of GPU execution.
    add_executable(hipblaslt-jit-process-test
        "${hipblaslt_jit_tensilelite_tests}/provider_process_test.cpp"
        "${hipblaslt_jit_tensilelite_source}/hipblaslt-jit-process.cpp")
    target_include_directories(hipblaslt-jit-process-test PRIVATE
        "${hipblaslt_jit_tensilelite_source}")
    target_link_libraries(hipblaslt-jit-process-test PRIVATE Threads::Threads)

    foreach(test_target hipblaslt-jit-api-test hipblaslt-jit-generic-api-test
                        hipblaslt-jit-direct-gemm-test hipblaslt-jit-generic-gemm-test
                        hipblaslt-jit-process-test)
        target_compile_features(${test_target} PRIVATE cxx_std_17)
        set_target_properties(${test_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    endforeach()
endif()
