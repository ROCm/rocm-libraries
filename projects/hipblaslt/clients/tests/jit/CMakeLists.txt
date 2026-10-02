# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu.
enable_testing()
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

set(hipblaslt_jit_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_scratch "${CMAKE_CURRENT_BINARY_DIR}/scratch")

# Each test writes to its own directory in an emptied scratch directory.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reset-scratch.cmake"
     "file(REMOVE_RECURSE \"${hipblaslt_jit_scratch}\")\n"
     "file(MAKE_DIRECTORY \"${hipblaslt_jit_scratch}\")\n")
add_test(NAME jit-scratch
         COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/reset-scratch.cmake")
set_tests_properties(jit-scratch PROPERTIES FIXTURES_SETUP hipblaslt-jit-scratch)

function(hipblaslt_jit_test name label)
    add_test(NAME ${name} COMMAND ${ARGN})
    set_tests_properties(${name} PROPERTIES
        LABELS ${label} FIXTURES_REQUIRED hipblaslt-jit-scratch)
endfunction()

function(hipblaslt_jit_stage)
    foreach(test_target ${ARGN})
        target_compile_features(${test_target} PRIVATE cxx_std_17)
        set_target_properties(${test_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    endforeach()
endfunction()

# Public API only, so it also builds without JIT.
add_executable(hipblaslt-jit-heuristic-test heuristic_test.cpp)
target_link_libraries(hipblaslt-jit-heuristic-test
    PRIVATE roc::hipblaslt hip::device Threads::Threads)

add_executable(hipblaslt-jit-artifacts-test tensilelite_artifacts_test.cpp)
target_include_directories(hipblaslt-jit-artifacts-test PRIVATE "${hipblaslt_jit_source}")
target_link_libraries(hipblaslt-jit-artifacts-test PRIVATE ZLIB::ZLIB)
hipblaslt_jit_test(jit-artifacts jit-cpu
    $<TARGET_FILE:hipblaslt-jit-artifacts-test> "${hipblaslt_jit_scratch}/artifacts")
hipblaslt_jit_stage(hipblaslt-jit-heuristic-test hipblaslt-jit-artifacts-test)

if(NOT HIPBLASLT_ENABLE_JIT)
    add_executable(hipblaslt-jit-disabled-test disabled_api_test.cpp)
    target_link_libraries(hipblaslt-jit-disabled-test PRIVATE roc::hipblaslt hip::host)
    hipblaslt_jit_stage(hipblaslt-jit-disabled-test)
    hipblaslt_jit_test(jit-disabled jit-cpu $<TARGET_FILE:hipblaslt-jit-disabled-test>)
    return()
endif()

# Compiles the Jit stages directly; runs without a GPU.
add_executable(hipblaslt-jit-component-test
    jit_component_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-component.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-process.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-tensilelite-defaults.cpp")
target_include_directories(hipblaslt-jit-component-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_compile_definitions(hipblaslt-jit-component-test PRIVATE HIPBLASLT_ENABLE_JIT)
target_link_libraries(hipblaslt-jit-component-test PRIVATE
    roc::hipblaslt hip::host Threads::Threads)
hipblaslt_jit_test(jit-component jit-cpu
    $<TARGET_FILE:hipblaslt-jit-component-test> "${hipblaslt_jit_scratch}/component")

# Compiles HIPBLASLT_JIT_DEBUG parsing, the line sink and the child observer directly.
add_executable(hipblaslt-jit-debug-test
    jit_debug_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-process.cpp")
target_include_directories(hipblaslt-jit-debug-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_compile_definitions(hipblaslt-jit-debug-test PRIVATE HIPBLASLT_ENABLE_JIT)
target_link_libraries(hipblaslt-jit-debug-test PRIVATE Threads::Threads)
hipblaslt_jit_test(jit-debug jit-cpu
    $<TARGET_FILE:hipblaslt-jit-debug-test> "${hipblaslt_jit_scratch}/debug")

# Compiles the comgr builder directly; runs without a GPU unless --gpu is given.
add_executable(hipblaslt-jit-code-object-test
    code_object_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
target_include_directories(hipblaslt-jit-code-object-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-code-object-test PRIVATE amd_comgr hip::host Threads::Threads)
hipblaslt_jit_test(jit-code-object jit-cpu
    $<TARGET_FILE:hipblaslt-jit-code-object-test>
    --target gfx950 --out "${hipblaslt_jit_scratch}/code-object")
hipblaslt_jit_test(jit-code-object-gpu jit-gpu
    $<TARGET_FILE:hipblaslt-jit-code-object-test>
    --gpu --out "${hipblaslt_jit_scratch}/code-object-gpu")
hipblaslt_jit_stage(hipblaslt-jit-component-test hipblaslt-jit-debug-test
                    hipblaslt-jit-code-object-test)

if(HIPBLASLT_JIT_TESTING)
    add_executable(hipblaslt-jit-mock-backend-test mock_backend_test.cpp)
    target_include_directories(hipblaslt-jit-mock-backend-test PRIVATE
        "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/include")
    target_link_libraries(hipblaslt-jit-mock-backend-test PRIVATE roc::hipblaslt hip::device)
    hipblaslt_jit_stage(hipblaslt-jit-mock-backend-test)
endif()

# Compiles the JIT solution library directly; runs without a GPU.
if(NOT HIPBLASLT_ENABLE_YAML)
    add_executable(hipblaslt-jit-library-test
        jit_library_test.cpp
        "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-fs.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-library.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-msgpack.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-problem-type.cpp")
    target_include_directories(hipblaslt-jit-library-test PRIVATE
        "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_compile_definitions(hipblaslt-jit-library-test PRIVATE
        HIPBLASLT_ENABLE_JIT HIPBLASLT_JIT_LIBRARY_TESTING)
    target_link_libraries(hipblaslt-jit-library-test PRIVATE
        roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} ZLIB::ZLIB amd_comgr
        hip::device Threads::Threads)
    hipblaslt_jit_stage(hipblaslt-jit-library-test)
endif()
