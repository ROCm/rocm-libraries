# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu.
enable_testing()
find_package(Threads REQUIRED)

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

# Compiles the Jit stages directly; runs without a GPU.
add_executable(hipblaslt-jit-component-test
    jit_component_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-component.cpp")
target_include_directories(hipblaslt-jit-component-test PRIVATE "${hipblaslt_jit_source}")
target_link_libraries(hipblaslt-jit-component-test PRIVATE
    roc::hipblaslt hip::host Threads::Threads)
hipblaslt_jit_test(jit-component jit-cpu
    $<TARGET_FILE:hipblaslt-jit-component-test> "${hipblaslt_jit_scratch}/component")

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
hipblaslt_jit_stage(hipblaslt-jit-component-test hipblaslt-jit-code-object-test)
