# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu. Each test empties its own scratch directory.
enable_testing()

set(hipblaslt_jit_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_scratch "${CMAKE_CURRENT_BINARY_DIR}/scratch")
# The jit-bundles test writes each bundle that bundle_writer.cpp describes here,
# under its architecture, with its library entry and manifest; data/README.md
# lists how the assembly was generated.
set(hipblaslt_jit_bundles "${hipblaslt_jit_scratch}/bundles")

# hipblaslt_jit_test(name label target args...) stages target and runs it.
function(hipblaslt_jit_test name label test_target)
    target_compile_features(${test_target} PRIVATE cxx_std_17)
    set_target_properties(${test_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${test_target}> ${ARGN})
    set_tests_properties(${name} PROPERTIES LABELS ${label})
endfunction()

add_executable(hipblaslt-jit-source-bundle-test source_bundle_test.cpp)
target_include_directories(hipblaslt-jit-source-bundle-test PRIVATE "${hipblaslt_jit_source}")
hipblaslt_jit_test(jit-source-bundle jit-cpu hipblaslt-jit-source-bundle-test
    "${hipblaslt_jit_scratch}/source-bundle")

if(NOT HIPBLASLT_ENABLE_JIT)
    add_executable(hipblaslt-jit-disabled-test disabled_api_test.cpp)
    target_link_libraries(hipblaslt-jit-disabled-test PRIVATE roc::hipblaslt hip::host)
    hipblaslt_jit_test(jit-disabled jit-cpu hipblaslt-jit-disabled-test)

    # HIPBLASLT_JIT must not change heuristic queries when JIT is not built.
    set(hipblaslt_jit_gpu OFF)
    foreach(target IN LISTS GPU_TARGETS)
        string(REGEX REPLACE ":.*" "" architecture "${target}")
        if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/${architecture}")
            set(hipblaslt_jit_gpu ON)
        endif()
    endforeach()
    if(hipblaslt_jit_gpu AND NOT HIPBLASLT_ENABLE_YAML)
        add_executable(hipblaslt-jit-heuristic-test jit_heuristic_test.cpp)
        target_link_libraries(hipblaslt-jit-heuristic-test PRIVATE roc::hipblaslt hip::device)
        hipblaslt_jit_test(jit-heuristic-ignored jit-gpu hipblaslt-jit-heuristic-test ignored)
        set_tests_properties(jit-heuristic-ignored PROPERTIES ENVIRONMENT "HIPBLASLT_JIT=2")
    endif()
    return()
endif()

# Writes the bundles. CTest runs it before each test that reads one.
add_executable(hipblaslt-jit-bundle-writer bundle_writer.cpp)
hipblaslt_jit_test(jit-bundles jit-cpu hipblaslt-jit-bundle-writer
    "${CMAKE_CURRENT_SOURCE_DIR}/data" "${hipblaslt_jit_scratch}/bundles")
set_tests_properties(jit-bundles PROPERTIES FIXTURES_SETUP jit-bundles)

# Compiles the comgr builder directly and builds a hand-written HIP kernel and
# every written bundle for its target. Needs no GPU, so every architecture runs
# it.
add_executable(hipblaslt-jit-builder-test
    builder_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
target_include_directories(hipblaslt-jit-builder-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-builder-test PRIVATE
    roc::hipblaslt amd_comgr hip::host)
hipblaslt_jit_test(jit-builder jit-cpu hipblaslt-jit-builder-test
    "${hipblaslt_jit_bundles}" "${CMAKE_CURRENT_SOURCE_DIR}/builder_test_kernel.hip"
    "${hipblaslt_jit_scratch}/builder")
set_tests_properties(jit-builder PROPERTIES FIXTURES_REQUIRED jit-bundles)

# The GPU tests run the bundles of the device's architecture, so they need a GPU
# target with committed bundles. A YAML build cannot read the MsgPack entries.
set(hipblaslt_jit_gpu OFF)
foreach(target IN LISTS GPU_TARGETS)
    string(REGEX REPLACE ":.*" "" architecture "${target}")
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/${architecture}")
        set(hipblaslt_jit_gpu ON)
    endif()
endforeach()
if(NOT hipblaslt_jit_gpu OR HIPBLASLT_ENABLE_YAML)
    return()
endif()

# Compiles the comgr builder and the TensileLite loader directly.
add_executable(hipblaslt-jit-loader-test
    loader_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-loader.cpp")
target_include_directories(hipblaslt-jit-loader-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-loader-test PRIVATE
    roc::hipblaslt roc::tensilelite-host amd_comgr hip::device)
hipblaslt_jit_test(jit-loader jit-gpu hipblaslt-jit-loader-test
    "${hipblaslt_jit_bundles}" "${hipblaslt_jit_scratch}/loader")
set_tests_properties(jit-loader PROPERTIES FIXTURES_REQUIRED jit-bundles)

# Compiles the JIT solution library directly. It reads the gfx950 plain entry
# and loads no code, but TensileLite queries the current device when it reads
# an entry.
find_package(Threads REQUIRED)
add_executable(hipblaslt-jit-library-test
    jit_library_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-fs.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-library.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-msgpack.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-problem-type.cpp")
target_include_directories(hipblaslt-jit-library-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_compile_definitions(hipblaslt-jit-library-test PRIVATE
    HIPBLASLT_ENABLE_JIT HIPBLASLT_JIT_LIBRARY_TESTING)
target_link_libraries(hipblaslt-jit-library-test PRIVATE
    roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} amd_comgr
    hip::device Threads::Threads)
hipblaslt_jit_test(jit-library jit-gpu hipblaslt-jit-library-test
    "${hipblaslt_jit_bundles}/gfx950/plain" "${hipblaslt_jit_scratch}/library")
hipblaslt_jit_test(jit-library-concurrency jit-gpu hipblaslt-jit-library-test
    "${hipblaslt_jit_bundles}/gfx950/plain" "${hipblaslt_jit_scratch}/library-concurrency"
    --writers 8 --per-writer 4)
set_tests_properties(jit-library jit-library-concurrency PROPERTIES
    FIXTURES_REQUIRED jit-bundles)

add_executable(hipblaslt-jit-end-to-end-test jit_end_to_end_test.cpp)
target_include_directories(hipblaslt-jit-end-to-end-test PRIVATE "${hipblaslt_jit_source}")
target_link_libraries(hipblaslt-jit-end-to-end-test PRIVATE roc::hipblaslt hip::device)
hipblaslt_jit_test(jit-end-to-end jit-gpu hipblaslt-jit-end-to-end-test
    "${hipblaslt_jit_bundles}")
set_tests_properties(jit-end-to-end PROPERTIES FIXTURES_REQUIRED jit-bundles)
# Publishes the solutions in one process and runs their indices in a second.
hipblaslt_jit_test(jit-end-to-end-library jit-gpu hipblaslt-jit-end-to-end-test
    "${hipblaslt_jit_bundles}" --library)
set_tests_properties(jit-end-to-end-library PROPERTIES
    FIXTURES_REQUIRED jit-bundles
    ENVIRONMENT HIPBLASLT_JIT_LIBRARY_PATH=${hipblaslt_jit_scratch}/end-to-end-library)

# One process per mode: HIPBLASLT_JIT is read once. The test sets
# HIPBLASLT_JIT_TEST_REPLAY to the device's plain-pair bundle.
add_executable(hipblaslt-jit-heuristic-test jit_heuristic_test.cpp)
target_link_libraries(hipblaslt-jit-heuristic-test PRIVATE roc::hipblaslt hip::device)
foreach(mode IN ITEMS off fallback forced)
    if(mode STREQUAL "off")
        set(hipblaslt_jit_mode_value 0)
    elseif(mode STREQUAL "fallback")
        set(hipblaslt_jit_mode_value 1)
    else()
        set(hipblaslt_jit_mode_value 2)
    endif()
    hipblaslt_jit_test(jit-heuristic-${mode} jit-gpu hipblaslt-jit-heuristic-test
        ${mode} "${hipblaslt_jit_bundles}")
    set_tests_properties(jit-heuristic-${mode} PROPERTIES
        FIXTURES_REQUIRED jit-bundles
        ENVIRONMENT "HIPBLASLT_JIT=${hipblaslt_jit_mode_value}")
endforeach()

if(HIPBLASLT_JIT_BUILT_HIPKITTENS)
    add_executable(hipblaslt-jit-hipkittens-test hipkittens_backend_test.cpp)
    target_include_directories(hipblaslt-jit-hipkittens-test PRIVATE
        "${hipblaslt_jit_source}"
        "${hipblaslt_jit_source}/rocblaslt/include"
        "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_link_libraries(hipblaslt-jit-hipkittens-test PRIVATE roc::hipblaslt hip::device)
    hipblaslt_jit_test(jit-hipkittens jit-gpu hipblaslt-jit-hipkittens-test)
endif()
