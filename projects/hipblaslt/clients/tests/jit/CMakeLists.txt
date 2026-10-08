# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu. A test that writes under this build tree empties its
# directory. End-to-end, heuristic and library tests create a temporary directory.
enable_testing()

set(hipblaslt_jit_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_scratch "${CMAKE_CURRENT_BINARY_DIR}/scratch")
set(hipblaslt_jit_data "${CMAKE_CURRENT_SOURCE_DIR}/data")
# The publish test writes the JIT solution library here. Replay tests stage a
# temporary source directory themselves, from the same descriptions.
set(hipblaslt_jit_library "${hipblaslt_jit_scratch}/published")
set(hipblaslt_jit_library_sources
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-fs.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-library.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-msgpack.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-problem-type.cpp")

find_package(Catch2 3 CONFIG QUIET)
if(NOT Catch2_FOUND)
    find_package(Catch2 3 CONFIG QUIET PATHS /usr)
endif()
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(Catch2)
endif()

# hipblaslt_jit_executable(name label target args...) stages target and runs it.
function(hipblaslt_jit_executable name label test_target)
    target_include_directories(${test_target} PRIVATE
        "${hipblaslt_jit_source}" "${PROJECT_SOURCE_DIR}/library/include")
    target_compile_features(${test_target} PRIVATE cxx_std_17)
    set_target_properties(${test_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${test_target}> ${ARGN})
    set_tests_properties(${name} PROPERTIES LABELS ${label})
endfunction()

# Catch2 tests. Extra arguments select one case. Paths are compile definitions.
function(hipblaslt_jit_catch_test name label test_target)
    target_link_libraries(${test_target} PRIVATE Catch2::Catch2WithMain)
    hipblaslt_jit_executable(${name} ${label} ${test_target} ${ARGN})
endfunction()

# Tests with their own main. Extra arguments are passed to the executable.
function(hipblaslt_jit_test name label test_target)
    target_link_libraries(${test_target} PRIVATE hip::host roc::hipblaslt)
    hipblaslt_jit_executable(${name} ${label} ${test_target} ${ARGN})
endfunction()

add_executable(hipblaslt-jit-source-bundle-test source_bundle_test.cpp)
target_link_libraries(hipblaslt-jit-source-bundle-test PRIVATE roc::hipblaslt hip::host)
target_compile_definitions(hipblaslt-jit-source-bundle-test PRIVATE
    "HIPBLASLT_JIT_SCRATCH=\"${hipblaslt_jit_scratch}/source-bundle\"")
hipblaslt_jit_catch_test(jit-source-bundle jit-cpu hipblaslt-jit-source-bundle-test)

if(NOT HIPBLASLT_ENABLE_JIT)
    add_executable(hipblaslt-jit-disabled-test disabled_api_test.cpp)
    target_link_libraries(hipblaslt-jit-disabled-test PRIVATE roc::hipblaslt hip::host)
    hipblaslt_jit_catch_test(jit-disabled jit-cpu hipblaslt-jit-disabled-test)

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
        target_compile_definitions(hipblaslt-jit-heuristic-test PRIVATE HIPBLASLT_JIT_HEURISTIC_IGNORED)
        hipblaslt_jit_catch_test(jit-heuristic-ignored jit-gpu hipblaslt-jit-heuristic-test
            "heuristic ignored")
    endif()
    return()
endif()

# Compiles the comgr builder directly and builds a hand-written HIP kernel and
# every committed assembly file for its target. Needs no GPU, so every
# architecture runs it.
add_executable(hipblaslt-jit-builder-test
    builder_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
target_include_directories(hipblaslt-jit-builder-test PRIVATE
    "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-builder-test PRIVATE
    roc::hipblaslt amd_comgr hip::host)
target_compile_definitions(hipblaslt-jit-builder-test PRIVATE
    "HIPBLASLT_JIT_DATA=\"${hipblaslt_jit_data}\""
    "HIPBLASLT_JIT_KERNEL=\"${CMAKE_CURRENT_SOURCE_DIR}/builder_test_kernel.hip\""
    "HIPBLASLT_JIT_SCRATCH=\"${hipblaslt_jit_scratch}/builder\"")
hipblaslt_jit_catch_test(jit-builder jit-cpu hipblaslt-jit-builder-test)

# The GPU tests need a GPU target with committed assembly. A YAML build cannot
# read the MsgPack entries.
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

# Publishes the device's committed kernels into the JIT solution library.
# CTest runs it before the loader.
add_executable(hipblaslt-jit-library-writer
    library_writer.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    ${hipblaslt_jit_library_sources})
target_include_directories(hipblaslt-jit-library-writer PRIVATE
    "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-library-writer PRIVATE
    roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} amd_comgr hip::device)
target_compile_definitions(hipblaslt-jit-library-writer PRIVATE
    "HIPBLASLT_JIT_DATA=\"${hipblaslt_jit_data}\""
    "HIPBLASLT_JIT_LIBRARY=\"${hipblaslt_jit_library}\""
    "HIPBLASLT_JIT_SCRATCH=\"${hipblaslt_jit_scratch}/publish\"")
hipblaslt_jit_catch_test(jit-publish jit-gpu hipblaslt-jit-library-writer)
set_tests_properties(jit-publish PROPERTIES FIXTURES_SETUP jit-publish)

# Loads that library and resolves the pre-generated kernel.
add_executable(hipblaslt-jit-loader-test
    loader_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-loader.cpp"
    ${hipblaslt_jit_library_sources})
target_include_directories(hipblaslt-jit-loader-test PRIVATE
    "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-loader-test PRIVATE
    roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} amd_comgr hip::device)
target_compile_definitions(hipblaslt-jit-loader-test PRIVATE
    "HIPBLASLT_JIT_DATA=\"${hipblaslt_jit_data}\""
    "HIPBLASLT_JIT_LIBRARY=\"${hipblaslt_jit_library}\"")
hipblaslt_jit_catch_test(jit-loader jit-gpu hipblaslt-jit-loader-test)
set_tests_properties(jit-loader PROPERTIES FIXTURES_REQUIRED jit-publish)

# Compiles the JIT solution library directly. It builds the plain entry of the
# current device's architecture from the committed assembly and loads no code,
# but TensileLite queries the current device when it reads an entry.
find_package(Threads REQUIRED)
add_executable(hipblaslt-jit-library-test
    jit_library_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-fs.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-library.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-msgpack.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-problem-type.cpp")
target_include_directories(hipblaslt-jit-library-test PRIVATE
    "${hipblaslt_jit_source}/rocblaslt/src/include")
target_compile_definitions(hipblaslt-jit-library-test PRIVATE
    HIPBLASLT_ENABLE_JIT
    "HIPBLASLT_JIT_DATA=\"${hipblaslt_jit_data}\"")
target_link_libraries(hipblaslt-jit-library-test PRIVATE
    roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} amd_comgr
    hip::device Threads::Threads)
hipblaslt_jit_catch_test(jit-library jit-gpu hipblaslt-jit-library-test "[library-checks]")
hipblaslt_jit_catch_test(jit-library-concurrency jit-gpu hipblaslt-jit-library-test
    "[library-concurrency]")

add_executable(hipblaslt-jit-end-to-end-test jit_end_to_end_test.cpp)
target_compile_definitions(hipblaslt-jit-end-to-end-test PRIVATE
    "HIPBLASLT_JIT_DATA=\"${hipblaslt_jit_data}\"")
target_link_libraries(hipblaslt-jit-end-to-end-test PRIVATE roc::hipblaslt hip::device)
hipblaslt_jit_catch_test(jit-end-to-end jit-gpu hipblaslt-jit-end-to-end-test
    "replayed JIT GEMM solutions run end to end")
hipblaslt_jit_catch_test(jit-end-to-end-library jit-gpu hipblaslt-jit-end-to-end-test
    "getLibraryAlgos publishes plain-pair solutions")

# One process per mode: HIPBLASLT_JIT is read once. The test sets that variable
# and points HIPBLASLT_JIT_LIBRARY_PATH at a scratch directory it creates.
add_executable(hipblaslt-jit-heuristic-test jit_heuristic_test.cpp)
target_link_libraries(hipblaslt-jit-heuristic-test PRIVATE roc::hipblaslt hip::device)
if(HIPBLASLT_JIT_BUILT_HIPKITTENS)
    target_compile_definitions(hipblaslt-jit-heuristic-test PRIVATE HIPBLASLT_JIT_HIPKITTENS)
endif()
foreach(mode IN ITEMS off fallback forced)
    hipblaslt_jit_catch_test(jit-heuristic-${mode} jit-gpu hipblaslt-jit-heuristic-test
        "heuristic ${mode}")
endforeach()

if(HIPBLASLT_JIT_BUILT_HIPKITTENS)
    add_executable(hipblaslt-jit-hipkittens-test hipkittens_backend_test.cpp)
    target_include_directories(hipblaslt-jit-hipkittens-test PRIVATE
        "${hipblaslt_jit_source}"
        "${hipblaslt_jit_source}/rocblaslt/include"
        "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_link_libraries(hipblaslt-jit-hipkittens-test PRIVATE roc::hipblaslt hip::device)
    hipblaslt_jit_catch_test(jit-hipkittens jit-gpu hipblaslt-jit-hipkittens-test)
endif()
