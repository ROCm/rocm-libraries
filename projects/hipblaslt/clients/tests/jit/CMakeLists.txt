# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu. A test that writes under this build tree empties its
# directory. End-to-end, heuristic and library tests create a temporary
# directory.
enable_testing()

set(hipblaslt_jit_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_scratch "${CMAKE_CURRENT_BINARY_DIR}/scratch")
set(hipblaslt_jit_data "${CMAKE_CURRENT_SOURCE_DIR}/data")
# The publish test writes the JIT solution library here.
# Replay tests read the committed bundles in place.
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
    if(HIPBLASLT_ENABLE_FETCH)
        message(STATUS "Catch2 3 was not found; fetching v3.4.0")
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.4.0
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(Catch2)
    elseif(HIPBLASLT_ENABLE_JIT)
        message(FATAL_ERROR
            "Catch2 3 was not found. The JIT tests need it: install Catch2 3, or configure "
            "with -DHIPBLASLT_ENABLE_FETCH=ON to download it at configure time.")
    else()
        # A build without JIT gains no test dependency.
        message(STATUS "Catch2 3 was not found and HIPBLASLT_ENABLE_FETCH is OFF; "
                       "the JIT tests are not built")
        return()
    endif()
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

# Catch2 tests. Extra arguments select cases. Paths are compile definitions.
function(hipblaslt_jit_catch_test name label test_target)
    target_link_libraries(${test_target} PRIVATE Catch2::Catch2WithMain)
    hipblaslt_jit_executable(${name} ${label} ${test_target} ${ARGN})
endfunction()

# What a test that compiles JIT sources directly needs to build them.
function(hipblaslt_jit_compiles_sources test_target)
    target_include_directories(${test_target} PRIVATE
        "${hipblaslt_jit_source}/rocblaslt/include"
        "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_link_libraries(${test_target} PRIVATE
        roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} amd_comgr hip::device)
    target_compile_definitions(${test_target} PRIVATE
        "HIPBLASLT_JIT_DATA=\"${hipblaslt_jit_data}\"")
endfunction()

add_executable(hipblaslt-jit-source-bundle-test source_bundle_test.cpp)
target_link_libraries(hipblaslt-jit-source-bundle-test PRIVATE
    roc::hipblaslt roc::tensilelite-host hip::device)
target_compile_definitions(hipblaslt-jit-source-bundle-test PRIVATE
    "HIPBLASLT_JIT_SCRATCH=\"${hipblaslt_jit_scratch}/source-bundle\"")
hipblaslt_jit_catch_test(jit-source-bundle jit-cpu hipblaslt-jit-source-bundle-test)

# The GPU tests need a GPU target with committed assembly. A YAML build cannot
# read the MsgPack entries.
set(hipblaslt_jit_gpu OFF)
foreach(target IN LISTS GPU_TARGETS)
    string(REGEX REPLACE ":.*" "" architecture "${target}")
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/${architecture}")
        set(hipblaslt_jit_gpu ON)
    endif()
endforeach()

if(NOT HIPBLASLT_ENABLE_JIT)
    # HIPBLASLT_JIT must not change heuristic queries when JIT is not built.
    if(hipblaslt_jit_gpu AND NOT HIPBLASLT_ENABLE_YAML)
        add_executable(hipblaslt-jit-heuristic-test jit_heuristic_test.cpp)
        target_link_libraries(hipblaslt-jit-heuristic-test PRIVATE
            roc::hipblaslt roc::tensilelite-host hip::device)
        target_compile_definitions(hipblaslt-jit-heuristic-test PRIVATE HIPBLASLT_JIT_HEURISTIC_IGNORED)
        hipblaslt_jit_catch_test(jit-heuristic-ignored jit-gpu hipblaslt-jit-heuristic-test
            "heuristic ignored")
    endif()
    return()
endif()

# Compiles the comgr builder directly and builds a hand-written HIP kernel and
# every committed assembly file for its target, alone and linked with that
# kernel. Needs no GPU, so every architecture runs it.
add_executable(hipblaslt-jit-builder-test
    builder_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
hipblaslt_jit_compiles_sources(hipblaslt-jit-builder-test)
target_compile_definitions(hipblaslt-jit-builder-test PRIVATE
    "HIPBLASLT_JIT_KERNEL=\"${CMAKE_CURRENT_SOURCE_DIR}/builder_test_kernel.hip\""
    "HIPBLASLT_JIT_SCRATCH=\"${hipblaslt_jit_scratch}/builder\"")
hipblaslt_jit_catch_test(jit-builder jit-cpu hipblaslt-jit-builder-test)

if(NOT hipblaslt_jit_gpu OR HIPBLASLT_ENABLE_YAML)
    return()
endif()

# Publishes the device's committed kernels into the JIT solution library.
# CTest runs it before the loader.
add_executable(hipblaslt-jit-library-writer
    library_writer.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    ${hipblaslt_jit_library_sources})
hipblaslt_jit_compiles_sources(hipblaslt-jit-library-writer)
target_compile_definitions(hipblaslt-jit-library-writer PRIVATE
    "HIPBLASLT_JIT_LIBRARY=\"${hipblaslt_jit_library}\""
    "HIPBLASLT_JIT_SCRATCH=\"${hipblaslt_jit_scratch}/publish\"")
hipblaslt_jit_catch_test(jit-publish jit-gpu hipblaslt-jit-library-writer)
set_tests_properties(jit-publish PROPERTIES FIXTURES_SETUP jit-publish)

# Loads that library and finds the pre-generated kernel's solutions.
add_executable(hipblaslt-jit-loader-test
    loader_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    ${hipblaslt_jit_library_sources})
hipblaslt_jit_compiles_sources(hipblaslt-jit-loader-test)
target_compile_definitions(hipblaslt-jit-loader-test PRIVATE
    "HIPBLASLT_JIT_LIBRARY=\"${hipblaslt_jit_library}\""
    "HIPBLASLT_JIT_SCRATCH=\"${hipblaslt_jit_scratch}/loader\"")
hipblaslt_jit_catch_test(jit-loader jit-gpu hipblaslt-jit-loader-test)
set_tests_properties(jit-loader PROPERTIES FIXTURES_REQUIRED jit-publish)

# Compiles the committed assembly, publishes it, retrieves it, launches it and
# checks D. Its jit-cpu cases store a helper relocatable and time out on a held
# file lock.
add_executable(hipblaslt-jit-launch-test
    launch_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    ${hipblaslt_jit_library_sources})
hipblaslt_jit_compiles_sources(hipblaslt-jit-launch-test)
target_compile_definitions(hipblaslt-jit-launch-test PRIVATE
    "HIPBLASLT_JIT_HELPER_LIBRARY=\"${hipblaslt_jit_scratch}/helper-library\""
    "HIPBLASLT_JIT_KERNEL=\"${CMAKE_CURRENT_SOURCE_DIR}/builder_test_kernel.hip\""
    "HIPBLASLT_JIT_LAUNCH_LIBRARY=\"${hipblaslt_jit_scratch}/launch-library\""
    "HIPBLASLT_JIT_LAUNCH_SCRATCH=\"${hipblaslt_jit_scratch}/launch\""
    "HIPBLASLT_JIT_LOCK_SCRATCH=\"${hipblaslt_jit_scratch}/lock\"")
hipblaslt_jit_catch_test(jit-launch jit-gpu hipblaslt-jit-launch-test "[jit-gpu]")
hipblaslt_jit_catch_test(jit-launch-cpu jit-cpu hipblaslt-jit-launch-test "[jit-cpu]")

# Compiles the JIT solution library directly. It builds the committed assembly
# of the current device's architecture and publishes it, and loads no code,
# but TensileLite queries the current device when it reads a solution.
find_package(Threads REQUIRED)
add_executable(hipblaslt-jit-library-test
    jit_library_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    ${hipblaslt_jit_library_sources})
hipblaslt_jit_compiles_sources(hipblaslt-jit-library-test)
target_link_libraries(hipblaslt-jit-library-test PRIVATE Threads::Threads Catch2::Catch2)
hipblaslt_jit_executable(jit-library jit-gpu hipblaslt-jit-library-test "[library-checks]")
hipblaslt_jit_executable(jit-library-concurrency jit-gpu hipblaslt-jit-library-test
    "[library-concurrency]")

add_executable(hipblaslt-jit-end-to-end-test
    jit_end_to_end_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
hipblaslt_jit_compiles_sources(hipblaslt-jit-end-to-end-test)
hipblaslt_jit_catch_test(jit-end-to-end jit-gpu hipblaslt-jit-end-to-end-test)

# One process per mode: HIPBLASLT_JIT is read once. The test sets that variable
# and points HIPBLASLT_JIT_LIBRARY_PATH at a scratch directory it creates.
add_executable(hipblaslt-jit-heuristic-test jit_heuristic_test.cpp)
target_link_libraries(hipblaslt-jit-heuristic-test PRIVATE
    roc::hipblaslt roc::tensilelite-host hip::device)
target_compile_definitions(hipblaslt-jit-heuristic-test PRIVATE
    "HIPBLASLT_JIT_DATA=\"${hipblaslt_jit_data}\"")
foreach(mode IN ITEMS off fallback forced)
    hipblaslt_jit_catch_test(jit-heuristic-${mode} jit-gpu hipblaslt-jit-heuristic-test
        "heuristic ${mode}")
endforeach()
