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
