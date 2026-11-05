# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT HIPDNN_ENABLE_SDPA)
    message(FATAL_ERROR
        "HIPDNN_IT_ENABLE_SDPA_CASE_FLOW_RUNNER needs hipDNN built with HIPDNN_ENABLE_SDPA=ON")
endif()

set(SDPA_CASE_FLOW_RUNNER_EXE hipdnn_sdpa_case_flow_runner)
set(_it_src ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(_runner_hash_header ${CMAKE_CURRENT_BINARY_DIR}/generated/RunnerSourceHash.hpp)
add_custom_command(
    OUTPUT ${_runner_hash_header}
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT=${_runner_hash_header}
        -DSOURCE_1=${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
        -DSOURCE_2=${_it_src}/sdpa-case-flow-poc/CaseGraph.cpp
        -DSOURCE_3=${_it_src}/sdpa-case-flow-poc/CaseGraph.hpp
        -DSOURCE_4=${CMAKE_CURRENT_SOURCE_DIR}/ShiftedStarts.hpp
        -DSOURCE_5=${CMAKE_CURRENT_SOURCE_DIR}/RaggedCompareView.hpp
        -P ${CMAKE_CURRENT_SOURCE_DIR}/GenerateSourceHash.cmake
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
        ${_it_src}/sdpa-case-flow-poc/CaseGraph.cpp
        ${_it_src}/sdpa-case-flow-poc/CaseGraph.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ShiftedStarts.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RaggedCompareView.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GenerateSourceHash.cmake
    COMMENT "Hashing the SDPA case-flow runner sources"
    VERBATIM
)

add_executable(${SDPA_CASE_FLOW_RUNNER_EXE}
    Main.cpp
    ${_runner_hash_header}
    ${_it_src}/sdpa-case-flow-poc/CaseGraph.cpp
    ${_it_src}/harness/CpuReferenceGraphExecutorAdapter.cpp
    ${_it_src}/harness/bundle/OutputComparison.cpp
    ${_it_src}/harness/bundle/VariantPackBuilder.cpp
    ${_it_src}/harness/gpu-graph-executor/GpuReferenceGraphExecutor.cpp
    ${_it_src}/harness/input-init/FillInputs.cpp
    ${_it_src}/harness/input-init/SdpaBackwardInputs.cpp
)

target_include_directories(${SDPA_CASE_FLOW_RUNNER_EXE}
    PRIVATE
        ${_it_src}
        ${_it_src}/sdpa-case-flow-poc
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(${SDPA_CASE_FLOW_RUNNER_EXE}
    PRIVATE
        hip::host
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        Threads::Threads
)

# Warnings are reported but do not fail the build.
target_compile_options(${SDPA_CASE_FLOW_RUNNER_EXE}
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS} -Wno-error)

set_target_properties(${SDPA_CASE_FLOW_RUNNER_EXE}
    PROPERTIES
        BUILD_RPATH_USE_ORIGIN TRUE
)

add_subdirectory(${_it_src}/sdpa-case-flow-poc/tests ${CMAKE_CURRENT_BINARY_DIR}/case-graph-tests)

# Pytest suite of the SDPA case-flow bundle tools (superset template grouping); needs no build.
if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest, yaml"
        RESULT_VARIABLE _case_flow_pytest_probe
        OUTPUT_QUIET ERROR_QUIET
    )
endif()
if(Python3_FOUND AND NOT _case_flow_pytest_probe)
    set(SDPA_CASE_FLOW_POC_PYTHON_TESTS hipdnn_sdpa_case_flow_poc_python_tests)
    add_test(
        NAME ${SDPA_CASE_FLOW_POC_PYTHON_TESTS}
        COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
                ${_it_src}/../scripts/sdpa_case_flow_poc/bundles/tests
    )
    set_tests_properties(${SDPA_CASE_FLOW_POC_PYTHON_TESTS}
                         PROPERTIES ENVIRONMENT "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1;PYTHONDONTWRITEBYTECODE=1")
    set(DNN_PROVIDER_CTEST_CATEGORIES_YAML
        "${_it_src}/sdpa-case-flow-poc/tests/test_categories_case_flow.yaml")
    _apply_provider_ctest_category_labels(${SDPA_CASE_FLOW_POC_PYTHON_TESTS})
elseif(Python3_FOUND)
    message(STATUS "hipdnn_sdpa_case_flow_poc_python_tests not registered: "
                   "${Python3_EXECUTABLE} cannot import pytest and yaml")
endif()
