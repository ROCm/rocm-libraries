# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipdnn_list_engines ListEngines.cpp)
# hipdnn_data_sdk supplies the canonical engine-ID hex rendering (formatEngineIdHex). It is a
# header-only INTERFACE target, and hipdnn_backend links it privately, so the tool names it here.
target_link_libraries(hipdnn_list_engines PRIVATE hipdnn_backend hipdnn_data_sdk)
target_compile_definitions(hipdnn_list_engines PRIVATE HIPDNN_BACKEND_COMPILATION)
install(TARGETS hipdnn_list_engines RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Smoke test: run the tool against a real, loadable engine plugin and assert it prints that plugin's
# own engine name rather than a hexadecimal fallback.
#
# Discovery is pointed at ${HIPDNN_TEST_PLUGIN_DIR}/default via HIPDNN_PLUGIN_DIR. The /default
# suffix is required: plugin discovery is non-recursive, the flat test-plugin directory and its
# custom/ subdirectory both hold deliberate load-rejection fixtures, and default/ holds exactly one
# plugin, test_good_default_plugin, which reports the engine name asserted below.
if(NOT HIPDNN_SKIP_TESTS AND TARGET test_good_default_plugin)
    # Mirrors K_GOOD_DEFAULT_PLUGIN_ENGINE_NAME in tests/test_plugins/TestPluginEngineIdMap.hpp and
    # the engine ID mapped to GoodDefaultPlugin in the same header, rendered by ListEngines.cpp as a
    # zero-padded uppercase 64-bit hexadecimal value.
    set(HIPDNN_LIST_ENGINES_SMOKE_EXPECTED "TEST_GOOD_DEFAULT_ENGINE (0xFFFFFFFFFFFFFFFD)")

    # The tool is a plain CLI, not a gtest binary, so the checks live in a driver script: it asserts
    # a zero exit status, an explicitly non-empty engine list, and the expected name anywhere in the
    # output. The substring search keeps the assertion independent of the listing order, which
    # varies with the engine names present.
    # cmake-format: off
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ListEnginesSmokeTest.cmake" [[
if(NOT DEFINED HIPDNN_LIST_ENGINES_TOOL OR NOT DEFINED HIPDNN_LIST_ENGINES_EXPECTED)
    message(FATAL_ERROR "HIPDNN_LIST_ENGINES_TOOL and HIPDNN_LIST_ENGINES_EXPECTED are required")
endif()

execute_process(
    COMMAND "${HIPDNN_LIST_ENGINES_TOOL}"
    OUTPUT_VARIABLE toolStdout
    ERROR_VARIABLE toolStderr
    RESULT_VARIABLE toolResult
)

message(STATUS "hipdnn_list_engines stdout:\n${toolStdout}")
message(STATUS "hipdnn_list_engines stderr:\n${toolStderr}")

if(NOT toolResult EQUAL 0)
    message(FATAL_ERROR "hipdnn_list_engines exited with '${toolResult}'")
endif()

string(FIND "${toolStdout}" "Loaded engines:" headerPosition)
if(headerPosition EQUAL -1)
    message(FATAL_ERROR
        "hipdnn_list_engines enumerated no engines; expected at least the plugin in "
        "$ENV{HIPDNN_PLUGIN_DIR}")
endif()

string(FIND "${toolStdout}" "${HIPDNN_LIST_ENGINES_EXPECTED}" namePosition)
if(namePosition EQUAL -1)
    message(FATAL_ERROR
        "hipdnn_list_engines output does not contain '${HIPDNN_LIST_ENGINES_EXPECTED}'")
endif()
]])
    # cmake-format: on

    add_test(
        NAME hipdnn_list_engines_smoke
        COMMAND
            ${CMAKE_COMMAND} "-DHIPDNN_LIST_ENGINES_TOOL=$<TARGET_FILE:hipdnn_list_engines>"
            "-DHIPDNN_LIST_ENGINES_EXPECTED=${HIPDNN_LIST_ENGINES_SMOKE_EXPECTED}" -P
            "${CMAKE_CURRENT_BINARY_DIR}/ListEnginesSmokeTest.cmake"
    )

    # The tool discovers plugins from HIPDNN_PLUGIN_DIR and, unlike the backend test binaries,
    # inherits no default for it.
    set(HIPDNN_LIST_ENGINES_SMOKE_ENVIRONMENT "HIPDNN_PLUGIN_DIR=${HIPDNN_TEST_PLUGIN_DIR}/default")
    if(DEFINED TEST_ENVIRONMENT)
        list(APPEND HIPDNN_LIST_ENGINES_SMOKE_ENVIRONMENT ${TEST_ENVIRONMENT})
    endif()
    set_tests_properties(
        hipdnn_list_engines_smoke PROPERTIES ENVIRONMENT "${HIPDNN_LIST_ENGINES_SMOKE_ENVIRONMENT}"
    )

    # PATH prepends (Windows ASAN runtime / ROCm / build DLL dirs) go through
    # ENVIRONMENT_MODIFICATION so the runtime PATH is extended rather than replaced.
    if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
        set_tests_properties(
            hipdnn_list_engines_smoke PROPERTIES ENVIRONMENT_MODIFICATION
                                                 "${TEST_ENVIRONMENT_MODIFICATION}"
        )
    endif()

    _apply_hipdnn_test_category_labels(hipdnn_list_engines_smoke)

    add_dependencies(hipdnn_list_engines test_good_default_plugin)
endif()
