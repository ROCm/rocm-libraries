# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(bench)

# corpus_gen generates UHD training corpora, and three of its headers use the ingestor's
# JsonLogic evaluator to read the constraint expressions in an .opmeta.json declaration. That
# evaluator's header is empty unless HIPDNN_ENABLE_KERNEL_INGESTOR is set, and the option
# defaults to OFF -- so building this tool unconditionally made the *default* configure fail to
# compile. The tool is part of the feature, so it is gated with it rather than papered over with
# a second gate inside the headers.
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    add_subdirectory(corpus_gen)
endif()

add_executable(hipdnn_list_engines ListEngines.cpp)
# hipdnn_data_sdk supplies the canonical engine-ID hex rendering (formatEngineIdHex). It is a
# header-only INTERFACE target, and hipdnn_backend links it privately, so the tool names it here.
target_link_libraries(hipdnn_list_engines PRIVATE hipdnn_backend hipdnn_data_sdk)
target_compile_definitions(hipdnn_list_engines PRIVATE HIPDNN_BACKEND_COMPILATION)
install(TARGETS hipdnn_list_engines RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Register the uhd_gen converter's Python tests with CTest so CI actually runs them.
# These pin the cross-language features_hash contract against the C++ FeatureExtractor;
# without this registration nothing executes them and the two sides drift silently.
# The suite self-skips (pytest.importorskip) when lightgbm/flatbuffers/numpy are
# absent, so environments without the training dependencies report skips, not failures.
# Registered unconditionally, matching how this project adds its other test
# directories (cmake/Tests.cmake calls enable_testing(); there is no BUILD_TESTING gate).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    # Probe for pytest rather than discovering its absence as a test failure:
    # `python -m pytest` exits 1 when the module is missing, which is
    # indistinguishable from a genuine test failure.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE UHD_GEN_PYTEST_PROBE
        OUTPUT_QUIET ERROR_QUIET)
    if(UHD_GEN_PYTEST_PROBE EQUAL 0)
        add_test(
            NAME uhd_gen_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/uhd_gen/tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        # pytest exits 5 when it collects nothing, which is what the suite's
        # importorskip guards produce when lightgbm/flatbuffers/numpy are absent.
        set_tests_properties(uhd_gen_tests PROPERTIES SKIP_RETURN_CODE 5 LABELS "unit")

        # results_import derives RFC 0019.13 §8.3's tflops and gbs from each operation's
        # declared flops/elements. Registered beside uhd_gen's suite and for the same reason:
        # a wrong formula or byte width does not raise, it yields a plausible rate, and every
        # model trained on that corpus ranks by a number wrong by a constant nobody sees.
        # Needs no ML dependencies -- it is arithmetic over a declaration and a row.
        add_test(
            NAME results_import_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                    ${CMAKE_CURRENT_SOURCE_DIR}/results_import/tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(results_import_tests PROPERTIES SKIP_RETURN_CODE 5 LABELS "unit")
    else()
        message(WARNING
                "uhd_gen tests NOT registered: pytest is not importable by "
                "${Python3_EXECUTABLE}. These pin the features_hash contract between "
                "tools/uhd_gen and the C++ FeatureExtractor; install pytest (plus "
                "lightgbm, flatbuffers, numpy) so they run.")
    endif()
else()
    message(WARNING "uhd_gen tests NOT registered: no Python3 interpreter found")
endif()

# Smoke test: run the tool against a real, loadable engine plugin and assert it prints that plugin's
# own engine name rather than a hexadecimal fallback.
#
# Discovery is pointed at ${HIPDNN_TEST_PLUGIN_DIR}/default via HIPDNN_PLUGIN_DIR. The /default
# suffix is required: discovery is non-recursive, and default/ holds exactly one plugin,
# ${TEST_GOOD_DEFAULT_PLUGIN_NAME}, which reports the engine name asserted below.
if(NOT HIPDNN_SKIP_TESTS AND TARGET ${TEST_GOOD_DEFAULT_PLUGIN_NAME})
    # Mirrors K_GOOD_DEFAULT_PLUGIN_ENGINE_NAME in tests/test_plugins/TestPluginEngineIdMap.hpp. The
    # ID is left out: NamedPluginFixtureIdsAreTheHashOfTheirNames covers it, and a hand-copied
    # literal here would only give it a second place to drift.
    set(HIPDNN_LIST_ENGINES_SMOKE_EXPECTED "TEST_GOOD_DEFAULT_ENGINE (")

    # The tool is not a gtest binary, so the checks live in a driver script: it asserts a zero exit
    # status, a non-empty engine list, and the expected name anywhere in the output.
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

    # The tool only enumerates engines, so anything past a few seconds is a hang -- most likely in
    # plugin dlopen/LoadLibrary. Fail fast rather than sitting until ctest's 1500 s default.
    set_tests_properties(hipdnn_list_engines_smoke PROPERTIES TIMEOUT 120)

    # add_hipdnn_test() assumes a gtest executable registered as the test command; this test runs a
    # CMake driver script instead, so its two run-time prerequisites -- the tool binary and the
    # plugin it discovers -- are appended to CHECK_DEPENDS_GLOBAL by hand.
    #
    # CHECK_EXECUTABLE_PATHS_GLOBAL and HIPDNN_TEST_TARGETS are deliberately not touched: both
    # assume a gtest executable installed under its own test name. HIPDNN_TEST_TARGETS is what
    # install_hipdnn_ctest_files() iterates, so this test is build-tree only -- the driver script
    # and the plugin it discovers are not installed either. Its test_categories.yaml entry
    # therefore labels it for `ctest` in the build tree and has no effect on the install tree.
    set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} hipdnn_list_engines
                             ${TEST_GOOD_DEFAULT_PLUGIN_NAME}
        CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE
    )

    # On Windows, stage the shadowed ROCm DLLs before the tool is built so a partial build plus a
    # manual ctest doesn't load the stale System32 amd_comgr.dll. add_hipdnn_test() does this for
    # gtest binaries; mirror it here.
    if(TARGET stage_shadowed_rocm_dlls)
        add_dependencies(hipdnn_list_engines stage_shadowed_rocm_dlls)
    endif()

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

    # Require the build-local MIOpen cache to be cleared before this test runs (ASAN builds only;
    # HIPDNN_TEST_MIOPEN_CACHE_DIR is unset otherwise). See the hipdnn_clear_miopen_test_cache
    # fixture in cmake/Tests.cmake.
    if(DEFINED HIPDNN_TEST_MIOPEN_CACHE_DIR)
        set_tests_properties(
            hipdnn_list_engines_smoke PROPERTIES FIXTURES_REQUIRED hipdnn_clear_miopen_test_cache
        )
    endif()

    _apply_hipdnn_test_category_labels(hipdnn_list_engines_smoke)
endif()
