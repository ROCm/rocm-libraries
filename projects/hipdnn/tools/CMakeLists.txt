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
    # manual ctest doesn't load the stale System32 ROCm DLLs. add_hipdnn_test() does this for
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

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # ValidateDescriptors.cpp transitively includes <hip/hip_runtime.h> through
    # DescriptorLoader.hpp, so it needs the same HIP wiring every ingestor test target
    # carries (plugin_sdk/tests/CMakeLists.txt).
    find_package(hip REQUIRED)

    add_executable(hipdnn_validate_descriptors IngestorGenerator/ValidateDescriptors.cpp)
    target_compile_definitions(hipdnn_validate_descriptors PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(hipdnn_validate_descriptors PRIVATE hipdnn_plugin_sdk hip::host)
    install(TARGETS hipdnn_validate_descriptors RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # Smoke test: run the validator over a real descriptor bundle and assert it reports
    # that bundle's engine. Same generated-driver-script shape as
    # hipdnn_list_engines_smoke above.
    #
    # The input is the baseline bundle the tool's mutation fixtures derive from: authored
    # JSON in this source tree, needing neither a packing step nor a device, and the bundle
    # tests/fixtures/validate_descriptors/README.md declares must exit 0. The staged
    # provider trees under dnn-providers/ exist only after a per-architecture packing rule
    # has run, which a hipDNN-only build need not configure.
    if(NOT HIPDNN_SKIP_TESTS)
        set(HIPDNN_VALIDATE_DESCRIPTORS_SMOKE_ROOT
            "${CMAKE_CURRENT_SOURCE_DIR}/IngestorGenerator/tests/fixtures/validate_descriptors/valid"
        )

        # The `name` the bundle's UED authors, and the engine tests/test_round_trip.py
        # names. Passed as --expect-engine so a bundle whose engine the loader drops exits
        # non-zero, and so a catalog loading some other engine is caught too.
        set(HIPDNN_VALIDATE_DESCRIPTORS_SMOKE_ENGINE "hipkernel:ValidateFixture")

        # The tool is not a gtest binary, so the checks live in a driver script: zero exit
        # status, a printed engine list, and the expected engine name in it.
        # cmake-format: off
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ValidateDescriptorsSmokeTest.cmake" [[
if(NOT DEFINED HIPDNN_VALIDATE_DESCRIPTORS_TOOL
   OR NOT DEFINED HIPDNN_VALIDATE_DESCRIPTORS_ROOT
   OR NOT DEFINED HIPDNN_VALIDATE_DESCRIPTORS_ENGINE)
    message(FATAL_ERROR
        "HIPDNN_VALIDATE_DESCRIPTORS_TOOL, HIPDNN_VALIDATE_DESCRIPTORS_ROOT and "
        "HIPDNN_VALIDATE_DESCRIPTORS_ENGINE are required")
endif()

execute_process(
    COMMAND "${HIPDNN_VALIDATE_DESCRIPTORS_TOOL}" "${HIPDNN_VALIDATE_DESCRIPTORS_ROOT}"
            --expect-engine "${HIPDNN_VALIDATE_DESCRIPTORS_ENGINE}"
    OUTPUT_VARIABLE toolStdout
    ERROR_VARIABLE toolStderr
    RESULT_VARIABLE toolResult
)

message(STATUS "hipdnn_validate_descriptors stdout:\n${toolStdout}")
message(STATUS "hipdnn_validate_descriptors stderr:\n${toolStderr}")

if(NOT toolResult EQUAL 0)
    message(FATAL_ERROR
        "hipdnn_validate_descriptors exited with '${toolResult}' on "
        "${HIPDNN_VALIDATE_DESCRIPTORS_ROOT}")
endif()

string(FIND "${toolStdout}" "Loaded engines:" headerPosition)
if(headerPosition EQUAL -1)
    message(FATAL_ERROR
        "hipdnn_validate_descriptors printed no engine list for "
        "${HIPDNN_VALIDATE_DESCRIPTORS_ROOT}")
endif()

string(FIND "${toolStdout}" "${HIPDNN_VALIDATE_DESCRIPTORS_ENGINE}" namePosition)
if(namePosition EQUAL -1)
    message(FATAL_ERROR
        "hipdnn_validate_descriptors output does not contain "
        "'${HIPDNN_VALIDATE_DESCRIPTORS_ENGINE}'")
endif()
]])
        # cmake-format: on

        add_test(
            NAME hipdnn_validate_descriptors_smoke
            COMMAND
                ${CMAKE_COMMAND}
                "-DHIPDNN_VALIDATE_DESCRIPTORS_TOOL=$<TARGET_FILE:hipdnn_validate_descriptors>"
                "-DHIPDNN_VALIDATE_DESCRIPTORS_ROOT=${HIPDNN_VALIDATE_DESCRIPTORS_SMOKE_ROOT}"
                "-DHIPDNN_VALIDATE_DESCRIPTORS_ENGINE=${HIPDNN_VALIDATE_DESCRIPTORS_SMOKE_ENGINE}"
                -P "${CMAKE_CURRENT_BINARY_DIR}/ValidateDescriptorsSmokeTest.cmake"
        )

        # The tool walks one small directory and parses its JSON, so anything past a few
        # seconds is a hang. Fail fast rather than ctest's 1500 s default.
        set_tests_properties(hipdnn_validate_descriptors_smoke PROPERTIES TIMEOUT 120)

        # add_hipdnn_test() assumes a gtest executable registered as the test command; this
        # test runs a CMake driver script, so its one run-time prerequisite (the tool
        # binary) is appended to CHECK_DEPENDS_GLOBAL by hand.
        # CHECK_EXECUTABLE_PATHS_GLOBAL and HIPDNN_TEST_TARGETS are left untouched for the
        # reason given at hipdnn_list_engines_smoke above; this test is build-tree only.
        set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} hipdnn_validate_descriptors
            CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE
        )

        # On Windows, stage the shadowed ROCm DLLs before the tool is built so a partial
        # build plus a manual ctest does not load the stale System32 ROCm DLLs.
        if(TARGET stage_shadowed_rocm_dlls)
            add_dependencies(hipdnn_validate_descriptors stage_shadowed_rocm_dlls)
        endif()

        # No discovery environment of its own: the root is a command-line argument and the
        # validator registers stubs for the native symbols rather than loading a plugin.
        if(DEFINED TEST_ENVIRONMENT)
            set_tests_properties(
                hipdnn_validate_descriptors_smoke PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}"
            )
        endif()

        # PATH prepends (Windows ASAN runtime / ROCm / build DLL dirs) go through
        # ENVIRONMENT_MODIFICATION so the runtime PATH is extended rather than replaced.
        if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
            set_tests_properties(
                hipdnn_validate_descriptors_smoke
                PROPERTIES ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}"
            )
        endif()

        _apply_hipdnn_test_category_labels(hipdnn_validate_descriptors_smoke)
    endif()
endif()
