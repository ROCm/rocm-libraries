# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # Headers both command-line tools below share; tools-internal, never installed.
    add_library(hipdnn_tools_common INTERFACE)
    target_include_directories(hipdnn_tools_common
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>)
    target_link_libraries(hipdnn_tools_common INTERFACE hipdnn_data_sdk hipdnn_flatbuffers_sdk)

    # The training-corpus harvester; it feeds uhd_gen, so it ships with the rest of the UHD
    # training tooling.
    add_subdirectory(bench)

    # Both evaluate UHD descriptor expressions through plugin_sdk's heuristics/uhd/, which
    # exists only with the kernel ingestor.
    add_subdirectory(corpus_gen)

    add_executable(hipdnn_uhd_features uhd_gen/UhdFeatures.cpp)
    target_compile_options(hipdnn_uhd_features PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    target_link_libraries(hipdnn_uhd_features PRIVATE hipdnn_plugin_sdk)
    target_compile_definitions(hipdnn_uhd_features PRIVATE HIPDNN_ENABLE_KERNEL_INGESTOR)
    clang_tidy_check(hipdnn_uhd_features)
    install(TARGETS hipdnn_uhd_features RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(hipdnn_list_engines ListEngines.cpp)
# hipdnn_data_sdk supplies the canonical engine-ID hex rendering (formatEngineIdHex). It is a
# header-only INTERFACE target, and hipdnn_backend links it privately, so the tool names it here.
target_link_libraries(hipdnn_list_engines PRIVATE hipdnn_backend hipdnn_data_sdk)
target_compile_definitions(hipdnn_list_engines PRIVATE HIPDNN_BACKEND_COMPILATION)
# CMAKE_INSTALL_RPATH_USE_LINK_PATH adds only link directories outside the build tree, so it
# never adds the directory of the in-tree hipdnn_backend; without this the installed tool
# needs LD_LIBRARY_PATH to start. Same layout as hipdnn_sample_dynamic_backend_loading
# (samples/CMakeLists.txt): bin/ -> lib/.
set_target_properties(
    hipdnn_list_engines PROPERTIES INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)
install(TARGETS hipdnn_list_engines RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# uhd_gen's Python tests pin the features_hash contract against the C++ FeatureExtractor, and
# mine_shapes mines the shapes the IngestorGenerator consumes; both are ingestor tooling, so
# a build without the ingestor, or with HIPDNN_SKIP_TESTS, registers neither. The suites self-skip (pytest.importorskip)
# when lightgbm/flatbuffers/numpy/pandas are absent; HIPDNN_REQUIRE_UHD_GEN_TESTS turns
# the skips of all four suites into failures (uhd_gen/conftest.py, which the two suites outside
# uhd_gen/ load with `-p`) and a missing pytest into a configure error.
if(HIPDNN_ENABLE_KERNEL_INGESTOR AND NOT HIPDNN_SKIP_TESTS)
    set(_uhd_gen_missing_severity ${HIPDNN_UHD_GEN_MISSING_SEVERITY})

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        # Probe for pytest: `python -m pytest` exits 1 when it is missing, which is
        # indistinguishable from a test failure.
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pytest"
            RESULT_VARIABLE UHD_GEN_PYTEST_PROBE
            OUTPUT_QUIET ERROR_QUIET)
        if(UHD_GEN_PYTEST_PROBE EQUAL 0)
            add_test(
                NAME uhd_gen_tests
                COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
                        ${CMAKE_CURRENT_SOURCE_DIR}/uhd_gen/tests
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
            # pytest exits 5 when it collects nothing (the importorskip guards skipped
            # everything).
            set_tests_properties(uhd_gen_tests PROPERTIES
                SKIP_RETURN_CODE 5
                ENVIRONMENT "HIPDNN_UHD_FEATURE_EVALUATOR=$<TARGET_FILE:hipdnn_uhd_features>")

            # Separate test because the suite above does not collect uhd_gen/dataset/tests,
            # and this one needs no ML dependencies.
            add_test(
                NAME uhd_gen_dataset_tests
                COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
                        ${CMAKE_CURRENT_SOURCE_DIR}/uhd_gen/dataset/tests
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
            set_tests_properties(uhd_gen_dataset_tests PROPERTIES SKIP_RETURN_CODE 5)

            # Only this file: the rest of IngestorGenerator/tests needs PyYAML/Jinja2 via its
            # conftest. --noconftest because test_mine_shapes.py must not inherit those
            # imports.
            add_test(
                NAME mine_shapes_tests
                COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider --noconftest -p uhd_gen.conftest
                        ${CMAKE_CURRENT_SOURCE_DIR}/IngestorGenerator/tests/test_mine_shapes.py
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
            set_tests_properties(mine_shapes_tests PROPERTIES SKIP_RETURN_CODE 5)

            # The hipdnn-uhd skill's scripts, against a fake hipdnn_bench. Its uhd_gen-backed
            # cases (promotion into a candidate tree) need the evaluator too.
            add_test(
                NAME uhd_skill_scripts_tests
                COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider -p uhd_gen.conftest
                        ${CMAKE_CURRENT_SOURCE_DIR}/ai/tests/test_uhd_skill_scripts.py
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
            set_tests_properties(uhd_skill_scripts_tests PROPERTIES
                SKIP_RETURN_CODE 5
                ENVIRONMENT "HIPDNN_UHD_FEATURE_EVALUATOR=$<TARGET_FILE:hipdnn_uhd_features>")

            if(HIPDNN_REQUIRE_UHD_GEN_TESTS)
                set_property(TEST uhd_gen_tests uhd_gen_dataset_tests mine_shapes_tests
                             uhd_skill_scripts_tests APPEND PROPERTY
                             ENVIRONMENT "HIPDNN_REQUIRE_UHD_GEN_TESTS=1")
            endif()
            # LightGBM takes every core by default, which stalls these suites for many
            # minutes beside a parallel ctest. Bytecode and pytest's cache stay out of the
            # source tree.
            set_property(TEST uhd_gen_tests uhd_gen_dataset_tests mine_shapes_tests
                         uhd_skill_scripts_tests APPEND PROPERTY
                         ENVIRONMENT "OMP_NUM_THREADS=4" "PYTHONDONTWRITEBYTECODE=1")

            # The two suites that run hipdnn_uhd_features load the sanitizer, ROCm and build
            # DLL directories the C++ tests do (PATH prepends on Windows, via
            # ENVIRONMENT_MODIFICATION so the runtime PATH is extended rather than replaced).
            if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
                set_tests_properties(uhd_gen_tests uhd_skill_scripts_tests PROPERTIES
                                     ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
            endif()

            # Build tree only: none of the four is installed.
            _apply_hipdnn_test_category_labels(uhd_gen_tests)
            _apply_hipdnn_test_category_labels(uhd_gen_dataset_tests)
            _apply_hipdnn_test_category_labels(mine_shapes_tests)
            _apply_hipdnn_test_category_labels(uhd_skill_scripts_tests)
        else()
            message(${_uhd_gen_missing_severity}
                    "uhd_gen tests NOT registered: pytest is not importable by "
                    "${Python3_EXECUTABLE}. These pin the features_hash contract between "
                    "tools/uhd_gen and the C++ FeatureExtractor; install pytest (plus "
                    "${HIPDNN_UHD_GEN_PYTHON_MODULES}) so they run.")
        endif()
    else()
        message(${_uhd_gen_missing_severity}
                "uhd_gen tests NOT registered: no Python3 interpreter found")
    endif()
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

# DescriptorGenerator pytest suite, then the IngestorGenerator one below. Build tree only, like
# the smoke tests above. Both run under the hipdnn_strict_skips plugin: ctest reports a suite
# whose cases all skip as a pass, so a missing compiler or binary would otherwise go unnoticed.
if(HIPDNN_ENABLE_TOOLING_TESTS AND HIPDNN_SKIP_TESTS)
    message(STATUS "hipDNN: HIPDNN_ENABLE_TOOLING_TESTS is ignored because HIPDNN_SKIP_TESTS is ON")
endif()
if(HIPDNN_ENABLE_TOOLING_TESTS AND NOT HIPDNN_SKIP_TESTS)
    if(NOT Python3_FOUND)
        message(FATAL_ERROR "hipDNN: HIPDNN_ENABLE_TOOLING_TESTS=ON needs a Python3 interpreter")
    endif()
    if(NOT HIPDNN_ENABLE_KERNEL_INGESTOR)
        message(
            FATAL_ERROR
                "hipDNN: HIPDNN_ENABLE_TOOLING_TESTS=ON also needs HIPDNN_ENABLE_KERNEL_INGESTOR=ON: "
                "the IngestorGenerator suite runs the hipdnn_validate_descriptors tool that option "
                "builds. Enable it, or configure with -DHIPDNN_ENABLE_TOOLING_TESTS=OFF."
        )
    endif()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pytest, yaml, jinja2"
        RESULT_VARIABLE _hipdnn_tooling_imports
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _hipdnn_tooling_imports EQUAL 0)
        message(
            FATAL_ERROR
                "hipDNN: pytest, PyYAML and Jinja2 are not all importable by "
                "${Python3_EXECUTABLE}, so the tooling tests cannot run. Install "
                "tools/IngestorGenerator/requirements.txt for that interpreter, or configure "
                "with -DHIPDNN_ENABLE_TOOLING_TESTS=OFF."
        )
    endif()
    set(HIPDNN_STRICT_SKIPS_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pytest_plugins")
    add_test(
        NAME hipdnn_descriptor_generator_pytest
        COMMAND "${Python3_EXECUTABLE}" -m pytest -p hipdnn_strict_skips
                "${CMAKE_CURRENT_SOURCE_DIR}/DescriptorGenerator/tests" -v
    )
    set_tests_properties(
        hipdnn_descriptor_generator_pytest
        PROPERTIES ENVIRONMENT_MODIFICATION
                   "PYTHONPATH=path_list_prepend:${HIPDNN_STRICT_SKIPS_PLUGIN_DIR}"
    )
    _apply_hipdnn_test_category_labels(hipdnn_descriptor_generator_pytest)
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
        set(HIPDNN_VALIDATE_DESCRIPTORS_FIXTURES
            "${CMAKE_CURRENT_SOURCE_DIR}/IngestorGenerator/tests/fixtures/validate_descriptors"
        )
        set(HIPDNN_VALIDATE_DESCRIPTORS_SMOKE_ROOT "${HIPDNN_VALIDATE_DESCRIPTORS_FIXTURES}/valid")

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

        # The role fixtures: the exit status, and every model_checks entry's role and verdict
        # in report order. HIPDNN_VALIDATE_DESCRIPTORS_MODEL_CHECKS is a comma-separated list
        # of <role>=<true|false>, comma rather than semicolon so it survives add_test().
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ValidateDescriptorsModelChecksTest.cmake" [[
if(NOT DEFINED HIPDNN_VALIDATE_DESCRIPTORS_TOOL
   OR NOT DEFINED HIPDNN_VALIDATE_DESCRIPTORS_ROOT
   OR NOT DEFINED HIPDNN_VALIDATE_DESCRIPTORS_ENGINE
   OR NOT DEFINED HIPDNN_VALIDATE_DESCRIPTORS_EXIT
   OR NOT DEFINED HIPDNN_VALIDATE_DESCRIPTORS_MODEL_CHECKS)
    message(FATAL_ERROR
        "HIPDNN_VALIDATE_DESCRIPTORS_TOOL, HIPDNN_VALIDATE_DESCRIPTORS_ROOT, "
        "HIPDNN_VALIDATE_DESCRIPTORS_ENGINE, HIPDNN_VALIDATE_DESCRIPTORS_EXIT and "
        "HIPDNN_VALIDATE_DESCRIPTORS_MODEL_CHECKS are required")
endif()

execute_process(
    COMMAND "${HIPDNN_VALIDATE_DESCRIPTORS_TOOL}" "${HIPDNN_VALIDATE_DESCRIPTORS_ROOT}"
            --expect-engine "${HIPDNN_VALIDATE_DESCRIPTORS_ENGINE}" --json
    OUTPUT_VARIABLE toolStdout
    ERROR_VARIABLE toolStderr
    RESULT_VARIABLE toolResult
)

message(STATUS "hipdnn_validate_descriptors stdout:\n${toolStdout}")
message(STATUS "hipdnn_validate_descriptors stderr:\n${toolStderr}")

if(NOT toolResult STREQUAL HIPDNN_VALIDATE_DESCRIPTORS_EXIT)
    message(FATAL_ERROR
        "hipdnn_validate_descriptors exited with '${toolResult}' on "
        "${HIPDNN_VALIDATE_DESCRIPTORS_ROOT}; expected '${HIPDNN_VALIDATE_DESCRIPTORS_EXIT}'")
endif()

string(JSON engines ERROR_VARIABLE jsonError GET "${toolStdout}" engines)
if(jsonError)
    message(FATAL_ERROR "hipdnn_validate_descriptors printed no JSON report: ${jsonError}")
endif()
string(FIND "${engines}" "\"${HIPDNN_VALIDATE_DESCRIPTORS_ENGINE}\"" namePosition)
if(namePosition EQUAL -1)
    message(FATAL_ERROR
        "'${HIPDNN_VALIDATE_DESCRIPTORS_ENGINE}' did not load; its model checks never ran")
endif()

string(REPLACE "," ";" expectedChecks "${HIPDNN_VALIDATE_DESCRIPTORS_MODEL_CHECKS}")
list(LENGTH expectedChecks expectedCount)
string(JSON checkCount LENGTH "${toolStdout}" model_checks)
if(NOT checkCount EQUAL expectedCount)
    message(FATAL_ERROR
        "hipdnn_validate_descriptors reported ${checkCount} model checks; expected "
        "${expectedCount} (${HIPDNN_VALIDATE_DESCRIPTORS_MODEL_CHECKS})")
endif()

set(index 0)
foreach(expected IN LISTS expectedChecks)
    string(REPLACE "=" ";" expectedPair "${expected}")
    list(GET expectedPair 0 expectedRole)
    list(GET expectedPair 1 expectedSuccess)
    string(JSON role GET "${toolStdout}" model_checks ${index} role)
    string(JSON success GET "${toolStdout}" model_checks ${index} success)
    if(success)
        set(success true)
    else()
        set(success false)
    endif()
    if(NOT role STREQUAL expectedRole OR NOT success STREQUAL expectedSuccess)
        message(FATAL_ERROR
            "model_checks[${index}] is role=${role} success=${success}; expected "
            "role=${expectedRole} success=${expectedSuccess}")
    endif()
    math(EXPR index "${index} + 1")
endforeach()
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
        set(HIPDNN_VALIDATE_DESCRIPTORS_TESTS hipdnn_validate_descriptors_smoke)

        # The multi-launch bundle: kernels carrying a `sources` map, one pair naming two
        # kernels and one naming a kernel under both launch names. It has its own engine,
        # and the same driver script. A validator that stopped declaring the pack's launch
        # names drops this engine and fails here, which the single-launch root cannot show.
        set(HIPDNN_VALIDATE_DESCRIPTORS_MULTI_LAUNCH_ROOT
            "${HIPDNN_VALIDATE_DESCRIPTORS_FIXTURES}/multi_launch"
        )
        set(HIPDNN_VALIDATE_DESCRIPTORS_MULTI_LAUNCH_ENGINE "hipkernel:ValidateFixtureMultiLaunch")
        add_test(
            NAME hipdnn_validate_descriptors_smoke_multi_launch
            COMMAND
                ${CMAKE_COMMAND}
                "-DHIPDNN_VALIDATE_DESCRIPTORS_TOOL=$<TARGET_FILE:hipdnn_validate_descriptors>"
                "-DHIPDNN_VALIDATE_DESCRIPTORS_ROOT=${HIPDNN_VALIDATE_DESCRIPTORS_MULTI_LAUNCH_ROOT}"
                "-DHIPDNN_VALIDATE_DESCRIPTORS_ENGINE=${HIPDNN_VALIDATE_DESCRIPTORS_MULTI_LAUNCH_ENGINE}"
                -P "${CMAKE_CURRENT_BINARY_DIR}/ValidateDescriptorsSmokeTest.cmake"
        )
        list(APPEND HIPDNN_VALIDATE_DESCRIPTORS_TESTS hipdnn_validate_descriptors_smoke_multi_launch)

        # <fixture>:<exit status>:<model checks>, as the fixtures' README states them. A
        # predict_engine model is admitted only through the L1 guards, so a static_order
        # estimate fails the run while a static_order kernel ranker does not. A ranker whose
        # features need a sample, run without one, is admitted with extraction skipped.
        foreach(
            roleFixture
            "l2_static_order:0:sort_kernel_catalog=true"
            "l2_dynamic_features:0:sort_kernel_catalog=true"
            "l1_static_order:1:sort_kernel_catalog=true,predict_engine=false"
            "l1_native:0:sort_kernel_catalog=true,predict_engine=true"
        )
            string(REPLACE ":" ";" roleFixture "${roleFixture}")
            list(GET roleFixture 0 fixtureName)
            list(GET roleFixture 1 fixtureExit)
            list(GET roleFixture 2 fixtureChecks)
            add_test(
                NAME hipdnn_validate_descriptors_${fixtureName}
                COMMAND
                    ${CMAKE_COMMAND}
                    "-DHIPDNN_VALIDATE_DESCRIPTORS_TOOL=$<TARGET_FILE:hipdnn_validate_descriptors>"
                    "-DHIPDNN_VALIDATE_DESCRIPTORS_ROOT=${HIPDNN_VALIDATE_DESCRIPTORS_FIXTURES}/${fixtureName}"
                    "-DHIPDNN_VALIDATE_DESCRIPTORS_ENGINE=${HIPDNN_VALIDATE_DESCRIPTORS_SMOKE_ENGINE}"
                    "-DHIPDNN_VALIDATE_DESCRIPTORS_EXIT=${fixtureExit}"
                    "-DHIPDNN_VALIDATE_DESCRIPTORS_MODEL_CHECKS=${fixtureChecks}"
                    -P "${CMAKE_CURRENT_BINARY_DIR}/ValidateDescriptorsModelChecksTest.cmake"
            )
            list(APPEND HIPDNN_VALIDATE_DESCRIPTORS_TESTS hipdnn_validate_descriptors_${fixtureName})
        endforeach()

        # The tool walks one small directory and parses its JSON, so anything past a few
        # seconds is a hang. Fail fast rather than ctest's 1500 s default.
        set_tests_properties(${HIPDNN_VALIDATE_DESCRIPTORS_TESTS} PROPERTIES TIMEOUT 120)

        # add_hipdnn_test() assumes a gtest executable registered as the test command; these
        # tests run CMake driver scripts, so their one run-time prerequisite (the tool
        # binary) is appended to CHECK_DEPENDS_GLOBAL by hand.
        # CHECK_EXECUTABLE_PATHS_GLOBAL and HIPDNN_TEST_TARGETS are left untouched for the
        # reason given at hipdnn_list_engines_smoke above; these tests are build-tree only.
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
                ${HIPDNN_VALIDATE_DESCRIPTORS_TESTS} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}"
            )
        endif()

        # PATH prepends (Windows ASAN runtime / ROCm / build DLL dirs) go through
        # ENVIRONMENT_MODIFICATION so the runtime PATH is extended rather than replaced.
        if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
            set_tests_properties(
                ${HIPDNN_VALIDATE_DESCRIPTORS_TESTS}
                PROPERTIES ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}"
            )
        endif()

        _apply_hipdnn_test_category_labels(${HIPDNN_VALIDATE_DESCRIPTORS_TESTS})

        # IngestorGenerator pytest suite. It is registered here, after the validator target:
        # tests/test_round_trip.py runs that binary (HIPDNN_VALIDATE_DESCRIPTORS) and
        # tests/test_native_stub.py needs the include list written below.
        if(HIPDNN_ENABLE_TOOLING_TESTS)
            # TestAgainstTheRealBuild gates a build packed with the gfx950 attention engine; the
            # lanes pack other descriptors, so it would fail there.
            add_test(
                NAME hipdnn_ingestor_generator_pytest
                COMMAND
                    "${Python3_EXECUTABLE}" -m pytest -p hipdnn_strict_skips
                    "${CMAKE_CURRENT_SOURCE_DIR}/IngestorGenerator/tests" -v
                    --deselect=tests/test_coverage_gate.py::TestAgainstTheRealBuild
            )

            set(_hipdnn_ingestor_pytest_env
                "HIPDNN_VALIDATE_DESCRIPTORS=$<TARGET_FILE:hipdnn_validate_descriptors>"
            )

            # test_verify_variant_sets_multi_launch_rocke.py packs with the real compiler and
            # fails without one. Name the hipcc find_package(hip) resolved, as the packaging
            # suite's own ctest entries do: the ROCm bin directory is not on every lane's PATH.
            if(HIP_HIPCC_EXECUTABLE)
                list(APPEND _hipdnn_ingestor_pytest_env "HKP_HIPCC=${HIP_HIPCC_EXECUTABLE}")
            endif()

            # Include directories for test_native_stub.py, one per line. TARGET_PROPERTY does not
            # follow ALIAS targets, so resolve to the real target.
            set(_hipdnn_stub_include_exprs "")
            foreach(_stub_dep flatbuffers::flatbuffers hip::host hip::amdhip64)
                if(TARGET ${_stub_dep})
                    get_target_property(_stub_real ${_stub_dep} ALIASED_TARGET)
                    if(NOT _stub_real)
                        set(_stub_real ${_stub_dep})
                    endif()
                    string(
                        APPEND
                        _hipdnn_stub_include_exprs
                        "$<JOIN:$<TARGET_PROPERTY:${_stub_real},INTERFACE_INCLUDE_DIRECTORIES>,\n>\n"
                    )
                endif()
            endforeach()
            file(
                GENERATE
                OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ingestor_stub_include_dirs.txt"
                CONTENT "${_hipdnn_stub_include_exprs}"
            )
            list(
                APPEND
                _hipdnn_ingestor_pytest_env
                "HIPDNN_INGESTOR_STUB_INCLUDE_FILE=${CMAKE_CURRENT_BINARY_DIR}/ingestor_stub_include_dirs.txt"
            )

            # test_packs_formatting.py takes the formatter from HIPDNN_CLANG_FORMAT, else PATH.
            # The lanes install clang-format into the interpreter's own environment, whose
            # scripts directory is not on PATH, so name it here.
            get_filename_component(_hipdnn_py_dir "${Python3_EXECUTABLE}" DIRECTORY)
            find_program(
                HIPDNN_INGESTOR_CLANG_FORMAT
                NAMES clang-format
                HINTS "${_hipdnn_py_dir}"
                NO_DEFAULT_PATH
            )
            if(HIPDNN_INGESTOR_CLANG_FORMAT)
                list(APPEND _hipdnn_ingestor_pytest_env
                     "HIPDNN_CLANG_FORMAT=${HIPDNN_INGESTOR_CLANG_FORMAT}"
                )
            endif()

            if(DEFINED TEST_ENVIRONMENT)
                list(APPEND _hipdnn_ingestor_pytest_env ${TEST_ENVIRONMENT})
            endif()
            set_tests_properties(
                hipdnn_ingestor_generator_pytest PROPERTIES ENVIRONMENT
                                                            "${_hipdnn_ingestor_pytest_env}"
            )

            # PYTHONPATH makes the plugin named by -p importable.
            set(_hipdnn_ingestor_pytest_env_mod
                "PYTHONPATH=path_list_prepend:${HIPDNN_STRICT_SKIPS_PLUGIN_DIR}"
            )
            if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
                list(APPEND _hipdnn_ingestor_pytest_env_mod ${TEST_ENVIRONMENT_MODIFICATION})
            endif()
            # Give test_native_stub.py the compiler that builds this tree; its lib/llvm/bin is not
            # on the lanes' PATH.
            get_filename_component(_hipdnn_stub_cxx_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
            list(APPEND _hipdnn_ingestor_pytest_env_mod
                 "PATH=path_list_prepend:${_hipdnn_stub_cxx_dir}")
            set_tests_properties(
                hipdnn_ingestor_generator_pytest PROPERTIES ENVIRONMENT_MODIFICATION
                                                            "${_hipdnn_ingestor_pytest_env_mod}"
            )

            _apply_hipdnn_test_category_labels(hipdnn_ingestor_generator_pytest)
        endif()
    endif()
endif()
