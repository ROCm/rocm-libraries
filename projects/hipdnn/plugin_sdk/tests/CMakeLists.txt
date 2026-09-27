# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping Plugin SDK tests")
    return()
endif()

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# TestUhdGenArtifact runs tools/uhd_gen and loads what it writes, which is the only check
# that a real artifact from the trainer is loadable by the runtime -- the features_hash
# contract is pinned on both sides, but pinning a hash is not loading a file.
#
# It needs the trainer's own dependencies, so probe for them by importing exactly what the
# tool imports. Gating compilation rather than skipping at run time keeps a developer
# without lightgbm from seeing a failure they cannot act on; the warning is there so the
# absence is visible, because uhd_gen's own suite spent its life unregistered by a probe
# that failed quietly.
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import lightgbm, numpy, flatbuffers, pandas, sklearn"
            RESULT_VARIABLE _uhd_gen_deps_missing
            OUTPUT_QUIET ERROR_QUIET)
        if(_uhd_gen_deps_missing EQUAL 0)
            set(HIPDNN_UHD_GEN_RUNNABLE ON)
        else()
            message(WARNING
                    "TestUhdGenArtifact NOT built: uhd_gen's dependencies are not importable "
                    "by ${Python3_EXECUTABLE}. Nothing then checks that a real uhd_gen "
                    "artifact loads at run time. Install "
                    "projects/hipdnn/tools/uhd_gen/requirements.txt, or use the hipDNN dev "
                    "image, which does.")
        endif()
    else()
        message(WARNING "TestUhdGenArtifact NOT built: no Python3 interpreter found")
    endif()
endif()

add_executable(hipdnn_plugin_sdk_tests
    main.cpp
    TestArchMatch.cpp
    TestDeviceQuery.cpp
    TestPluginDataTypeHelpers.cpp
    TestPluginException.cpp
    TestPluginHelpers.cpp
    TestPluginLastErrorManager.cpp
    TestKnobFactory.cpp
    TestEngineManager.cpp
    TestEnginePluginContainer.cpp
    TestKnobSettingFactory.cpp
    TestGlobalKnobDefines.cpp
    TestSdkLogging.cpp
    TestVersion.cpp
    TestPluginDeviceBuffers.cpp
    TestRuntimePassByValue.cpp
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestJsonExpression.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestJsonDataSource.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestJsonValue.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestLruCache.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestNativeRegistry.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestKernelHeuristic.cpp>
    heuristics/uhd/TestCategoricalEncoding.cpp
    heuristics/uhd/TestCustomLibraryAdapter.cpp
    heuristics/uhd/TestInlineExpressions.cpp
    heuristics/uhd/TestFeatureExtractor.cpp
    heuristics/uhd/TestJsonLogicEvaluator.cpp
    heuristics/uhd/TestNativeAdapter.cpp
    heuristics/uhd/TestScoreTransform.cpp
    heuristics/uhd/TestTableAdapter.cpp
    heuristics/uhd/TestTreeDataAdapter.cpp
    heuristics/uhd/TestSha256.cpp
    heuristics/uhd/TestUhdAdapters.cpp
    heuristics/uhd/TestEnginePredictor.cpp
    heuristics/TestEngineFeatures.cpp
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestUhdKernelHeuristic.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestUhdGeneratedModel.cpp>
    $<$<BOOL:${HIPDNN_UHD_GEN_RUNNABLE}>:ingestor/uhd/TestUhdGenArtifact.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestMatchContext.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestDescriptors.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestArchGate.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestKernelDefinition.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestKernelIngestorStateManager.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestGenericPlan.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestGenericEngine.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestGenericPlanBuilder.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestBenchmarkPlan.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestDescriptorLoader.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestDeviceKey.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestWinnerCache.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:TestKernelIngestorConcurrency.cpp>
)

target_compile_options(hipdnn_plugin_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS})

# TestUhdGeneratedModel loads committed `uhd_gen` output from the source tree.
# Passed as a definition rather than copied into the build dir: the fixture is
# source, and a runtime search for it would turn a missing file into a skip when
# it should be a failure.
target_compile_definitions(
    hipdnn_plugin_sdk_tests
    PRIVATE HIPDNN_UHD_GENERATED_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/ingestor/uhd/fixtures"
)
# TestCustomLibraryAdapter dlopen's this at runtime; it is the only consumer, and it moved here
# with the adapter it covers. HIPDNN_TEST_PLUGIN_DIR is set by the top-level CMakeLists, so both
# suites can name the same output directory.
    add_library(hipdnn_test_scorer_lib SHARED heuristics/uhd/test_scorer_lib.cpp)
    set_target_properties(hipdnn_test_scorer_lib PROPERTIES CXX_VISIBILITY_PRESET hidden)
    target_compile_options(hipdnn_test_scorer_lib PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    add_dependencies(hipdnn_plugin_sdk_tests hipdnn_test_scorer_lib)

    set_target_properties(
        hipdnn_test_scorer_lib
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
            THEROCK_INSTALL_RPATH_ORIGIN "${HIPDNN_TEST_PLUGIN_INSTALL_DIR}"
    )

    install(TARGETS hipdnn_test_scorer_lib LIBRARY DESTINATION ${HIPDNN_TEST_PLUGIN_INSTALL_DIR}
            RUNTIME DESTINATION ${HIPDNN_TEST_PLUGIN_INSTALL_DIR}
    )

    target_compile_definitions(hipdnn_plugin_sdk_tests PRIVATE
        HIPDNN_TEST_PLUGIN_DIR="${HIPDNN_TEST_PLUGIN_DIR}")

target_link_libraries(
    hipdnn_plugin_sdk_tests
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_plugin_sdk
    hipdnn_test_sdk
    hip::host
)

# ContentCarryingTestGraph.hpp lives in flatbuffers_sdk's test tree (plugin_sdk depends on
# flatbuffers_sdk, not the reverse, so the shared fixture cannot live here). No existing
# include path exposes cross-SDK test headers, so add it explicitly.
target_include_directories(hipdnn_plugin_sdk_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../flatbuffers_sdk/tests)
if(HIPDNN_UHD_GEN_RUNNABLE)
    # The interpreter that satisfied the probe above, not whatever `python3` resolves to in
    # the test's environment; and the directory `python -m uhd_gen` has to run from, since
    # the tool is a source tree rather than an installed package.
    target_compile_definitions(hipdnn_plugin_sdk_tests
        PRIVATE HIPDNN_UHD_GEN_PYTHON="${Python3_EXECUTABLE}"
                HIPDNN_UHD_GEN_TOOLS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tools")

    # uhd_gen resolves the evaluator from this variable on its own, so naming it here covers
    # every invocation rather than the ones that remembered to pass --feature-evaluator. It
    # names a target `tools` declares, which is configured after this directory; the
    # generator expression is resolved at generate time, and the guard above already implies
    # HIPDNN_ENABLE_KERNEL_INGESTOR, which is what decides that target exists.
    #
    # Appended, not set: Sanitizers.cmake puts the ASAN symbolizer and the MIOpen cache
    # directory in this same variable, and add_hipdnn_test() applies it through
    # set_tests_properties, which replaces the property rather than adding to it.
    list(APPEND TEST_ENVIRONMENT
         "HIPDNN_UHD_FEATURE_EVALUATOR=$<TARGET_FILE:hipdnn_uhd_features>")

    # The variable only names the binary; this is what makes it exist first.
    add_dependencies(hipdnn_plugin_sdk_tests hipdnn_uhd_features)
endif()


clang_tidy_check(hipdnn_plugin_sdk_tests)
add_hipdnn_test(hipdnn_plugin_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})

# Cross-process coverage: one process writes a winner record, a SECOND process reads it
# back. Two ctest invocations of the same binary, sequenced with
# FIXTURES_SETUP/FIXTURES_REQUIRED, following the shape hipdnn/cmake/Tests.cmake's
# MIOpen-cache fixture already uses. A single-process double-compute proves nothing here:
# the in-memory map would satisfy it without the file being read at all.
#
# Both halves are DISABLED_ in gtest so a bare developer run of the suite skips them --
# neither is meaningful alone -- and ctest opts in with --gtest_also_run_disabled_tests.
# RUN_SERIAL because they share one shard file, and the shared HIPDNN_CACHE_DIR lives
# under the build tree so a clean build starts from an empty cache.
#
# The category labels are not optional decoration: CI selects tests by label tier, so an
# unlabelled test is invisible to every tier and would never run there while passing
# locally. _apply_hipdnn_test_category_labels() is what add_hipdnn_test() applies to every
# other test in this tree; these are registered by hand (one binary, three ctest entries,
# which add_hipdnn_test() cannot express) so they must apply it themselves.
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    set(_disk_cache_cross_process_dir "${CMAKE_CURRENT_BINARY_DIR}/disk_cache_cross_process")
    set(_disk_cache_cross_process_env
        "HIPDNN_CACHE_DIR=${_disk_cache_cross_process_dir};HIPDNN_DISABLE_CACHE=")

    # Single source of truth for the two DISABLED_ test names and the pass regex, so a
    # rename or a regex change is a one-line edit instead of two hand-typed copies (build
    # tree below, installed-tree snippet further down) silently drifting apart.
    set(_disk_cache_writer_filter
        "TestIngestorWinnerCacheCrossProcess.DISABLED_WriterLeavesARecordOnDisk")
    set(_disk_cache_reader_filter
        "TestIngestorWinnerCacheCrossProcess.DISABLED_ReaderServesTheRecordTheWriterLeft")
    # A filter matching zero tests still exits 0 ("[  PASSED  ] 0 tests"), so a rename or
    # move of either DISABLED_ test above would silently turn its half into a no-op pass.
    # Require gtest's own "ran exactly one" summary line instead of trusting the exit code.
    set(_disk_cache_pass_regex "\\[==========\\] 1 test from 1 test suite ran\\.")

    add_test(NAME hipdnn_plugin_sdk_disk_cache_cross_process_setup
             COMMAND ${CMAKE_COMMAND} -E rm -rf "${_disk_cache_cross_process_dir}")
    set_tests_properties(hipdnn_plugin_sdk_disk_cache_cross_process_setup PROPERTIES
                         FIXTURES_SETUP DiskCacheCrossProcessClean)

    add_test(NAME hipdnn_plugin_sdk_disk_cache_cross_process_writer
             COMMAND hipdnn_plugin_sdk_tests
                     --gtest_also_run_disabled_tests
                     --gtest_filter=${_disk_cache_writer_filter})
    set_tests_properties(hipdnn_plugin_sdk_disk_cache_cross_process_writer PROPERTIES
                         RUN_SERIAL TRUE
                         FIXTURES_REQUIRED DiskCacheCrossProcessClean
                         FIXTURES_SETUP DiskCacheCrossProcessWritten
                         ENVIRONMENT "${_disk_cache_cross_process_env}"
                         PASS_REGULAR_EXPRESSION "${_disk_cache_pass_regex}")

    add_test(NAME hipdnn_plugin_sdk_disk_cache_cross_process_reader
             COMMAND hipdnn_plugin_sdk_tests
                     --gtest_also_run_disabled_tests
                     --gtest_filter=${_disk_cache_reader_filter})
    set_tests_properties(hipdnn_plugin_sdk_disk_cache_cross_process_reader PROPERTIES
                         RUN_SERIAL TRUE
                         FIXTURES_REQUIRED "DiskCacheCrossProcessClean;DiskCacheCrossProcessWritten"
                         ENVIRONMENT "${_disk_cache_cross_process_env}"
                         PASS_REGULAR_EXPRESSION "${_disk_cache_pass_regex}")

    # One call per test: the helper forwards its arguments positionally to the parser,
    # which takes a single explicit test name.
    _apply_hipdnn_test_category_labels(hipdnn_plugin_sdk_disk_cache_cross_process_setup)
    _apply_hipdnn_test_category_labels(hipdnn_plugin_sdk_disk_cache_cross_process_writer)
    _apply_hipdnn_test_category_labels(hipdnn_plugin_sdk_disk_cache_cross_process_reader)

    # install_hipdnn_ctest_files() (cmake/Tests.cmake) writes the installed
    # CTestTestfile.cmake only from one bare add_test() per HIPDNN_TEST_TARGETS entry, so
    # these three hand-registered, fixture-sequenced entries never ship. Stage the same
    # three add_test()/set_tests_properties() pairs as text on a global property, which
    # install_hipdnn_ctest_files() appends into that one file ahead of its category-label
    # pass -- so these entries carry the same test_categories.yaml labels as every other
    # installed test. Only what genuinely differs between trees is retyped here: a
    # relative command (matching the relative form install_hipdnn_ctest_files() already
    # emits for every other test) and a cache directory relative to ctest's installed
    # working directory (${CMAKE_INSTALL_BINDIR}/hipdnn). The filter names, the pass regex
    # and the fixture names all come from the variables the build-tree entries above use.
    #
    # Escaping note: _disk_cache_pass_regex holds single backslashes, and this staged text
    # is reparsed as ctest script syntax when ctest loads the installed
    # CTestTestfile.cmake -- so anything containing a backslash must be re-escaped once
    # more, or that reparse strips it. string(REPLACE) makes this explicit instead of
    # hand-counting backslashes.
    string(REPLACE "\\" "\\\\" _disk_cache_pass_regex_for_file "${_disk_cache_pass_regex}")

    # A bare "cmake" (not the configure-time absolute ${CMAKE_COMMAND}) so the installed
    # tree keeps working on a host where CMake lives at a different path than the one
    # this project was configured on.
    set(_disk_cache_installed_tests "")
    string(APPEND _disk_cache_installed_tests
           "\n"
           "# Installed-tree copies of the hipdnn_plugin_sdk_disk_cache_cross_process_* entries registered in\n"
           "# projects/hipdnn/plugin_sdk/tests/CMakeLists.txt.\n"
           "add_test(hipdnn_plugin_sdk_disk_cache_cross_process_setup \"cmake\" -E rm -rf disk_cache_cross_process)\n"
           "set_tests_properties(hipdnn_plugin_sdk_disk_cache_cross_process_setup PROPERTIES FIXTURES_SETUP DiskCacheCrossProcessClean)\n\n")

    string(APPEND _disk_cache_installed_tests
           "add_test(hipdnn_plugin_sdk_disk_cache_cross_process_writer \"../hipdnn_plugin_sdk_tests\" --gtest_also_run_disabled_tests --gtest_filter=${_disk_cache_writer_filter})\n"
           "set_tests_properties(hipdnn_plugin_sdk_disk_cache_cross_process_writer PROPERTIES\n"
           "    RUN_SERIAL TRUE\n"
           "    FIXTURES_REQUIRED DiskCacheCrossProcessClean\n"
           "    FIXTURES_SETUP DiskCacheCrossProcessWritten\n"
           "    ENVIRONMENT \"HIPDNN_CACHE_DIR=disk_cache_cross_process;HIPDNN_DISABLE_CACHE=\"\n"
           "    PASS_REGULAR_EXPRESSION \"${_disk_cache_pass_regex_for_file}\")\n\n")

    string(APPEND _disk_cache_installed_tests
           "add_test(hipdnn_plugin_sdk_disk_cache_cross_process_reader \"../hipdnn_plugin_sdk_tests\" --gtest_also_run_disabled_tests --gtest_filter=${_disk_cache_reader_filter})\n"
           "set_tests_properties(hipdnn_plugin_sdk_disk_cache_cross_process_reader PROPERTIES\n"
           "    RUN_SERIAL TRUE\n"
           "    FIXTURES_REQUIRED \"DiskCacheCrossProcessClean;DiskCacheCrossProcessWritten\"\n"
           "    ENVIRONMENT \"HIPDNN_CACHE_DIR=disk_cache_cross_process;HIPDNN_DISABLE_CACHE=\"\n"
           "    PASS_REGULAR_EXPRESSION \"${_disk_cache_pass_regex_for_file}\")\n")

    set_property(GLOBAL APPEND_STRING PROPERTY
                 HIPDNN_INSTALLED_CTEST_STAGING "${_disk_cache_installed_tests}")
endif()
