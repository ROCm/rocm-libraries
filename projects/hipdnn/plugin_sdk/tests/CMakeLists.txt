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
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestLruCache.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestNativeRegistry.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/TestKernelHeuristic.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestCustomLibraryAdapter.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestDerivedValues.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestFeatureExtractor.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestJsonLogicEvaluator.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestNativeAdapter.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestScoreTransform.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestTableAdapter.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestTreeDataAdapter.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestUhdLoader.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestUhdLoaderMultiArch.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestSha256.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestUhdAdapters.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_KERNEL_INGESTOR}>:ingestor/uhd/TestUhdKernelHeuristic.cpp>
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

# TestCustomLibraryAdapter dlopen's this at runtime; it is the only consumer, and it moved here
# with the adapter it covers. HIPDNN_TEST_PLUGIN_DIR is set by the top-level CMakeLists, so both
# suites can name the same output directory.
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    add_library(hipdnn_test_scorer_lib SHARED ingestor/uhd/test_scorer_lib.cpp)
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
endif()

target_link_libraries(
    hipdnn_plugin_sdk_tests
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_plugin_sdk
    hipdnn_test_sdk
    hip::host
)

if(HIPDNN_UHD_GEN_RUNNABLE)
    # The interpreter that satisfied the probe above, not whatever `python3` resolves to in
    # the test's environment; and the directory `python -m uhd_gen` has to run from, since
    # the tool is a source tree rather than an installed package.
    target_compile_definitions(hipdnn_plugin_sdk_tests
        PRIVATE HIPDNN_UHD_GEN_PYTHON="${Python3_EXECUTABLE}"
                HIPDNN_UHD_GEN_TOOLS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tools")
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
    set(_disk_cache_cross_process_env "HIPDNN_CACHE_DIR=${_disk_cache_cross_process_dir}")

    add_test(NAME DiskCacheCrossProcess_Setup
             COMMAND ${CMAKE_COMMAND} -E rm -rf "${_disk_cache_cross_process_dir}")
    set_tests_properties(DiskCacheCrossProcess_Setup PROPERTIES
                         FIXTURES_SETUP DiskCacheCrossProcessClean)

    add_test(NAME DiskCacheCrossProcess_Writer
             COMMAND hipdnn_plugin_sdk_tests
                     --gtest_also_run_disabled_tests
                     --gtest_filter=TestIngestorWinnerCacheCrossProcess.DISABLED_WriterLeavesARecordOnDisk)
    set_tests_properties(DiskCacheCrossProcess_Writer PROPERTIES
                         RUN_SERIAL TRUE
                         FIXTURES_REQUIRED DiskCacheCrossProcessClean
                         FIXTURES_SETUP DiskCacheCrossProcessWritten
                         ENVIRONMENT "${_disk_cache_cross_process_env}"
                         # A filter matching zero tests still exits 0 ("[  PASSED  ] 0
                         # tests"), so a rename or move of the DISABLED_ test above would
                         # silently turn this into a no-op pass. Require gtest's own
                         # "ran exactly one" summary line instead of trusting the exit code.
                         PASS_REGULAR_EXPRESSION "\\[==========\\] 1 test from 1 test suite ran\\.")

    add_test(NAME DiskCacheCrossProcess_Reader
             COMMAND hipdnn_plugin_sdk_tests
                     --gtest_also_run_disabled_tests
                     --gtest_filter=TestIngestorWinnerCacheCrossProcess.DISABLED_ReaderServesTheRecordTheWriterLeft)
    set_tests_properties(DiskCacheCrossProcess_Reader PROPERTIES
                         RUN_SERIAL TRUE
                         FIXTURES_REQUIRED "DiskCacheCrossProcessClean;DiskCacheCrossProcessWritten"
                         ENVIRONMENT "${_disk_cache_cross_process_env}"
                         PASS_REGULAR_EXPRESSION "\\[==========\\] 1 test from 1 test suite ran\\.")

    # One call per test: the helper forwards its arguments positionally to the parser,
    # which takes a single explicit test name.
    _apply_hipdnn_test_category_labels(DiskCacheCrossProcess_Setup)
    _apply_hipdnn_test_category_labels(DiskCacheCrossProcess_Writer)
    _apply_hipdnn_test_category_labels(DiskCacheCrossProcess_Reader)
endif()
