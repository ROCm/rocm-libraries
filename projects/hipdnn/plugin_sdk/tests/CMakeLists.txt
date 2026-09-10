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

    add_test(NAME DiskCacheCrossProcess_Setup
             COMMAND ${CMAKE_COMMAND} -E rm -rf "${_disk_cache_cross_process_dir}")
    set_tests_properties(DiskCacheCrossProcess_Setup PROPERTIES
                         FIXTURES_SETUP DiskCacheCrossProcessClean)

    add_test(NAME DiskCacheCrossProcess_Writer
             COMMAND hipdnn_plugin_sdk_tests
                     --gtest_also_run_disabled_tests
                     --gtest_filter=${_disk_cache_writer_filter})
    set_tests_properties(DiskCacheCrossProcess_Writer PROPERTIES
                         RUN_SERIAL TRUE
                         FIXTURES_REQUIRED DiskCacheCrossProcessClean
                         FIXTURES_SETUP DiskCacheCrossProcessWritten
                         ENVIRONMENT "${_disk_cache_cross_process_env}"
                         PASS_REGULAR_EXPRESSION "${_disk_cache_pass_regex}")

    add_test(NAME DiskCacheCrossProcess_Reader
             COMMAND hipdnn_plugin_sdk_tests
                     --gtest_also_run_disabled_tests
                     --gtest_filter=${_disk_cache_reader_filter})
    set_tests_properties(DiskCacheCrossProcess_Reader PROPERTIES
                         RUN_SERIAL TRUE
                         FIXTURES_REQUIRED "DiskCacheCrossProcessClean;DiskCacheCrossProcessWritten"
                         ENVIRONMENT "${_disk_cache_cross_process_env}"
                         PASS_REGULAR_EXPRESSION "${_disk_cache_pass_regex}")

    # One call per test: the helper forwards its arguments positionally to the parser,
    # which takes a single explicit test name.
    _apply_hipdnn_test_category_labels(DiskCacheCrossProcess_Setup)
    _apply_hipdnn_test_category_labels(DiskCacheCrossProcess_Writer)
    _apply_hipdnn_test_category_labels(DiskCacheCrossProcess_Reader)

    # install_hipdnn_ctest_files() (cmake/Tests.cmake) writes the installed
    # CTestTestfile.cmake only from one bare add_test() per HIPDNN_TEST_TARGETS entry, so
    # these three hand-registered, fixture-sequenced entries never ship. Generate the same
    # three add_test()/set_tests_properties() pairs into a snippet installed beside the
    # binary, and register the snippet's file name so install_hipdnn_ctest_files() includes
    # it. Only what genuinely differs between trees is retyped here: a relative command
    # (matching the relative form install_hipdnn_ctest_files() already emits for every
    # other test) and a cache directory relative to ctest's installed working directory
    # (${CMAKE_INSTALL_BINDIR}/hipdnn). The filter names, the pass regex, the fixture
    # names, and the labels are all read back from the build-tree properties above via
    # get_test_property(), so this snippet cannot drift from them.
    #
    # Escaping note: a property value read back with get_test_property() is already
    # backslash-collapsed once (e.g. the regex above reads back as `\[==========...`,
    # single backslash). Writing it into this snippet file with file(APPEND) is itself a
    # second round of CMake string handling, and the snippet file is then reparsed a
    # THIRD time as ctest script syntax when ctest loads it -- so anything containing a
    # backslash must be re-escaped once more before being written, or ctest's own reparse
    # strips it. string(REPLACE) makes that explicit instead of hand-counting backslashes.
    string(REPLACE "\\" "\\\\" _disk_cache_pass_regex_for_file "${_disk_cache_pass_regex}")

    set(_disk_cache_installed_snippet
        "${CMAKE_CURRENT_BINARY_DIR}/DiskCacheCrossProcessTests.cmake")
    file(WRITE "${_disk_cache_installed_snippet}"
         "# Autogenerated: installed-tree copy of the DiskCacheCrossProcess_* ctest entries\n"
         "# in projects/hipdnn/plugin_sdk/tests/CMakeLists.txt. Regenerated from those\n"
         "# entries' own properties at configure time -- do not hand-edit.\n\n")

    # A bare "cmake" (not the configure-time absolute ${CMAKE_COMMAND}) so the installed
    # tree keeps working on a host where CMake lives at a different path than the one
    # this project was configured on.
    file(APPEND "${_disk_cache_installed_snippet}"
         "add_test(DiskCacheCrossProcess_Setup \"cmake\" -E rm -rf disk_cache_cross_process)\n"
         "set_tests_properties(DiskCacheCrossProcess_Setup PROPERTIES FIXTURES_SETUP DiskCacheCrossProcessClean)\n\n")

    file(APPEND "${_disk_cache_installed_snippet}"
         "add_test(DiskCacheCrossProcess_Writer \"../hipdnn_plugin_sdk_tests\" --gtest_also_run_disabled_tests --gtest_filter=${_disk_cache_writer_filter})\n"
         "set_tests_properties(DiskCacheCrossProcess_Writer PROPERTIES\n"
         "    RUN_SERIAL TRUE\n"
         "    FIXTURES_REQUIRED DiskCacheCrossProcessClean\n"
         "    FIXTURES_SETUP DiskCacheCrossProcessWritten\n"
         "    ENVIRONMENT \"HIPDNN_CACHE_DIR=disk_cache_cross_process;HIPDNN_DISABLE_CACHE=\"\n"
         "    PASS_REGULAR_EXPRESSION \"${_disk_cache_pass_regex_for_file}\")\n\n")

    file(APPEND "${_disk_cache_installed_snippet}"
         "add_test(DiskCacheCrossProcess_Reader \"../hipdnn_plugin_sdk_tests\" --gtest_also_run_disabled_tests --gtest_filter=${_disk_cache_reader_filter})\n"
         "set_tests_properties(DiskCacheCrossProcess_Reader PROPERTIES\n"
         "    RUN_SERIAL TRUE\n"
         "    FIXTURES_REQUIRED \"DiskCacheCrossProcessClean;DiskCacheCrossProcessWritten\"\n"
         "    ENVIRONMENT \"HIPDNN_CACHE_DIR=disk_cache_cross_process;HIPDNN_DISABLE_CACHE=\"\n"
         "    PASS_REGULAR_EXPRESSION \"${_disk_cache_pass_regex_for_file}\")\n\n")

    # ctest's CTestTestfile.cmake script interpreter understands add_test()/set_tests_properties()
    # but not the generic set_property() command (it reports the test as unknown), so labels are
    # emitted through set_tests_properties() instead, one call per test as its own statement.
    foreach(_disk_cache_test DiskCacheCrossProcess_Setup DiskCacheCrossProcess_Writer DiskCacheCrossProcess_Reader)
        get_test_property(${_disk_cache_test} LABELS _disk_cache_test_labels)
        if(_disk_cache_test_labels)
            file(APPEND "${_disk_cache_installed_snippet}"
                 "set_tests_properties(${_disk_cache_test} PROPERTIES LABELS \"${_disk_cache_test_labels}\")\n")
        endif()
    endforeach()

    install(FILES "${_disk_cache_installed_snippet}" DESTINATION ${CMAKE_INSTALL_BINDIR}/hipdnn)
    set_property(GLOBAL APPEND PROPERTY HIPDNN_INSTALLED_CTEST_INCLUDES DiskCacheCrossProcessTests.cmake)
endif()
