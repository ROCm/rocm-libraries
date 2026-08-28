# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping Data SDK tests")
    return()
endif()

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    hipdnn_data_sdk_tests
    main.cpp
    types/TestBfloat16.cpp
    types/TestCrossTypeConversion.cpp
    types/TestForwardingFunctions.cpp
    types/TestFp4E2M1.cpp
    types/TestFp6E2M3.cpp
    types/TestFp6E3M2.cpp
    types/TestFp8E4M3.cpp
    types/TestFp8E4M3Fnuz.cpp
    types/TestFp8E5M2.cpp
    types/TestFp8E5M2Fnuz.cpp
    types/TestFp8E8M0.cpp
    types/TestHalf.cpp
    types/TestPortableTypes.cpp
    types/TestUserDefinedLiterals.cpp
    utilities/TestAllocators.cpp
    utilities/TestCacheRoot.cpp
    utilities/TestEngineNames.cpp
    utilities/TestFloats.cpp
    utilities/TestLineStore.cpp
    utilities/TestMigratableMemory.cpp
    utilities/TestPackedFp4Tensor.cpp
    utilities/TestPackedFp6Tensor.cpp
    utilities/TestPathSanitizer.cpp
    utilities/TestPlatformUtils.cpp
    utilities/TestScopedResource.cpp
    utilities/TestRaggedTensor.cpp
    utilities/TestShallowHostOnlyMigratableMemory.cpp
    utilities/TestShallowRaggedTensor.cpp
    utilities/TestShallowTensor.cpp
    utilities/TestShapeUtilities.cpp
    utilities/TestStringUtil.cpp
    utilities/TestTensor.cpp
    utilities/TestTensorIterator.cpp
    utilities/TestTensorView.cpp
    utilities/TestTimingStatistics.cpp
    utilities/TestVersionUtils.cpp
    utilities/TestVersion.cpp
    utilities/TestVisitor.cpp
)

# Standalone helper process for TestLineStore's two-process lock contention case (see
# LineStoreLockHelper.cpp). Not a gtest binary and not registered via add_hipdnn_test().
#
# Linux-only: its sole consumer is the fork()/waitpid() test guarded by
# `#if defined(__linux__)` in TestLineStore.cpp, so building and shipping it elsewhere
# would install a binary nothing can invoke.
#
# The name ends in `_test` on purpose. TheRock packages hipDNN's test artifact by the
# glob `bin/*hipdnn_*_test*` (ml-libs/artifact-hipdnn.toml), which requires `_test`
# AFTER the `hipdnn_` prefix; a name that misses the glob is built but never staged, and
# the test then fails at runtime because the helper is absent beside the test binary.
# For the same reason the install stays uncomponented: restricting it to the development
# component would drop it from the test artifact again.
#
# Only the helper's FILENAME is baked in and resolved next to the test's own executable at
# runtime, since an absolute $<TARGET_FILE:...> path would name the build tree, which does
# not exist once CI installs and runs the test binaries from the install prefix.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(hipdnn_line_store_lock_helper_test utilities/LineStoreLockHelper.cpp)
    target_link_libraries(hipdnn_line_store_lock_helper_test PRIVATE hipdnn_data_sdk)
    target_compile_options(
        hipdnn_line_store_lock_helper_test PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS}
    )
    clang_tidy_check(hipdnn_line_store_lock_helper_test)
    set_target_properties(
        hipdnn_line_store_lock_helper_test
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )
    target_compile_definitions(
        hipdnn_data_sdk_tests PRIVATE
        HIPDNN_LINESTORE_LOCK_HELPER_NAME="$<TARGET_FILE_NAME:hipdnn_line_store_lock_helper_test>"
    )

    # The test spawns the helper, so it must exist before the test runs and ship alongside it.
    add_dependencies(hipdnn_data_sdk_tests hipdnn_line_store_lock_helper_test)
    install(TARGETS hipdnn_line_store_lock_helper_test RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

target_compile_options(hipdnn_data_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS})

target_link_libraries(
    hipdnn_data_sdk_tests
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_data_sdk
    hipdnn_test_sdk
    hip::host
    ${CMAKE_DL_LIBS}
)

clang_tidy_check(hipdnn_data_sdk_tests)
add_hipdnn_test(hipdnn_data_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
