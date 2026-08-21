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
    utilities/TestVersionUtils.cpp
    utilities/TestVersion.cpp
    utilities/TestVisitor.cpp
)

# Standalone helper process for TestLineStore's two-process lock contention case (see
# LineStoreLockHelper.cpp). Not a gtest binary and not registered via add_hipdnn_test().
#
# Only the helper's FILENAME is baked in and resolved next to the test's own executable at
# runtime, since an absolute $<TARGET_FILE:...> path would name the build tree, which does
# not exist once CI installs and runs the test binaries from the install prefix.
add_executable(LineStoreLockHelper utilities/LineStoreLockHelper.cpp)
target_link_libraries(LineStoreLockHelper hipdnn_data_sdk)
set_target_properties(
    LineStoreLockHelper PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
)
target_compile_definitions(
    hipdnn_data_sdk_tests PRIVATE
    HIPDNN_LINESTORE_LOCK_HELPER_NAME="$<TARGET_FILE_NAME:LineStoreLockHelper>"
)

# The test spawns the helper, so it must exist before the test runs and ship alongside it.
add_dependencies(hipdnn_data_sdk_tests LineStoreLockHelper)
install(TARGETS LineStoreLockHelper RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

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
