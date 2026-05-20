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
    utilities/TestEngineNames.cpp
    utilities/TestFloats.cpp
    utilities/TestMigratableMemory.cpp
    utilities/TestPlatformUtils.cpp
    utilities/TestScopedResource.cpp
    utilities/TestShallowHostOnlyMigratableMemory.cpp
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
add_unit_test_target(hipdnn_data_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
