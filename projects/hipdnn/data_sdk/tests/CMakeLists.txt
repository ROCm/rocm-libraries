# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping SDK tests")
    return()
endif()

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    hipdnn_sdk_tests
    main.cpp
    plugin/flatbuffer_utilities/TestEngineConfigWrapper.cpp
    plugin/flatbuffer_utilities/TestEngineDetailsWrapper.cpp
    plugin/flatbuffer_utilities/TestGraphWrapper.cpp
    plugin/flatbuffer_utilities/TestNodeWrapper.cpp
    utilities/TestAllocators.cpp
    utilities/TestCallbackLogger.cpp
    utilities/TestFloats.cpp
    utilities/TestJson.cpp
    utilities/TestMigratableMemory.cpp
    utilities/TestPlatformUtils.cpp
    utilities/TestPointwiseValidation.cpp
    utilities/TestScopedResource.cpp
    utilities/TestShallowHostOnlyMigratableMemory.cpp
    utilities/TestShallowTensor.cpp
    utilities/TestShapeUtilities.cpp
    utilities/TestStaticCast.cpp
    utilities/TestTensor.cpp
    utilities/TestTensorIterator.cpp
    utilities/TestTensorView.cpp
)

target_compile_options(hipdnn_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(
    hipdnn_sdk_tests
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_test_sdk
    hip::host
    ${CMAKE_DL_LIBS}
)

clang_tidy_check(hipdnn_sdk_tests)
add_unit_test_target(hipdnn_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
