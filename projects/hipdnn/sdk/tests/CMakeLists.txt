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
enable_language(HIP)

add_executable(hipdnn_sdk_tests
    main.cpp
    plugin/TestPluginException.cpp
    plugin/TestPluginHelpers.cpp
    plugin/TestPluginLastErrorManager.cpp
    plugin/TestPluginDataTypeHelpers.cpp
    plugin/flatbuffer_utilities/TestGraphWrapper.cpp
    plugin/flatbuffer_utilities/TestEngineConfigWrapper.cpp
    plugin/flatbuffer_utilities/TestEngineDetailsWrapper.cpp
    utilities/TestAllocators.cpp
    utilities/TestCallbackLogger.cpp
    utilities/TestMigratableMemory.cpp
    utilities/TestFloats.cpp
    utilities/TestJson.cpp
    utilities/TestScopedResource.cpp
    utilities/TestShapeUtilities.cpp
    utilities/TestTensor.cpp
    utilities/TestPlatformUtils.cpp
    utilities/TestShallowHostOnlyMigratableMemory.cpp
    utilities/TestShallowTensor.cpp
    test_utilities/TestCpuFpReferenceBatchnorm.cpp
    test_utilities/TestCpuFpReferenceConvolution.cpp
    test_utilities/TestCpuFpReferenceUtilities.cpp
    test_utilities/TestCpuFpReferenceValidation.cpp
    test_utilities/TestCpuFpReferenceMiopenRmsValidation.cpp
    test_utilities/TestCpuReferencePointwise.cpp
    test_utilities/TestTempDirectory.cpp
    test_utilities/TestScopedEnvironmentVariableSetter.cpp
    test_utilities/TestLoggingUtils.cpp
    test_utilities/TestFlatbufferDatatypeMapping.cpp
    test_utilities/cpu_graph_executor/TestCpuReferenceGraphExecutor.cpp
    test_utilities/cpu_graph_executor/TestTensorVariant.cpp
    test_utilities/cpu_graph_executor/TestBatchnormSignatureRegistryKey.cpp



)

target_compile_definitions(hipdnn_sdk_tests PRIVATE
    COMPONENT_NAME="hipdnn_sdk_tests"
)

target_compile_options(hipdnn_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(hipdnn_sdk_tests 
    GTest::gtest
    Threads::Threads
    hipdnn_sdk
    hip::host
    spdlog_header_only
    ${CMAKE_DL_LIBS}
)

clang_tidy_check(hipdnn_sdk_tests)
add_unit_test_target(hipdnn_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
