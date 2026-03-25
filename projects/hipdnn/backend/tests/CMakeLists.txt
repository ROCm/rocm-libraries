# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Backend Unit tests")
    return()
endif()

find_package(hip REQUIRED)
find_package(Threads REQUIRED)

set(TEST_PLUGIN1_NAME "hipdnn_test_plugin1")
set(TEST_PLUGIN2_NAME "hipdnn_test_plugin2")
set(TEST_NO_API_VERSION_PLUGIN_NAME "hipdnn_test_no_api_version_plugin_name")
set(TEST_ENGINE_PLUGIN1_NAME "hipdnn_test_engine_plugin1")

add_executable(
    hipdnn_backend_tests
    descriptors/TestBackendDescriptor.cpp
    descriptors/TestBlockScaleDequantizeOperationDescriptor.cpp
    descriptors/TestBlockScaleQuantizeOperationDescriptor.cpp
    descriptors/TestDescriptorFactory.cpp
    descriptors/TestEngineConfigDescriptor.cpp
    descriptors/TestEngineDescriptor.cpp
    descriptors/TestEngineHeuristicDescriptor.cpp
    descriptors/TestExecutionPlanDescriptor.cpp
    descriptors/TestGraphDescriptor.cpp
    descriptors/TestTensorDescriptor.cpp
    descriptors/TestCustomOpOperationDescriptor.cpp
    descriptors/TestGraphDescriptorCustomOp.cpp
    descriptors/TestConvolutionFwdOperationDescriptor.cpp
    descriptors/TestConvolutionWrwOperationDescriptor.cpp
    descriptors/TestGraphDescriptorConvolutionWrw.cpp
    descriptors/TestBatchnormInferenceOperationDescriptor.cpp
    descriptors/TestBatchnormOperationDescriptor.cpp
    descriptors/TestGraphDescriptorBatchnorm.cpp
    descriptors/TestGraphDescriptorBatchnormInference.cpp
    descriptors/TestGraphDescriptorBlockScaleDequantize.cpp
    descriptors/TestKnobDescriptor.cpp
    descriptors/TestKnobSettingDescriptor.cpp
    descriptors/TestBatchnormBackwardOperationDescriptor.cpp
    descriptors/TestConvolutionBwdOperationDescriptor.cpp
    descriptors/TestGraphDescriptorBlockScaleQuantize.cpp
    descriptors/TestMatmulOperationDescriptor.cpp
    descriptors/TestGraphDescriptorMatmul.cpp
    descriptors/TestGraphDescriptorBatchnormBackward.cpp
    descriptors/TestGraphDescriptorConvolutionBwd.cpp
    descriptors/TestBatchnormInferenceVarianceExtOperationDescriptor.cpp
    descriptors/TestGraphDescriptorBatchnormInferenceVarianceExt.cpp
    descriptors/TestGraphDescriptorLayernorm.cpp
    descriptors/TestGraphDescriptorOps.cpp
    descriptors/TestGraphDescriptorLifting.cpp
    descriptors/TestGraphDescriptorPointwise.cpp
    descriptors/TestGraphDescriptorRMSNorm.cpp
    descriptors/TestGraphDescriptorSdpaFprop.cpp
    descriptors/TestLayernormOperationDescriptor.cpp
    descriptors/TestLayernormOperationFromNode.cpp
    descriptors/TestPointwiseOperationDescriptor.cpp
    descriptors/TestPointwiseOperationFromNode.cpp
    descriptors/TestRMSNormOperationDescriptor.cpp
    descriptors/TestSdpaBpropOperationDescriptor.cpp
    descriptors/TestSdpaBpropOperationFromNode.cpp
    descriptors/TestSdpaFpropOperationDescriptor.cpp
    descriptors/TestGraphDescriptorSdpaBprop.cpp
    descriptors/TestDescriptorAttributeUtils.cpp
    descriptors/TestDataTypeConversion.cpp
    descriptors/TestDescriptorFromFlatBuffer.cpp
    descriptors/TestNodeFactory.cpp
    descriptors/TestVariantPackDescriptor.cpp
    main.cpp
    TestEnginePluginResourceManager.cpp
    TestLastErrorManager.cpp
    TestBackendEnumStringUtils.cpp
    TestFlatbufferUtilities.cpp
    TestHandle.cpp
    TestHelpers.cpp
    TestBackendLogger.cpp
    TestGraphLogger.cpp
    TestUserLoggingApis.cpp
    TestPlatformUtils.cpp
    TestPlugin.cpp
    TestSharedLibrary.cpp
    TestVersion.cpp
    utilities/TestEngineOrdering.cpp
)

target_link_libraries(
    hipdnn_backend_tests
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_backend_private
    hipdnn_test_sdk
    ${CMAKE_DL_LIBS}
)

target_compile_definitions(
    hipdnn_backend_tests
    PRIVATE HIPDNN_BACKEND_COMPILATION TEST_PLUGIN1_NAME="${TEST_PLUGIN1_NAME}"
            TEST_PLUGIN2_NAME="${TEST_PLUGIN2_NAME}"
            TEST_NO_API_VERSION_PLUGIN_NAME="${TEST_NO_API_VERSION_PLUGIN_NAME}"
            TEST_ENGINE_PLUGIN1_NAME="${TEST_ENGINE_PLUGIN1_NAME}"
)
target_compile_options(hipdnn_backend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS})
clang_tidy_check(hipdnn_backend_tests)

# libhipdnn_test_plugin1.so or hipdnn_test_plugin1.dll (used by TestPlugin.cpp and
# TestSharedLibrary.cpp)
add_library(${TEST_PLUGIN1_NAME} SHARED plugins/PluginApiImpl.cpp plugins/Plugin1.cpp)
target_link_libraries(${TEST_PLUGIN1_NAME} PRIVATE hipdnn_data_sdk hipdnn_plugin_sdk)
set_target_properties(${TEST_PLUGIN1_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(${TEST_PLUGIN1_NAME} PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(${TEST_PLUGIN1_NAME} PRIVATE HIPDNN_COMPONENT_NAME="plugin1")
clang_tidy_check(${TEST_PLUGIN1_NAME})
add_dependencies(hipdnn_backend_tests ${TEST_PLUGIN1_NAME})

set_target_properties(
    ${TEST_PLUGIN1_NAME}
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        THEROCK_INSTALL_RPATH_ORIGIN "${HIPDNN_TEST_PLUGIN_INSTALL_DIR}"
)

install(TARGETS ${TEST_PLUGIN1_NAME} LIBRARY DESTINATION ${HIPDNN_TEST_PLUGIN_INSTALL_DIR}
        RUNTIME DESTINATION ${HIPDNN_TEST_PLUGIN_INSTALL_DIR}
)

# libhipdnn_test_plugin2.so or hipdnn_test_plugin2.dll (used by TestPlugin.cpp)
add_library(${TEST_PLUGIN2_NAME} SHARED plugins/PluginApiImpl.cpp plugins/Plugin2.cpp)
target_link_libraries(${TEST_PLUGIN2_NAME} PRIVATE hipdnn_data_sdk hipdnn_plugin_sdk)
set_target_properties(${TEST_PLUGIN2_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(${TEST_PLUGIN2_NAME} PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(${TEST_PLUGIN2_NAME} PRIVATE HIPDNN_COMPONENT_NAME="plugin2")
clang_tidy_check(${TEST_PLUGIN2_NAME})
add_dependencies(hipdnn_backend_tests ${TEST_PLUGIN2_NAME})

set_target_properties(
    ${TEST_PLUGIN2_NAME}
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        THEROCK_INSTALL_RPATH_ORIGIN "${HIPDNN_TEST_PLUGIN_INSTALL_DIR}"
)

install(TARGETS ${TEST_PLUGIN2_NAME} LIBRARY DESTINATION ${HIPDNN_TEST_PLUGIN_INSTALL_DIR}
        RUNTIME DESTINATION ${HIPDNN_TEST_PLUGIN_INSTALL_DIR}
)

# libhipdnn_test_no_api_version_plugin.so or libhipdnn_test_no_api_version_plugin.dll (used by TestPlugin.cpp)
add_library(${TEST_NO_API_VERSION_PLUGIN_NAME} SHARED plugins/PluginApiImpl.cpp plugins/NoApiVersionPlugin.cpp)
target_link_libraries(${TEST_NO_API_VERSION_PLUGIN_NAME} PRIVATE hipdnn_data_sdk hipdnn_plugin_sdk)
set_target_properties(${TEST_NO_API_VERSION_PLUGIN_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(${TEST_NO_API_VERSION_PLUGIN_NAME} PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(${TEST_NO_API_VERSION_PLUGIN_NAME} PRIVATE HIPDNN_COMPONENT_NAME="noApiVersionPlugin" HIPDNN_API_VERSION_UNDEFINED)
clang_tidy_check(${TEST_NO_API_VERSION_PLUGIN_NAME})
add_dependencies(hipdnn_backend_tests ${TEST_NO_API_VERSION_PLUGIN_NAME})

set_target_properties(
    ${TEST_NO_API_VERSION_PLUGIN_NAME}
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_TEST_PLUGIN_DIR}"
        THEROCK_INSTALL_RPATH_ORIGIN "${HIPDNN_TEST_PLUGIN_INSTALL_DIR}"
)

install(TARGETS ${TEST_NO_API_VERSION_PLUGIN_NAME} LIBRARY DESTINATION ${HIPDNN_TEST_PLUGIN_INSTALL_DIR}
        RUNTIME DESTINATION ${HIPDNN_TEST_PLUGIN_INSTALL_DIR}
)

add_unit_test_target(hipdnn_backend_tests ${CMAKE_CURRENT_BINARY_DIR})
