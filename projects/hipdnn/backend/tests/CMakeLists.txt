# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Backend Unit tests")
    return()
endif()

find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# Test plugin target names come from cmake/TestPluginNames.cmake (included by
# the root CMakeLists.txt before any subdirectory). The targets themselves are
# created later by tests/test_plugins/, but the names are needed here for
# compile definitions and add_dependencies forward references.

add_executable(
    hipdnn_backend_tests
    descriptors/TestBackendDescriptor.cpp
    descriptors/TestBlockScaleDequantizeOperationDescriptor.cpp
    descriptors/TestBlockScaleDequantizeOperationFromNode.cpp
    descriptors/TestBlockScaleQuantizeOperationDescriptor.cpp
    descriptors/TestBlockScaleQuantizeOperationFromNode.cpp
    descriptors/TestDescriptorFactory.cpp
    descriptors/TestEngineConfigDescriptor.cpp
    descriptors/TestEngineDescriptor.cpp
    descriptors/TestEngineHeuristicDescriptor.cpp
    descriptors/TestExecutionPlanDescriptor.cpp
    descriptors/TestGraphDescriptor.cpp
    descriptors/TestTensorDescriptor.cpp
    descriptors/TestCustomOpOperationDescriptor.cpp
    descriptors/TestCustomOpOperationFromNode.cpp
    descriptors/TestGraphDescriptorCustomOp.cpp
    descriptors/TestConvolutionFwdOperationDescriptor.cpp
    descriptors/TestConvolutionWrwOperationDescriptor.cpp
    descriptors/TestConvolutionWrwOperationFromNode.cpp
    descriptors/TestGraphDescriptorConvolutionWrw.cpp
    descriptors/TestBatchnormInferenceOperationDescriptor.cpp
    descriptors/TestBatchnormInferenceOperationFromNode.cpp
    descriptors/TestBatchnormOperationDescriptor.cpp
    descriptors/TestBatchnormOperationFromNode.cpp
    descriptors/TestGraphDescriptorBatchnorm.cpp
    descriptors/TestGraphDescriptorBatchnormInference.cpp
    descriptors/TestGraphDescriptorBlockScaleDequantize.cpp
    descriptors/TestKnobDescriptor.cpp
    descriptors/TestKnobSettingDescriptor.cpp
    descriptors/TestBatchnormBackwardOperationDescriptor.cpp
    descriptors/TestBatchnormBackwardOperationFromNode.cpp
    descriptors/TestConvolutionBwdOperationDescriptor.cpp
    descriptors/TestGraphDescriptorBlockScaleQuantize.cpp
    descriptors/TestMatmulOperationDescriptor.cpp
    descriptors/TestMatmulOperationFromNode.cpp
    descriptors/TestGraphDescriptorMatmul.cpp
    descriptors/TestGraphDescriptorBatchnormBackward.cpp
    descriptors/TestGraphDescriptorConvolutionBwd.cpp
    descriptors/TestBatchnormInferenceVarianceExtOperationDescriptor.cpp
    descriptors/TestBatchnormInferenceVarianceExtOperationFromNode.cpp
    descriptors/TestGraphDescriptorBatchnormInferenceVarianceExt.cpp
    descriptors/TestGraphDescriptorLayernorm.cpp
    descriptors/TestGraphDescriptorOps.cpp
    descriptors/TestGraphDescriptorLifting.cpp
    descriptors/TestGraphDescriptorPointwise.cpp
    descriptors/TestGraphDescriptorRMSNorm.cpp
    descriptors/TestGraphDescriptorSdpaFwd.cpp
    descriptors/TestConvolutionBwdOperationFromNode.cpp
    descriptors/TestLayernormOperationDescriptor.cpp
    descriptors/TestLayernormOperationFromNode.cpp
    descriptors/TestPointwiseOperationDescriptor.cpp
    descriptors/TestPointwiseOperationFromNode.cpp
    descriptors/TestReductionOperationDescriptor.cpp
    descriptors/TestReductionOperationFromNode.cpp
    descriptors/TestGraphDescriptorReduction.cpp
    descriptors/TestRMSNormOperationDescriptor.cpp
    descriptors/TestRMSNormOperationFromNode.cpp
    descriptors/TestRMSNormBackwardOperationDescriptor.cpp
    descriptors/TestRMSNormBackwardOperationFromNode.cpp
    descriptors/TestGraphDescriptorRMSNormBackward.cpp
    descriptors/TestSdpaBwdOperationDescriptor.cpp
    descriptors/TestSdpaBwdOperationFromNode.cpp
    descriptors/TestSdpaFwdOperationDescriptor.cpp
    descriptors/TestSdpaFwdOperationFromNode.cpp
    descriptors/TestGraphDescriptorSdpaBwd.cpp
    descriptors/TestResampleFwdOperationDescriptor.cpp
    descriptors/TestResampleFwdOperationFromNode.cpp
    descriptors/TestGraphDescriptorResampleFwd.cpp
    descriptors/TestDescriptorAttributeUtils.cpp
    descriptors/TestDataTypeConversion.cpp
    descriptors/TestDescriptorFromFlatBuffer.cpp
    descriptors/TestNodeFactory.cpp
    descriptors/TestVariantPackDescriptor.cpp
    main.cpp
    TestEnginePluginResourceManager.cpp
    TestExecutionPlanSerializationApi.cpp
    TestLastErrorManager.cpp
    TestBackendEnumStringUtils.cpp
    TestFlatbufferUtilities.cpp
    TestHandle.cpp
    TestHelpers.cpp
    IntegrationHeuristicPlugin.cpp
    TestHeuristicPolicyFramework.cpp
    IntegrationHeuristicPolicyPlugins.cpp
    TestBackendLogger.cpp
    TestGraphLogger.cpp
    TestHeuristicPlugin.cpp
    TestHeuristicPluginManager.cpp
    TestHeuristicPluginManagerValidationPaths.cpp
    TestSelectionHeuristic.cpp
    TestHeuristicPluginResourceManager.cpp
    heuristics/TestConfigBuiltIn.cpp
    heuristics/TestDeviceProperties.cpp
    heuristics/TestEngineOverrideConfig.cpp
    heuristics/TestStaticOrderingBuiltIn.cpp
    TestUserLoggingApis.cpp
    TestPlatformUtils.cpp
    TestEnginePluginOverride.cpp
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
            TEST_GOOD_HEURISTIC_PLUGIN_NAME="${TEST_GOOD_HEURISTIC_PLUGIN_NAME}"
            TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME="${TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME}"
            TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME="${TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME}"
            TEST_BAD_API_VERSION_HEURISTIC_PLUGIN_NAME="${TEST_BAD_API_VERSION_HEURISTIC_PLUGIN_NAME}"
            TEST_EMPTY_NAME_HEURISTIC_PLUGIN_NAME="${TEST_EMPTY_NAME_HEURISTIC_PLUGIN_NAME}"
            TEST_DUPLICATE_POLICY_ID_A_PLUGIN_NAME="${TEST_DUPLICATE_POLICY_ID_A_PLUGIN_NAME}"
            TEST_DUPLICATE_POLICY_ID_B_PLUGIN_NAME="${TEST_DUPLICATE_POLICY_ID_B_PLUGIN_NAME}"
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

# Add dependencies on heuristic test plugins (built in tests/test_plugins/)
# Note: These targets are forward references - they will be created when tests/test_plugins/
# is processed later. CMake resolves these dependencies at build time.
add_dependencies(hipdnn_backend_tests
    ${TEST_GOOD_HEURISTIC_PLUGIN_NAME}
    ${TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME}
    ${TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME}
    ${TEST_BAD_API_VERSION_HEURISTIC_PLUGIN_NAME}
    ${TEST_EMPTY_NAME_HEURISTIC_PLUGIN_NAME}
    ${TEST_DUPLICATE_POLICY_ID_A_PLUGIN_NAME}
    ${TEST_DUPLICATE_POLICY_ID_B_PLUGIN_NAME}
)

add_unit_test_target(hipdnn_backend_tests ${CMAKE_CURRENT_BINARY_DIR})
