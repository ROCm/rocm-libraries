# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Frontend Unit tests")
    return()
endif()

find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(hipdnn_frontend_tests 
    main.cpp
    TestBackendLoggingHelpers.cpp
    TestBatchnormAttributes.cpp
    TestBatchnormBackwardAttributes.cpp
    TestBatchnormBackwardNode.cpp
    TestBatchnormInferenceAttributes.cpp
    TestBatchnormInferenceNode.cpp
    TestBatchnormNode.cpp
    TestConvolutionDgradAttributes.cpp
    TestConvolutionFwdAttributes.cpp
    TestConvolutionDgradNode.cpp
    TestConvolutionNode.cpp
    TestError.cpp
    TestGraph.cpp
    TestGraphAttributes.cpp
    TestGraphGenerated.cpp
    TestPointwiseAttributes.cpp
    TestPointwiseNode.cpp
    TestScopedHipdnnBackendDescriptor.cpp
    TestTensorAttributes.cpp
    TestTensorValueAttributes.cpp
    TestTypes.cpp
    TestUtilities.cpp
    TestTopologicalSortingUtils.cpp
)

target_compile_options(hipdnn_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(hipdnn_frontend_tests 
    GTest::gtest 
    GTest::gmock
    hip::host
    Threads::Threads 
    hipdnn_sdk 
    hipdnn_frontend
)
clang_tidy_check(hipdnn_frontend_tests)

add_unit_test_target(hipdnn_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})
