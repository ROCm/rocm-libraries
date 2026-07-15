# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    rocke_client_integration_tests
    main.cpp
    TestRockeClientLoad.cpp
    TestRockeClientApplicability.cpp
)

target_compile_options(rocke_client_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# rocke-client is gated entirely behind HIPKERNELPROVIDER_ENABLE_ROCKE, so its
# integration tests always drive the SDPA graph path regardless of the backend's
# HIPDNN_ENABLE_SDPA setting. Define it unconditionally for this target.
target_compile_definitions(
    rocke_client_integration_tests
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/rocke-client"
        HIPDNN_ENABLE_SDPA
)

target_link_libraries(
    rocke_client_integration_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk
                                          hipdnn_plugin_sdk Threads::Threads
)

add_dependencies(rocke_client_integration_tests rocke_client)

clang_tidy_check(rocke_client_integration_tests)

add_integration_test_target(rocke_client_integration_tests ${CMAKE_CURRENT_BINARY_DIR}
    LABELS quick)
