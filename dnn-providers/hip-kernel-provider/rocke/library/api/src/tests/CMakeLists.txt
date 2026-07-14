# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    rocke_client_tests
    main.cpp
    TestRockeClientContainer.cpp
    TestRockeClientContext.cpp
    TestRockeClientEngine.cpp
    TestPluginPublic.cpp
    dispatcher/TestAotCatalog.cpp
    dispatcher/TestKpackModuleLoad.cpp
    dispatcher/TestRockeClientDispatcher.cpp
    dispatcher/TestSdpaGraphAdapter.cpp
    dispatcher/TestSelectionConstraints.cpp
)

target_compile_options(rocke_client_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    rocke_client_tests PUBLIC GTest::gtest hip::host rocke_client_private hipdnn_test_sdk
                              hipdnn_plugin_sdk Threads::Threads
)

# Absolute path to the committed synthetic .kpack fixture (host-tier ELF magic test).
target_compile_definitions(rocke_client_tests PRIVATE
    "ROCKE_KPACK_TEST_FIXTURE=\"${CMAKE_CURRENT_SOURCE_DIR}/data/rocke_client_gfx942.kpack\""
)

# Optional: bake in a build-tree bundle dir so the GPU tier can find a real
# AOT-produced .kpack without setting ROCKE_CLIENT_BUNDLE_DIR at runtime.
# Pass -DROCKE_CLIENT_TEST_BUNDLE_DIR=/path/to/arch_content/rocke to cmake.
if(ROCKE_CLIENT_TEST_BUNDLE_DIR)
    target_compile_definitions(rocke_client_tests PRIVATE
        "ROCKE_CLIENT_BUNDLE_DIR_DEFAULT=\"${ROCKE_CLIENT_TEST_BUNDLE_DIR}\""
    )
endif()

target_include_directories(rocke_client_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

clang_tidy_check(rocke_client_tests)

add_unit_test_target(rocke_client_tests ${CMAKE_CURRENT_BINARY_DIR} LABELS quick)
