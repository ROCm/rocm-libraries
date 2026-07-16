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
    dispatcher/TestRockeClientLaunchGpu.cpp
    dispatcher/TestAotManifestLoader.cpp
    dispatcher/TestRockeClientDispatcher.cpp
    dispatcher/TestSdpaGraphAdapter.cpp
    dispatcher/TestSelectionConstraints.cpp
    plans/TestLaunchAbi.cpp
)

target_compile_options(rocke_client_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# Locate the per-arch AOT bundle (rocke_client_<arch>.{json,kpack}) the build
# packs beside the plugin; the GPU-tier launch test reads the running arch's
# bundle from here and self-skips when it is absent (kpack target not built).
target_compile_definitions(rocke_client_tests PRIVATE
    ROCKE_AOT_BUNDLE_ROOT="${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/arch_content/rocke")

target_link_libraries(
    rocke_client_tests PUBLIC GTest::gtest hip::host rocke_client_private hipdnn_test_sdk
                              hipdnn_plugin_sdk Threads::Threads
)

target_include_directories(rocke_client_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

clang_tidy_check(rocke_client_tests)

add_unit_test_target(rocke_client_tests ${CMAKE_CURRENT_BINARY_DIR} LABELS quick)
