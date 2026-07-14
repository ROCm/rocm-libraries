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
                              hipdnn_plugin_sdk Threads::Threads ${CMAKE_DL_LIBS}
)

# Copy the committed test fixture tree beside the test binary so testModuleDir()
# resolves arch_content/rocke/<arch>/rocke_client_<arch>.kpack at runtime.
add_custom_command(
    TARGET rocke_client_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data/arch_content"
        "$<TARGET_FILE_DIR:rocke_client_tests>/arch_content"
    COMMENT "Copying rocke-client test fixture tree"
)

target_include_directories(rocke_client_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

clang_tidy_check(rocke_client_tests)

add_unit_test_target(rocke_client_tests ${CMAKE_CURRENT_BINARY_DIR} LABELS quick)
