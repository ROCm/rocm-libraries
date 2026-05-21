# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

find_package(hip REQUIRED)

# Plugin target names are defined in cmake/TestPluginNames.cmake (included by
# the root CMakeLists.txt) so they are available to both this directory and
# the test executables that depend on these targets, regardless of the order
# subdirectories are added.

# Function to create a test plugin with common configuration
function(add_test_plugin target_name)
    set(source_files ${ARGN})

    add_library(${target_name} SHARED ${source_files})

    target_compile_definitions(
        ${target_name} PRIVATE HIPDNN_COMPONENT_NAME="${target_name}" __HIP_PLATFORM_AMD__
    )

    target_include_directories(${target_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(${target_name} PUBLIC hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hip::host)

    target_compile_options(${target_name} PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

    # Default to custom test plugin directory unless overridden
    if(NOT DEFINED ${target_name}_OUTPUT_DIR)
        set(${target_name}_OUTPUT_DIR "${HIPDNN_TEST_PLUGIN_DIR}/custom")
        set(${target_name}_INSTALL_DIR "${HIPDNN_TEST_PLUGIN_INSTALL_DIR}/custom")
    endif()

    set_target_properties(
        ${target_name}
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${${target_name}_OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${${target_name}_OUTPUT_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY "${${target_name}_OUTPUT_DIR}"
            THEROCK_INSTALL_RPATH_ORIGIN "${${target_name}_INSTALL_DIR}"
    )

    install(TARGETS ${target_name} LIBRARY DESTINATION "${${target_name}_INSTALL_DIR}"
            RUNTIME DESTINATION "${${target_name}_INSTALL_DIR}"
    )

    clang_tidy_check(${target_name})

endfunction()

# NOTE: If you add a new test plugin, then ensure it is listed in the frontend and backend
# CMakeLists.txt files as a dependency, and add the plugin name define.

# test_good_plugin is a well-behaved plugin that returns a happy path
add_test_plugin(${TEST_GOOD_PLUGIN_NAME} TestGoodPlugin.cpp)

# test_execute_fails_plugin is a plugin that fails during execution
add_test_plugin(${TEST_EXECUTE_FAILS_PLUGIN_NAME} TestExecuteFailsPlugin.cpp)

# test_no_applicable_engines_a/b_plugin is a plugin that reports no applicable engines
add_test_plugin(${TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME} TestNoApplicableEnginesAPlugin.cpp)
add_test_plugin(${TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME} TestNoApplicableEnginesBPlugin.cpp)

# test_good_plugin_duplicate_id_a/b shares an engine id with each other
add_test_plugin(${TEST_DUPLICATE_ID_A_PLUGIN_NAME} TestDuplicateIdAPlugin.cpp)
add_test_plugin(${TEST_DUPLICATE_ID_B_PLUGIN_NAME} TestDuplicateIdBPlugin.cpp)

# test_incomplete_api_plugin only implements part of the plugin API
add_test_plugin(${TEST_INCOMPLETE_API_PLUGIN_NAME} TestIncompleteApiPlugin.cpp)

# test_knobs_plugin is a plugin that exposes engine configuration knobs
add_test_plugin(${TEST_KNOBS_PLUGIN_NAME} TestKnobsPlugin.cpp)

# test_knob_constraint_validation_plugin is a plugin that validates knob constraint types
add_test_plugin(${TEST_KNOB_CONSTRAINT_VALIDATION_PLUGIN_NAME} TestKnobConstraintValidationPlugin.cpp)

# test_incompatible_version_plugin is a plugin that returns an incompatible plugin API version
add_test_plugin(${TEST_INCOMPATIBLE_VERSION_PLUGIN_NAME} TestIncompatibleVersionPlugin.cpp)

# Create the default plugin in test_plugins/default
set(test_good_default_plugin_OUTPUT_DIR "${HIPDNN_TEST_PLUGIN_DIR}/default")
set(test_good_default_plugin_INSTALL_DIR "${HIPDNN_TEST_PLUGIN_INSTALL_DIR}/default")
add_test_plugin(${TEST_GOOD_DEFAULT_PLUGIN_NAME} TestGoodDefaultPlugin.cpp)

# Heuristic test plugins
# test_good_heuristic_plugin implements the full heuristic plugin API
add_test_plugin(${TEST_GOOD_HEURISTIC_PLUGIN_NAME} TestGoodHeuristicPlugin.cpp)

# test_incomplete_heuristic_api_plugin is missing required symbols (for error testing)
add_test_plugin(${TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME} TestIncompleteHeuristicApiPlugin.cpp)

# test_no_optional_heuristic_plugin omits optional symbols (GetPolicyName, SetLogLevel)
add_test_plugin(${TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME} TestNoOptionalHeuristicPlugin.cpp)

# test_bad_api_version_heuristic_plugin returns incompatible API version
add_test_plugin(${TEST_BAD_API_VERSION_HEURISTIC_PLUGIN_NAME} TestBadApiVersionHeuristicPlugin.cpp)

# test_empty_name_heuristic_plugin returns empty policy name
add_test_plugin(${TEST_EMPTY_NAME_HEURISTIC_PLUGIN_NAME} TestEmptyNameHeuristicPlugin.cpp)

# test_duplicate_policy_id_a/b_plugin return same policy name (same ID)
add_test_plugin(${TEST_DUPLICATE_POLICY_ID_A_PLUGIN_NAME} TestDuplicatePolicyIdAPlugin.cpp)
add_test_plugin(${TEST_DUPLICATE_POLICY_ID_B_PLUGIN_NAME} TestDuplicatePolicyIdBPlugin.cpp)

# Plugin names are exported via cmake/TestPluginNames.cmake; no PARENT_SCOPE
# re-export needed.
