# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

find_package(hip REQUIRED)

# Pull in the RFC 0008 override-execute fake-plugin target name constants
# (Stream B → Stream C handoff per task B.6a).
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestPluginTargets.cmake")

# Define plugin names as variables
set(TEST_GOOD_PLUGIN_NAME "test_good_plugin")
set(TEST_EXECUTE_FAILS_PLUGIN_NAME "test_execute_fails_plugin")
set(TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME "test_no_applicable_engines_a_plugin")
set(TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME "test_no_applicable_engines_b_plugin")
set(TEST_DUPLICATE_ID_A_PLUGIN_NAME "test_duplicate_id_a_plugin")
set(TEST_DUPLICATE_ID_B_PLUGIN_NAME "test_duplicate_id_b_plugin")
set(TEST_INCOMPLETE_API_PLUGIN_NAME "test_incomplete_api_plugin")
set(TEST_GOOD_DEFAULT_PLUGIN_NAME "test_good_default_plugin")
set(TEST_KNOBS_PLUGIN_NAME "test_knobs_plugin")
set(TEST_KNOB_CONSTRAINT_VALIDATION_PLUGIN_NAME "test_knob_constraint_validation_plugin")
set(TEST_INCOMPATIBLE_VERSION_PLUGIN_NAME "test_incompatible_version_plugin")

# Heuristic plugin test names (RFC 0007)
set(TEST_GOOD_HEURISTIC_PLUGIN_NAME "test_good_heuristic_plugin")
set(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME "test_incomplete_heuristic_api_plugin")
set(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME "test_no_optional_heuristic_plugin")

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

# RFC 0008 override-execute fake plugins (target names are published in
# `cmake/TestPluginTargets.cmake` for Stream C consumption per B.6a).
#
# Override-implementing fake: reports `K_OVERRIDE_EXECUTE_MIN_API_VERSION`
# and exports the optional override-execute symbol — admitted by the
# version filter, dispatched through the override path.
add_test_plugin(${HIPDNN_TEST_OVERRIDE_IMPLEMENTING_PLUGIN_TARGET} TestOverrideImplementingPlugin.cpp)

# Override-omitting fake: reports the baseline "1.0.0" — filtered out for
# override-flag graphs, admitted for non-override graphs (binary-compat
# regression).
add_test_plugin(${HIPDNN_TEST_OVERRIDE_OMITTING_PLUGIN_TARGET} TestOverrideOmittingPlugin.cpp)

# Version-liar fake: reports `K_OVERRIDE_EXECUTE_MIN_API_VERSION` but does
# NOT export the override symbol — exercises the dispatch-time
# `hasOverrideExecute()` safety net.
add_test_plugin(${HIPDNN_TEST_VERSION_LIAR_PLUGIN_TARGET} TestVersionLiarPlugin.cpp)

# Second override-implementing fake (distinct engine id) — Test #9
# coverage for multiple plugins serving the same override graph.
add_test_plugin(${HIPDNN_TEST_SECOND_OVERRIDE_PLUGIN_TARGET} TestSecondOverridePlugin.cpp)

# Malformed-version fake (RFC 0008 post-review fix #1): exercises the
# load-time `parsedApiVersion()` rejection so a bad version string is never
# re-parsed in the dispatch hot path.
add_test_plugin(${HIPDNN_TEST_MALFORMED_VERSION_PLUGIN_TARGET} TestMalformedVersionPlugin.cpp)

# Version-zero fake (RFC 0008 plan T-missing #2): reports a parseable but
# too-low API version ("0.0.0") — exercises the parsed-but-too-low
# rejection path independent of the malformed-version fake's unparseable
# rejection path.
add_test_plugin(${HIPDNN_TEST_VERSION_ZERO_PLUGIN_TARGET} TestVersionZeroPlugin.cpp)

# Create the default plugin in test_plugins/default
set(test_good_default_plugin_OUTPUT_DIR "${HIPDNN_TEST_PLUGIN_DIR}/default")
set(test_good_default_plugin_INSTALL_DIR "${HIPDNN_TEST_PLUGIN_INSTALL_DIR}/default")
add_test_plugin(${TEST_GOOD_DEFAULT_PLUGIN_NAME} TestGoodDefaultPlugin.cpp)

# Heuristic test plugins (RFC 0007 Part 1)
# test_good_heuristic_plugin implements the full heuristic plugin API
add_test_plugin(${TEST_GOOD_HEURISTIC_PLUGIN_NAME} TestGoodHeuristicPlugin.cpp)

# test_incomplete_heuristic_api_plugin is missing required symbols (for error testing)
add_test_plugin(${TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME} TestIncompleteHeuristicApiPlugin.cpp)

# test_no_optional_heuristic_plugin omits optional symbols (GetPolicyName, SetLogLevel)
add_test_plugin(${TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME} TestNoOptionalHeuristicPlugin.cpp)

# Export plugin names to parent scope for use in test compilation
set(TEST_GOOD_PLUGIN_NAME "${TEST_GOOD_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_EXECUTE_FAILS_PLUGIN_NAME "${TEST_EXECUTE_FAILS_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME "${TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME}"
    PARENT_SCOPE
)
set(TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME "${TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME}"
    PARENT_SCOPE
)
set(TEST_DUPLICATE_ID_A_PLUGIN_NAME "${TEST_DUPLICATE_ID_A_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_DUPLICATE_ID_B_PLUGIN_NAME "${TEST_DUPLICATE_ID_B_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_INCOMPLETE_API_PLUGIN_NAME "${TEST_INCOMPLETE_API_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_GOOD_DEFAULT_PLUGIN_NAME "${TEST_GOOD_DEFAULT_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_KNOBS_PLUGIN_NAME "${TEST_KNOBS_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_KNOB_CONSTRAINT_VALIDATION_PLUGIN_NAME "${TEST_KNOB_CONSTRAINT_VALIDATION_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_INCOMPATIBLE_VERSION_PLUGIN_NAME "${TEST_INCOMPATIBLE_VERSION_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_GOOD_HEURISTIC_PLUGIN_NAME "${TEST_GOOD_HEURISTIC_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME "${TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME "${TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME}" PARENT_SCOPE)

# RFC 0008 override-execute fake-plugin target names (B.6a) are set with
# `CACHE INTERNAL` in cmake/TestPluginTargets.cmake (included at the top of
# this file), which already gives them global visibility — re-exporting
# via PARENT_SCOPE here is redundant and was removed.
