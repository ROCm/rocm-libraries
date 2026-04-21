# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# HipdnnIntegrationTestHelpers
# ----------------------------
#
# Provides the ``add_external_integration_test_target()`` function for creating
# custom targets that run the ``hipdnn_integration_tests`` binary against a
# specific plugin.
#
# This module is distributed as part of the ``hipdnn_integration_tests``
# CMake package and is automatically included by
# ``find_package(hipdnn_integration_tests)``.

#   Create a custom target that runs integration tests against a plugin::
#
#     add_external_integration_test_target(
#         TARGET_NAME   <name>
#         PLUGIN_TARGET <target>
#         ENGINE_NAME   <engine>
#         [GTEST_FILTER <filter>...]
#     )
#
#   ``TARGET_NAME``
#     Name of the custom target to create.
#
#   ``PLUGIN_TARGET``
#     CMake target for the plugin shared library. The target must produce
#     a shared library (.so). ``$<TARGET_FILE:...>`` is used to resolve
#     the path at build time.
#
#   ``ENGINE_NAME``
#     Engine name passed via ``--test-engine`` to the test binary.
#
#   ``TEST_CONFIG``
#     Optional path to a TOML configuration file for per-test tolerance
#     overrides. Passed via ``--test-config`` to the test binary.
#
#   ``GTEST_FILTER``
#     Optional list of Google Test filter expressions. Each entry is joined
#     with ``:`` to form the final filter string passed via ``--gtest_filter``.
#     If omitted, all tests run. Patterns can be specified one per line for
#     readability.
function(add_external_integration_test_target)
    cmake_parse_arguments(ARG "" "TARGET_NAME;PLUGIN_TARGET;ENGINE_NAME;TEST_CONFIG" "GTEST_FILTER" ${ARGN})

    # Validate required arguments
    if(NOT ARG_TARGET_NAME)
        message(FATAL_ERROR "add_external_integration_test_target: TARGET_NAME is required")
    endif()
    if(NOT ARG_PLUGIN_TARGET)
        message(FATAL_ERROR "add_external_integration_test_target: PLUGIN_TARGET is required")
    endif()
    if(NOT ARG_ENGINE_NAME)
        message(FATAL_ERROR "add_external_integration_test_target: ENGINE_NAME is required")
    endif()

    # Build command
    set(_CMD
        $<TARGET_FILE:hipdnn_integration_tests>
        --test-article $<TARGET_FILE:${ARG_PLUGIN_TARGET}>
        --test-engine ${ARG_ENGINE_NAME}
    )
    if(ARG_TEST_CONFIG)
        list(APPEND _CMD "--test-config" "${ARG_TEST_CONFIG}")
    endif()
    if(ARG_GTEST_FILTER)
        list(JOIN ARG_GTEST_FILTER ":" _GTEST_FILTER_STR)
        list(APPEND _CMD "--gtest_filter=${_GTEST_FILTER_STR}")
    endif()

    add_custom_target(${ARG_TARGET_NAME}
        COMMAND ${_CMD}
        DEPENDS ${ARG_PLUGIN_TARGET}
        COMMENT "Running integration tests for ${ARG_ENGINE_NAME}"
        USES_TERMINAL
        VERBATIM
    )
endfunction()
