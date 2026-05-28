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
#         [INSTALL_SUBDIR <subdir>]
#         [TEST_CONFIG <path>]
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
#   ``INSTALL_SUBDIR``
#     Optional. When provided, also stage an install-tree ``add_test()``
#     entry so this test appears in the installed ``CTestTestfile.cmake``
#     produced by ``install_provider_ctest_files(<subdir>)``. The value
#     must match the subdir passed to that helper. The test config TOML
#     (if any) is installed alongside the CTestTestfile so it resolves
#     relative to ctest's working directory.
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
    cmake_parse_arguments(ARG "" "TARGET_NAME;PLUGIN_TARGET;ENGINE_NAME;INSTALL_SUBDIR;TEST_CONFIG" "GTEST_FILTER" ${ARGN})

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
        DEPENDS ${ARG_PLUGIN_TARGET} hipdnn_integration_tests
        COMMENT "Running integration tests for ${ARG_ENGINE_NAME}"
        USES_TERMINAL
        VERBATIM
    )

    # Register with ctest so the cross-provider integration suite is picked up
    # by the calling project's `<project>-integration-check` target (which runs
    # `ctest -L integration_test`) and by direct `ctest` invocations from the
    # project's build subdir. Labels mirror add_integration_test_target so the
    # test is selected the same way as the provider's own integration tests,
    # plus an `external_integration_test` label and the engine name for filtering.
    set(_LABELS "integration_test;slow;external_integration_test;${ARG_ENGINE_NAME}")
    add_test(NAME ${ARG_TARGET_NAME} COMMAND ${_CMD})
    set_tests_properties(${ARG_TARGET_NAME} PROPERTIES LABELS "${_LABELS}")

    # Stage an install-tree add_test() snippet so install_provider_ctest_files
    # can include this test in the installed CTestTestfile.cmake. Required for
    # CI flows that invoke ctest from the install tree (e.g. TheRock).
    if(ARG_INSTALL_SUBDIR)
        if(ARG_TEST_CONFIG)
            get_filename_component(_install_config "${ARG_TEST_CONFIG}" NAME)
            install(FILES "${ARG_TEST_CONFIG}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/${ARG_INSTALL_SUBDIR}"
            )
        endif()

        # Install-tree paths for the test binary and plugin are relative to the
        # directory ctest runs from, which is
        # ${CMAKE_INSTALL_BINDIR}/${ARG_INSTALL_SUBDIR}/. Compute via
        # file(RELATIVE_PATH) against the real install locations so the result
        # tracks the platform layout — the plugin lives under bin/ on Windows
        # but lib/ on Linux (HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR encodes
        # that split). Use a synthetic absolute root because file(RELATIVE_PATH)
        # requires absolute inputs and the real prefix may differ between
        # configure-time and the deploy target.
        set(_synthetic_root "/__hipdnn_install_root__")
        set(_install_cwd_abs
            "${_synthetic_root}/${CMAKE_INSTALL_BINDIR}/${ARG_INSTALL_SUBDIR}"
        )
        set(_bin_abs
            "${_synthetic_root}/${CMAKE_INSTALL_BINDIR}/hipdnn_integration_tests${CMAKE_EXECUTABLE_SUFFIX}"
        )
        set(_plugin_abs
            "${_synthetic_root}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${ARG_PLUGIN_TARGET}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )
        file(RELATIVE_PATH _install_bin "${_install_cwd_abs}" "${_bin_abs}")
        file(RELATIVE_PATH _install_plugin "${_install_cwd_abs}" "${_plugin_abs}")

        set(_install_cmd "add_test([=[${ARG_TARGET_NAME}]=] \"${_install_bin}\" \"--test-article\" \"${_install_plugin}\" \"--test-engine\" \"${ARG_ENGINE_NAME}\"")
        if(ARG_TEST_CONFIG)
            string(APPEND _install_cmd " \"--test-config\" \"${_install_config}\"")
        endif()
        if(ARG_GTEST_FILTER)
            string(APPEND _install_cmd " \"--gtest_filter=${_GTEST_FILTER_STR}\"")
        endif()
        string(APPEND _install_cmd ")\n")
        string(APPEND _install_cmd
            "set_tests_properties([=[${ARG_TARGET_NAME}]=] PROPERTIES LABELS \"${_LABELS}\")\n"
        )

        set_property(GLOBAL APPEND_STRING
            PROPERTY "EXTERNAL_TEST_INSTALL_STAGING_${ARG_INSTALL_SUBDIR}"
            "${_install_cmd}"
        )
    endif()
endfunction()
