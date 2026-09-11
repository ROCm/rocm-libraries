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
#         [TEST_CATEGORIES_YAML <path>]
#         [INSTALL_TEST_FILE <path>]
#         [TEST_NAME_PREFIX <prefix>]
#         [TEST_CONFIG <path>]
#         [REFERENCE_EXECUTOR <cpu|gpu>]
#         [ENVIRONMENT <VAR=value>...]
#         [INSTALL_ENVIRONMENT <VAR=value>...]
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
#   ``REFERENCE_EXECUTOR``
#     Optional reference executor passed via ``--reference-executor``.
#
#   ``ENVIRONMENT``
#     Optional list of ``VAR=value`` entries applied to the build-tree custom
#     target and CTest test.
#
#   ``INSTALL_ENVIRONMENT``
#     Optional list of ``VAR=value`` entries for the staged install-tree CTest
#     test. Defaults to ``ENVIRONMENT`` when omitted.
#
#   ``GTEST_FILTER``
#     Optional list of Google Test filter expressions. Each entry is joined
#     with ``:`` to form the final filter string passed via ``--gtest_filter``.
#     If omitted, all tests run. Patterns can be specified one per line for
#     readability.
#
#   ``TEST_CATEGORIES_YAML``
#     Optional path to a GTest ``test_categories.yaml``. When provided, this
#     helper also creates category-specific CTest suites for
#     ``hipdnn_integration_tests`` with the plugin/test-engine/test-config
#     arguments plus generated ``--gtest_filter`` values.
#
#   ``INSTALL_TEST_FILE``
#     Optional install-tree ``CTestTestfile.cmake`` staging file used with
#     ``TEST_CATEGORIES_YAML``. The generated category suites use install-relative
#     paths for both ``hipdnn_integration_tests`` and the plugin.
#
#   ``TEST_NAME_PREFIX``
#     Optional prefix for generated category suite CTest names. Defaults to
#     ``TARGET_NAME``.

# Builds the build-tree command for an external integration test.
#
# Arguments:
#   out_var - Variable to receive the command list
# ~~~
macro(_build_external_integration_command out_var)
    set(${out_var}
        $<TARGET_FILE:hipdnn_integration_tests>
        --test-article $<TARGET_FILE:${ARG_PLUGIN_TARGET}>
        --test-engine ${ARG_ENGINE_NAME}
    )
    if(ARG_TEST_CONFIG)
        list(APPEND ${out_var} "--test-config" "${ARG_TEST_CONFIG}")
    endif()
    if(ARG_REFERENCE_EXECUTOR)
        list(APPEND ${out_var} "--reference-executor" "${ARG_REFERENCE_EXECUTOR}")
    endif()
    if(ARG_GTEST_FILTER)
        list(JOIN ARG_GTEST_FILTER ":" _GTEST_FILTER_STR)
        list(APPEND ${out_var} "--gtest_filter=${_GTEST_FILTER_STR}")
    endif()
endmacro()

# Stages the install-tree CTest entry for an external integration test.
#
# Uses the ARG_* variables parsed by add_external_integration_test_target().
# ~~~
macro(_stage_external_integration_install_test)
    set(_install_bin "")
    set(_install_plugin "")
    set(_install_config "")
    if(ARG_INSTALL_SUBDIR)
        if(ARG_TEST_CONFIG)
            get_filename_component(_install_config "${ARG_TEST_CONFIG}" NAME)
            install(FILES "${ARG_TEST_CONFIG}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/${ARG_INSTALL_SUBDIR}"
            )
        endif()

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

        if(NOT _GENERATE_EXTERNAL_CATEGORY_SUITES)
            set(_install_cmd "add_test(\"${ARG_TARGET_NAME}\" \"${_install_bin}\" \"--test-article\" \"${_install_plugin}\" \"--test-engine\" \"${ARG_ENGINE_NAME}\"")
            if(ARG_TEST_CONFIG)
                string(APPEND _install_cmd " \"--test-config\" \"${_install_config}\"")
            endif()
            if(ARG_REFERENCE_EXECUTOR)
                string(APPEND _install_cmd " \"--reference-executor\" \"${ARG_REFERENCE_EXECUTOR}\"")
            endif()
            if(ARG_GTEST_FILTER)
                string(APPEND _install_cmd " \"--gtest_filter=${_GTEST_FILTER_STR}\"")
            endif()
            string(APPEND _install_cmd ")\n")

            set(_install_properties "LABELS \"${_LABELS}\"")
            if(ARG_INSTALL_ENVIRONMENT)
                set(_install_environment "${ARG_INSTALL_ENVIRONMENT}")
            else()
                set(_install_environment "${ARG_ENVIRONMENT}")
            endif()
            if(_install_environment)
                string(REPLACE ";" "\\;" _install_environment_escaped "${_install_environment}")
                string(APPEND _install_properties " ENVIRONMENT \"${_install_environment_escaped}\"")
            endif()

            string(APPEND _install_cmd
                "set_tests_properties(\"${ARG_TARGET_NAME}\" PROPERTIES ${_install_properties})\n"
            )

            set_property(GLOBAL APPEND_STRING
                PROPERTY "EXTERNAL_TEST_INSTALL_STAGING_${ARG_INSTALL_SUBDIR}"
                "${_install_cmd}"
            )
        endif()
    endif()
endmacro()

# Adds category-specific GTest-filtered CTest suites.
#
# Uses the ARG_* variables parsed by add_external_integration_test_target().
# ~~~
macro(_add_external_integration_category_suites)
    if(_GENERATE_EXTERNAL_CATEGORY_SUITES)
        set(_category_prefix "${ARG_TARGET_NAME}")
        if(ARG_TEST_NAME_PREFIX)
            set(_category_prefix "${ARG_TEST_NAME_PREFIX}")
        endif()

        set(_category_command_args
            "--test-article" "$<TARGET_FILE:${ARG_PLUGIN_TARGET}>"
            "--test-engine" "${ARG_ENGINE_NAME}"
        )
        if(ARG_TEST_CONFIG)
            list(APPEND _category_command_args "--test-config" "${ARG_TEST_CONFIG}")
        endif()
        if(ARG_REFERENCE_EXECUTOR)
            list(APPEND _category_command_args "--reference-executor" "${ARG_REFERENCE_EXECUTOR}")
        endif()

        set(_apply_category_args
            TEST_NAME_PREFIX "${_category_prefix}"
            COMMAND_ARGS ${_category_command_args}
            ADDITIONAL_LABELS "integration_test" "slow" "external_integration_test" "${ARG_ENGINE_NAME}"
        )
        if(ARG_ENVIRONMENT)
            list(APPEND _apply_category_args ENVIRONMENT ${ARG_ENVIRONMENT})
        endif()
        if(ARG_ENVIRONMENT_MODIFICATION)
            list(APPEND _apply_category_args
                ENVIRONMENT_MODIFICATION ${ARG_ENVIRONMENT_MODIFICATION})
        endif()
        if(ARG_FIXTURES_REQUIRED)
            list(APPEND _apply_category_args FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
        endif()

        if(ARG_INSTALL_TEST_FILE AND _install_bin)
            set(_category_install_command_args
                "--test-article" "${_install_plugin}"
                "--test-engine" "${ARG_ENGINE_NAME}"
            )
            if(ARG_TEST_CONFIG)
                list(APPEND _category_install_command_args "--test-config" "${_install_config}")
            endif()
            if(ARG_REFERENCE_EXECUTOR)
                list(APPEND _category_install_command_args "--reference-executor" "${ARG_REFERENCE_EXECUTOR}")
            endif()

            list(APPEND _apply_category_args
                INSTALL_TEST_FILE "${ARG_INSTALL_TEST_FILE}"
                INSTALL_EXECUTABLE "${_install_bin}"
                INSTALL_COMMAND_ARGS ${_category_install_command_args}
            )
        endif()

        apply_test_category_labels(
            hipdnn_integration_tests
            "${ARG_TEST_CATEGORIES_YAML}"
            "${CMAKE_CURRENT_BINARY_DIR}"
            ${_apply_category_args}
        )
    endif()
endmacro()

# Adds a custom target and optional CTest entries for an external integration test.
#
# When TEST_CATEGORIES_YAML is provided, the base target stays CMake-only and
# category-specific GTest-filtered suites cover the CTest surface.
# ~~~
function(add_external_integration_test_target)
    cmake_parse_arguments(
        ARG
        ""
        "TARGET_NAME;PLUGIN_TARGET;ENGINE_NAME;INSTALL_SUBDIR;TEST_CONFIG;REFERENCE_EXECUTOR;TEST_CATEGORIES_YAML;INSTALL_TEST_FILE;TEST_NAME_PREFIX"
        "GTEST_FILTER;ENVIRONMENT;INSTALL_ENVIRONMENT;ENVIRONMENT_MODIFICATION;FIXTURES_REQUIRED"
        ${ARGN}
    )

    if(NOT ARG_TARGET_NAME)
        message(FATAL_ERROR "add_external_integration_test_target: TARGET_NAME is required")
    endif()
    if(NOT ARG_PLUGIN_TARGET)
        message(FATAL_ERROR "add_external_integration_test_target: PLUGIN_TARGET is required")
    endif()
    if(NOT ARG_ENGINE_NAME)
        message(FATAL_ERROR "add_external_integration_test_target: ENGINE_NAME is required")
    endif()

    _build_external_integration_command(_CMD)

    set(_TARGET_CMD ${_CMD})
    if(ARG_ENVIRONMENT)
        set(_TARGET_CMD ${CMAKE_COMMAND} -E env ${ARG_ENVIRONMENT} ${_CMD})
    endif()

    add_custom_target(${ARG_TARGET_NAME}
        COMMAND ${_TARGET_CMD}
        DEPENDS ${ARG_PLUGIN_TARGET} hipdnn_integration_tests
        COMMENT "Running integration tests for ${ARG_ENGINE_NAME}"
        USES_TERMINAL
        VERBATIM
    )

    set(_GENERATE_EXTERNAL_CATEGORY_SUITES FALSE)
    if(ARG_TEST_CATEGORIES_YAML AND COMMAND apply_test_category_labels)
        set(_GENERATE_EXTERNAL_CATEGORY_SUITES TRUE)
    endif()

    # Register with ctest so the cross-provider integration suite is picked up
    # by the calling project's `<project>-integration-check` target (which runs
    # `ctest -L integration_test`) and by direct `ctest` invocations from the
    # project's build subdir. Labels mirror add_integration_test_target so the
    # test is selected the same way as the provider's own integration tests,
    # plus an `external_integration_test` label and the engine name for filtering.
    set(_LABELS "integration_test;slow;external_integration_test;${ARG_ENGINE_NAME}")
    if(NOT _GENERATE_EXTERNAL_CATEGORY_SUITES)
        add_test(NAME ${ARG_TARGET_NAME} COMMAND ${_CMD})
        set_tests_properties(${ARG_TARGET_NAME} PROPERTIES LABELS "${_LABELS}")
        if(ARG_ENVIRONMENT)
            set_tests_properties(${ARG_TARGET_NAME} PROPERTIES ENVIRONMENT "${ARG_ENVIRONMENT}")
        endif()
        if(ARG_ENVIRONMENT_MODIFICATION)
            set_tests_properties(${ARG_TARGET_NAME} PROPERTIES
                ENVIRONMENT_MODIFICATION "${ARG_ENVIRONMENT_MODIFICATION}")
        endif()
        if(ARG_FIXTURES_REQUIRED)
            set_tests_properties(${ARG_TARGET_NAME} PROPERTIES
                FIXTURES_REQUIRED "${ARG_FIXTURES_REQUIRED}")
        endif()
    endif()

    # Stage an install-tree add_test() snippet so install_provider_ctest_files
    # can include this test in the installed CTestTestfile.cmake. Required for
    # CI flows that invoke ctest from the install tree (e.g. TheRock).
    _stage_external_integration_install_test()
    _add_external_integration_category_suites()
endfunction()
