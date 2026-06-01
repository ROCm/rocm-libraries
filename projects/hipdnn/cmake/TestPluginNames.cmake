# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared definitions for test plugin target names.
#
# Several test executables (hipdnn_backend_tests, hipdnn_public_frontend_tests, …)
# reference test plugin targets that are added by tests/test_plugins/. CMake
# processes that subdirectory after backend/ and frontend/, so any test that
# uses the names without a shared definition would either silently expand them
# to empty strings (producing -DTEST_FOO=\"\" compile defs and no-op
# add_dependencies entries) or have to duplicate the literal name and rely on
# the two copies staying in sync.
#
# Defining the names here, before any subdirectory is processed, avoids both
# pitfalls. Both tests/test_plugins/ (which creates the targets) and the test
# executables (which depend on them) read from the same source of truth.

# Engine / generic test plugins
set(TEST_PLUGIN1_NAME "hipdnn_test_plugin1")
set(TEST_PLUGIN2_NAME "hipdnn_test_plugin2")
set(TEST_NO_API_VERSION_PLUGIN_NAME "hipdnn_test_no_api_version_plugin_name")
set(TEST_ENGINE_PLUGIN1_NAME "hipdnn_test_engine_plugin1")

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

# Override-execute fake plugin names
set(HIPDNN_TEST_OVERRIDE_IMPLEMENTING_PLUGIN_TARGET "test_override_implementing_plugin")
set(HIPDNN_TEST_OVERRIDE_OMITTING_PLUGIN_TARGET "test_override_omitting_plugin")
set(HIPDNN_TEST_VERSION_LIAR_PLUGIN_TARGET "test_version_liar_plugin")
set(HIPDNN_TEST_SECOND_OVERRIDE_PLUGIN_TARGET "test_second_override_plugin")
set(HIPDNN_TEST_MALFORMED_VERSION_PLUGIN_TARGET "test_malformed_version_plugin")
set(HIPDNN_TEST_VERSION_ZERO_PLUGIN_TARGET "test_version_zero_plugin")

# Heuristic plugin test names
set(TEST_GOOD_HEURISTIC_PLUGIN_NAME "test_good_heuristic_plugin")
set(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME "test_incomplete_heuristic_api_plugin")
set(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME "test_no_optional_heuristic_plugin")
set(TEST_BAD_API_VERSION_HEURISTIC_PLUGIN_NAME "test_bad_api_version_heuristic_plugin")
set(TEST_EMPTY_NAME_HEURISTIC_PLUGIN_NAME "test_empty_name_heuristic_plugin")
set(TEST_DUPLICATE_POLICY_ID_A_PLUGIN_NAME "test_duplicate_policy_id_a_plugin")
set(TEST_DUPLICATE_POLICY_ID_B_PLUGIN_NAME "test_duplicate_policy_id_b_plugin")
