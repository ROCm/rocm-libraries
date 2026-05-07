# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Stream B → Stream C handoff for override-execute dispatch testing
# (RFC 0008 §B.6a).
#
# Publishes the finalized CMake target names for the six fake plugins
# Stream B contributes for override-execute dispatch testing:
#   1. test_override_implementing_plugin
#   2. test_override_omitting_plugin
#   3. test_version_liar_plugin
#   4. test_second_override_plugin
#   5. test_malformed_version_plugin (post-review fix #1)
#   6. test_version_zero_plugin       (plan T-missing #2)
# Stream C integration tests (`tests/frontend/Integration*.cpp`) include
# this file from `tests/frontend/CMakeLists.txt` so they can reference
# the targets in `add_dependencies(...)` and emit `*_PLUGIN_NAME` defines
# for use by `getTestCustomFilepathForPlugin(...)`.
#
# Loader-discovery contract for Stream C
# --------------------------------------
# All four plugins are built via `add_test_plugin(<target>)` in the
# sibling `tests/test_plugins/CMakeLists.txt`, which places the resulting
# `.so` files in `${HIPDNN_TEST_PLUGIN_DIR}/custom`. Stream C tests load
# them via `hipdnn_tests::plugin_constants::getTestCustomFilepathForPlugin(<name>)`
# (see `tests/test_plugins/TestPluginConstants.hpp:30`) — the same
# loader-discovery mechanism every existing test plugin uses. No new
# discovery infrastructure is required.
#
# Centralized `K_OVERRIDE_EXECUTE_MIN_API_VERSION` constant
# --------------------------------------------------------
# The shared version-constant header (Foundation task A.0) lives at
# `plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp` so all
# consumers (backend, fake test plugins, frontend integration tests) reach
# it through the public `hipdnn_plugin_sdk` link, with no layering
# inversion into `backend/src`. Stream C frontend tests include it
# directly via `<hipdnn_plugin_sdk/PluginVersionConstants.hpp>` to avoid
# re-spelling the literal `"1.1.0"` (RFC 0008 §4.5).

# --- Override-implementing fake (admitted by version filter, exports the
#     optional override symbol). Tests assert that dispatch routes through
#     `hipdnnEnginePluginExecuteOpGraphWithOverrides`.
set(HIPDNN_TEST_OVERRIDE_IMPLEMENTING_PLUGIN_TARGET test_override_implementing_plugin
    CACHE INTERNAL "Override-implementing fake plugin target (RFC 0008)")

# --- Override-omitting fake (filtered out by version filter for override
#     graphs; admitted for non-override graphs as the binary-compat
#     regression).
set(HIPDNN_TEST_OVERRIDE_OMITTING_PLUGIN_TARGET test_override_omitting_plugin
    CACHE INTERNAL "Override-omitting fake plugin target (RFC 0008)")

# --- Version-liar fake (admitted by version filter, but does NOT export
#     the override symbol). Exercises the `hasOverrideExecute()` dispatch-
#     time safety net (RFC §4.6, §7.2).
set(HIPDNN_TEST_VERSION_LIAR_PLUGIN_TARGET test_version_liar_plugin
    CACHE INTERNAL "Version-liar fake plugin target (RFC 0008)")

# --- Second override-implementing fake (distinct engine id) used by
#     Test #9 to cover multiple plugins serving the same override graph.
set(HIPDNN_TEST_SECOND_OVERRIDE_PLUGIN_TARGET test_second_override_plugin
    CACHE INTERNAL "Second override-implementing fake plugin target (RFC 0008)")

# --- Malformed-version fake (RFC 0008 post-review fix #1): reports
#     a non-parseable plugin API version string. The host's load-time
#     `parsedApiVersion()` cache yields `nullopt`; `validateBeforeAdding`
#     throws inside the existing `tryCatch` wrapper so the plugin is
#     skipped instead of crashing every dispatch.
set(HIPDNN_TEST_MALFORMED_VERSION_PLUGIN_TARGET test_malformed_version_plugin
    CACHE INTERNAL "Post-review fix malformed-version fake plugin target (RFC 0008)")

# --- Version-zero fake (RFC 0008 plan T-missing #2): reports a parseable
#     but too-low API version ("0.0.0"). Distinct from the
#     malformed-version fake — exercises the parsed-but-too-low rejection
#     path so the version baseline is verified independent of the
#     malformed/unparseable code path.
set(HIPDNN_TEST_VERSION_ZERO_PLUGIN_TARGET test_version_zero_plugin
    CACHE INTERNAL "Plan T-missing #2 version-zero fake plugin target (RFC 0008)")
