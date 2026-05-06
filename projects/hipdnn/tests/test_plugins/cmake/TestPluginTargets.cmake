# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# RFC 0008 Phase 1 — Stream B → Stream C handoff (task B.6a).
#
# Publishes the finalized CMake target names for the four fake plugins
# Stream B contributes for RFC 0008 Phase 1 dispatch testing. Stream C
# integration tests (`tests/frontend/Integration*.cpp`) include this file
# from `tests/frontend/CMakeLists.txt` so they can reference the targets
# in `add_dependencies(...)` and emit `*_PLUGIN_NAME` defines for use by
# `getTestCustomFilepathForPlugin(...)`.
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
# Centralized `K_PHASE1_OVERRIDE_MIN_VERSION` constant
# ------------------------------------------------
# The shared version-constant header (Foundation task A.0) lives at
# `plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp` so all
# consumers (backend, fake test plugins, frontend integration tests) reach
# it through the public `hipdnn_plugin_sdk` link, with no layering
# inversion into `backend/src`. Stream C frontend tests include it
# directly via `<hipdnn_plugin_sdk/PluginVersionConstants.hpp>` to avoid
# re-spelling the placeholder literal `"1.1.0"` (RFC §4.5).

# --- Override-implementing fake (admitted by version filter, exports the
#     optional override symbol). Tests assert that dispatch routes through
#     `hipdnnEnginePluginExecuteOpGraphWithOverrides`.
set(HIPDNN_TEST_OVERRIDE_IMPLEMENTING_PLUGIN_TARGET test_override_implementing_plugin
    CACHE INTERNAL "RFC 0008 Phase 1 override-implementing fake plugin target")

# --- Override-omitting fake (filtered out by version filter for override
#     graphs; admitted for non-override graphs as the binary-compat
#     regression).
set(HIPDNN_TEST_OVERRIDE_OMITTING_PLUGIN_TARGET test_override_omitting_plugin
    CACHE INTERNAL "RFC 0008 Phase 1 override-omitting fake plugin target")

# --- Version-liar fake (admitted by version filter, but does NOT export
#     the override symbol). Exercises the `hasOverrideExecute()` dispatch-
#     time safety net (RFC §4.6, §7.2).
set(HIPDNN_TEST_VERSION_LIAR_PLUGIN_TARGET test_version_liar_plugin
    CACHE INTERNAL "RFC 0008 Phase 1 version-liar fake plugin target")

# --- Second override-implementing fake (distinct engine id) used by
#     Test #9 to cover multiple plugins serving the same override graph.
set(HIPDNN_TEST_SECOND_OVERRIDE_PLUGIN_TARGET test_second_override_plugin
    CACHE INTERNAL "RFC 0008 Phase 1 second override-implementing fake plugin target")
