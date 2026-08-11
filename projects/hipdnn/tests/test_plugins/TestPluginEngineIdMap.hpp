// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace hipdnn_tests::plugin_constants
{
template <class T>
constexpr int64_t engineId() = delete;
} // namespace hipdnn_tests::plugin_constants

// NOLINTBEGIN(bugprone-macro-parentheses) ClassName is used as a type identifier
#define HIPDNN_MAP_TO_ID(ClassName, id)      \
    class ClassName;                         \
    namespace hipdnn_tests::plugin_constants \
    {                                        \
    template <>                              \
    constexpr int64_t engineId<ClassName>()  \
    {                                        \
        return (id);                         \
    };                                       \
    }
// NOLINTEND(bugprone-macro-parentheses)

HIPDNN_MAP_TO_ID(GoodPlugin, -2);
HIPDNN_MAP_TO_ID(GoodDefaultPlugin, -3);
HIPDNN_MAP_TO_ID(NoApplicableEnginesAPlugin, -4);
HIPDNN_MAP_TO_ID(NoApplicableEnginesBPlugin, -5);
HIPDNN_MAP_TO_ID(ExecuteFailsPlugin, -6);
HIPDNN_MAP_TO_ID(DuplicateIdAPlugin, -7);
HIPDNN_MAP_TO_ID(DuplicateIdBPlugin, -7);
HIPDNN_MAP_TO_ID(KnobsPlugin, -8);
HIPDNN_MAP_TO_ID(KnobsPluginEngineB, -9);
HIPDNN_MAP_TO_ID(KnobConstraintValidationPlugin, -10);
HIPDNN_MAP_TO_ID(IncompatibleVersionPlugin, -11);

// Override-execute fake plugins. Each receives a distinct id.
HIPDNN_MAP_TO_ID(OverrideImplementingPlugin, -12);
HIPDNN_MAP_TO_ID(OverrideOmittingPlugin, -13);
HIPDNN_MAP_TO_ID(VersionLiarPlugin, -14);
HIPDNN_MAP_TO_ID(SecondOverridePlugin, -15);

// Malformed-version plugin used for load-time API-version parse rejection.
HIPDNN_MAP_TO_ID(MalformedVersionPlugin, -16);

// Version-zero plugin reports a parseable but too-low API version.
HIPDNN_MAP_TO_ID(VersionZeroPlugin, -17);

// Runtime pass-by-value fake reports K_PASS_BY_VALUE_MIN_API_VERSION ("1.2.0").
HIPDNN_MAP_TO_ID(PassByValuePlugin, -24);

// Runtime pass-by-value RECORDER fake reports "1.2.0" and records the scalar it
// resolves from device_buffers at execute (delivery-verification plugin).
HIPDNN_MAP_TO_ID(PassByValueRecorderPlugin, -25);

// Autotune test plugins.
HIPDNN_MAP_TO_ID(AutotunePlugin, -18);
HIPDNN_MAP_TO_ID(AutotunePluginEngineB, -19);
HIPDNN_MAP_TO_ID(AutotunePluginEngineC, -20);
HIPDNN_MAP_TO_ID(AutotunePluginEngineFails, -21);
HIPDNN_MAP_TO_ID(AutotunePluginEnginePrimingOnlyFails, -22);
HIPDNN_MAP_TO_ID(AutotunePluginEngineWorkspaceGrows, -23);

// Hashed-name fake: its engine id is the FNV-1a-64 hash of its own engine name,
// "TEST_HASHED_NAME_ENGINE", reproducing the id/name identity that
// HIPDNN_REGISTER_ENGINE establishes for production plugins. The literal is
// precomputed because engineNameToId() is not usable in a constant expression:
// it delegates to fnv1aHash(), which uses reinterpret_cast. Tests that rely on
// the identity assert it at runtime against engineNameToId().
HIPDNN_MAP_TO_ID(HashedNamePlugin, static_cast<int64_t>(0xD134891277747B22ULL));

namespace hipdnn_tests::plugin_constants
{
// Engine names reported by the named test plugins, both through the optional
// `hipdnnEnginePluginGetEngineName` entry point and in `EngineDetails.name`.
// All of these names are plugin-supplied and deliberately absent from the
// data_sdk engine-name registry.
//
// The good-default and execute-fails plugins keep hardcoded engine ids that
// their names do not hash back to, which is what makes them fixtures for
// name/id disagreement. The hashed-name plugin is the opposite fixture: its id
// is exactly engineNameToId() of its name, so filters that resolve a name by
// hashing it -- deselect_engines(names), set_preferred_engine_id_ext -- reach
// its engine.
inline constexpr const char* K_GOOD_DEFAULT_PLUGIN_ENGINE_NAME = "TEST_GOOD_DEFAULT_ENGINE";
inline constexpr const char* K_EXECUTE_FAILS_PLUGIN_ENGINE_NAME = "TEST_EXECUTE_FAILS_ENGINE";
inline constexpr const char* K_HASHED_NAME_PLUGIN_ENGINE_NAME = "TEST_HASHED_NAME_ENGINE";
} // namespace hipdnn_tests::plugin_constants
