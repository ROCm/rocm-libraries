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
// The good-default and execute-fails fakes report engine names, so their ids must be
// the FNV-1a-64 hash of those names or the backend drops the engine at load. The
// literals are precomputed because engineNameToId() is not constexpr.
HIPDNN_MAP_TO_ID(GoodDefaultPlugin, static_cast<int64_t>(0x5FB06B52DB2039ACULL));
HIPDNN_MAP_TO_ID(NoApplicableEnginesAPlugin, -4);
HIPDNN_MAP_TO_ID(NoApplicableEnginesBPlugin, -5);
HIPDNN_MAP_TO_ID(ExecuteFailsPlugin, static_cast<int64_t>(0x637C0BB90F065BA2ULL));
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

// Hashed-name fake: its engine id is the FNV-1a-64 hash of "TEST_HASHED_NAME_ENGINE",
// precomputed for the same reason as the ids above.
HIPDNN_MAP_TO_ID(HashedNamePlugin, static_cast<int64_t>(0xD134891277747B22ULL));

// Lying-engine-name fake. Each id selects one malformed answer from the engine-name
// entry point. See TestLyingEngineNamePlugin.cpp.
HIPDNN_MAP_TO_ID(LyingEngineNamePlugin, -26);
HIPDNN_MAP_TO_ID(LyingEngineNamePluginEmptyName, -27);
HIPDNN_MAP_TO_ID(LyingEngineNamePluginErrorStatus, -28);

// Mismatched-name fake: a well-formed engine name that does not hash back to
// this id. See TestMismatchedNamePlugin.cpp.
HIPDNN_MAP_TO_ID(MismatchedNamePlugin, -29);

namespace hipdnn_tests::plugin_constants
{
// Engine names reported by the named test plugins. All are deliberately absent from
// the data_sdk engine-name registry. Every id above is engineNameToId() of the
// matching name, except the mismatched-name fake, which exists to violate that.
inline constexpr const char* K_GOOD_DEFAULT_PLUGIN_ENGINE_NAME = "TEST_GOOD_DEFAULT_ENGINE";
inline constexpr const char* K_EXECUTE_FAILS_PLUGIN_ENGINE_NAME = "TEST_EXECUTE_FAILS_ENGINE";
inline constexpr const char* K_HASHED_NAME_PLUGIN_ENGINE_NAME = "TEST_HASHED_NAME_ENGINE";
inline constexpr const char* K_MISMATCHED_NAME_PLUGIN_ENGINE_NAME = "TEST_MISMATCHED_NAME_ENGINE";

// Handed out by the lying-engine-name fake alongside a failure status. The host
// ignores any name that arrives with a non-success status, so this string must
// never reach an engine info.
inline constexpr const char* K_LYING_ENGINE_NAME_UNUSABLE_NAME
    = "TEST_LYING_ENGINE_NAME_MUST_BE_IGNORED";
} // namespace hipdnn_tests::plugin_constants
