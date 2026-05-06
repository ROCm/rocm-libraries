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

// RFC 0008 Phase 1 fake plugins. Each receives a distinct id; the second
// override-implementing fake intentionally differs from the first so
// Test #9 can distinguish them in the applicable-engine set.
HIPDNN_MAP_TO_ID(OverrideImplementingPlugin, -12);
HIPDNN_MAP_TO_ID(OverrideOmittingPlugin, -13);
HIPDNN_MAP_TO_ID(VersionLiarPlugin, -14);
HIPDNN_MAP_TO_ID(SecondOverridePlugin, -15);
