// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for the MIOpen public-wrapper runtime dispatch seam
// (src/private/routing.{hpp,cpp}, RFC 0001 hipDNN forwarding). The seam lives in
// the wrapper library, which only exists when MIOPEN_ENABLE_HIPDNN_WRAPPER is
// ON, so the whole file compiles to zero tests when the wrapper is OFF. The seam
// object is compiled into the test-common library (see gtest/CMakeLists.txt)
// because the wrapper exports its symbols with hidden visibility.
//
// The tests cover the pure decision logic directly (ParseForwardingMode,
// IsInForwardingSet, ResolveRoute) and the process-global Dispatch() /
// GetForwardingMode() on the default (variable-unset) path. They assume the
// MIOPEN_HIPDNN_FORWARDING environment variable is not set in the test
// environment, which the ctest harness does not set.

#include <gtest/gtest.h>

#ifdef MIOPEN_ENABLE_HIPDNN_WRAPPER

#include "routing.hpp"

#include <ostream>

namespace miopen {
namespace wrapper {
// Readable failure diagnostics for the routing enums (picked up by GoogleTest
// via argument-dependent lookup).
static void PrintTo(ForwardingMode mode, std::ostream* os)
{
    *os << (mode == ForwardingMode::Enabled ? "ForwardingMode::Enabled"
                                            : "ForwardingMode::Disabled");
}
static void PrintTo(Route route, std::ostream* os)
{
    *os << (route == Route::Hipdnn ? "Route::Hipdnn" : "Route::Miopen");
}
} // namespace wrapper
} // namespace miopen

namespace {

using miopen::wrapper::Dispatch;
using miopen::wrapper::ForwardingMode;
using miopen::wrapper::GetForwardingMode;
using miopen::wrapper::IsInForwardingSet;
using miopen::wrapper::ParseForwardingMode;
using miopen::wrapper::ResolveRoute;
using miopen::wrapper::Route;

// ---------------------------------------------------------------------------
// ParseForwardingMode: raw MIOPEN_HIPDNN_FORWARDING value -> ForwardingMode.
// ---------------------------------------------------------------------------

struct ParseCase
{
    const char* value; // raw env value; nullptr models the variable being unset
    ForwardingMode expected;
};

class CPU_WrapperRoutingParse_NONE : public ::testing::TestWithParam<ParseCase>
{
};

// cppcheck-suppress syntaxError
TEST_P(CPU_WrapperRoutingParse_NONE, Maps)
{
    const ParseCase& c = GetParam();
    EXPECT_EQ(ParseForwardingMode(c.value), c.expected)
        << "value=" << (c.value == nullptr ? "<null>" : c.value);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_WrapperRoutingParse_NONE,
                         ::testing::Values(
                             // Unset and everything unrecognized resolve to Disabled so forwarding
                             // is never enabled by accident.
                             ParseCase{nullptr, ForwardingMode::Disabled},
                             ParseCase{"", ForwardingMode::Disabled},
                             ParseCase{"0", ForwardingMode::Disabled},
                             ParseCase{"disabled", ForwardingMode::Disabled},
                             ParseCase{"off", ForwardingMode::Disabled},
                             ParseCase{"false", ForwardingMode::Disabled},
                             ParseCase{"no", ForwardingMode::Disabled},
                             ParseCase{"garbage", ForwardingMode::Disabled},
                             // Leading whitespace is not trimmed: " enabled" is not the token
                             // "enabled" and therefore stays Disabled.
                             ParseCase{" enabled", ForwardingMode::Disabled},
                             // The accepted enable tokens, and their case-insensitivity.
                             ParseCase{"enabled", ForwardingMode::Enabled},
                             ParseCase{"ENABLED", ForwardingMode::Enabled},
                             ParseCase{"Enabled", ForwardingMode::Enabled},
                             ParseCase{"1", ForwardingMode::Enabled},
                             ParseCase{"on", ForwardingMode::Enabled},
                             ParseCase{"ON", ForwardingMode::Enabled},
                             ParseCase{"true", ForwardingMode::Enabled},
                             ParseCase{"TRUE", ForwardingMode::Enabled},
                             ParseCase{"yes", ForwardingMode::Enabled},
                             ParseCase{"Yes", ForwardingMode::Enabled}));

// ---------------------------------------------------------------------------
// IsInForwardingSet: empty in Phase 1, so false for every entry point.
// ---------------------------------------------------------------------------

class CPU_WrapperRoutingForwardingSet_NONE : public ::testing::TestWithParam<const char*>
{
};

TEST_P(CPU_WrapperRoutingForwardingSet_NONE, EmptyInPhase1)
{
    EXPECT_FALSE(IsInForwardingSet(GetParam())) << "entryPoint=" << GetParam();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_WrapperRoutingForwardingSet_NONE,
                         ::testing::Values("miopenConvolutionForward",
                                           "miopenCreate",
                                           "miopenGetVersion",
                                           "miopenDestroy",
                                           ""));

// ---------------------------------------------------------------------------
// ResolveRoute: pure (mode, entryPoint) -> Route decision.
// ---------------------------------------------------------------------------

struct ResolveCase
{
    ForwardingMode mode;
    const char* entryPoint;
    Route expected;
};

class CPU_WrapperRoutingResolve_NONE : public ::testing::TestWithParam<ResolveCase>
{
};

TEST_P(CPU_WrapperRoutingResolve_NONE, Decides)
{
    const ResolveCase& c = GetParam();
    EXPECT_EQ(ResolveRoute(c.mode, c.entryPoint), c.expected) << "entryPoint=" << c.entryPoint;
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_WrapperRoutingResolve_NONE,
    ::testing::Values(
        // Disabled routes everything to the MIOpen implementation.
        ResolveCase{ForwardingMode::Disabled, "miopenConvolutionForward", Route::Miopen},
        ResolveCase{ForwardingMode::Disabled, "miopenCreate", Route::Miopen},
        ResolveCase{ForwardingMode::Disabled, "", Route::Miopen},
        // Enabled with an empty forwarding set is functionally identical to
        // Disabled: every call still routes to the MIOpen implementation. When an
        // entry point is added to the forwarding set, its Enabled case here must
        // be updated to expect Route::Hipdnn.
        ResolveCase{ForwardingMode::Enabled, "miopenConvolutionForward", Route::Miopen},
        ResolveCase{ForwardingMode::Enabled, "miopenCreate", Route::Miopen},
        ResolveCase{ForwardingMode::Enabled, "", Route::Miopen}));

// ---------------------------------------------------------------------------
// Dispatch() / GetForwardingMode(): the real process-global path. With the
// environment variable unset, the resolved mode is Disabled and every call
// routes to the MIOpen implementation.
// ---------------------------------------------------------------------------

class CPU_WrapperRoutingDispatch_NONE : public ::testing::TestWithParam<const char*>
{
};

TEST_P(CPU_WrapperRoutingDispatch_NONE, DefaultRoutesToMiopen)
{
    const char* entryPoint = GetParam();
    EXPECT_EQ(GetForwardingMode(), ForwardingMode::Disabled);
    // Dispatch is exactly ResolveRoute over the process-wide mode...
    EXPECT_EQ(Dispatch(entryPoint), ResolveRoute(GetForwardingMode(), entryPoint))
        << "entryPoint=" << entryPoint;
    // ...which, on the default disabled path, is the MIOpen implementation.
    EXPECT_EQ(Dispatch(entryPoint), Route::Miopen) << "entryPoint=" << entryPoint;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_WrapperRoutingDispatch_NONE,
                         ::testing::Values("miopenConvolutionForward",
                                           "miopenCreate",
                                           "miopenGetErrorString"));

} // namespace

#endif // MIOPEN_ENABLE_HIPDNN_WRAPPER
