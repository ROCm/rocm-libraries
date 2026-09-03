// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for the wrapper dispatch seam (src/private/routing.{hpp,cpp}). The
// seam only exists when MIOPEN_ENABLE_HIPDNN_WRAPPER is ON, so this file
// compiles to zero tests otherwise. The wrapper does not export these symbols,
// so routing.cpp is compiled into the test-common library instead (see
// gtest/CMakeLists.txt).
//
// ParseForwardingMode, IsInForwardingSet and ResolveRoute take their inputs
// explicitly, so both routes and all reporting are testable without touching the
// environment. GetForwardingMode()/Dispatch() are covered on the default path
// only: their cache is a function-local static that cannot be reset once
// initialized.

#include <gtest/gtest.h>

#ifdef MIOPEN_ENABLE_HIPDNN_WRAPPER

#include "../../src/private/routing.hpp"

#include <cstdlib>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace miopen {
namespace wrapper {
// Readable failure diagnostics for the routing enums; GoogleTest finds these via
// argument-dependent lookup.
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
using miopen::wrapper::DispatchFromStub;
using miopen::wrapper::EntryPointNameMatches;
using miopen::wrapper::ForwardingMode;
using miopen::wrapper::ForwardingSet;
using miopen::wrapper::GetForwardingMode;
using miopen::wrapper::IsInForwardingSet;
using miopen::wrapper::ParseForwardingMode;
using miopen::wrapper::ResolveRoute;
using miopen::wrapper::Route;

// Duplicated rather than included: it is internal to routing.cpp.
constexpr const char* kForwardingEnvVar = "MIOPEN_HIPDNN_FORWARDING";

// GetForwardingMode() caches the parsed mode on first use, so whatever the
// launching shell had MIOPEN_HIPDNN_FORWARDING set to would otherwise leak into
// CPU_WrapperRoutingDispatch_NONE with no way to undo it. Clearing it from a
// global test environment happens before any test body runs.
class ForwardingEnvSetup : public ::testing::Environment
{
public:
    void SetUp() override
    {
#ifdef _WIN32
        // Windows has no unsetenv; the empty string is how a variable is
        // removed, and the parser treats empty the same as unset.
        _putenv_s(kForwardingEnvVar, "");
#else
        unsetenv(kForwardingEnvVar);
#endif
    }
};

// GoogleTest takes ownership; the returned handle is unused.
[[maybe_unused]] const ::testing::Environment* const kForwardingEnvSetup =
    ::testing::AddGlobalTestEnvironment(new ForwardingEnvSetup);

// Injected so the Route::Hipdnn half of the decision is reachable while the
// build's own forwarding set is empty.
constexpr std::string_view kTestEntries[] = {"miopenConvolutionForward", "miopenCreate"};
const ForwardingSet kTestSet{kTestEntries};

// ---------------------------------------------------------------------------
// ParseForwardingMode: raw env value -> mode, plus what the user is told.
// ---------------------------------------------------------------------------

struct ParseResult
{
    ForwardingMode mode;
    std::string output;
};

ParseResult Parse(const char* value)
{
    std::ostringstream os;
    const ForwardingMode mode = ParseForwardingMode(value, os);
    return {mode, os.str()};
}

struct ParseCase
{
    const char* value; // raw env value; nullptr models the variable being unset
    ForwardingMode expectedMode;
    bool expectsOutput; // false means the parser must say nothing at all
};

class CPU_WrapperRoutingParse_NONE : public ::testing::TestWithParam<ParseCase>
{
};

// cppcheck-suppress syntaxError
TEST_P(CPU_WrapperRoutingParse_NONE, Maps)
{
    const ParseCase& c        = GetParam();
    const ParseResult result  = Parse(c.value);
    const char* const printed = c.value == nullptr ? "<null>" : c.value;

    EXPECT_EQ(result.mode, c.expectedMode) << "value=" << printed;
    EXPECT_EQ(!result.output.empty(), c.expectsOutput)
        << "value=" << printed << " output=" << result.output;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_WrapperRoutingParse_NONE,
                         ::testing::Values(
                             // Unset is the default path, and silent: a wrapper build must look
                             // the same on stderr as a non-wrapper build.
                             ParseCase{nullptr, ForwardingMode::Disabled, false},
                             ParseCase{"", ForwardingMode::Disabled, false},
                             // Explicit disable lands on the default behavior, so it is silent
                             // too.
                             ParseCase{"disable", ForwardingMode::Disabled, false},
                             ParseCase{"disabled", ForwardingMode::Disabled, false},
                             ParseCase{"0", ForwardingMode::Disabled, false},
                             ParseCase{"no", ForwardingMode::Disabled, false},
                             ParseCase{"off", ForwardingMode::Disabled, false},
                             ParseCase{"false", ForwardingMode::Disabled, false},
                             ParseCase{"FALSE", ForwardingMode::Disabled, false},
                             // "enable" without the trailing 'd' works everywhere else in MIOpen
                             // and must work here too. Enabling is the one case worth announcing.
                             ParseCase{"enable", ForwardingMode::Enabled, true},
                             ParseCase{"enabled", ForwardingMode::Enabled, true},
                             ParseCase{"ENABLED", ForwardingMode::Enabled, true},
                             ParseCase{"Enabled", ForwardingMode::Enabled, true},
                             ParseCase{"1", ForwardingMode::Enabled, true},
                             ParseCase{"on", ForwardingMode::Enabled, true},
                             ParseCase{"ON", ForwardingMode::Enabled, true},
                             ParseCase{"true", ForwardingMode::Enabled, true},
                             ParseCase{"TRUE", ForwardingMode::Enabled, true},
                             ParseCase{"yes", ForwardingMode::Enabled, true},
                             ParseCase{"Yes", ForwardingMode::Enabled, true},
                             // Anything else stays disabled, but is reported: a typo'd enable
                             // deserves better than silence.
                             ParseCase{"garbage", ForwardingMode::Disabled, true},
                             ParseCase{"enabeld", ForwardingMode::Disabled, true},
                             // Leading whitespace is not trimmed.
                             ParseCase{" enabled", ForwardingMode::Disabled, true},
                             ParseCase{"2", ForwardingMode::Disabled, true}));

// ---------------------------------------------------------------------------
// The table above pins which values produce output; these pin what it says.
// ---------------------------------------------------------------------------

TEST(CPU_WrapperRoutingReport_NONE, WarnsOnUnrecognizedValue)
{
    const std::string report = Parse("enabeld").output;
    EXPECT_NE(report.find("Warning"), std::string::npos) << report;
    // Echoing the offending value back is the whole point of warning.
    EXPECT_NE(report.find("enabeld"), std::string::npos) << report;
    EXPECT_NE(report.find(kForwardingEnvVar), std::string::npos) << report;
}

TEST(CPU_WrapperRoutingReport_NONE, AnnouncesForwardingWhenEnabled)
{
    const std::string report = Parse("enabled").output;
    // Not a warning: the user asked for this and got it.
    EXPECT_EQ(report.find("Warning"), std::string::npos) << report;
    EXPECT_NE(report.find(kForwardingEnvVar), std::string::npos) << report;
    EXPECT_NE(report.find("hipDNN"), std::string::npos) << report;
}

// ---------------------------------------------------------------------------
// IsInForwardingSet: membership against an injected set, and against this
// build's own set.
// ---------------------------------------------------------------------------

TEST(CPU_WrapperRoutingForwardingSet_NONE, MatchesInjectedMembers)
{
    EXPECT_TRUE(IsInForwardingSet("miopenConvolutionForward", kTestSet));
    EXPECT_TRUE(IsInForwardingSet("miopenCreate", kTestSet));
}

TEST(CPU_WrapperRoutingForwardingSet_NONE, RejectsNonMembers)
{
    EXPECT_FALSE(IsInForwardingSet("miopenGetVersion", kTestSet));
    EXPECT_FALSE(IsInForwardingSet("miopenDestroy", kTestSet));
    // Prefixes and suffixes of a member are not members.
    EXPECT_FALSE(IsInForwardingSet("miopenConvolutionForwardBias", kTestSet));
    EXPECT_FALSE(IsInForwardingSet("miopenConvolutionForwar", kTestSet));
    EXPECT_FALSE(IsInForwardingSet("", kTestSet));
    EXPECT_FALSE(IsInForwardingSet(nullptr, kTestSet));
}

TEST(CPU_WrapperRoutingForwardingSet_NONE, EmptySetMatchesNothing)
{
    const ForwardingSet empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_FALSE(IsInForwardingSet("miopenConvolutionForward", empty));
}

// Adding an entry point to the build's forwarding set has to break this test,
// so the addition is deliberate rather than silent.
TEST(CPU_WrapperRoutingForwardingSet_NONE, DefaultSetIsEmpty)
{
    EXPECT_TRUE(miopen::wrapper::DefaultForwardingSet().empty());
    EXPECT_FALSE(IsInForwardingSet("miopenConvolutionForward"));
    EXPECT_FALSE(IsInForwardingSet("miopenCreate"));
}

// ---------------------------------------------------------------------------
// ResolveRoute: (mode, entryPoint, set) -> Route. The injected set is what makes
// the Enabled rows real assertions rather than a restatement of "set is empty".
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
    EXPECT_EQ(ResolveRoute(c.mode, c.entryPoint, kTestSet), c.expected)
        << "entryPoint=" << c.entryPoint;
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_WrapperRoutingResolve_NONE,
    ::testing::Values(
        // The kill switch: Disabled routes everything to MIOpen, set membership
        // notwithstanding.
        ResolveCase{ForwardingMode::Disabled, "miopenConvolutionForward", Route::Miopen},
        ResolveCase{ForwardingMode::Disabled, "miopenCreate", Route::Miopen},
        ResolveCase{ForwardingMode::Disabled, "miopenGetVersion", Route::Miopen},
        ResolveCase{ForwardingMode::Disabled, "", Route::Miopen},
        // Enabled forwards exactly the members of the set...
        ResolveCase{ForwardingMode::Enabled, "miopenConvolutionForward", Route::Hipdnn},
        ResolveCase{ForwardingMode::Enabled, "miopenCreate", Route::Hipdnn},
        // ...and nothing else: a non-member still falls through to MIOpen.
        ResolveCase{ForwardingMode::Enabled, "miopenGetVersion", Route::Miopen},
        ResolveCase{ForwardingMode::Enabled, "miopenDestroy", Route::Miopen},
        ResolveCase{ForwardingMode::Enabled, "", Route::Miopen}));

// Kept separate from the injected-set cases so the difference between "the logic
// works" and "this build forwards nothing" stays explicit.
TEST(CPU_WrapperRoutingResolve_NONE, DefaultSetRoutesEverythingToMiopen)
{
    EXPECT_EQ(ResolveRoute(ForwardingMode::Enabled, "miopenConvolutionForward"), Route::Miopen);
    EXPECT_EQ(ResolveRoute(ForwardingMode::Disabled, "miopenConvolutionForward"), Route::Miopen);
}

// ---------------------------------------------------------------------------
// The guard that keeps a stub from dispatching under a neighbour's name.
// ---------------------------------------------------------------------------

TEST(CPU_WrapperRoutingStubName_NONE, MatchesOnlyTheSameName)
{
    EXPECT_TRUE(EntryPointNameMatches("miopenCreate", "miopenCreate"));
    EXPECT_FALSE(EntryPointNameMatches("miopenCreate", "miopenDestroy"));
    EXPECT_FALSE(EntryPointNameMatches("miopenCreate", "miopenCreateWithStream"));
    EXPECT_FALSE(EntryPointNameMatches(nullptr, "miopenCreate"));
    EXPECT_FALSE(EntryPointNameMatches("miopenCreate", nullptr));
    EXPECT_FALSE(EntryPointNameMatches(nullptr, nullptr));
}

TEST(CPU_WrapperRoutingStubName_NONE, DispatchFromStubAgreesWithDispatch)
{
    // __func__ here is not an entry-point name, so pass the name twice to
    // satisfy DispatchFromStub's assertion.
    EXPECT_EQ(DispatchFromStub("miopenCreate", "miopenCreate"), Dispatch("miopenCreate"));
}

// ---------------------------------------------------------------------------
// The process-global path. ForwardingEnvSetup clears the environment variable
// before any test runs, so the mode is Disabled and everything routes to MIOpen.
// ---------------------------------------------------------------------------

class CPU_WrapperRoutingDispatch_NONE : public ::testing::TestWithParam<const char*>
{
};

TEST_P(CPU_WrapperRoutingDispatch_NONE, DefaultRoutesToMiopen)
{
    const char* entryPoint = GetParam();
    EXPECT_EQ(GetForwardingMode(), ForwardingMode::Disabled);
    // Dispatch is ResolveRoute over the process-wide mode...
    EXPECT_EQ(Dispatch(entryPoint), ResolveRoute(GetForwardingMode(), entryPoint))
        << "entryPoint=" << entryPoint;
    // ...which on the default path is MIOpen.
    EXPECT_EQ(Dispatch(entryPoint), Route::Miopen) << "entryPoint=" << entryPoint;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_WrapperRoutingDispatch_NONE,
                         ::testing::Values("miopenConvolutionForward",
                                           "miopenCreate",
                                           "miopenGetErrorString"));

} // namespace

#endif // MIOPEN_ENABLE_HIPDNN_WRAPPER
