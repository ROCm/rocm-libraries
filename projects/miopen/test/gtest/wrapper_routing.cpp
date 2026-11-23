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
// environment. GetForwardingMode()/Dispatch() cache the mode on first use, so
// their tests accept whatever the variable is set to rather than clearing it:
// the parity harness sets it for shim tests in this same binary.

#include <gtest/gtest.h>

#ifdef MIOPEN_ENABLE_HIPDNN_WRAPPER

#include "../../src/private/hipdnn_graph.hpp"
#include "../../src/private/routing.hpp"

#include <miopen/logger.hpp>

#include <climits>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

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
    *os << (route == Route::Hipdnn     ? "Route::Hipdnn"
            : route == Route::Rejected ? "Route::Rejected"
                                       : "Route::Miopen");
}
static void PrintTo(RouteReason reason, std::ostream* os)
{
    switch(reason)
    {
    case RouteReason::ForwardingOff: *os << "RouteReason::ForwardingOff"; return;
    case RouteReason::NotInForwardingSet: *os << "RouteReason::NotInForwardingSet"; return;
    case RouteReason::DisabledByEnv: *os << "RouteReason::DisabledByEnv"; return;
    case RouteReason::HipdnnUnavailable: *os << "RouteReason::HipdnnUnavailable"; return;
    case RouteReason::InForwardingSet: *os << "RouteReason::InForwardingSet"; return;
    case RouteReason::ReentrantFromHipdnn: *os << "RouteReason::ReentrantFromHipdnn"; return;
    }
    *os << "RouteReason(" << static_cast<int>(reason) << ')';
}
} // namespace wrapper
} // namespace miopen

namespace {

using miopen::wrapper::DefaultForwardingSet;
using miopen::wrapper::DisabledSet;
using miopen::wrapper::Dispatch;
using miopen::wrapper::DispatchFromStub;
using miopen::wrapper::EntryPointNameMatches;
using miopen::wrapper::ForwardingMode;
using miopen::wrapper::ForwardingSet;
using miopen::wrapper::GetForwardingMode;
using miopen::wrapper::IsInForwardingSet;
using miopen::wrapper::ParseDisabledSet;
using miopen::wrapper::ParseForwardingMode;
using miopen::wrapper::ResolveRoute;
using miopen::wrapper::ResolveRouteWithReason;
using miopen::wrapper::Route;
using miopen::wrapper::RouteDecision;
using miopen::wrapper::RouteReason;

// Duplicated rather than included: they are internal to routing.cpp.
constexpr const char* kForwardingEnvVar = "MIOPEN_HIPDNN_FORWARDING";
constexpr const char* kDisableEnvVar    = "MIOPEN_DISABLE_HIPDNN_FOR";

// Injected so the routing logic is exercised against a set this test controls,
// independently of whatever the build happens to forward. "miopenCreate" is here
// precisely because the real set does not contain it.
constexpr std::string_view kTestEntries[] = {"miopenConvolutionForward", "miopenCreate"};
const ForwardingSet kTestSet{kTestEntries};

// gtest_discover_tests names each parameterized case after its printed value.
// Without a PrintTo, gtest prints a struct's raw bytes, and the string-literal
// addresses in them change on every run, so the ctest names would too. The
// PrintTo overloads below print only a case's input, which is unique per case.
void PrintEnvValue(const char* value, std::ostream* os)
{
    if(value == nullptr)
        *os << "<unset>";
    else
        *os << '"' << value << '"';
}

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

void PrintTo(const ParseCase& c, std::ostream* os) { PrintEnvValue(c.value, os); }

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
// so the addition is deliberate rather than silent. It cannot check that the new entry
// point has a HipdnnShim test; the checklist above kForwardingEntries in
// src/private/routing.cpp requires one, so review must check.
TEST(CPU_WrapperRoutingForwardingSet_NONE, DefaultSetIsTheConvolutionFamily)
{
    EXPECT_EQ(DefaultForwardingSet().size(), 4u);
    EXPECT_TRUE(IsInForwardingSet("miopenConvolutionForward"));
    EXPECT_TRUE(IsInForwardingSet("miopenConvolutionBackwardData"));
    EXPECT_TRUE(IsInForwardingSet("miopenConvolutionBackwardWeights"));
    EXPECT_TRUE(IsInForwardingSet("miopenConvolutionBiasActivationForward"));

    // The bias-only entry points are standalone bias kernels rather than
    // convolutions, so they stay on the MIOpen path.
    EXPECT_FALSE(IsInForwardingSet("miopenConvolutionForwardBias"));
    EXPECT_FALSE(IsInForwardingSet("miopenConvolutionBackwardBias"));

    EXPECT_FALSE(IsInForwardingSet("miopenSetTensorDescriptor"));
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

void PrintTo(const ResolveCase& c, std::ostream* os)
{
    *os << (c.mode == ForwardingMode::Enabled ? "Enabled " : "Disabled ");
    PrintEnvValue(c.entryPoint, os);
}

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
// works" and "this build forwards these four names" stays explicit.
TEST(CPU_WrapperRoutingResolve_NONE, DefaultSetRoutesTheConvolutionFamily)
{
    EXPECT_EQ(ResolveRoute(ForwardingMode::Enabled, "miopenConvolutionForward"), Route::Hipdnn);
    EXPECT_EQ(ResolveRoute(ForwardingMode::Enabled, "miopenSetTensorDescriptor"), Route::Miopen);
    EXPECT_EQ(ResolveRoute(ForwardingMode::Disabled, "miopenConvolutionForward"), Route::Miopen);
}

// ---------------------------------------------------------------------------
// ParseDisabledSet: raw MIOPEN_DISABLE_HIPDNN_FOR value -> deny list.
// ---------------------------------------------------------------------------

struct DisabledParse
{
    DisabledSet set;
    std::string output;
};

DisabledParse ParseDisabled(const char* value)
{
    std::ostringstream os;
    DisabledSet set = ParseDisabledSet(value, kTestSet, os);
    return {std::move(set), os.str()};
}

struct DisabledCase
{
    const char* value; // raw env value; nullptr models the variable being unset
    bool disablesEverything;
    bool disablesConvolutionForward;
    bool disablesCreate;
    bool expectsWarning;
};

void PrintTo(const DisabledCase& c, std::ostream* os) { PrintEnvValue(c.value, os); }

class CPU_WrapperRoutingDisabledSet_NONE : public ::testing::TestWithParam<DisabledCase>
{
};

TEST_P(CPU_WrapperRoutingDisabledSet_NONE, Parses)
{
    const DisabledCase& c      = GetParam();
    const DisabledParse result = ParseDisabled(c.value);
    const char* const printed  = c.value == nullptr ? "<null>" : c.value;

    EXPECT_EQ(result.set.DisablesEverything(), c.disablesEverything) << "value=" << printed;
    EXPECT_EQ(result.set.Contains("miopenConvolutionForward"), c.disablesConvolutionForward)
        << "value=" << printed;
    EXPECT_EQ(result.set.Contains("miopenCreate"), c.disablesCreate) << "value=" << printed;
    EXPECT_EQ(!result.output.empty(), c.expectsWarning)
        << "value=" << printed << " output=" << result.output;
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_WrapperRoutingDisabledSet_NONE,
    ::testing::Values(
        // Unset and empty disable nothing, silently.
        DisabledCase{nullptr, false, false, false, false},
        DisabledCase{"", false, false, false, false},
        // A list of nothing but separators is still a list of nothing.
        DisabledCase{",,", false, false, false, false},
        DisabledCase{"   ", false, false, false, false},
        // One name, several names, and whitespace around each.
        DisabledCase{"miopenConvolutionForward", false, true, false, false},
        DisabledCase{"miopenConvolutionForward,miopenCreate", false, true, true, false},
        DisabledCase{"  miopenConvolutionForward , miopenCreate  ", false, true, true, false},
        // "*" disables everything, on its own or mixed in anywhere.
        DisabledCase{"*", true, false, false, false},
        DisabledCase{"miopenCreate,*", true, false, true, false},
        DisabledCase{"*,miopenCreate", true, false, true, false},
        // A name this build does not forward is a typo, not a request: warned
        // about, and with no effect.
        DisabledCase{"miopenGetVersion", false, false, false, true},
        // Naming one real entry point and one typo keeps the real one.
        DisabledCase{"miopenConvolutionForward,miopenGetVersoin", false, true, false, true},
        // A prefix or a suffix of a real name is not that name.
        DisabledCase{"miopenConvolutionForwardBias", false, false, false, true},
        DisabledCase{"miopenConvolutionForwar", false, false, false, true}));

TEST(CPU_WrapperRoutingDisabledSet_NONE, WarningNamesTheVariableAndTheValue)
{
    const std::string report = ParseDisabled("miopenGetVersion").output;
    EXPECT_NE(report.find("Warning"), std::string::npos) << report;
    EXPECT_NE(report.find(kDisableEnvVar), std::string::npos) << report;
    EXPECT_NE(report.find("miopenGetVersion"), std::string::npos) << report;
}

TEST(CPU_WrapperRoutingDisabledSet_NONE, MatchesNothingWhenAskedAboutNull)
{
    EXPECT_FALSE(ParseDisabled("*").set.Contains(nullptr));
    EXPECT_FALSE(ParseDisabled("miopenCreate").set.Contains(nullptr));
}

// ---------------------------------------------------------------------------
// ResolveRouteWithReason: every reason the seam can report.
// ---------------------------------------------------------------------------

RouteDecision Resolve(ForwardingMode mode,
                      const char* entryPoint,
                      const DisabledSet& disabled,
                      bool hipdnnAvailable)
{
    return ResolveRouteWithReason(mode, entryPoint, kTestSet, disabled, hipdnnAvailable);
}

TEST(CPU_WrapperRoutingReason_NONE, ForwardingOff)
{
    const RouteDecision d =
        Resolve(ForwardingMode::Disabled, "miopenConvolutionForward", DisabledSet(), true);
    EXPECT_EQ(d.route, Route::Miopen);
    EXPECT_EQ(d.reason, RouteReason::ForwardingOff);
}

TEST(CPU_WrapperRoutingReason_NONE, NotInForwardingSet)
{
    const RouteDecision d =
        Resolve(ForwardingMode::Enabled, "miopenGetVersion", DisabledSet(), true);
    EXPECT_EQ(d.route, Route::Miopen);
    EXPECT_EQ(d.reason, RouteReason::NotInForwardingSet);
}

TEST(CPU_WrapperRoutingReason_NONE, DisabledByEnvName)
{
    const DisabledSet disabled = ParseDisabled("miopenConvolutionForward").set;
    const RouteDecision d =
        Resolve(ForwardingMode::Enabled, "miopenConvolutionForward", disabled, true);
    EXPECT_EQ(d.route, Route::Miopen);
    EXPECT_EQ(d.reason, RouteReason::DisabledByEnv);

    // The deny list is per name: the other member of the set still forwards.
    EXPECT_EQ(Resolve(ForwardingMode::Enabled, "miopenCreate", disabled, true).reason,
              RouteReason::InForwardingSet);
}

TEST(CPU_WrapperRoutingReason_NONE, DisabledByEnvStar)
{
    const DisabledSet disabled = ParseDisabled("*").set;
    const RouteDecision d =
        Resolve(ForwardingMode::Enabled, "miopenConvolutionForward", disabled, true);
    EXPECT_EQ(d.route, Route::Miopen);
    EXPECT_EQ(d.reason, RouteReason::DisabledByEnv);
}

// The one case where the route stays on hipDNN despite something being wrong.
// Routing back to MIOpen here would turn a broken hipDNN installation into
// results that look correct, which is exactly what this design exists to avoid.
TEST(CPU_WrapperRoutingReason_NONE, HipdnnUnavailableStillRoutesAwayFromMiopen)
{
    const RouteDecision d = Resolve(
        ForwardingMode::Enabled, "miopenConvolutionForward", DisabledSet(), /*available=*/false);
    EXPECT_EQ(d.route, Route::Hipdnn);
    EXPECT_EQ(d.reason, RouteReason::HipdnnUnavailable);
}

// An unavailable hipDNN does not override the earlier reasons: a call that was
// never going to be forwarded is not reported as a hipDNN problem.
TEST(CPU_WrapperRoutingReason_NONE, UnavailableIsCheckedLast)
{
    EXPECT_EQ(
        Resolve(ForwardingMode::Disabled, "miopenConvolutionForward", DisabledSet(), false).reason,
        RouteReason::ForwardingOff);
    EXPECT_EQ(Resolve(ForwardingMode::Enabled, "miopenGetVersion", DisabledSet(), false).reason,
              RouteReason::NotInForwardingSet);
    EXPECT_EQ(
        Resolve(ForwardingMode::Enabled, "miopenConvolutionForward", ParseDisabled("*").set, false)
            .reason,
        RouteReason::DisabledByEnv);
}

TEST(CPU_WrapperRoutingReason_NONE, InForwardingSet)
{
    const RouteDecision d =
        Resolve(ForwardingMode::Enabled, "miopenConvolutionForward", DisabledSet(), true);
    EXPECT_EQ(d.route, Route::Hipdnn);
    EXPECT_EQ(d.reason, RouteReason::InForwardingSet);
}

// ---------------------------------------------------------------------------
// The trace line. Its shape is what a user reads to find out why a call went
// where it did, so it is asserted rather than left to drift.
// ---------------------------------------------------------------------------

TEST(CPU_WrapperRoutingReport_NONE, TraceLineNamesEntryPointModeRouteAndReason)
{
    std::ostringstream os;
    miopen::wrapper::ReportRouteDecision("miopenConvolutionForward",
                                         ForwardingMode::Enabled,
                                         {Route::Hipdnn, RouteReason::InForwardingSet},
                                         os);
    const std::string line = os.str();
    EXPECT_NE(line.find("miopenConvolutionForward"), std::string::npos) << line;
    EXPECT_NE(line.find("mode=enabled"), std::string::npos) << line;
    EXPECT_NE(line.find("route=hipdnn"), std::string::npos) << line;
    EXPECT_NE(line.find("reason=in-forwarding-set"), std::string::npos) << line;
}

TEST(CPU_WrapperRoutingReport_NONE, TraceLineHandlesANullEntryPoint)
{
    std::ostringstream os;
    miopen::wrapper::ReportRouteDecision(
        nullptr, ForwardingMode::Disabled, {Route::Miopen, RouteReason::ForwardingOff}, os);
    const std::string line = os.str();
    EXPECT_NE(line.find("<null>"), std::string::npos) << line;
    EXPECT_NE(line.find("reason=forwarding-off"), std::string::npos) << line;
}

// stderr is unbuffered, so each write is a separate system call, and two
// threads' lines can mix between them.
class WriteCountingBuf : public std::streambuf
{
public:
    int writes = 0;
    std::string text;

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override
    {
        ++writes;
        text.append(s, static_cast<std::size_t>(n));
        return n;
    }
    int_type overflow(int_type ch) override
    {
        ++writes;
        if(!traits_type::eq_int_type(ch, traits_type::eof()))
            text.push_back(traits_type::to_char_type(ch));
        return traits_type::not_eof(ch);
    }
};

TEST(CPU_WrapperRoutingReport_NONE, TraceLineIsOneWrite)
{
    WriteCountingBuf buf;
    std::ostream os(&buf);
    miopen::wrapper::ReportRouteDecision("miopenConvolutionForward",
                                         ForwardingMode::Enabled,
                                         {Route::Hipdnn, RouteReason::HipdnnUnavailable},
                                         os);
    EXPECT_EQ(buf.writes, 1) << buf.text;
    ASSERT_FALSE(buf.text.empty());
    EXPECT_EQ(buf.text.find('\n'), buf.text.size() - 1) << buf.text;
}

TEST(CPU_WrapperRoutingReport_NONE, ForwardedCallLineIsOneWrite)
{
    WriteCountingBuf buf;
    std::ostream os(&buf);
    miopen::wrapper::ReportForwardedCall("miopenConvolutionForward", os);
    EXPECT_EQ(buf.writes, 1) << buf.text;
    EXPECT_EQ(buf.text, "[MIOpen] hipDNN forwarded: miopenConvolutionForward\n");
}

TEST(CPU_WrapperRoutingReport_NONE, ForwardedCallLineHandlesANullEntryPoint)
{
    std::ostringstream os;
    miopen::wrapper::ReportForwardedCall(nullptr, os);
    EXPECT_EQ(os.str(), "[MIOpen] hipDNN forwarded: <null>\n");
}

// ---------------------------------------------------------------------------
// MIOPEN_LOG_LEVEL. Parsed the way MIOpen parses it, so the wrapper's trace and
// error lines switch on at the same values as MIOpen's own.
// ---------------------------------------------------------------------------

TEST(CPU_WrapperRoutingLogLevel_NONE, ParsesLikeMiopen)
{
    using miopen::wrapper::ParseLogLevel;
    EXPECT_EQ(ParseLogLevel(nullptr), 0ULL);
    EXPECT_EQ(ParseLogLevel(""), 0ULL);
    EXPECT_EQ(ParseLogLevel("abc"), 0ULL);
    EXPECT_EQ(ParseLogLevel("6"), 6ULL);
    EXPECT_EQ(ParseLogLevel("7"), 7ULL);
    // Base 0: a leading 0 means octal and 0x means hex, so "08" stops at the 8.
    EXPECT_EQ(ParseLogLevel("06"), 6ULL);
    EXPECT_EQ(ParseLogLevel("08"), 0ULL);
    EXPECT_EQ(ParseLogLevel("0x6"), 6ULL);
    EXPECT_EQ(ParseLogLevel("6abc"), 6ULL);
    // strtoull wraps a negative value; the two checks below then narrow it to int.
    EXPECT_EQ(ParseLogLevel("-1"), ULLONG_MAX);
}

// Thresholds use MIOpen's own enum so renumbering a level there breaks these
// tests instead of leaving the wrapper behind.
constexpr unsigned long long Level(miopen::LoggingLevel level)
{
    return static_cast<unsigned long long>(level);
}

TEST(CPU_WrapperRoutingLogLevel_NONE, TracesFromInfo2Up)
{
    using miopen::LoggingLevel;
    using miopen::wrapper::RouteTraceEnabledFor;
    for(unsigned long long level = 0; level < Level(LoggingLevel::Info2); ++level)
        EXPECT_FALSE(RouteTraceEnabledFor(level)) << level;
    EXPECT_TRUE(RouteTraceEnabledFor(Level(LoggingLevel::Info2)));
    EXPECT_TRUE(RouteTraceEnabledFor(Level(LoggingLevel::Trace)));
}

TEST(CPU_WrapperRoutingLogLevel_NONE, RouteTraceNarrowsTheLevelToIntLikeMiopen)
{
    using miopen::wrapper::RouteTraceEnabledFor;
    // MIOpen reads -1 as negative, which is below every level.
    EXPECT_FALSE(RouteTraceEnabledFor(ULLONG_MAX));
    // 0x100000006 and 0x100000002 read as 6 (Info2) and 2 (Fatal).
    EXPECT_TRUE(RouteTraceEnabledFor(0x100000006ULL));
    EXPECT_FALSE(RouteTraceEnabledFor(0x100000002ULL));
}

TEST(CPU_WrapperRoutingLogLevel_NONE, ErrorsShownOnlyAtDefaultAndFromErrorUp)
{
    using miopen::LoggingLevel;
    using miopen::wrapper::ErrorLoggingEnabledFor;
    EXPECT_FALSE(ErrorLoggingEnabledFor(Level(LoggingLevel::Quiet)));
    EXPECT_FALSE(ErrorLoggingEnabledFor(Level(LoggingLevel::Fatal)));
    EXPECT_TRUE(ErrorLoggingEnabledFor(Level(LoggingLevel::Default)));
    EXPECT_TRUE(ErrorLoggingEnabledFor(Level(LoggingLevel::Error)));
    EXPECT_TRUE(ErrorLoggingEnabledFor(Level(LoggingLevel::Info2)));
    EXPECT_TRUE(ErrorLoggingEnabledFor(Level(LoggingLevel::Trace)));
}

TEST(CPU_WrapperRoutingLogLevel_NONE, ErrorsNarrowTheLevelToIntLikeMiopen)
{
    using miopen::wrapper::ErrorLoggingEnabledFor;
    // MIOpen reads -1 as negative and prints nothing, errors included.
    EXPECT_FALSE(ErrorLoggingEnabledFor(ULLONG_MAX));
    // 0x100000002 reads as 2 (Fatal); 0x100000003 reads as 3 (Error).
    EXPECT_FALSE(ErrorLoggingEnabledFor(0x100000002ULL));
    EXPECT_TRUE(ErrorLoggingEnabledFor(0x100000003ULL));
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
// The process-global path.
// ---------------------------------------------------------------------------

class CPU_WrapperRoutingDispatch_NONE : public ::testing::TestWithParam<const char*>
{
};

TEST_P(CPU_WrapperRoutingDispatch_NONE, AgreesWithResolveRoute)
{
    const char* entryPoint = GetParam();
    std::ostringstream ignored;
    EXPECT_EQ(GetForwardingMode(), ParseForwardingMode(std::getenv(kForwardingEnvVar), ignored));
    EXPECT_EQ(Dispatch(entryPoint),
              ResolveRoute(GetForwardingMode(),
                           entryPoint,
                           miopen::wrapper::DefaultForwardingSet(),
                           miopen::wrapper::GetDisabledSet()))
        << "entryPoint=" << entryPoint;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_WrapperRoutingDispatch_NONE,
                         ::testing::Values("miopenConvolutionForward",
                                           "miopenCreate",
                                           "miopenGetErrorString"));

class CPU_WrapperRoutingDispatchUnforwarded_NONE : public ::testing::TestWithParam<const char*>
{
};

// In either mode: forwarding only ever redirects entry points in the set.
TEST_P(CPU_WrapperRoutingDispatchUnforwarded_NONE, RoutesToMiopen)
{
    const char* entryPoint = GetParam();
    ASSERT_FALSE(IsInForwardingSet(entryPoint, miopen::wrapper::DefaultForwardingSet()));
    EXPECT_EQ(Dispatch(entryPoint), Route::Miopen) << "entryPoint=" << entryPoint;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_WrapperRoutingDispatchUnforwarded_NONE,
                         ::testing::Values("miopenCreate", "miopenGetErrorString"));

// ---------------------------------------------------------------------------
// The record behind miopenGetErrorString's "[hipDNN-forwarded]" prefix. These
// cover its logic only, not that the stubs MIOpen serves actually clear it.
// ---------------------------------------------------------------------------

namespace hipdnn = miopen::wrapper::hipdnn;

TEST(CPU_WrapperRoutingLastError_NONE, PrefixesTheForwardedFailure)
{
    testing::internal::CaptureStderr();
    hipdnn::RecordFailure(miopenStatusUnsupportedOp, "hipDNN's reason");
    const std::string logged   = testing::internal::GetCapturedStderr();
    const char* const prefixed = hipdnn::PrefixedErrorString(miopenStatusUnsupportedOp, "native");
    hipdnn::ClearForwardedFailure();

    EXPECT_STREQ(prefixed, "[hipDNN-forwarded] native");
    if(miopen::wrapper::ErrorLoggingEnabled())
        EXPECT_NE(logged.find("hipDNN's reason"), std::string::npos) << logged;
}

// MIOpen's strings live for the whole process, so a caller may keep the pointer.
TEST(CPU_WrapperRoutingLastError_NONE, KeptTextOutlivesLaterFailuresAndItsThread)
{
    testing::internal::CaptureStderr();
    hipdnn::RecordFailure(miopenStatusUnsupportedOp, "first reason");
    const char* const first = hipdnn::PrefixedErrorString(miopenStatusUnsupportedOp, "native");
    hipdnn::RecordFailure(miopenStatusBadParm, "second reason");
    const char* const second = hipdnn::PrefixedErrorString(miopenStatusBadParm, "other");
    hipdnn::RecordFailure(miopenStatusUnsupportedOp, "third reason");
    const char* const third = hipdnn::PrefixedErrorString(miopenStatusUnsupportedOp, "native");
    hipdnn::ClearForwardedFailure();

    const char* fromExitedThread = nullptr;
    std::thread([&fromExitedThread] {
        hipdnn::RecordFailure(miopenStatusNotImplemented, "reason");
        fromExitedThread = hipdnn::PrefixedErrorString(miopenStatusNotImplemented, "thread");
    }).join();
    static_cast<void>(testing::internal::GetCapturedStderr());

    EXPECT_STREQ(first, "[hipDNN-forwarded] native");
    EXPECT_STREQ(second, "[hipDNN-forwarded] other");
    EXPECT_EQ(third, first);
    EXPECT_STREQ(fromExitedThread, "[hipDNN-forwarded] thread");
}

TEST(CPU_WrapperRoutingLastError_NONE, IgnoresADifferentStatus)
{
    hipdnn::RecordFailure(miopenStatusUnsupportedOp, "hipDNN's reason");
    const char* const prefixed = hipdnn::PrefixedErrorString(miopenStatusBadParm, "native");
    hipdnn::ClearForwardedFailure();

    EXPECT_EQ(prefixed, nullptr) << prefixed;
}

TEST(CPU_WrapperRoutingLastError_NONE, ClearedFailureIsNotPrefixed)
{
    hipdnn::RecordFailure(miopenStatusUnsupportedOp, "hipDNN's reason");
    hipdnn::ClearForwardedFailure();

    EXPECT_EQ(hipdnn::PrefixedErrorString(miopenStatusUnsupportedOp, "native"), nullptr);
}

TEST(CPU_WrapperRoutingLastError_NONE, RecordsAnException)
{
    miopenStatus_t status = miopenStatusSuccess;
    testing::internal::CaptureStderr();
    try
    {
        throw std::runtime_error("what went wrong");
    }
    catch(...)
    {
        status = hipdnn::RecordCurrentException();
    }
    const std::string logged   = testing::internal::GetCapturedStderr();
    const char* const prefixed = hipdnn::PrefixedErrorString(miopenStatusUnknownError, "native");
    hipdnn::ClearForwardedFailure();

    EXPECT_EQ(status, miopenStatusUnknownError);
    EXPECT_STREQ(prefixed, "[hipDNN-forwarded] native");
    if(miopen::wrapper::ErrorLoggingEnabled())
        EXPECT_NE(logged.find("what went wrong"), std::string::npos) << logged;
}

// An entry point in the forwarding set that has no hipDNN implementation yet
// must fail loudly instead of running MIOpen.
TEST(CPU_WrapperRoutingLastError_NONE, ForwardingNotImplementedFailsAndRecordsTheFailure)
{
    miopenStatus_t recorded = miopenStatusSuccess;
    hipdnn::ClearForwardedFailure();

    testing::internal::CaptureStderr();
    const miopenStatus_t status = hipdnn::ForwardingNotImplemented("miopenFoo");
    const std::string logged    = testing::internal::GetCapturedStderr();
    const bool wasRecorded      = hipdnn::LastForwardedFailure(&recorded);
    hipdnn::ClearForwardedFailure();

    EXPECT_EQ(status, miopenStatusNotImplemented);
    EXPECT_TRUE(wasRecorded);
    EXPECT_EQ(recorded, miopenStatusNotImplemented);
    if(miopen::wrapper::ErrorLoggingEnabled())
        EXPECT_NE(
            logged.find("miopenFoo is in the forwarding set but has no hipDNN implementation"),
            std::string::npos)
            << logged;
}

// The record behind miopenGetLastForwardedError.
TEST(CPU_WrapperRoutingLastError_NONE, LastForwardedFailureFollowsRecordAndClear)
{
    miopenStatus_t status = miopenStatusSuccess;
    hipdnn::ClearForwardedFailure();
    EXPECT_FALSE(hipdnn::LastForwardedFailure(&status));

    // Not InternalError: every forwarded failure the process tests can cause has
    // that status, so only this test would catch a record that ignores its input.
    testing::internal::CaptureStderr();
    hipdnn::RecordFailure(miopenStatusUnsupportedOp, "hipDNN's reason");
    static_cast<void>(testing::internal::GetCapturedStderr());
    EXPECT_TRUE(hipdnn::LastForwardedFailure(&status));
    EXPECT_EQ(status, miopenStatusUnsupportedOp);

    status = miopenStatusSuccess;
    EXPECT_TRUE(hipdnn::LastForwardedFailure(&status));
    EXPECT_EQ(status, miopenStatusUnsupportedOp);

    hipdnn::ClearForwardedFailure();
    EXPECT_FALSE(hipdnn::LastForwardedFailure(&status));
}

TEST(CPU_WrapperRoutingLastError_NONE, LastForwardedFailureIsPerThread)
{
    hipdnn::ClearForwardedFailure();
    bool seenOnItsThread = false;
    testing::internal::CaptureStderr();
    std::thread([&seenOnItsThread] {
        hipdnn::RecordFailure(miopenStatusInternalError, "reason");
        miopenStatus_t status = miopenStatusSuccess;
        seenOnItsThread       = hipdnn::LastForwardedFailure(&status);
    }).join();
    static_cast<void>(testing::internal::GetCapturedStderr());

    miopenStatus_t status = miopenStatusSuccess;
    EXPECT_TRUE(seenOnItsThread);
    EXPECT_FALSE(hipdnn::LastForwardedFailure(&status));
}

// ---------------------------------------------------------------------------
// Re-entry: a hipDNN plugin built against the public libMIOpen.so calls a forwarded
// entry point from inside hipDNN, which must fail instead of looping.
// ---------------------------------------------------------------------------

// Drops what the code under test writes to stderr, even if an assertion ends the test early.
class SilencedStderr
{
public:
    SilencedStderr() { testing::internal::CaptureStderr(); }
    ~SilencedStderr() { static_cast<void>(testing::internal::GetCapturedStderr()); }
    SilencedStderr(const SilencedStderr&)            = delete;
    SilencedStderr& operator=(const SilencedStderr&) = delete;
};

TEST(CPU_WrapperRoutingReentry_NONE, IsOnlyActiveInsideAScope)
{
    EXPECT_FALSE(hipdnn::InHipdnnCall());
    {
        const hipdnn::ScopedHipdnnCall scope;
        EXPECT_TRUE(hipdnn::InHipdnnCall());
    }
    EXPECT_FALSE(hipdnn::InHipdnnCall());
}

// hipDNN reports failures as values, but the plugin under it is not obliged to.
TEST(CPU_WrapperRoutingReentry_NONE, LeavesTheScopeWhenAnExceptionPassesThrough)
{
    try
    {
        const hipdnn::ScopedHipdnnCall scope;
        throw std::runtime_error("from the plugin");
    }
    catch(const std::runtime_error&)
    {
    }
    EXPECT_FALSE(hipdnn::InHipdnnCall());
}

TEST(CPU_WrapperRoutingReentry_NONE, AnInnerScopeLeavesTheOuterOneActive)
{
    const hipdnn::ScopedHipdnnCall outer;
    {
        const hipdnn::ScopedHipdnnCall inner;
        EXPECT_TRUE(hipdnn::InHipdnnCall());
    }
    EXPECT_TRUE(hipdnn::InHipdnnCall());
}

TEST(CPU_WrapperRoutingReentry_NONE, RejectingFailsTheCallAndMarksTheScope)
{
    const hipdnn::ScopedHipdnnCall scope;
    ASSERT_FALSE(scope.ReentryDetected());

    std::ostringstream os;
    const miopenStatus_t status =
        hipdnn::RejectReentrantCallTo("miopenConvolutionForward", false, os);

    EXPECT_EQ(status, miopenStatusInternalError);
    EXPECT_TRUE(scope.ReentryDetected());
}

// A later scope must not inherit the mark, or the next call of a well-behaved plugin would
// fail.
TEST(CPU_WrapperRoutingReentry_NONE, TheMarkDoesNotOutliveTheOutermostScope)
{
    std::ostringstream os;
    {
        const hipdnn::ScopedHipdnnCall scope;
        hipdnn::RejectReentrantCallTo("miopenConvolutionForward", false, os);
        ASSERT_TRUE(scope.ReentryDetected());
    }

    const hipdnn::ScopedHipdnnCall next;
    EXPECT_FALSE(next.ReentryDetected());
}

// The call that opened the scope reports the failure once, so the rejected call itself adds
// no error line or failure record. Its only output is the route trace line, if asked for.
TEST(CPU_WrapperRoutingReentry_NONE, RejectingWithTracingWritesOnlyTheRouteLine)
{
    hipdnn::ClearForwardedFailure();
    const hipdnn::ScopedHipdnnCall scope;

    std::ostringstream os;
    hipdnn::RejectReentrantCallTo("miopenConvolutionForward", true, os);
    const std::string logged = os.str();

    miopenStatus_t status = miopenStatusSuccess;
    EXPECT_FALSE(hipdnn::LastForwardedFailure(&status));
    EXPECT_NE(logged.find("miopenConvolutionForward"), std::string::npos) << logged;
    EXPECT_NE(logged.find("route=rejected reason=reentrant-from-hipdnn"), std::string::npos)
        << logged;
    EXPECT_EQ(logged.find("[hipDNN-forwarded]"), std::string::npos) << logged;
}

TEST(CPU_WrapperRoutingReentry_NONE, RejectingWithoutTracingWritesNothing)
{
    hipdnn::ClearForwardedFailure();
    const hipdnn::ScopedHipdnnCall scope;

    std::ostringstream os;
    hipdnn::RejectReentrantCallTo("miopenConvolutionForward", false, os);

    miopenStatus_t status = miopenStatusSuccess;
    EXPECT_FALSE(hipdnn::LastForwardedFailure(&status));
    EXPECT_TRUE(os.str().empty()) << os.str();
}

TEST(CPU_WrapperRoutingReentry_NONE, RecordedFailureNamesTheEntryPointAndTheRemedy)
{
    hipdnn::ClearForwardedFailure();
    std::ostringstream os;
    {
        const hipdnn::ScopedHipdnnCall scope;
        hipdnn::RejectReentrantCallTo("miopenConvolutionBackwardData", false, os);
    }
    const miopenStatus_t status = hipdnn::RecordReentryFailureTo(true, os);
    const std::string logged    = os.str();

    miopenStatus_t recorded = miopenStatusSuccess;
    EXPECT_EQ(status, miopenStatusInternalError);
    EXPECT_TRUE(hipdnn::LastForwardedFailure(&recorded));
    EXPECT_EQ(recorded, miopenStatusInternalError);
    EXPECT_STREQ(hipdnn::PrefixedErrorString(miopenStatusInternalError, "native"),
                 "[hipDNN-forwarded] native");
    hipdnn::ClearForwardedFailure();

    // check_plugin_reentry.py looks for the first sentence of this text.
    EXPECT_NE(logged.find("a hipDNN plugin called the public MIOpen entry point "
                          "miopenConvolutionBackwardData"),
              std::string::npos)
        << logged;
    EXPECT_NE(logged.find("Rebuild or reinstall the plugin"), std::string::npos) << logged;
    EXPECT_NE(logged.find("MIOpen_private"), std::string::npos) << logged;
}

TEST(CPU_WrapperRoutingReentry_NONE, EveryFailingCallReportsNotJustTheFirst)
{
    std::ostringstream os;
    {
        const hipdnn::ScopedHipdnnCall scope;
        hipdnn::RejectReentrantCallTo("miopenConvolutionForward", false, os);
    }
    hipdnn::RecordReentryFailureTo(true, os);
    hipdnn::RecordReentryFailureTo(true, os);
    const std::string logged = os.str();

    size_t lines = 0;
    for(size_t at = logged.find("[hipDNN-forwarded]"); at != std::string::npos;
        at        = logged.find("[hipDNN-forwarded]", at + 1))
        ++lines;
    EXPECT_EQ(lines, 2u) << logged;
    hipdnn::ClearForwardedFailure();
}

// MIOPEN_LOG_LEVEL can hide the error line; the failure is still recorded for
// miopenGetLastForwardedError.
TEST(CPU_WrapperRoutingReentry_NONE, AHiddenErrorLineStillRecordsTheFailure)
{
    hipdnn::ClearForwardedFailure();
    std::ostringstream os;

    const miopenStatus_t status = hipdnn::RecordReentryFailureTo(false, os);

    miopenStatus_t recorded = miopenStatusSuccess;
    EXPECT_EQ(status, miopenStatusInternalError);
    EXPECT_TRUE(hipdnn::LastForwardedFailure(&recorded));
    EXPECT_EQ(recorded, miopenStatusInternalError);
    EXPECT_TRUE(os.str().empty()) << os.str();
    hipdnn::ClearForwardedFailure();
}

// Pins the documented limit: the mark is per thread, so a plugin that moves the call to a
// thread of its own is not seen.
TEST(CPU_WrapperRoutingReentry_NONE, AnotherThreadIsNotInsideTheCall)
{
    const hipdnn::ScopedHipdnnCall scope;
    bool otherThreadInside = true;
    std::thread([&otherThreadInside] { otherThreadInside = hipdnn::InHipdnnCall(); }).join();

    EXPECT_TRUE(hipdnn::InHipdnnCall());
    EXPECT_FALSE(otherThreadInside);
}

TEST(CPU_WrapperRoutingReentry_NONE, ARejectionOnAnotherThreadDoesNotMarkThisScope)
{
    const hipdnn::ScopedHipdnnCall scope;
    std::thread([] {
        std::ostringstream os;
        const hipdnn::ScopedHipdnnCall other;
        hipdnn::RejectReentrantCallTo("miopenConvolutionForward", false, os);
    }).join();

    EXPECT_FALSE(scope.ReentryDetected());
}

TEST(CPU_WrapperRoutingReport_NONE, TraceLineGivesARejectedCallItsOwnReason)
{
    std::ostringstream os;
    miopen::wrapper::ReportRouteDecision("miopenConvolutionForward",
                                         ForwardingMode::Enabled,
                                         {Route::Rejected, RouteReason::ReentrantFromHipdnn},
                                         os);
    const std::string line = os.str();
    EXPECT_NE(line.find("route=rejected"), std::string::npos) << line;
    EXPECT_NE(line.find("reason=reentrant-from-hipdnn"), std::string::npos) << line;
    EXPECT_EQ(line.find("forwarding-off"), std::string::npos) << line;
    EXPECT_EQ(line.find("not-in-forwarding-set"), std::string::npos) << line;
}

// The two helpers AcquireGraph and RunGraph report through: the scope is opened inside
// CallInsideHipdnn, and FailureOfGuardedCall puts a re-entry ahead of hipDNN's result.

TEST(CPU_WrapperRoutingReentry_NONE, CallInsideHipdnnMarksTheThreadOnlyWhileItRuns)
{
    bool insideWhileRunning = false;
    const auto guarded      = hipdnn::CallInsideHipdnn([&] {
        insideWhileRunning = hipdnn::InHipdnnCall();
        return 7;
    });

    EXPECT_TRUE(insideWhileRunning);
    EXPECT_FALSE(hipdnn::InHipdnnCall());
    EXPECT_EQ(guarded.result, 7);
    EXPECT_FALSE(guarded.reentered);
}

TEST(CPU_WrapperRoutingReentry_NONE, CallInsideHipdnnReportsARejectionMadeWhileItRan)
{
    std::ostringstream os;
    const auto guarded = hipdnn::CallInsideHipdnn([&os] {
        hipdnn::RejectReentrantCallTo("miopenConvolutionForward", false, os);
        return 0;
    });

    EXPECT_TRUE(guarded.reentered);
}

TEST(CPU_WrapperRoutingReentry_NONE, CallInsideHipdnnDoesNotInheritAnEarlierRejection)
{
    std::ostringstream os;
    hipdnn::CallInsideHipdnn([&os] {
        hipdnn::RejectReentrantCallTo("miopenConvolutionForward", false, os);
        return 0;
    });

    const auto next = hipdnn::CallInsideHipdnn([] { return 0; });
    EXPECT_FALSE(next.reentered);
}

TEST(CPU_WrapperRoutingReentry_NONE, AReentryFailsTheCallEvenWhenHipdnnReportedSuccess)
{
    const SilencedStderr silenced;
    hipdnn::ClearForwardedFailure();
    int hipdnnFailureCalls = 0;
    const hipdnn::GuardedResult<int> guarded{0, true};

    const auto failure = hipdnn::FailureOfGuardedCall(
        guarded, [&hipdnnFailureCalls](int) -> std::optional<miopenStatus_t> {
            ++hipdnnFailureCalls;
            return std::nullopt;
        });

    miopenStatus_t recorded = miopenStatusSuccess;
    ASSERT_TRUE(failure.has_value());
    EXPECT_EQ(*failure, miopenStatusInternalError);
    EXPECT_TRUE(hipdnn::LastForwardedFailure(&recorded));
    EXPECT_EQ(recorded, miopenStatusInternalError);
    EXPECT_EQ(hipdnnFailureCalls, 0);
    hipdnn::ClearForwardedFailure();
}

// hipDNN's own failure would hide the cause: the application would see a generic status
// and no message about the plugin.
TEST(CPU_WrapperRoutingReentry_NONE, AReentryIsReportedInsteadOfHipdnnsOwnFailure)
{
    const SilencedStderr silenced;
    const hipdnn::GuardedResult<int> guarded{1, true};

    const auto failure = hipdnn::FailureOfGuardedCall(
        guarded, [](int) -> std::optional<miopenStatus_t> { return miopenStatusBadParm; });
    hipdnn::ClearForwardedFailure();

    ASSERT_TRUE(failure.has_value());
    EXPECT_EQ(*failure, miopenStatusInternalError);
}

TEST(CPU_WrapperRoutingReentry_NONE, WithoutAReentryHipdnnsResultDecides)
{
    const hipdnn::GuardedResult<int> failed{1, false};
    const hipdnn::GuardedResult<int> worked{0, false};
    const auto map = [](int value) -> std::optional<miopenStatus_t> {
        if(value == 0)
            return std::nullopt;
        return miopenStatusBadParm;
    };

    const auto failure = hipdnn::FailureOfGuardedCall(failed, map);
    ASSERT_TRUE(failure.has_value());
    EXPECT_EQ(*failure, miopenStatusBadParm);
    EXPECT_FALSE(hipdnn::FailureOfGuardedCall(worked, map).has_value());
}

// ---------------------------------------------------------------------------
// The two steps AcquireGraph and RunGraph are made of, with a stand-in for hipDNN. A plugin
// that re-enters the wrapper is a call to RejectReentrantCall from inside the build or the
// execute.
// ---------------------------------------------------------------------------

namespace graph_steps {

using SilencedStderr = ::SilencedStderr;

struct FakeGraph
{
};

class FakeCache
{
public:
    const std::shared_ptr<FakeGraph>* Find(const hipdnn::PlanKey& key) const
    {
        for(const auto& entry : entries_)
            if(entry.first == key)
                return &entry.second;
        return nullptr;
    }
    void Insert(const hipdnn::PlanKey& key, std::shared_ptr<FakeGraph> graph)
    {
        entries_.emplace_back(key, std::move(graph));
    }
    size_t Size() const { return entries_.size(); }

private:
    std::vector<std::pair<hipdnn::PlanKey, std::shared_ptr<FakeGraph>>> entries_;
};

// What a call into hipDNN returns here: 0 worked, anything else failed.
std::optional<miopenStatus_t> FailureOf(int result)
{
    if(result == 0)
        return std::nullopt;
    return miopenStatusBadParm;
}

constexpr const char* kEntryPoint = "miopenConvolutionForward";

struct Acquire
{
    FakeCache cache;
    std::mutex mutex;
    hipdnn::PlanKey key;
    std::function<bool(FakeGraph&)> populate = [](FakeGraph&) { return true; };
    int builds                               = 0;

    template <typename Build>
    std::pair<std::shared_ptr<FakeGraph>, miopenStatus_t> Run(Build&& build)
    {
        return hipdnn::AcquireGraphWith<FakeGraph>(
            cache, mutex, key, populate, std::forward<Build>(build), FailureOf);
    }
};

struct Run
{
    std::mutex mutex;
    int64_t needed             = 0;
    int workspaceQueries       = 0;
    int ensures                = 0;
    int executions             = 0;
    int recordedEnds           = 0;
    bool inHipdnnDuringExecute = false;
    void* handleBuffer         = reinterpret_cast<void*>(0x2000);
    void* executedOn           = nullptr;

    template <typename Execute>
    miopenStatus_t Go(Execute&& execute,
                      void* callerWorkspace  = nullptr,
                      size_t callerSize      = 0,
                      int queryResult        = 0,
                      bool recordEndSucceeds = true)
    {
        return hipdnn::RunGraphWith(
            mutex,
            callerWorkspace,
            callerSize,
            [&](int64_t& out) {
                ++workspaceQueries;
                out = needed;
                return queryResult;
            },
            [&](size_t) -> void* {
                ++ensures;
                return handleBuffer;
            },
            [&](void* workspace) {
                ++executions;
                executedOn            = workspace;
                inHipdnnDuringExecute = hipdnn::InHipdnnCall();
                return execute();
            },
            [&] {
                ++recordedEnds;
                return recordEndSucceeds;
            },
            FailureOf);
    }
};

} // namespace graph_steps

TEST(CPU_WrapperRoutingReentry_NONE, ABuiltGraphIsCachedAndBuiltOnce)
{
    graph_steps::Acquire acquire;
    const auto build = [&](graph_steps::FakeGraph&) {
        ++acquire.builds;
        return 0;
    };

    const auto first  = acquire.Run(build);
    const auto second = acquire.Run(build);

    EXPECT_EQ(first.second, miopenStatusSuccess);
    ASSERT_NE(first.first, nullptr);
    EXPECT_EQ(second.first, first.first);
    EXPECT_EQ(acquire.builds, 1);
    EXPECT_EQ(acquire.cache.Size(), 1u);
}

TEST(CPU_WrapperRoutingReentry_NONE, TheBuildRunsInsideAHipdnnCall)
{
    graph_steps::Acquire acquire;
    bool inside = false;

    acquire.Run([&](graph_steps::FakeGraph&) {
        inside = hipdnn::InHipdnnCall();
        return 0;
    });

    EXPECT_TRUE(inside);
    EXPECT_FALSE(hipdnn::InHipdnnCall());
}

// A plugin that swallows the error it was given makes hipDNN report success.
TEST(CPU_WrapperRoutingReentry_NONE, AReentryDuringABuildFailsTheCallAndIsNotCached)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Acquire acquire;

    const auto result = acquire.Run([](graph_steps::FakeGraph&) {
        hipdnn::RejectReentrantCall(graph_steps::kEntryPoint);
        return 0;
    });

    EXPECT_EQ(result.second, miopenStatusInternalError);
    EXPECT_EQ(result.first, nullptr);
    EXPECT_EQ(acquire.cache.Size(), 0u);
    miopenStatus_t recorded = miopenStatusSuccess;
    EXPECT_TRUE(hipdnn::LastForwardedFailure(&recorded));
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, AReentryDuringABuildOutranksHipdnnsOwnFailure)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Acquire acquire;

    const auto result = acquire.Run([](graph_steps::FakeGraph&) {
        hipdnn::RejectReentrantCall(graph_steps::kEntryPoint);
        return 1;
    });

    EXPECT_EQ(result.second, miopenStatusInternalError);
    EXPECT_EQ(acquire.cache.Size(), 0u);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, AHipdnnBuildFailureIsReportedAndNotCached)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Acquire acquire;

    const auto result = acquire.Run([](graph_steps::FakeGraph&) { return 1; });

    EXPECT_EQ(result.second, miopenStatusBadParm);
    EXPECT_EQ(result.first, nullptr);
    EXPECT_EQ(acquire.cache.Size(), 0u);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, AProblemThatCannotBePopulatedIsNeverBuilt)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Acquire acquire;
    acquire.populate = [](graph_steps::FakeGraph&) { return false; };

    const auto result = acquire.Run([&](graph_steps::FakeGraph&) {
        ++acquire.builds;
        return 0;
    });

    EXPECT_EQ(result.second, miopenStatusUnsupportedOp);
    EXPECT_EQ(acquire.builds, 0);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, AGraphRunsInsideAHipdnnCallAndClearsEarlierFailures)
{
    const graph_steps::SilencedStderr silenced;
    hipdnn::ClearForwardedFailure();
    graph_steps::Run run;
    void* const callerBuffer = reinterpret_cast<void*>(0x1000);

    // An earlier forwarded failure must not outlive a call that worked.
    static_cast<void>(hipdnn::RecordFailure(miopenStatusBadParm, "earlier"));
    const auto status = run.Go([] { return 0; }, callerBuffer, 64);

    miopenStatus_t recorded = miopenStatusSuccess;
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_TRUE(run.inHipdnnDuringExecute);
    EXPECT_FALSE(hipdnn::InHipdnnCall());
    EXPECT_FALSE(hipdnn::LastForwardedFailure(&recorded));
    EXPECT_EQ(run.recordedEnds, 1);
}

TEST(CPU_WrapperRoutingReentry_NONE, AReentryDuringAnExecuteFailsTheCallEvenIfHipdnnSucceeded)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Run run;

    const auto status = run.Go([] {
        hipdnn::RejectReentrantCall(graph_steps::kEntryPoint);
        return 0;
    });

    EXPECT_EQ(status, miopenStatusInternalError);
    miopenStatus_t recorded = miopenStatusSuccess;
    EXPECT_TRUE(hipdnn::LastForwardedFailure(&recorded));
    EXPECT_EQ(recorded, miopenStatusInternalError);
    // The work it queued is still marked on the handle's stream.
    EXPECT_EQ(run.recordedEnds, 1);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, AReentryDuringAnExecuteOutranksHipdnnsOwnFailure)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Run run;

    const auto status = run.Go([] {
        hipdnn::RejectReentrantCall(graph_steps::kEntryPoint);
        return 1;
    });

    EXPECT_EQ(status, miopenStatusInternalError);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, AHipdnnExecuteFailureIsReportedAfterTheEndIsMarked)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Run run;

    const auto status = run.Go([] { return 1; });

    EXPECT_EQ(status, miopenStatusBadParm);
    EXPECT_EQ(run.recordedEnds, 1);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, AWorkspaceQueryFailureStopsBeforeTheExecute)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Run run;

    const auto status = run.Go([] { return 0; }, nullptr, 0, 1);

    EXPECT_EQ(status, miopenStatusBadParm);
    EXPECT_EQ(run.executions, 0);
    EXPECT_EQ(run.recordedEnds, 0);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, ACallersWorkspaceIsUsedOnlyWhenItIsBigEnough)
{
    const graph_steps::SilencedStderr silenced;
    void* const callerBuffer = reinterpret_cast<void*>(0x1000);
    graph_steps::Run big;
    big.needed = 64;
    graph_steps::Run small;
    small.needed = 64;

    EXPECT_EQ(big.Go([] { return 0; }, callerBuffer, 64), miopenStatusSuccess);
    EXPECT_EQ(big.executedOn, callerBuffer);
    EXPECT_EQ(big.ensures, 0);

    EXPECT_EQ(small.Go([] { return 0; }, callerBuffer, 63), miopenStatusSuccess);
    EXPECT_EQ(small.executedOn, small.handleBuffer);
    EXPECT_EQ(small.ensures, 1);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, ARunThatCannotGetItsWorkspaceIsNeverExecuted)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Run run;
    run.needed       = 64;
    run.handleBuffer = nullptr;

    const auto status = run.Go([] { return 0; });

    EXPECT_EQ(status, miopenStatusAllocFailed);
    EXPECT_EQ(run.executions, 0);
    hipdnn::ClearForwardedFailure();
}

TEST(CPU_WrapperRoutingReentry_NONE, AnEndThatCannotBeMarkedFailsTheCall)
{
    const graph_steps::SilencedStderr silenced;
    graph_steps::Run run;

    const auto status = run.Go([] { return 0; }, nullptr, 0, 0, false);

    EXPECT_EQ(status, miopenStatusInternalError);
    hipdnn::ClearForwardedFailure();
}

// ---------------------------------------------------------------------------
// The dispatch macro itself, through a stub named like an entry point. The route is read
// from MIOPEN_HIPDNN_FORWARDING once per process, so these skip unless it enables
// forwarding; the wrapper_reentry_macro ctest entry sets it.
// ---------------------------------------------------------------------------

namespace reentry_stub {

int forwardedCalls = 0;

miopenStatus_t MarkForwarded()
{
    ++forwardedCalls;
    return miopenStatusSuccess;
}

// Named like the entry point because the macro compares its first argument to __func__.
miopenStatus_t miopenConvolutionForward()
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForward, MarkForwarded());
    return miopenStatusBadParm; // what the MIOpen route would return
}

// Not in the forwarding set, so MIOpen serves it; named like the entry point for the macro.
miopenStatus_t miopenGetVersion()
{
    MIOPEN_WRAPPER_FORWARD(miopenGetVersion, MarkForwarded());
    return miopenStatusBadParm; // what the MIOpen route returns
}

} // namespace reentry_stub

class CPU_WrapperRoutingReentryMacro_NONE : public ::testing::Test
{
protected:
    void SetUp() override
    {
        Route route = Route::Miopen;
        {
            const SilencedStderr silenced;
            route = miopen::wrapper::Dispatch("miopenConvolutionForward");
        }
        if(route != Route::Hipdnn)
            GTEST_SKIP() << "miopenConvolutionForward is not on the hipDNN route; run with "
                            "MIOPEN_HIPDNN_FORWARDING=1";
        reentry_stub::forwardedCalls = 0;
        hipdnn::ClearForwardedFailure();
    }
};

TEST_F(CPU_WrapperRoutingReentryMacro_NONE, ForwardsOutsideHipdnn)
{
    EXPECT_EQ(reentry_stub::miopenConvolutionForward(), miopenStatusSuccess);
    EXPECT_EQ(reentry_stub::forwardedCalls, 1);
}

TEST_F(CPU_WrapperRoutingReentryMacro_NONE, FailsInsteadOfForwardingFromInsideHipdnn)
{
    const hipdnn::ScopedHipdnnCall scope;
    const SilencedStderr silenced;
    const miopenStatus_t first  = reentry_stub::miopenConvolutionForward();
    const miopenStatus_t second = reentry_stub::miopenConvolutionForward();

    EXPECT_EQ(first, miopenStatusInternalError);
    EXPECT_EQ(second, miopenStatusInternalError);
    EXPECT_EQ(reentry_stub::forwardedCalls, 0);
    EXPECT_TRUE(scope.ReentryDetected());
}

TEST_F(CPU_WrapperRoutingReentryMacro_NONE, ForwardsAgainOnceHipdnnHasReturned)
{
    {
        const hipdnn::ScopedHipdnnCall scope;
        const SilencedStderr silenced;
        reentry_stub::miopenConvolutionForward();
    }
    EXPECT_EQ(reentry_stub::miopenConvolutionForward(), miopenStatusSuccess);
    EXPECT_EQ(reentry_stub::forwardedCalls, 1);
}

// An entry point served by MIOpen cannot loop, so it is not rejected from inside hipDNN. A
// deny-listed entry point takes the same route.
TEST_F(CPU_WrapperRoutingReentryMacro_NONE, AnEntryPointServedByMiopenIsNotRejected)
{
    const hipdnn::ScopedHipdnnCall scope;

    EXPECT_EQ(reentry_stub::miopenGetVersion(), miopenStatusBadParm);
    EXPECT_EQ(reentry_stub::forwardedCalls, 0);
    EXPECT_FALSE(scope.ReentryDetected());
}

} // namespace

#endif // MIOPEN_ENABLE_HIPDNN_WRAPPER
