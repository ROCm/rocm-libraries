// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Implementation of the dispatch seam declared in src/private/routing.hpp.

#include "routing.hpp"

#include "hipdnn_graph.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>

namespace miopen {
namespace wrapper {

namespace {

const char* const kForwardingEnvVar = "MIOPEN_HIPDNN_FORWARDING";
const char* const kDisableEnvVar    = "MIOPEN_DISABLE_HIPDNN_FOR";
const char* const kLogLevelEnvVar   = "MIOPEN_LOG_LEVEL";
const char* const kApiLoggingEnvVar = "MIOPEN_ENABLE_LOGGING";

// MIOpen's Default, Error and Info2 levels, as numbers because the wrapper cannot
// include MIOpen's logging header.
constexpr int kDefaultLogLevel = 0;
constexpr int kErrorLogLevel   = 3;
constexpr int kInfo2LogLevel   = 6;

// Entry points redirected to hipDNN when forwarding is enabled. To add one, list
// it here and bump the array size.
//
// The forwarding parity tests (script/run_forwarding_parity.py) run the shim
// tests once with forwarding disabled and once enabled and require the two runs
// to agree, so every name listed here is checked against the MIOpen path.
//
// The bias entry points (miopenConvolutionForwardBias, miopenConvolutionBackwardBias)
// are deliberately absent: they are standalone bias kernels rather than
// convolutions, and hipDNN has no matching single-op graph for them.
//
// constexpr so it lands in .rodata: no initialization order or exit-time
// destructor to worry about.
constexpr std::array<std::string_view, 4> kForwardingEntries{
    "miopenConvolutionForward",
    "miopenConvolutionBackwardData",
    "miopenConvolutionBackwardWeights",
    "miopenConvolutionBiasActivationForward",
};

std::string ToLower(std::string_view value)
{
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered;
}

std::string_view Trim(std::string_view value)
{
    const auto is_space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    while(!value.empty() && is_space(value.front()))
        value.remove_prefix(1);
    while(!value.empty() && is_space(value.back()))
        value.remove_suffix(1);
    return value;
}

const char* ModeName(ForwardingMode mode)
{
    return mode == ForwardingMode::Enabled ? "enabled" : "disabled";
}

const char* RouteName(Route route) { return route == Route::Hipdnn ? "hipdnn" : "miopen"; }

const char* ReasonName(RouteReason reason)
{
    switch(reason)
    {
    case RouteReason::ForwardingOff: return "forwarding-off";
    case RouteReason::NotInForwardingSet: return "not-in-forwarding-set";
    case RouteReason::DisabledByEnv: return "disabled-by-env";
    case RouteReason::HipdnnUnavailable: return "hipdnn-unavailable";
    case RouteReason::InForwardingSet: return "in-forwarding-set";
    }
    return "unknown";
}

enum class BoolSetting
{
    Enabled,
    Disabled,
    Unrecognized,
};

// Same tokens as MIOpen's own boolean env parser (src/include/miopen/env.hpp),
// so these variables behave like every other MIOpen boolean variable.
BoolSetting ParseBoolSetting(const char* value)
{
    if(value == nullptr)
        return BoolSetting::Unrecognized;

    const std::string lowered = ToLower(value);
    if(lowered == "enable" || lowered == "enabled" || lowered == "1" || lowered == "yes" ||
       lowered == "on" || lowered == "true")
        return BoolSetting::Enabled;
    if(lowered == "disable" || lowered == "disabled" || lowered == "0" || lowered == "no" ||
       lowered == "off" || lowered == "false")
        return BoolSetting::Disabled;
    return BoolSetting::Unrecognized;
}

// No warning for an unrecognized value: MIOpen reports this variable's errors
// itself.
bool ApiLoggingEnabled()
{
    static const bool enabled =
        ParseBoolSetting(std::getenv(kApiLoggingEnvVar)) == BoolSetting::Enabled;
    return enabled;
}

unsigned long long LogLevel()
{
    static const unsigned long long level = ParseLogLevel(std::getenv(kLogLevelEnvVar));
    return level;
}

// MIOpen stores the level as uint64 but compares it as int, so out-of-range
// values wrap (-1 hides everything, 0x100000002 reads as Fatal). Wrap the same
// way to agree with MIOpen.
int AsMiopenLevel(unsigned long long level) { return static_cast<int>(level); }

} // namespace

ForwardingMode ParseForwardingMode(const char* value, std::ostream& diagnostics)
{
    // Unset and empty mean the user did not ask for forwarding, so they take the
    // silent default rather than the unrecognized-value warning below.
    if(value == nullptr || *value == '\0')
        return ForwardingMode::Disabled;

    switch(ParseBoolSetting(value))
    {
    case BoolSetting::Enabled:
        diagnostics << std::string("[MIOpen] ") + kForwardingEnvVar + '=' + value +
                           ": entry points in the forwarding set are redirected to hipDNN, all "
                           "others dispatch to the MIOpen implementation.\n";
        return ForwardingMode::Enabled;
    case BoolSetting::Disabled:
        // Silent, like the default: nothing to tell a user who asked for the
        // behavior they were going to get anyway.
        return ForwardingMode::Disabled;
    case BoolSetting::Unrecognized: break;
    }

    // Warn rather than throw: the common failure is a typo'd enable, and this
    // runs on the C ABI boundary of every entry point, where an escaping
    // exception would be worse than a loud warning. (MIOpen's own parser throws
    // miopenStatusInvalidValue here.)
    diagnostics << std::string("[MIOpen] Warning: ") + kForwardingEnvVar + " is set to '" + value +
                       "', which is not a recognized value. hipDNN forwarding stays disabled. "
                       "Recognized values are enable/enabled/1/yes/on/true and "
                       "disable/disabled/0/no/off/false.\n";
    return ForwardingMode::Disabled;
}

ForwardingSet DefaultForwardingSet()
{
    return ForwardingSet(kForwardingEntries.data(), kForwardingEntries.size());
}

bool IsInForwardingSet(const char* entryPoint, ForwardingSet set)
{
    if(entryPoint == nullptr)
        return false;

    const std::string_view name(entryPoint);
    return std::find(set.begin(), set.end(), name) != set.end();
}

bool IsInForwardingSet(const char* entryPoint)
{
    return IsInForwardingSet(entryPoint, DefaultForwardingSet());
}

bool DisabledSet::Contains(const char* entryPoint) const
{
    if(entryPoint == nullptr)
        return false;

    const std::string_view name(entryPoint);
    return std::find(names_.begin(), names_.end(), name) != names_.end();
}

DisabledSet ParseDisabledSet(const char* value, ForwardingSet set, std::ostream& diagnostics)
{
    DisabledSet disabled;
    if(value == nullptr)
        return disabled;

    std::string_view rest(value);
    while(true)
    {
        const auto comma   = rest.find(',');
        const auto element = Trim(rest.substr(0, comma));

        if(element == "*")
        {
            disabled.SetDisableAll();
        }
        else if(!element.empty())
        {
            if(IsInForwardingSet(std::string(element).c_str(), set))
            {
                disabled.Add(std::string(element));
            }
            else
            {
                // Warn rather than accept: a name this build never forwards is
                // almost always a typo, and accepting it silently would leave the
                // user believing they had turned something off.
                diagnostics << std::string("[MIOpen] Warning: ") + kDisableEnvVar + " names '" +
                                   std::string(element) +
                                   "', which is not an entry point this build forwards to "
                                   "hipDNN. It has no effect.\n";
            }
        }

        if(comma == std::string_view::npos)
            break;
        rest.remove_prefix(comma + 1);
    }

    return disabled;
}

const DisabledSet& GetDisabledSet()
{
    // Run-once, like the mode: MIOPEN_WRAPPER_DISPATCH caches each entry point's
    // route in a function-local static, so a deny list re-read later would have no
    // effect after that entry point's first call. Keep this the only production
    // caller of ParseDisabledSet.
    static const DisabledSet disabled =
        ParseDisabledSet(std::getenv(kDisableEnvVar), DefaultForwardingSet(), std::cerr);
    return disabled;
}

unsigned long long ParseLogLevel(const char* value)
{
    return value == nullptr ? 0 : std::strtoull(value, nullptr, 0);
}

bool RouteTraceEnabledFor(unsigned long long level)
{
    return AsMiopenLevel(level) >= kInfo2LogLevel;
}

bool ErrorLoggingEnabledFor(unsigned long long level)
{
    const int narrowed = AsMiopenLevel(level);
    return narrowed == kDefaultLogLevel || narrowed >= kErrorLogLevel;
}

bool RouteTraceEnabled() { return RouteTraceEnabledFor(LogLevel()); }

bool ErrorLoggingEnabled() { return ErrorLoggingEnabledFor(LogLevel()); }

// One write per line: stderr is unbuffered, so each << is a separate write and
// two threads' lines could mix.
void ReportRouteDecision(const char* entryPoint,
                         ForwardingMode mode,
                         RouteDecision decision,
                         std::ostream& diagnostics)
{
    std::string line = "[MIOpen] hipDNN routing: ";
    line += entryPoint == nullptr ? "<null>" : entryPoint;
    line += " mode=";
    line += ModeName(mode);
    line += " route=";
    line += RouteName(decision.route);
    line += " reason=";
    line += ReasonName(decision.reason);
    line += '\n';
    diagnostics << line;
}

void ReportForwardedCall(const char* entryPoint, std::ostream& diagnostics)
{
    std::string line = "[MIOpen] hipDNN forwarded: ";
    line += entryPoint == nullptr ? "<null>" : entryPoint;
    line += '\n';
    diagnostics << line;
}

void LogForwardedCall(const char* entryPoint) noexcept
{
    try
    {
        if(ApiLoggingEnabled())
            ReportForwardedCall(entryPoint, std::cerr);
    }
    catch(...)
    {
        // A lost log line must not fail the call it describes.
    }
}

RouteDecision ResolveRouteWithReason(ForwardingMode mode,
                                     const char* entryPoint,
                                     ForwardingSet set,
                                     const DisabledSet& disabled,
                                     bool hipdnnAvailable)
{
    if(mode != ForwardingMode::Enabled)
        return {Route::Miopen, RouteReason::ForwardingOff};
    if(!IsInForwardingSet(entryPoint, set))
        return {Route::Miopen, RouteReason::NotInForwardingSet};
    if(disabled.DisablesEverything() || disabled.Contains(entryPoint))
        return {Route::Miopen, RouteReason::DisabledByEnv};
    if(!hipdnnAvailable)
    {
        // Still routed away from MIOpen. The forwarding function turns this into
        // miopenStatusInternalError; routing back to the implementation would
        // instead make a broken hipDNN installation look like a working one.
        return {Route::Hipdnn, RouteReason::HipdnnUnavailable};
    }
    return {Route::Hipdnn, RouteReason::InForwardingSet};
}

Route ResolveRoute(ForwardingMode mode,
                   const char* entryPoint,
                   ForwardingSet set,
                   const DisabledSet& disabled)
{
    return ResolveRouteWithReason(mode, entryPoint, set, disabled, /*hipdnnAvailable=*/true).route;
}

Route ResolveRoute(ForwardingMode mode, const char* entryPoint, ForwardingSet set)
{
    return ResolveRoute(mode, entryPoint, set, DisabledSet());
}

Route ResolveRoute(ForwardingMode mode, const char* entryPoint)
{
    return ResolveRoute(mode, entryPoint, DefaultForwardingSet());
}

ForwardingMode GetForwardingMode()
{
    // Thread-safe run-once initialization, so the parse -- and whatever it
    // reports -- happens exactly once per process. Keep this the only production
    // caller of ParseForwardingMode, or "once" stops being true.
    static const ForwardingMode mode =
        ParseForwardingMode(std::getenv(kForwardingEnvVar), std::cerr);
    return mode;
}

Route Dispatch(const char* entryPoint)
{
    const ForwardingMode mode = GetForwardingMode();

    RouteDecision decision = ResolveRouteWithReason(
        mode, entryPoint, DefaultForwardingSet(), GetDisabledSet(), /*hipdnnAvailable=*/true);

    // Probed here rather than passed in, so a process that forwards nothing never
    // loads the hipDNN backend and never prints its diagnostics.
    if(decision.reason == RouteReason::InForwardingSet && !hipdnn::IsAvailable())
        decision.reason = RouteReason::HipdnnUnavailable;

    // Once per entry point, not once per call: the dispatch macro caches the
    // result, so this costs nothing in a hot loop.
    if(RouteTraceEnabled())
        ReportRouteDecision(entryPoint, mode, decision, std::cerr);

    return decision.route;
}

bool EntryPointNameMatches(const char* entryPoint, const char* enclosingFunction)
{
    if(entryPoint == nullptr || enclosingFunction == nullptr)
        return false;
    return std::string_view(entryPoint) == std::string_view(enclosingFunction);
}

Route DispatchFromStub(const char* entryPoint, const char* enclosingFunction)
{
    // A stub cloned from its neighbour still compiles with the neighbour's name
    // and silently becomes unroutable; only __func__ can catch that.
    assert(EntryPointNameMatches(entryPoint, enclosingFunction) &&
           "MIOPEN_WRAPPER_DISPATCH was given a name other than the enclosing function's");
    std::ignore = enclosingFunction; // unused when NDEBUG is defined
    return Dispatch(entryPoint);
}

} // namespace wrapper
} // namespace miopen
