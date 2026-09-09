// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Implementation of the dispatch seam declared in src/private/routing.hpp.

#include "routing.hpp"

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

// Entry points redirected to hipDNN when forwarding is enabled. To add one, list
// it here and bump the array size:
//
//     constexpr std::array<std::string_view, 1> kForwardingEntries{
//         "miopenConvolutionForward",
//     };
//
// constexpr so it lands in .rodata: no initialization order or exit-time
// destructor to worry about.
constexpr std::array<std::string_view, 0> kForwardingEntries{};

std::string ToLower(std::string_view value)
{
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered;
}

} // namespace

ForwardingMode ParseForwardingMode(const char* value, std::ostream& diagnostics)
{
    // Unset and empty mean the user did not ask for forwarding, so they take the
    // silent default rather than the unrecognized-value warning below.
    if(value == nullptr || *value == '\0')
        return ForwardingMode::Disabled;

    const std::string lowered = ToLower(value);

    // These token sets match MIOpen's own boolean env parser
    // (src/include/miopen/env.hpp) so MIOPEN_HIPDNN_FORWARDING behaves like every
    // other MIOpen boolean variable.
    if(lowered == "enable" || lowered == "enabled" || lowered == "1" || lowered == "yes" ||
       lowered == "on" || lowered == "true")
    {
        diagnostics << "[MIOpen] " << kForwardingEnvVar << '=' << value
                    << ": entry points in the forwarding set are redirected to hipDNN, all others "
                       "dispatch to the MIOpen implementation.\n";
        return ForwardingMode::Enabled;
    }

    // Silent, like the default: nothing to tell a user who asked for the
    // behavior they were going to get anyway.
    if(lowered == "disable" || lowered == "disabled" || lowered == "0" || lowered == "no" ||
       lowered == "off" || lowered == "false")
        return ForwardingMode::Disabled;

    // Warn rather than throw: the common failure is a typo'd enable, and this
    // runs on the C ABI boundary of every entry point, where an escaping
    // exception would be worse than a loud warning. (MIOpen's own parser throws
    // miopenStatusInvalidValue here.)
    diagnostics << "[MIOpen] Warning: " << kForwardingEnvVar << " is set to '" << value
                << "', which is not a recognized value. hipDNN forwarding stays disabled. "
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

Route ResolveRoute(ForwardingMode mode, const char* entryPoint, ForwardingSet set)
{
    if(mode == ForwardingMode::Enabled && IsInForwardingSet(entryPoint, set))
        return Route::Hipdnn;
    return Route::Miopen;
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

Route Dispatch(const char* entryPoint) { return ResolveRoute(GetForwardingMode(), entryPoint); }

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
