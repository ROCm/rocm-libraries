// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Implementation of the MIOpen public-wrapper runtime dispatch seam declared in
// src/private/routing.hpp (RFC 0001 hipDNN forwarding). Parses the
// MIOPEN_HIPDNN_FORWARDING environment variable, emits the one-time
// configuration banner, and answers the per-call routing decision by consulting
// the compile-time forwarding set.
//
// The decision logic is split into pure, side-effect-free helpers
// (ParseForwardingMode, IsInForwardingSet, ResolveRoute) so it can be unit
// tested directly, and the process-global parts (env read, one-time banner,
// cached mode) live in GetForwardingMode()/Dispatch().

#include "routing.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>

namespace miopen {
namespace wrapper {

namespace {

const char* const kForwardingEnvVar = "MIOPEN_HIPDNN_FORWARDING";

void EmitBanner(ForwardingMode mode)
{
    if(mode == ForwardingMode::Enabled)
    {
        std::cerr << "[MIOpen] " << kForwardingEnvVar
                  << " resolved to 'enabled'; entry points in the forwarding set are redirected "
                     "to hipDNN, all others dispatch to the MIOpen implementation.\n";
    }
    else
    {
        std::cerr << "[MIOpen] " << kForwardingEnvVar
                  << " resolved to 'disabled'; every call dispatches to the MIOpen "
                     "implementation.\n";
    }
}

ForwardingMode ParseAndAnnounce()
{
    const ForwardingMode mode = ParseForwardingMode(std::getenv(kForwardingEnvVar));
    EmitBanner(mode);
    return mode;
}

} // namespace

ForwardingMode ParseForwardingMode(const char* value)
{
    if(value == nullptr)
        return ForwardingMode::Disabled;

    std::string lowered(value);
    for(char& c : lowered)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if(lowered == "enabled" || lowered == "1" || lowered == "on" || lowered == "true" ||
       lowered == "yes")
        return ForwardingMode::Enabled;

    // Unset, "disabled", and any unrecognized value all resolve to disabled so
    // that forwarding is never turned on by accident.
    return ForwardingMode::Disabled;
}

bool IsInForwardingSet(const char* entryPoint)
{
    // The compile-time forwarding set (RFC 0001 routing policy): entry points
    // listed here are redirected to hipDNN when forwarding is enabled. Adding an
    // entry point is a one-line change.
    for(const char* name : std::initializer_list<const char*>{
            // Add public entry-point names to forward them to hipDNN, e.g.
            // "miopenConvolutionForward".
        })
    {
        if(std::strcmp(name, entryPoint) == 0)
            return true;
    }
    return false;
}

Route ResolveRoute(ForwardingMode mode, const char* entryPoint)
{
    if(mode == ForwardingMode::Enabled && IsInForwardingSet(entryPoint))
        return Route::Hipdnn;
    return Route::Miopen;
}

ForwardingMode GetForwardingMode()
{
    // Thread-safe, run-once initialization (C++11 function-local static). The
    // initializer runs the first time control reaches this line; on every later
    // call the compiler-generated guard skips it and returns the cached value.
    // The banner is emitted inside the initializer, so it appears exactly once
    // per process, on the first wrapped call that reaches Dispatch().
    static const ForwardingMode mode = ParseAndAnnounce();
    return mode;
}

Route Dispatch(const char* entryPoint) { return ResolveRoute(GetForwardingMode(), entryPoint); }

} // namespace wrapper
} // namespace miopen
