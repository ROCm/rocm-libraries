// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Implementation of the MIOpen public-wrapper runtime dispatch seam declared in
// src/private/routing.hpp (RFC 0001 hipDNN forwarding). Parses the
// MIOPEN_HIPDNN_FORWARDING environment variable, emits the one-time
// configuration banner, and answers the per-call routing decision by consulting
// the compile-time forwarding set.

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

ForwardingMode ParseForwardingEnv()
{
    const char* raw = std::getenv(kForwardingEnvVar);
    if(raw == nullptr)
        return ForwardingMode::Disabled;

    std::string value(raw);
    for(char& c : value)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if(value == "enabled" || value == "1" || value == "on" || value == "true" || value == "yes")
        return ForwardingMode::Enabled;

    // Unset, "disabled", and any unrecognized value all resolve to disabled so
    // that forwarding is never turned on by accident.
    return ForwardingMode::Disabled;
}

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

// The compile-time forwarding set (RFC 0001 routing policy): entry points listed
// here are redirected to hipDNN when forwarding is enabled. Adding an entry point
// is a one-line change.
bool IsInForwardingSet(const char* entryPoint)
{
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

ForwardingMode ParseAndAnnounce()
{
    const ForwardingMode mode = ParseForwardingEnv();
    EmitBanner(mode);
    return mode;
}

} // namespace

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

Route Dispatch(const char* entryPoint)
{
    if(GetForwardingMode() == ForwardingMode::Enabled && IsInForwardingSet(entryPoint))
        return Route::Hipdnn;
    return Route::Miopen;
}

} // namespace wrapper
} // namespace miopen
