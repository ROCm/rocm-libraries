// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Runtime dispatch seam for the MIOpen public wrapper. The pass-through stubs in
// src/private/wrapper.cpp consult Dispatch() on every public call to decide
// whether the call is served by the MIOpen implementation (the _impl symbols in
// libMIOpen_private.so) or forwarded to hipDNN. The decision is governed at
// runtime by the MIOPEN_HIPDNN_FORWARDING environment variable together with a
// compile-time forwarding set: when forwarding is enabled, entry points in the
// forwarding set are redirected to hipDNN and all others fall through to the
// MIOpen implementation.
//
// This header is compiled only into the public wrapper library and is never
// installed.
#ifndef MIOPEN_PRIVATE_ROUTING_HPP
#define MIOPEN_PRIVATE_ROUTING_HPP

namespace miopen {
namespace wrapper {

// Resolved value of MIOPEN_HIPDNN_FORWARDING for the current process.
enum class ForwardingMode
{
    Disabled, // every call is served by the MIOpen implementation
    Enabled,  // forwarding permitted for entry points in the forwarding set
};

// Where a wrapped public entry point is served.
enum class Route
{
    Miopen, // the MIOpen implementation (the _impl symbol in libMIOpen_private.so)
    Hipdnn, // forwarded to hipDNN
};

// Pure mapping from a raw MIOPEN_HIPDNN_FORWARDING value to a ForwardingMode.
// value may be null (variable unset). Recognizes (case-insensitively) "enabled",
// "1", "on", "true", and "yes" as ForwardingMode::Enabled; null, "disabled", and
// every other value map to ForwardingMode::Disabled so that forwarding is never
// enabled by accident. Reads no environment, caches nothing, and emits no banner.
ForwardingMode ParseForwardingMode(const char* value);

// True when entryPoint is in the compile-time forwarding set: the entry points
// redirected to hipDNN when forwarding is enabled.
bool IsInForwardingSet(const char* entryPoint);

// Pure routing decision for an already-resolved mode: Route::Hipdnn only when
// mode is ForwardingMode::Enabled AND entryPoint is in the forwarding set,
// otherwise Route::Miopen. This is the decision Dispatch() applies on top of the
// process-wide GetForwardingMode().
Route ResolveRoute(ForwardingMode mode, const char* entryPoint);

// The process-wide forwarding mode, parsed once from MIOPEN_HIPDNN_FORWARDING on
// first use and cached for the lifetime of the process. The first call also
// emits the one-time configuration banner to stderr.
ForwardingMode GetForwardingMode();

// Routing decision for a single wrapped call. entryPoint is the public function
// name, e.g. "miopenConvolutionForward". Equivalent to
// ResolveRoute(GetForwardingMode(), entryPoint): Route::Hipdnn when forwarding is
// enabled AND entryPoint is in the compile-time forwarding set, otherwise
// Route::Miopen.
Route Dispatch(const char* entryPoint);

} // namespace wrapper
} // namespace miopen

#endif // MIOPEN_PRIVATE_ROUTING_HPP
