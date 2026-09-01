// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Runtime dispatch seam for the MIOpen public wrapper. The stubs in
// src/private/wrapper.cpp use it to decide whether a public call goes to the
// MIOpen implementation (the _impl symbols in libMIOpen_private.so) or is
// forwarded to hipDNN. A call is forwarded only when MIOPEN_HIPDNN_FORWARDING
// enables forwarding AND the entry point is in the compile-time forwarding set.
//
// ParseForwardingMode/IsInForwardingSet/ResolveRoute take every input
// explicitly (even the diagnostics stream) so tests can exercise both routes
// without touching the environment. GetForwardingMode() and Dispatch() hold the
// process-global state.
//
// Compiled only into the public wrapper library; never installed.
#ifndef MIOPEN_PRIVATE_ROUTING_HPP
#define MIOPEN_PRIVATE_ROUTING_HPP

#include <cstddef>
#include <iosfwd>
#include <string_view>

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

// Resolves a raw MIOPEN_HIPDNN_FORWARDING value; null means unset. Accepts the
// same tokens as MIOpen's own boolean env parser (src/include/miopen/env.hpp),
// case-insensitively:
//
//   enable, enabled, 1, yes, on, true    -> Enabled
//   disable, disabled, 0, no, off, false -> Disabled
//
// Anything else, including null and empty, is Disabled: forwarding is never
// turned on by accident.
//
// Writes to diagnostics only when forwarding is enabled or the value was not
// understood. The paths a user clearly meant to be off stay silent so a wrapper
// build looks the same on stderr as a non-wrapper build.
//
// Takes the stream rather than writing to stderr so tests can read the
// diagnostics. "Report once per process" comes from GetForwardingMode() being
// the only production caller.
ForwardingMode ParseForwardingMode(const char* value, std::ostream& diagnostics);

// Non-owning view of the entry-point names redirected to hipDNN when forwarding
// is enabled. Constexpr constructible so production and tests can each build one
// from their own static array without allocating.
class ForwardingSet
{
public:
    constexpr ForwardingSet() noexcept : entries_(nullptr), count_(0) {}

    constexpr ForwardingSet(const std::string_view* entries, std::size_t count) noexcept
        : entries_(entries), count_(count)
    {
    }

    template <std::size_t N>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    constexpr explicit ForwardingSet(const std::string_view (&entries)[N]) noexcept
        : entries_(entries), count_(N)
    {
    }

    constexpr const std::string_view* begin() const noexcept { return entries_; }
    constexpr const std::string_view* end() const noexcept { return entries_ + count_; }
    constexpr std::size_t size() const noexcept { return count_; }
    constexpr bool empty() const noexcept { return count_ == 0; }

private:
    const std::string_view* entries_;
    std::size_t count_;
};

// The set this build was compiled with; adding an entry point is a one-line
// change in routing.cpp.
ForwardingSet DefaultForwardingSet();

// The set-taking overloads exist so tests can inject a non-empty set; the others
// use DefaultForwardingSet().
bool IsInForwardingSet(const char* entryPoint, ForwardingSet set);
bool IsInForwardingSet(const char* entryPoint);

// Route::Hipdnn only when mode is Enabled AND entryPoint is in the set.
Route ResolveRoute(ForwardingMode mode, const char* entryPoint, ForwardingSet set);
Route ResolveRoute(ForwardingMode mode, const char* entryPoint);

// Parsed from MIOPEN_HIPDNN_FORWARDING on first use and cached for the lifetime
// of the process; the first call reports against std::cerr.
ForwardingMode GetForwardingMode();

// Route for one wrapped call, e.g. entryPoint "miopenConvolutionForward".
// Wrapper stubs should use MIOPEN_WRAPPER_DISPATCH below instead, which resolves
// the route once per entry point rather than on every call.
Route Dispatch(const char* entryPoint);

// True when entryPoint names the same function as enclosingFunction (a __func__
// value); null on either side is a mismatch.
bool EntryPointNameMatches(const char* entryPoint, const char* enclosingFunction);

// Dispatch() plus a debug-build assertion that entryPoint is the enclosing
// function's own name.
Route DispatchFromStub(const char* entryPoint, const char* enclosingFunction);

} // namespace wrapper
} // namespace miopen

// Dispatch hook, used as the first statement of the stub for entry point `fn`:
//
//     extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle)
//     {
//         MIOPEN_WRAPPER_DISPATCH(miopenCreate);
//         return miopenCreate_impl(handle);
//     }
//
// Takes the function token, not a string, so the name can be checked against
// __func__: a stub cloned from its neighbour then asserts in a debug build
// instead of silently never forwarding.
//
// Caching the route in a function-local static is safe because neither input
// changes after startup (the mode is read from the environment once, the
// forwarding set is compile-time). That keeps hot entry points such as
// miopenSetTensorDescriptor about as cheap as the plain tail-call they were
// before the seam existed.
//
// Requires a `forward_to_hipdnn(const char*)` returning the enclosing function's
// return type to be in scope; wrapper.cpp defines it.
#define MIOPEN_WRAPPER_DISPATCH(fn)                                   \
    do                                                                \
    {                                                                 \
        static const ::miopen::wrapper::Route miopen_wrapper_route_ = \
            ::miopen::wrapper::DispatchFromStub(#fn, __func__);       \
        if(miopen_wrapper_route_ == ::miopen::wrapper::Route::Hipdnn) \
            return forward_to_hipdnn(#fn);                            \
    } while(false)

#endif // MIOPEN_PRIVATE_ROUTING_HPP
