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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

// Why a call landed on the route it did. Carried alongside the route rather than
// recomputed for the trace line, so the two cannot drift apart.
enum class RouteReason
{
    ForwardingOff,      // MIOPEN_HIPDNN_FORWARDING is not enabled
    NotInForwardingSet, // this build does not forward this entry point
    DisabledByEnv,      // MIOPEN_DISABLE_HIPDNN_FOR named it, or named *
    HipdnnUnavailable,  // hipDNN missing or reporting an unsupported version
    InForwardingSet,    // forwarded
};

struct RouteDecision
{
    Route route;
    RouteReason reason;
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

// Entry points the user has taken back off the hipDNN path via
// MIOPEN_DISABLE_HIPDNN_FOR. Owns its storage because the names come from the
// environment rather than a compile-time table.
class DisabledSet
{
public:
    DisabledSet() = default;
    explicit DisabledSet(bool disableAll) : disableAll_(disableAll) {}

    bool Contains(const char* entryPoint) const;
    bool DisablesEverything() const { return disableAll_; }

    void Add(std::string name) { names_.push_back(std::move(name)); }
    void SetDisableAll() { disableAll_ = true; }

private:
    std::vector<std::string> names_;
    bool disableAll_ = false;
};

// Parses a raw MIOPEN_DISABLE_HIPDNN_FOR value; null or empty disables nothing.
// "*" anywhere in the list disables every entry point. A name that is not in the
// build's forwarding set is reported to diagnostics and otherwise ignored: it is
// almost always a typo, and silently accepting it would leave the user believing
// they had turned something off.
//
// Names are matched case-sensitively because they are C identifiers; whitespace
// around each element is trimmed. There is deliberately no opt-in counterpart:
// the forwarding set is compile-time, and the only runtime controls are the
// master switch and this list.
DisabledSet ParseDisabledSet(const char* value, ForwardingSet set, std::ostream& diagnostics);

// Parsed from MIOPEN_DISABLE_HIPDNN_FOR on first use and cached for the process.
//
// Reading it once is what keeps the dispatch macros' per-entry-point route cache
// honest: a list that could change mid-process would silently stop having any
// effect after each entry point's first call.
const DisabledSet& GetDisabledSet();

// True when MIOPEN_LOG_LEVEL selects MIOpen's trace level. Read straight from the
// environment because the wrapper cannot include MIOpen's logging header.
bool TracingEnabled();

// Writes one "[MIOpen] hipDNN routing: ..." line describing the decision.
void ReportRouteDecision(const char* entryPoint,
                         ForwardingMode mode,
                         RouteDecision decision,
                         std::ostream& diagnostics);

// Route::Hipdnn only when mode is Enabled AND entryPoint is in the set AND the
// deny list does not name it. hipdnnAvailable false still routes away from
// MIOpen: an enabled-but-unavailable hipDNN has to fail loudly, and a route back
// to the MIOpen implementation would instead make it look like success.
RouteDecision ResolveRouteWithReason(ForwardingMode mode,
                                     const char* entryPoint,
                                     ForwardingSet set,
                                     const DisabledSet& disabled,
                                     bool hipdnnAvailable);

Route ResolveRoute(ForwardingMode mode,
                   const char* entryPoint,
                   ForwardingSet set,
                   const DisabledSet& disabled);
Route ResolveRoute(ForwardingMode mode, const char* entryPoint, ForwardingSet set);
Route ResolveRoute(ForwardingMode mode, const char* entryPoint);

// Parsed from MIOPEN_HIPDNN_FORWARDING on first use and cached for the lifetime
// of the process; the first call reports against std::cerr.
ForwardingMode GetForwardingMode();

// Route for one wrapped call, e.g. entryPoint "miopenConvolutionForward".
// Wrapper stubs should use the dispatch macros below instead, which resolve the
// route once per entry point rather than on every call.
Route Dispatch(const char* entryPoint);

// True when entryPoint names the same function as enclosingFunction (a __func__
// value); null on either side is a mismatch.
bool EntryPointNameMatches(const char* entryPoint, const char* enclosingFunction);

// Dispatch() plus a debug-build assertion that entryPoint is the enclosing
// function's own name.
Route DispatchFromStub(const char* entryPoint, const char* enclosingFunction);

} // namespace wrapper
} // namespace miopen

// The dispatch seam, used as the first statement of the stub for entry point
// `fn`. `expr` is what the hipDNN route returns, and is evaluated only on that
// route, so the stub's arguments are untouched when the call is served by
// MIOpen:
//
//     extern "C" miopenStatus_t miopenConvolutionForward(miopenHandle_t handle, ...)
//     {
//         MIOPEN_WRAPPER_FORWARD(miopenConvolutionForward,
//                                ::miopen::wrapper::hipdnn::ConvolutionForward(handle, ...));
//         return miopenConvolutionForward_impl(handle, ...);
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
#define MIOPEN_WRAPPER_FORWARD(fn, expr)                              \
    do                                                                \
    {                                                                 \
        static const ::miopen::wrapper::Route miopen_wrapper_route_ = \
            ::miopen::wrapper::DispatchFromStub(#fn, __func__);       \
        if(miopen_wrapper_route_ == ::miopen::wrapper::Route::Hipdnn) \
            return (expr);                                            \
    } while(false)

// The same seam for the stubs with nowhere to forward to yet:
//
//     extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle)
//     {
//         MIOPEN_WRAPPER_DISPATCH(miopenCreate);
//         return miopenCreate_impl(handle);
//     }
//
// Requires a `forward_to_hipdnn(const char*)` returning the enclosing function's
// return type to be in scope; wrapper.cpp defines it.
#define MIOPEN_WRAPPER_DISPATCH(fn) MIOPEN_WRAPPER_FORWARD(fn, forward_to_hipdnn(#fn))

#endif // MIOPEN_PRIVATE_ROUTING_HPP
