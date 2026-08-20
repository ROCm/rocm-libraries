#pragma once

// This file implements the public interface to hipconv
//
// Error-handling policy: runtime failures are reported through a hipconvError_t
// return value, while a violated precondition (an invalid argument such as a
// null handle) throws std::invalid_argument.

#include "conv2d_params.hpp"
#include "export.hpp"
#include "tolerance.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hipconv
{

class ConvKernel;
struct ArchEntry;

// Opaque, copyable handle to a library-managed architecture.
//
// ArchHandle may become an object in the future that allocates resources
// in its constructor and deallocates them in its destructor (RAII). Now,
// there are no per-architecture resources to manage. Either way, callers
// treat it as a copyable value.
using ArchHandle = const ArchEntry*;

// Opaque handle to a library-owned ConvKernel.
//
// Callers must not delete or manage the pointee; the library owns it for the
// lifetime of the program.
using ConvKernelHandle = ConvKernel*;

// Resolve a GFX arch name (e.g. "gfx950", "gfx950:sramecc+:xnack-") to a handle.
// Returns nullopt if this build has no support for that architecture.
HIPCONV_API std::optional<ArchHandle> resolve_arch(std::string_view name);

// Error code returned by hipconv's launch API. Currently a hipError_t and
// carries the standard hipError_t values; may be extended with hipconv-specific
// codes in the future.
using hipconvError_t = hipError_t;

// How many ranked configs survive a merge, at every level of it.
inline constexpr std::size_t MAX_RANKED_CONFIGS = 8;

// Keeps every matching config, for a caller that selects by index or by descriptor.
inline constexpr std::size_t ALL_RANKED_CONFIGS = std::numeric_limits<std::size_t>::max();

// All valid kernels for the given params, best first. Empty if unsupported.
HIPCONV_API std::vector<ConvKernelHandle>
get_valid_configs(ArchHandle arch,
                  const Conv2dParams& par,
                  std::size_t max_ranked = MAX_RANKED_CONFIGS);

// All valid kernels for the given params and algorithm.
HIPCONV_API std::vector<ConvKernelHandle>
get_valid_configs(ArchHandle arch,
                  const Conv2dParams& par,
                  Algorithm algo,
                  std::size_t max_ranked = MAX_RANKED_CONFIGS);

// Best kernel, or nullopt if unsupported.
HIPCONV_API std::optional<ConvKernelHandle> find_config(ArchHandle arch, const Conv2dParams& par);

// Re-check whether a previously selected kernel still supports `par`.
//
// Lets a caller who mutates params slightly avoid recomputing the full
// valid-configs list. Returns false if `par` falls outside the kernel's family
// applicability or its tuning configuration.
HIPCONV_API bool is_applicable(ConvKernelHandle kernel, const Conv2dParams& par);

// Short kernel-family name of the handle (e.g. "direct_l1", "direct", "grouped").
//
// Stable for the program's lifetime; used to select a family by name or to label
// listings.
HIPCONV_API std::string_view name(ConvKernelHandle kernel);

// The algorithm the handle belongs to (e.g. Direct, Grouped, Pointwise).
HIPCONV_API Algorithm algorithm(ConvKernelHandle kernel);

// The handle's config field list, with no family name or brackets.
//
// E.g. "waves_k=2,wave_k16=4,kh=3,kw=3,direction=fprop", exactly a spec that
// matches_descriptor() accepts. A family with no descriptor fields returns "".
HIPCONV_API std::string describe_config(ConvKernelHandle kernel);

// Does this kernel satisfy a descriptor constraint string?
//
// The string is a comma-separated set of key=value pairs (e.g.
// "waves_k=2,kh=3,direction=fprop"); the kernel matches iff every constraint it
// understands is satisfied and it rejects any token it does not understand. This
// is the ONLY general interface for specifying a config: how a constraint maps to
// a kernel's tuning fields is encapsulated entirely inside the kernel. A family
// that has not opted in matches only the empty string. On a malformed/unknown
// token, returns false and (if `error` is non-null) sets it to a message.
HIPCONV_API bool
matches_descriptor(ConvKernelHandle kernel, std::string_view spec, std::string* error = nullptr);

HIPCONV_API size_t get_workspace_size(ConvKernelHandle kernel, const Conv2dParams& par);

// Weighted throughput index of `kernel` for `par`; larger is better.
//
// 1.0 means full hardware utilization. Pass a kernel from find_config/
// get_valid_configs; the caller uses it to rank hipconv against other providers.
HIPCONV_API float get_weighted_throughput_index(ConvKernelHandle kernel, const Conv2dParams& par);

// Enqueue the kernel on the stream, returning the launch-time HIP status.
//
// hipSuccess means the launch was submitted; otherwise it is the launch error
// (hipErrorInvalidValue if kernel is null). The kernel runs asynchronously, so
// an execution fault (e.g. an out-of-bounds access) surfaces at a later
// synchronization, not here.
HIPCONV_API hipconvError_t launch(ConvKernelHandle kernel,
                                  const Conv2dParams& par,
                                  const void* in,
                                  const void* wei,
                                  void* out,
                                  void* workspace    = nullptr,
                                  hipStream_t stream = nullptr);

// The error bound `kernel` admits on `par`: |kernel - exact| <= atol + rtol * conv(|A|,|B|).
// rtol is TOLERANCE_UNAVAILABLE when no model applies; check has_tolerance(rtol) first.
HIPCONV_API void
get_tolerance(ConvKernelHandle kernel, const Conv2dParams& par, float& atol, float& rtol);

// The tolerance for a kernel that accumulates by recursive summation over the whole contraction.
//
// Also the error a float CPU reference admits, since recursive summation is what one does. A
// kernel reporting a tolerance below this has a blocked accumulation that such a reference is too
// coarse to check, so the caller owes it a more accurate one; see
// docs/algorithms/direct/direct-wgrad-tolerance.md.
//
// rtol is TOLERANCE_UNAVAILABLE past the depth where the model applies.
HIPCONV_API void
get_recursive_summation_tolerance(const Conv2dParams& par, float& atol, float& rtol);

// A bound (kernel, params) pair ready to launch.
//
// Captures `par` by value and pre-computes the per-launch derived state, so
// repeated launches against the same shape skip the per-call setup and cannot
// drift out of sync with the kernel handle. To launch with a different `par`,
// build a new ConvLaunch. Coexists with the free launch(kernel, par, ...) API;
// existing callers need not migrate.
class HIPCONV_API ConvLaunch
{
public:
    // Build a ConvLaunch for `kernel` bound to `par`.
    //
    // Returns nullopt if the kernel does not support `par` (same predicate as
    // is_applicable). Throws std::invalid_argument if `kernel` is null.
    static std::optional<ConvLaunch> make(ConvKernelHandle kernel, Conv2dParams par);

    ConvLaunch(ConvLaunch&&) noexcept;
    ConvLaunch& operator=(ConvLaunch&&) noexcept;
    ~ConvLaunch();

    ConvLaunch(const ConvLaunch&)            = delete;
    ConvLaunch& operator=(const ConvLaunch&) = delete;

    // Cached at construction; no Conv2dParams argument needed.
    size_t workspace_size() const noexcept;
    // rtol is TOLERANCE_UNAVAILABLE when no model applies; see the free get_tolerance above.
    void get_tolerance(float& atol, float& rtol) const;

    // The bound parameters and kernel handle, for inspection or printing.
    const Conv2dParams& params() const noexcept;
    ConvKernelHandle kernel() const noexcept;

    // Launch the kernel on the bound parameters.
    //
    // Only the data pointers and stream vary between calls; the kernel and
    // parameters are reused. See the free launch() above; execution faults
    // surface at a later synchronization, not here.
    hipconvError_t launch(const void* in,
                          const void* wei,
                          void* out,
                          void* workspace    = nullptr,
                          hipStream_t stream = nullptr) const;

private:
    struct State;
    std::unique_ptr<State> state_;

    explicit ConvLaunch(std::unique_ptr<State> state) noexcept;
};

} // namespace hipconv
