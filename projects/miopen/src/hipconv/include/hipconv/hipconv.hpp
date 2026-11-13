#pragma once

// This file implements the public interface to hipconv
//
// Error-handling policy: runtime failures are reported through a hipconvError_t
// return value, while a violated precondition (an invalid argument such as a
// null handle) throws std::invalid_argument.

#include "conv_params.hpp"
#include "device.hpp"
#include <hipconv/export.hpp>
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

// Opaque handle to a library-owned ConvKernel.
//
// Callers must not delete or manage the pointee; the library owns it for the
// lifetime of the program.
using ConvKernelHandle = ConvKernel*;

// This build's hipconv version, as `git describe --tags` names it.
//
// "v0.4.0" at the tag, "v0.4.0-2-g1a2b3c4d" past it; "unknown" when neither a VERSION file
// nor git provides one.
HIPCONV_API std::string_view version();

// The major and minor version of version(), e.g. "v0.4".
//
// "unknown" when no release tag precedes this build. For keying a stored config
// description: within a minor version, a full description always selects the same config.
HIPCONV_API std::string_view config_version();

// Returns the handle for a GFX arch name (e.g. "gfx950", "gfx950:sramecc+:xnack-"), or nullopt
// if the name is unknown.
//
// Looks the name up in the registry, which is identical in every build. Makes no HIP calls.
// If the binary has no device code for the GPU, launch() fails with hipErrorNoBinaryForGpu or
// hipErrorInvalidDeviceFunction; has_device_code reports in advance whether the binary has it.
HIPCONV_API std::optional<ArchHandle> resolve_arch(std::string_view name);

// Same as resolve_arch.
HIPCONV_API std::optional<ArchHandle> find_arch(std::string_view name);

// Whether the loaded binary holds hipconv's device code for HIP device hip_device.
//
// False for an index that names no device. Makes hip_device current for one query and restores
// the calling thread's current device. Clears the HIP error its own failed calls raise. The answer
// is cached per device.
HIPCONV_API bool has_device_code(int hip_device);

// Returns the name of every architecture in the registry, in registry order. resolve_arch
// accepts each.
HIPCONV_API std::vector<std::string_view> arch_names();

// Error code returned by hipconv's launch API. Currently a hipError_t and
// carries the standard hipError_t values; may be extended with hipconv-specific
// codes in the future.
using hipconvError_t = hipError_t;

// How many ranked configs survive a merge, at every level of it.
inline constexpr std::size_t MAX_RANKED_CONFIGS = 8;

// Keeps every matching config, for a caller that selects by index or by descriptor.
inline constexpr std::size_t ALL_RANKED_CONFIGS = std::numeric_limits<std::size_t>::max();

// All valid kernels for the given params, best first. Empty if unsupported.
//
// Makes no HIP calls, so it runs without a GPU. A kernel that calls a runtime library, such as
// hipBLASLt, can still fail its first launch() with hipErrorNotSupported; the caller then tries
// the next kernel.
//
// `par` is matched as written; no folding happens here. A caller with a conv1d
// or conv3d layer passes ConvParams::unfolded() instead, which is what gets a
// degenerate conv3d served by a conv2d kernel; without it such a layer matches
// only the kernels that declare three dimensions: today direct_l1, for 3x3x3 fprop
// and stride-1 dgrad. The handle this returns is then bound to the unfolded params,
// so ConvLaunch::make and launch() must be given the same ones.
//
// Throws std::invalid_argument if device.arch is null or device.cu_count is not positive,
// as does every function below that takes a Device.
HIPCONV_API std::vector<ConvKernelHandle>
get_valid_configs(const Device& device,
                  const ConvParams& par,
                  std::size_t max_ranked = MAX_RANKED_CONFIGS);

// All valid kernels for the given params and algorithm.
HIPCONV_API std::vector<ConvKernelHandle>
get_valid_configs(const Device& device,
                  const ConvParams& par,
                  Algorithm algo,
                  std::size_t max_ranked = MAX_RANKED_CONFIGS);

// Best kernel, or nullopt if unsupported.
HIPCONV_API std::optional<ConvKernelHandle> find_config(const Device& device,
                                                        const ConvParams& par);

// Re-check whether a previously selected kernel still supports `par`.
//
// Lets a caller who mutates params slightly avoid recomputing the full
// valid-configs list. Returns true iff get_valid_configs(device, par, ALL_RANKED_CONFIGS)
// lists the kernel.
HIPCONV_API bool
is_applicable(ConvKernelHandle kernel, const Device& device, const ConvParams& par);

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
// `full` renders every field; otherwise default values are omitted. Only the full
// form is guaranteed to select this config alone through matches_descriptor().
HIPCONV_API std::string describe_config(ConvKernelHandle kernel, bool full = false);

// The kernel's family and full description encoded as a string.
//
// "family[full description]", or "family" when describe_config(kernel, true) is empty. Within
// one config_version(), the string names one kernel of an arch, which find_kernel returns.
HIPCONV_API std::string describe_kernel(ConvKernelHandle kernel);

// Returns the kernel of `arch` whose describe_kernel() is `description`, or nullopt.
//
// Thread-safe. Reads only `arch` and makes no HIP calls. Check the kernel against a problem
// with is_applicable(kernel, device, par). Throws std::invalid_argument if `arch` is null or
// did not come from find_arch or resolve_arch.
HIPCONV_API std::optional<ConvKernelHandle> find_kernel(ArchHandle arch,
                                                        std::string_view description);

// Does this kernel satisfy a descriptor constraint string?
//
// The string is a comma-separated set of key=value pairs (e.g.
// "waves_k=2,kh=3,direction=fprop"); the kernel matches iff every constraint it
// understands is satisfied and it rejects any token it does not understand. This
// is the ONLY general interface for specifying a config: how a constraint maps to
// a kernel's configuration fields is encapsulated entirely inside the kernel. A family
// that has not opted in matches only the empty string. On a malformed/unknown
// token, returns false and (if `error` is non-null) sets it to a message.
HIPCONV_API bool
matches_descriptor(ConvKernelHandle kernel, std::string_view spec, std::string* error = nullptr);

HIPCONV_API size_t get_workspace_size(ConvKernelHandle kernel,
                                      const Device& device,
                                      const ConvParams& par);

// Whether launching `kernel` twice on the same inputs writes bitwise-identical outputs.
//
// False where the kernel adds partial results into the output atomically, in an order that varies
// between launches. Pass a kernel from get_valid_configs for `device` and `par`.
HIPCONV_API bool
is_deterministic(ConvKernelHandle kernel, const Device& device, const ConvParams& par);

// Weighted throughput index of `kernel` for `par`; larger is better.
//
// 1.0 means full hardware utilization. Pass a kernel from find_config/
// get_valid_configs; the caller uses it to rank hipconv against other providers.
HIPCONV_API float
get_weighted_throughput_index(ConvKernelHandle kernel, const Device& device, const ConvParams& par);

// Enqueue the kernel on the stream, returning the launch-time HIP status.
//
// hipSuccess means the launch was submitted; otherwise it is the launch error
// (hipErrorInvalidValue if kernel is null or `device` is invalid, hipErrorNotSupported if
// the runtime library the kernel calls, such as hipBLASLt, has no algorithm for `par` on the
// stream's device). The kernel runs asynchronously, so an execution fault (e.g. an
// out-of-bounds access) surfaces at a later synchronization.
//
// The stream's device must be current, as for hipBLASLt and rocBLAS, and must be the one
// `device` describes.
HIPCONV_API hipconvError_t launch(ConvKernelHandle kernel,
                                  const Device& device,
                                  const ConvParams& par,
                                  const void* in,
                                  const void* wei,
                                  void* out,
                                  void* workspace    = nullptr,
                                  hipStream_t stream = nullptr);

// The error bound `kernel` admits on `par`: |kernel - exact| <= atol + rtol * conv(|A|,|B|).
// rtol is TOLERANCE_UNAVAILABLE when no model applies; check has_tolerance(rtol) first.
HIPCONV_API void get_tolerance(ConvKernelHandle kernel,
                               const Device& device,
                               const ConvParams& par,
                               float& atol,
                               float& rtol);

// The tolerance for a kernel that accumulates by recursive summation over the whole contraction.
//
// Also the error a float CPU reference admits, since recursive summation is what one does. A
// kernel reporting a tolerance below this has a blocked accumulation that such a reference is too
// coarse to check, so the caller owes it a more accurate one; see
// docs/algorithms/details/direct/direct-wgrad-tolerance.md.
//
// rtol is TOLERANCE_UNAVAILABLE past the depth where the model applies.
HIPCONV_API void get_recursive_summation_tolerance(const ConvParams& par, float& atol, float& rtol);

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
    // Build a ConvLaunch for `kernel` bound to `device` and `par`.
    //
    // Returns nullopt if the kernel does not support `par` (same predicate as
    // is_applicable). Makes no HIP calls: if the kernel's runtime library (hipBLASLt for the
    // explicit-GEMM kernels) has no algorithm for `par`, launch() returns hipErrorNotSupported.
    // Throws std::invalid_argument if `kernel` is null or `device` is invalid.
    static std::optional<ConvLaunch>
    make(ConvKernelHandle kernel, const Device& device, ConvParams par);

    ConvLaunch(ConvLaunch&&) noexcept;
    ConvLaunch& operator=(ConvLaunch&&) noexcept;
    ~ConvLaunch();

    ConvLaunch(const ConvLaunch&)            = delete;
    ConvLaunch& operator=(const ConvLaunch&) = delete;

    // Cached at construction; no ConvParams argument needed.
    size_t workspace_size() const noexcept;
    // rtol is TOLERANCE_UNAVAILABLE when no model applies; see the free get_tolerance above.
    void get_tolerance(float& atol, float& rtol) const;

    // The bound device, parameters and kernel handle, for inspection or printing.
    const Device& device() const noexcept;
    const ConvParams& params() const noexcept;
    ConvKernelHandle kernel() const noexcept;

    // Launch the kernel on the bound parameters.
    //
    // Only the data pointers and stream vary between calls; the kernel, device and
    // parameters are reused. See the free launch() above; execution faults
    // surface at a later synchronization.
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
