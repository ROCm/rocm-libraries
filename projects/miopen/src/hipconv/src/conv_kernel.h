#pragma once

#include "hipconv/device.hpp"
#include "internal_api.h"
#include "kv_descriptor.h"
#include "tolerance.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct LaunchParams;

typedef struct ihipStream_t* hipStream_t;

namespace hipconv
{

class ConvKernel;

class RepresentationHashPair
{
public:
    RepresentationHashPair(std::string_view repr);
    inline auto repr() const { return repr_; }
    inline auto hash() const { return hash_; }

private:
    std::string_view repr_;
    std::uint32_t hash_;
};

// A non-owning view over a contiguous array of ConvKernel pointers and a pointer to a function that
// looks up kernels by a string representation of its config. Each kernel TU exposes its kernels[]
// this way so per-arch backends can iterate kernels uniformly without templating on each TU's
// count.
struct ConvKernelGroup
{
    std::span<ConvKernel* const> kernels;
    ConvKernel* (*get_kernel_by_config)(RepresentationHashPair const& cfg);
};

// One kernel and what it scored on a layer.
struct ScoredKernel
{
    ConvKernel* kernel;
    float wti;
};

// Order the scored kernels best first, and drop all but `max_ranked` of them.
//
// Stable, so insertion order wins a tie: a family that hand-orders its table best-first keeps
// that order wherever the index cannot separate two configs. Ranking and truncation are the
// same operation at every level, an algorithm over its spans and the registry over its
// algorithms, so every level takes the caller's limit.
//
// min-heap or partial_sort would not be a stable alternative and would not be significantly
// faster for a small number of kernels.
inline void keep_top_ranked(std::vector<ScoredKernel>& scored, std::size_t max_ranked)
{
    std::stable_sort(scored.begin(),
                     scored.end(),
                     [](const ScoredKernel& a, const ScoredKernel& b) { return a.wti > b.wti; });
    if(scored.size() > max_ranked)
        scored.resize(max_ranked);
}

class HIPCONV_INTERNAL_API ConvKernel
{
public:
    using LaunchFn = void (*)(const LaunchParams&,
                              const hipconv::Device&,
                              const hipconv::ConvParams&,
                              const void*,
                              const void*,
                              void*,
                              void*,
                              hipStream_t);

    constexpr explicit ConvKernel(LaunchFn launch_fn) : launch_fn_(launch_fn) {}

    virtual ~ConvKernel() = default;

    // Short kernel-family name (e.g. "direct_l1", "direct").
    //
    // Used to select a family by name (the app's --variant filter) and to label
    // rows in listings. Kernels in the same family share one name.
    virtual std::string_view name() const = 0;

    // The algorithm this kernel's family belongs to. Set by the family base.
    virtual hipconv::Algorithm algorithm() const = 0;

    // The kernel's configuration as key=value fields; empty for a family without any.
    //
    // describe_config and matches_descriptor are built on it.
    virtual KVDescriptor config_descriptor() const { return {}; }

    // Return a specification of the kernel's configuration.
    //
    // The spec string is a comma-separated list of key=value pairs. `full` renders
    // every field; otherwise default values are omitted. No default argument: on a
    // virtual it would bind statically.
    virtual std::string describe_config(bool full) const
    {
        return config_descriptor().describe(full);
    }

    // Does this kernel's configuration satisfy the given spec?
    //
    // True if the configuration matches every key-value pair in the spec.
    // Set *error and return false on bad syntax or an unknown key.
    virtual bool matches_descriptor(std::string_view spec, std::string* error) const;

    // Does this kernel family support the given parameters?
    //
    // Must not read cfg_: the dispatcher asks the first kernel in a span and takes the answer
    // for the whole span.
    virtual bool is_applicable(const hipconv::Device& dev,
                               const hipconv::ConvParams& par) const = 0;

    // Does this kernel family support the given number of dimensions?
    //
    // Two-dimensional unless a family overrides this; direct_l1 does, for conv3d.
    // Same span contract as is_applicable, which is why the answer must not depend
    // on the config: a family serving both dimensionalities answers for both and
    // lets is_valid_config pick the config.
    virtual bool supports_dims(int dims) const { return dims == 2; }

    // Does this specific kernel configuration support the given parameters?
    virtual bool is_valid_config(const hipconv::Device& dev,
                                 const hipconv::ConvParams& par) const = 0;

    virtual LaunchParams get_launch_params(const hipconv::Device& dev,
                                           const hipconv::ConvParams& par) const = 0;

    // Enqueue the kernel; throw on a launch-time failure.
    //
    // Defined out-of-line in conv_kernel.cpp so this header stays free of HIP
    // headers, since every kernel translation unit includes it.
    void launch(const LaunchParams& lp,
                const hipconv::Device& dev,
                const hipconv::ConvParams& par,
                const void* in,
                const void* wei,
                void* out,
                void* workspace,
                hipStream_t stream) const;

    virtual size_t get_workspace_size(const hipconv::Device& /*dev*/,
                                      const hipconv::ConvParams& /*par*/) const
    {
        return 0;
    }

    // Whether two launches on the same inputs write bitwise-identical outputs.
    //
    // A family whose workgroups add into one output element atomically overrides this.
    virtual bool is_deterministic(const hipconv::Device& /*dev*/,
                                  const hipconv::ConvParams& /*par*/) const
    {
        return true;
    }

    // Weighted throughput index for `par`; larger is better.
    //
    // 1.0 means full hardware utilization (MIOpen's GetWti convention, which a
    // host uses to rank providers without benchmarking). Queried only on a kernel
    // already selected for `par`, so it need not report inapplicability; each
    // family answers for its own tuning story.
    virtual float get_weighted_throughput_index(const hipconv::Device& dev,
                                                const hipconv::ConvParams& par) const = 0;

    // The error bound this kernel admits on `par`, or TOLERANCE_UNAVAILABLE when none applies.
    // A family whose accumulation is blocked overrides this to pass its own depth.
    virtual void get_tolerance(const hipconv::Device& /*dev*/,
                               const hipconv::ConvParams& par,
                               float& atol,
                               float& rtol) const
    {
        hipconv::get_mixed_precision_tolerance(par, atol, rtol);
    }

private:
    LaunchFn launch_fn_;
};

} // namespace hipconv
