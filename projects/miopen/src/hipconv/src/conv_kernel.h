#pragma once

#include "tolerance.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct LaunchParams;

typedef struct ihipStream_t* hipStream_t;

namespace hipconv
{

class ConvKernel;

// A non-owning view over a contiguous array of ConvKernel pointers.
// Each kernel TU exposes its kernels[] this way so per-arch backends
// can iterate kernels uniformly without templating on each TU's count.
using ConvKernelSpan = std::span<ConvKernel* const>;

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

class ConvKernel
{
public:
    using LaunchFn = void (*)(const LaunchParams&,
                              const hipconv::Conv2dParams&,
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

    // Just the config field list (key=value,...), with no family name or brackets:
    // the bare form that matches_descriptor()/--config accepts.
    //
    // The default is empty (a family with no descriptor fields has no per-config
    // selector). Descriptor families override it.
    virtual std::string describe_config() const { return {}; }

    // Does this configuration satisfy a descriptor constraint string?
    //
    // The string form is the ONLY general interface; each family encapsulates how
    // a token maps to its own tuning fields. The default understands no fields, so
    // it matches only the empty spec and rejects any non-empty token. Override to
    // participate. On a malformed/unknown token, return false and set *error.
    virtual bool matches_descriptor(std::string_view spec, std::string* error) const
    {
        for(char c : spec)
            if(c != ' ' && c != '\t' && c != ',')
            {
                if(error)
                    *error = "kernel '" + std::string(name()) + "' has no descriptor fields";
                return false;
            }
        return true;
    }

    // Family-level applicability: does this kernel family support these
    // parameters at all? Must depend only on `par`, never on per-config
    // tuning state. The dispatcher relies on this: it calls is_applicable
    // on the first kernel in each ConvKernelSpan and assumes the answer
    // speaks for every kernel in the span. Because each span contains
    // instances of a single concrete leaf class, the contract reduces to:
    // is_applicable must not read cfg_. Override in family base classes
    // or in concrete leaf classes (to add group-wide checks); never read
    // per-instance config state.
    virtual bool is_applicable(const hipconv::Conv2dParams& par) const = 0;

    // Per-config validity: given that the family is applicable, does this
    // specific tuning configuration match the parameters? May depend on
    // both `par` and the leaf kernel's stored cfg_.
    virtual bool is_valid_config(const hipconv::Conv2dParams& par) const = 0;

    virtual LaunchParams get_launch_params(const hipconv::Conv2dParams& par) const = 0;

    // Enqueue the kernel; throw on a launch-time failure.
    //
    // Defined out-of-line in conv_kernel.cpp so this header stays free of HIP
    // headers, since every kernel translation unit includes it.
    void launch(const LaunchParams& lp,
                const hipconv::Conv2dParams& par,
                const void* in,
                const void* wei,
                void* out,
                void* workspace,
                hipStream_t stream) const;

    virtual size_t get_workspace_size(const hipconv::Conv2dParams& /*par*/) const { return 0; }

    // Weighted throughput index for `par`; larger is better.
    //
    // 1.0 means full hardware utilization (MIOpen's GetWti convention, which a
    // host uses to rank providers without benchmarking). Queried only on a kernel
    // already selected for `par`, so it need not report inapplicability; each
    // family answers for its own tuning story.
    virtual float get_weighted_throughput_index(const hipconv::Conv2dParams& par) const = 0;

    // The error bound this kernel admits on `par`, or TOLERANCE_UNAVAILABLE when none applies.
    // A family whose accumulation is blocked overrides this to pass its own depth.
    virtual void get_tolerance(const hipconv::Conv2dParams& par, float& atol, float& rtol) const
    {
        hipconv::get_mixed_precision_tolerance(par, atol, rtol);
    }

private:
    LaunchFn launch_fn_;
};

} // namespace hipconv
