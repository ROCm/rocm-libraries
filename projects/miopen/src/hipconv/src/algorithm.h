#pragma once

#include "conv_kernel.h"
#include "hipconv/conv_params.hpp"

#include <span>
#include <vector>

namespace hipconv
{

// A convolution algorithm for a specific hardware architecture.
//
// Holds an algorithm-level applicability function and a ConvKernelGroup containing the list of
// pointers to per-kernel-TU spans of ConvKernel instances. The indirection through pointers (rather
// than holding the spans by value) avoids a static-init ordering hazard: each ConvKernelGroup is
// itself dynamically initialized in its kernel TU, and capturing pointers postpones the read until
// get_valid_configs runs (after all dynamic initialization completes).
class ConvAlgorithm
{
public:
    using IsApplicableFn = bool (*)(const hipconv::ConvParams&);

    constexpr ConvAlgorithm(IsApplicableFn is_applicable_fn,
                            std::span<const ConvKernelGroup* const> kernel_groups)
        : is_applicable_(is_applicable_fn)
        , kernel_groups_(kernel_groups)
    {
    }

    // This algorithm's configs for `par`, best first, at most `max_ranked` of them.
    std::vector<ScoredKernel> get_valid_configs(const hipconv::Device& dev,
                                                const hipconv::ConvParams& par,
                                                std::size_t max_ranked) const
    {
        std::vector<ScoredKernel> result;
        if(!is_applicable_(par))
            return result;
        for(const ConvKernelGroup* group_ptr : kernel_groups_)
        {
            const auto& group = *group_ptr;
            if(group.kernels.empty())
                continue;
            // All kernels in a family share the same is_applicable and
            // supports_dims results.
            if(!group.kernels.front()->supports_dims(par.dims))
                continue;
            if(!group.kernels.front()->is_applicable(dev, par))
                continue;
            for(ConvKernel* kernel : group.kernels)
            {
                if(kernel->is_valid_config(dev, par))
                    result.push_back({kernel, kernel->get_weighted_throughput_index(dev, par)});
            }
        }
        keep_top_ranked(result, max_ranked);
        return result;
    }

    // The predicate get_valid_configs applies before asking any kernel.
    bool is_applicable(const hipconv::ConvParams& par) const { return is_applicable_(par); }

    // Every kernel of this algorithm, one or more groups per family.
    std::span<const ConvKernelGroup* const> kernel_groups() const { return kernel_groups_; }

private:
    IsApplicableFn is_applicable_;
    std::span<const ConvKernelGroup* const> kernel_groups_;
};

} // namespace hipconv
