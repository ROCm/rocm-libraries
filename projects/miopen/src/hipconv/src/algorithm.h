#pragma once

#include "conv_kernel.h"
#include "hipconv/conv2d_params.hpp"

#include <span>
#include <vector>

namespace hipconv
{

// A convolution algorithm for a specific hardware architecture.
//
// Holds an algorithm-level applicability function and a list of pointers
// to per-kernel-TU spans of ConvKernel instances. The indirection through
// pointers (rather than holding the spans by value) avoids a static-init
// ordering hazard: each ConvKernelSpan is itself dynamically initialized
// in its kernel TU, and capturing pointers postpones the read until
// get_valid_configs runs (after all dynamic initialization completes).
class ConvAlgorithm
{
public:
    using IsApplicableFn = bool (*)(const hipconv::Conv2dParams&);

    constexpr ConvAlgorithm(IsApplicableFn is_applicable_fn,
                            std::span<const ConvKernelSpan* const> kernel_groups)
        : is_applicable_(is_applicable_fn)
        , kernel_groups_(kernel_groups)
    {
    }

    std::vector<ConvKernel*> get_valid_configs(const hipconv::Conv2dParams& par) const
    {
        std::vector<ConvKernel*> result;
        if(!is_applicable_(par))
            return result;
        for(const ConvKernelSpan* group_ptr : kernel_groups_)
        {
            const auto& group = *group_ptr;
            if(group.empty())
                continue;
            // All kernels in a family share the same is_applicable result.
            if(!group.front()->is_applicable(par))
                continue;
            for(auto* k : group)
            {
                if(k->is_valid_config(par))
                    result.push_back(k);
            }
        }
        return result;
    }

private:
    IsApplicableFn is_applicable_;
    std::span<const ConvKernelSpan* const> kernel_groups_;
};

} // namespace hipconv
