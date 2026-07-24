#pragma once

// Boilerplate-eliminator for per-TU kernel tables. Each grouped/direct TU
// declares its own configs[] array, a per-config launch_impl<cfg> function
// template, and a Leaf class derived from ConvKernel. This header
// provides the templates that build, from those, the kernels[] array of
// concrete instances and the kernel_ptrs[] upcast array used by the per-arch
// backend. The macro at the bottom binds launch_impl (which can't be passed
// as a template argument, since it's a function template) at the TU's call
// site and forwards everything else to the templates below.

#include "conv_kernel.h"

#include <array>
#include <cstddef>
#include <iterator>
#include <span>
#include <utility>

namespace conv_kernel_table
{

template <typename KernelT, auto& Configs, typename LaunchFnGetter, size_t... Is>
constexpr std::array<KernelT, sizeof...(Is)> make_kernels_impl(LaunchFnGetter g,
                                                               std::index_sequence<Is...>)
{
    return { KernelT{Configs[Is], g.template operator()<Configs[Is]>()}... };
}

template <typename KernelT, auto& Configs, typename LaunchFnGetter>
constexpr auto make_kernels(LaunchFnGetter g)
{
    return make_kernels_impl<KernelT, Configs>(g, std::make_index_sequence<std::size(Configs)>{});
}

// Shard variant: build instances for the index range [Begin, End) of Configs,
// offsetting the local 0..(End-Begin) sequence by Begin so each shard captures
// &launch_impl<cfg> for its own slice. The returned array has End-Begin entries,
// in the same order as Configs. Used to spread one kernel family's device
// instantiations across several TUs that compile in parallel; an aggregator
// concatenates the per-shard spans back into one global-order span.
template <typename KernelT, auto& Configs, int Begin, typename LaunchFnGetter, size_t... Is>
constexpr std::array<KernelT, sizeof...(Is)> make_kernels_range_impl(LaunchFnGetter g,
                                                                     std::index_sequence<Is...>)
{
    return { KernelT{Configs[Begin + Is], g.template operator()<Configs[Begin + Is]>()}... };
}

template <typename KernelT, auto& Configs, int Begin, int End, typename LaunchFnGetter>
constexpr auto make_kernels_range(LaunchFnGetter g)
{
    return make_kernels_range_impl<KernelT, Configs, Begin>(
        g, std::make_index_sequence<End - Begin>{});
}

template <typename KernelT, size_t N, size_t... Is>
constexpr std::array<hipconv::ConvKernel*, N> make_kernel_ptrs_impl(std::array<KernelT, N>& kernels,
                                                                    std::index_sequence<Is...>)
{
    return {(&kernels[Is])...};
}

template <typename KernelT, size_t N>
constexpr auto make_kernel_ptrs(std::array<KernelT, N>& kernels)
{
    return make_kernel_ptrs_impl(kernels, std::make_index_sequence<N>{});
}

} // namespace conv_kernel_table

// Define `kernels` and `kernel_ptrs` for a kernel TU. Expects `configs` (a
// constexpr array) and `launch_impl<cfg>` (a function template) to be in scope.
#define HIPCONV_DEFINE_KERNEL_TABLE(LeafKernel)                                 \
    inline auto kernels = conv_kernel_table::make_kernels<LeafKernel, configs>( \
        []<auto cfg>() { return &launch_impl<cfg>; });                          \
    inline auto kernel_ptrs = conv_kernel_table::make_kernel_ptrs(kernels)

// Export a TU's kernel_ptrs as a std::span under an extern symbol that the
// per-arch backend can declare and iterate without knowing the TU's
// NUM_CONFIGS. Use at file scope, outside the TU's namespace.
#define HIPCONV_EXPORT_KERNEL_TABLE(ExportName, Namespace) \
    extern const hipconv::ConvKernelSpan ExportName = Namespace::kernel_ptrs

// Shard variant of HIPCONV_DEFINE_KERNEL_TABLE: build instances and pointers for
// the config index range [Begin, End) only, under shard-unique names
// kernels_##Id / kernel_ptrs_##Id. The plain `kernels` name cannot be reused
// across shards: it is `inline` in the same namespace, so multiple shards would
// collide under the one-definition rule. Expects `configs` and `launch_impl<cfg>`
// in scope. Each shard TU exports its kernel_ptrs_##Id with
// HIPCONV_EXPORT_KERNEL_TABLE_SYM; an aggregator TU concatenates the shard spans
// in Id order into the family's global-order span.
#define HIPCONV_DEFINE_KERNEL_TABLE_SHARD(LeafKernel, Id, Begin, End)               \
    inline auto kernels_##Id =                                                      \
        conv_kernel_table::make_kernels_range<LeafKernel, configs, (Begin), (End)>( \
            []<auto cfg>() { return &launch_impl<cfg>; });                          \
    inline auto kernel_ptrs_##Id = conv_kernel_table::make_kernel_ptrs(kernels_##Id)

// Export an arbitrary kernel_ptrs expression (e.g. a shard's kernel_ptrs_##Id)
// as an extern ConvKernelSpan symbol. Use at file scope, outside the namespace.
#define HIPCONV_EXPORT_KERNEL_TABLE_SYM(ExportName, PtrsExpr) \
    extern const hipconv::ConvKernelSpan ExportName = PtrsExpr
