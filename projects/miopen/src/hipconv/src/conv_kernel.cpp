#include "conv_kernel.h"

#include "arch_registry.h"
#include "hip_util.h"
#include "launch_params.h"
#include "hash.hpp"

#include <string>

namespace hipconv
{

RepresentationHashPair::RepresentationHashPair(std::string_view repr) : repr_{std::move(repr)}
{
    hash_ = murmur3_32(repr_);
}

std::size_t max_workgroups_yz(const hipconv::Device& dev)
{
    const auto name = dev.arch->name;
    if(name.starts_with("gfx1200") || name.starts_with("gfx1201"))
        return std::size_t{1} << 16;
    if(name.starts_with("gfx1250"))
        return (std::size_t{1} << 16) - 1;
    return no_workgroup_limit;
}

namespace
{

bool grid_yz_within(const dim3& grid, size_t limit)
{
    return grid.y <= limit && grid.z <= limit;
}

} // namespace

bool ConvKernel::grid_fits(const hipconv::Device& dev, const hipconv::ConvParams& par) const
{
    const auto limit = max_workgroups_yz(dev);
    return limit == no_workgroup_limit || grid_yz_within(get_launch_params(dev, par).grid, limit);
}

void ConvKernel::launch(const LaunchParams& lp,
                        const hipconv::Device& dev,
                        const hipconv::ConvParams& par,
                        const void* in,
                        const void* wei,
                        void* out,
                        void* workspace,
                        hipStream_t stream) const
{
    if(!grid_yz_within(lp.grid, max_workgroups_yz(dev)))
        throw HipError(hipErrorInvalidConfiguration,
                       std::string{name()} + ": grid Y/Z exceeds the workgroup-count limit of " +
                           std::string{dev.arch->name});
    launch_fn_(lp, dev, par, in, wei, out, workspace, stream);
    // Kernel launches have no return value.
    // Read the launch-time status right after the launch and throw on failure.
    HIP_CHECK(hipGetLastError());
}

bool ConvKernel::matches_descriptor(std::string_view spec, std::string* error) const
{
    KVDescriptor descriptor = config_descriptor();
    if(descriptor.match(spec))
        return true;
    if(error)
    {
        // With no fields every key is unknown, so name the cause instead.
        *error = descriptor.describe(/*include_defaults=*/true).empty()
                     ? "kernel '" + std::string(name()) + "' has no descriptor fields"
                     : descriptor.error();
    }
    return false;
}

} // namespace hipconv
