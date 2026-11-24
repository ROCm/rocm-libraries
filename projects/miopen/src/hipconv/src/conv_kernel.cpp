#include "conv_kernel.h"

#include "hip_util.h"
#include "launch_params.h"
#include "hash.hpp"

namespace hipconv
{

RepresentationHashPair::RepresentationHashPair(std::string_view repr) : repr_{std::move(repr)}
{
    hash_ = murmur3_32(repr_);
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
