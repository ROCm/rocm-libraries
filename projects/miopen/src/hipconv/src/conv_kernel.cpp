#include "conv_kernel.h"

#include "hip_util.h"
#include "launch_params.h"

namespace hipconv
{

void ConvKernel::launch(const LaunchParams& lp,
                        const hipconv::Conv2dParams& par,
                        const void* in,
                        const void* wei,
                        void* out,
                        void* workspace,
                        hipStream_t stream) const
{
    launch_fn_(lp, par, in, wei, out, workspace, stream);
    // Kernel launches have no return value.
    // Read the launch-time status right after the launch and throw on failure.
    HIP_CHECK(hipGetLastError());
}

} // namespace hipconv
