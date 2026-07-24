#include "conv_kernel_table.h"
#include "launch_params.h"
#include "pointwise/hipblaslt_matmul.hpp"
#include "pointwise_conv_kernel.h"
#include "hipconv/conv2d_params.hpp"

#include <array>

namespace hipconv::pointwise_fp16bf16
{

void launch_impl(const LaunchParams&,
                 const Conv2dParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void*,
                 hipStream_t stream)
{
    hipconv::pointwise::launch_pointwise_gemm(par, in, wei, out, stream);
}

// The pointwise family has a single, stateless implementation: the GEMM
// dispatches on `par` (direction and dtype) at runtime through hipBLASLt, so
// there is no per-config compile-time state to enumerate. Family-level
// applicability (dtype/layout/1x1/stride/pad/...) is enforced by
// PointwiseConvKernel::is_applicable, which the dispatcher checks before
// is_valid_config, so the latter is trivially true.
class PointwiseConvKernelImpl : public PointwiseConvKernel
{
public:
    constexpr PointwiseConvKernelImpl() : PointwiseConvKernel(&launch_impl) {}

    bool is_valid_config(const Conv2dParams&) const override { return true; }

    LaunchParams get_launch_params(const Conv2dParams&) const override { return LaunchParams{}; }
};

PointwiseConvKernelImpl kernel;
std::array<ConvKernel*, 1> kernel_ptrs = {&kernel};

} // namespace hipconv::pointwise_fp16bf16

HIPCONV_EXPORT_KERNEL_TABLE(pointwise_fp16bf16_kernels, hipconv::pointwise_fp16bf16);
