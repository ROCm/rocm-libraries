#include "conv_kernel_table.h"
#include "launch_params.h"
#include "explicit_gemm/hipblaslt_matmul.hpp"
#include "pointwise_conv_kernel.h"
#include "hipconv/conv_params.hpp"

#include <array>

namespace hipconv::pointwise_kernel
{

void launch_impl(const LaunchParams&,
                 const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void*,
                 hipStream_t stream)
{
    hipconv::explicit_gemm::launch_gemm(par, in, wei, out, stream);
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

    bool is_valid_config(const ConvParams&) const override { return true; }

    LaunchParams get_launch_params(const ConvParams&) const override { return LaunchParams{}; }
};

PointwiseConvKernelImpl kernel;
std::array<ConvKernel*, 1> kernel_ptrs = {&kernel};

} // namespace hipconv::pointwise_kernel

HIPCONV_EXPORT_KERNEL_TABLE(pointwise_kernels, hipconv::pointwise_kernel);
