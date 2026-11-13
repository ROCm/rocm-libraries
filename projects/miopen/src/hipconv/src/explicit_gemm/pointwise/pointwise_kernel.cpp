#include "launch_params.h"
#include "explicit_gemm/hipblaslt_matmul.hpp"
#include "pointwise_conv_kernel.h"
#include "hipconv/conv_params.hpp"

#include <array>

namespace hipconv::pointwise_kernel
{

void launch_impl(const LaunchParams& lp,
                 const Device&,
                 const ConvParams&,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    hipconv::explicit_gemm::launch_gemm(*lp.gemm_plan, in, wei, out, workspace, stream);
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

    bool is_valid_config(const Device& /*dev*/, const ConvParams&) const override { return true; }

    LaunchParams get_launch_params(const Device& /*dev*/, const ConvParams& par) const override
    {
        LaunchParams lp;
        lp.gemm_plan = explicit_gemm::make_gemm_plan(par);
        return lp;
    }
};

PointwiseConvKernelImpl kernel;
std::array<ConvKernel*, 1> kernel_ptrs = {&kernel};

// Only a single kernel in the pointwise family with empty config representation
auto get_kernel_by_config(RepresentationHashPair const& cfg) -> hipconv::ConvKernel*
{
    return cfg.repr().empty() ? &kernel : nullptr;
}

} // namespace hipconv::pointwise_kernel

extern const hipconv::ConvKernelGroup pointwise_kernels = {
    hipconv::pointwise_kernel::kernel_ptrs,
    hipconv::pointwise_kernel::get_kernel_by_config};
