#include "algorithm.h"
#include "conv_kernel_table.h"

#include <array>

using hipconv::Conv2dParams;
using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;

extern const ConvKernelSpan pointwise_fp16bf16_kernels;

namespace
{

// Coarse algorithm-level discriminator: pointwise handles 1x1 filters. The
// authoritative constraints (unit stride, zero pad, single group, dtype,
// layout, size limits) are enforced by PointwiseConvKernel::is_applicable.
bool is_applicable(const Conv2dParams& par)
{
    return par.kh == 1 && par.kw == 1;
}

constexpr std::array<const ConvKernelSpan*, 1> kernel_groups = {
    &pointwise_fp16bf16_kernels,
};

} // namespace

// The pointwise path is backed by hipBLASLt (arch-independent host code, no
// per-ISA kernels), so the same ConvAlgorithm is offered on every arch;
// hipBLASLt handles the actual hardware support at runtime. Named with external
// linkage so the generated arch registry can reference it in each arch's
// algorithm list. (A namespace-scope `const` has internal linkage by default,
// hence the explicit `extern`.)
//
// Host-only: ConvAlgorithm holds a host function pointer, so the device pass and
// device linker must not see this object (matches the arch backends' guard).
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm pointwise_algo{is_applicable, kernel_groups};
#endif
