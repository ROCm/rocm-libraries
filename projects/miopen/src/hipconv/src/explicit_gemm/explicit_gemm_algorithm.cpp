#include "algorithm.h"
#include "conv_kernel_table.h"

#include <array>

using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;
using hipconv::ConvParams;

extern const ConvKernelSpan pointwise_kernels;
extern const ConvKernelSpan patch_embed_fp16bf16_kernels;

namespace
{

// Coarse algorithm-level discriminator: a 1x1 filter, or a patch embedding that
// reduces to one. The authoritative constraints (unit stride, zero pad, single
// group, dtype, layout, size limits) are enforced by the families'
// is_applicable.
bool is_applicable(const ConvParams& par)
{
    const bool unit_filter = par.kh == 1 && par.kw == 1;
    const bool patch =
        par.kh == par.stride_h && par.kw == par.stride_w && par.pad_h == 0 && par.pad_w == 0;
    return unit_filter || patch;
}

constexpr std::array<const ConvKernelSpan*, 2> kernel_groups = {
    &pointwise_kernels,
    &patch_embed_fp16bf16_kernels,
};

} // namespace

// The explicit-GEMM path is backed by hipBLASLt (arch-independent host code, no
// per-ISA kernels), so the same ConvAlgorithm is offered on every arch;
// hipBLASLt handles the actual hardware support at runtime. Named with external
// linkage so the generated arch registry can reference it in each arch's
// algorithm list. (A namespace-scope `const` has internal linkage by default,
// hence the explicit `extern`.)
//
// Host-only: ConvAlgorithm holds a host function pointer, so the device pass and
// device linker must not see this object (matches the arch backends' guard).
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm explicit_gemm_algo{is_applicable, kernel_groups};
#endif
