// direct_l1 kernel-table aggregator (host-only, no device code). The kernel
// body and its per-config device instantiations live in direct_l1/kernel.h,
// compiled in parallel across direct_l1/shard0..7.cpp so the large kernel no
// longer builds as one monolithic translation unit. Each shard exports a span
// over a contiguous slice of the configs[] table; this TU concatenates those
// spans, in shard-id order, into one global-order span exported as
// direct_l1_cdna4_kernels.
//
// Global order matters: direct_l1_test addresses a specific kernel as
// direct_l1_cdna4_kernels[config_index(...)], where config_index() is a
// position in the whole configs[] table (config_table.h). Because the shards
// partition [0, NUM_CONFIGS) into contiguous ascending ranges (shard_begin/
// shard_end), concatenating them in id order reproduces that global order.

#include "conv_kernel.h"
#include "conv_kernel_table.h"
#include "direct_l1/config_table.h"

#include <array>
#include <span>

using hipconv::ConvKernelSpan;

// Per-shard spans, each defined in its own TU (direct_l1/shardN.cpp) over the
// config index range [shard_begin(N), shard_end(N)).
extern const ConvKernelSpan direct_l1_cdna4_kernels_0;
extern const ConvKernelSpan direct_l1_cdna4_kernels_1;
extern const ConvKernelSpan direct_l1_cdna4_kernels_2;
extern const ConvKernelSpan direct_l1_cdna4_kernels_3;
extern const ConvKernelSpan direct_l1_cdna4_kernels_4;
extern const ConvKernelSpan direct_l1_cdna4_kernels_5;
extern const ConvKernelSpan direct_l1_cdna4_kernels_6;
extern const ConvKernelSpan direct_l1_cdna4_kernels_7;

namespace hipconv::cdna4::direct_l1
{
static_assert(kShardCount == 8,
              "aggregator wires exactly kShardCount shard spans; "
              "update both together if kShardCount changes");

// Flatten the shard spans into one contiguous global-order array. Dynamically
// initialized (it reads the shard spans), which is safe: each shard span is
// constant-initialized from a constexpr make_kernels_range result, so every span
// is ready before this runs, and the array is only read at dispatch/test time.
inline std::array<ConvKernel*, NUM_CONFIGS> all_kernel_ptrs = [] {
    std::array<ConvKernel*, NUM_CONFIGS> out{};
    const ConvKernelSpan* shards[kShardCount] = {
        &direct_l1_cdna4_kernels_0,
        &direct_l1_cdna4_kernels_1,
        &direct_l1_cdna4_kernels_2,
        &direct_l1_cdna4_kernels_3,
        &direct_l1_cdna4_kernels_4,
        &direct_l1_cdna4_kernels_5,
        &direct_l1_cdna4_kernels_6,
        &direct_l1_cdna4_kernels_7,
    };
    size_t pos = 0;
    for(const ConvKernelSpan* s : shards)
        for(ConvKernel* k : *s)
            out[pos++] = k;
    return out;
}();

} // namespace hipconv::cdna4::direct_l1

// The cdna4 direct backend declares this extern span and adds it to its
// kernel_groups; the unit test indexes it by global config_index().
HIPCONV_EXPORT_KERNEL_TABLE_SYM(direct_l1_cdna4_kernels,
                                hipconv::cdna4::direct_l1::all_kernel_ptrs);
