// direct_l1 kernel shard 0 of kShardCount.
//
// Instantiates the GPU kernel template for a subset of its configurations.
// Multiple shard files can be compiled in parallel, speeding up compilation time.
//
// The kernel body is in kernel.h. The thin host-only direct_l1.cpp (one level
// up) concatenates the shard spans back into one global-order kernel table. See
// that aggregator and HIPCONV_DEFINE_KERNEL_TABLE_SHARD for the contract.

#include "kernel.h"

namespace hipconv::cdna4::direct_l1
{
HIPCONV_DEFINE_KERNEL_TABLE_SHARD(DirectL1_ConvKernel, 0, shard_begin(0), shard_end(0));
} // namespace hipconv::cdna4::direct_l1

HIPCONV_EXPORT_KERNEL_TABLE_SYM(direct_l1_cdna4_kernels_0,
                                hipconv::cdna4::direct_l1::kernel_ptrs_0);
