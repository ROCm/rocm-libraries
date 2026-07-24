// direct_l1 kernel shard 4 of kShardCount. See shard0.cpp for the shard
// contract and direct_l1.cpp for the aggregator.

#include "kernel.h"

namespace hipconv::cdna4::direct_l1
{
HIPCONV_DEFINE_KERNEL_TABLE_SHARD(DirectL1_ConvKernel, 4, shard_begin(4), shard_end(4));
} // namespace hipconv::cdna4::direct_l1

HIPCONV_EXPORT_KERNEL_TABLE_SYM(direct_l1_cdna4_kernels_4,
                                hipconv::cdna4::direct_l1::kernel_ptrs_4);
