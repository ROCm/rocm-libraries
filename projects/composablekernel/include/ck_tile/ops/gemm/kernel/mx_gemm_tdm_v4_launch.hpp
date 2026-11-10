// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <stdexcept>
#include <string>

#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_tdm_v4_kernel.hpp"

namespace ck_tile {

/**
 * @brief Launches a MxGemmTdmV4Kernel instance on `args`: the grid of GridSize(), 128 threads,
 *        and with a cluster tuning the cluster of ClusterSize() (the kernel's multicast masks and
 *        GL2 prefetch split assume it). Returns launch_kernel's time.
 *
 * @throws std::runtime_error if the instance does not run `args` (GetUnsupportedReason) or the
 *         build has no cluster launch.
 */
template <typename Kernel>
CK_TILE_HOST float launch_mx_gemm_tdm_v4(const stream_config& s, const typename Kernel::Args& args)
{
    if(const char* why = Kernel::GetUnsupportedReason(args))
        throw std::runtime_error(Kernel::GetName() + ": " + why);
    const dim3 grid  = Kernel::GridSize(args.M, args.N);
    const dim3 block = Kernel::BlockSize();
    if constexpr(Kernel::kHasCluster)
    {
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
        return launch_kernel(s,
                             make_kernel<1>(Kernel{}, Kernel::ClusterSize(), grid, block, 0, args));
#else
        throw std::runtime_error(Kernel::GetName() + ": this build has no cluster launch");
#endif
    }
    else
    {
        return launch_kernel(s, make_kernel<1>(Kernel{}, grid, block, 0, args));
    }
}

} // namespace ck_tile
