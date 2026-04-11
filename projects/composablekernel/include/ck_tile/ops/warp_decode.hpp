// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_policy.hpp"
#include "ck_tile/ops/warp_decode/kernel/warp_decode_gate_up_kernel.hpp"
#include "ck_tile/ops/warp_decode/kernel/warp_decode_down_reduce_kernel.hpp"

namespace ck_tile {

template <typename WarpDecodeGateUpKernel>
float launch_warp_decode_gate_up(const typename WarpDecodeGateUpKernel::Kargs& args,
                                 const stream_config& s)
{
    return launch_kernel(s,
                         WarpDecodeGateUpKernel{},
                         WarpDecodeGateUpKernel::GridSize(args),
                         WarpDecodeGateUpKernel::BlockSize(),
                         0,
                         WarpDecodeGateUpKernel::MakeKargs(args));
}

template <typename WarpDecodeDownReduceKernel>
float launch_warp_decode_down_reduce(const typename WarpDecodeDownReduceKernel::Kargs& args,
                                     const stream_config& s)
{
    return launch_kernel(s,
                         WarpDecodeDownReduceKernel{},
                         WarpDecodeDownReduceKernel::GridSize(args),
                         WarpDecodeDownReduceKernel::BlockSize(),
                         0,
                         WarpDecodeDownReduceKernel::MakeKargs(args));
}

} // namespace ck_tile
