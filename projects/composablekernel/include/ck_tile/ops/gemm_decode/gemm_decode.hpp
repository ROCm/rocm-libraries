// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/gemm_decode/pipeline/gemm_decode_problem.hpp"
#include "ck_tile/ops/gemm_decode/pipeline/gemm_decode_policy.hpp"
#include "ck_tile/ops/gemm_decode/kernel/gemm_decode_universal_kernel.hpp"

namespace ck_tile {

// Host-side launch wrapper for the warp-per-scalar dense GEMM kernel.
// Mirrors the shape of `launch_warp_decode_gate_up` from ops/warp_decode.hpp:
// validate via IsSupportedArgument (throw on rejection), then dispatch the
// device kernel through the standard CK Tile launch path. When k_batch > 1
// the caller is responsible for zeroing the C buffer before invocation;
// `launch_kernel` does not allocate scratch on its own.
template <typename Kernel>
float launch_gemm_decode_universal(const typename Kernel::Kargs& args, const stream_config& s)
{
    if(!Kernel::IsSupportedArgument(args))
    {
        throw std::invalid_argument("GemmDecodeUniversalKernel arguments are not supported.");
    }

    return launch_kernel(s,
                         make_kernel(Kernel{}, Kernel::GridSize(args), Kernel::BlockSize(), 0, args));
}

} // namespace ck_tile
