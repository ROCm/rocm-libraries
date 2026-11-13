// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Stand-in for a FlyDSL decode megakernel with the same ABI, kdec(tab: i64, glob: i64,
// nlayers: i32). A real megakernel needs model weights behind `tab`; this one writes a
// pattern naming each thread and nlayers into the buffer `glob` points to, and copies
// tab[0] after it, so a test sees that the launch ran with the packed geometry and
// received both addresses and nlayers.
#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void megakernel_standin(int64_t tab, int64_t glob, int32_t nlayers)
{
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t n = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    auto* out = reinterpret_cast<uint64_t*>(glob);
    const auto* in = reinterpret_cast<const uint64_t*>(tab);
    out[i] = (0x4D4Bull << 48) | (static_cast<uint64_t>(static_cast<uint32_t>(nlayers)) << 32) | i;
    if(i == 0)
    {
        out[n] = in[0];
    }
}
