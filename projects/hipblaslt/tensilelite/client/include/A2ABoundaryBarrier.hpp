// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Boundary barrier for the fused GEMM.A2A multi-GPU arm: one single-thread
// kernel per rank, add / wait / subtract over an arrival counter that each
// rank owns and its peers increment.

#include <Tensile/FusedA2AKernArg.hpp> // FUSED_A2A_MAX_RANKS

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace TensileLite
{
    namespace Client
    {
        // A whole line for the one live u32, and the region is memset once at
        // allocation and never again.
        constexpr size_t FUSED_A2A_ARRIVAL_BYTES = 64;

        // Entry j is rank j's arrival counter, reachable from every rank once
        // peer access is on.
        struct A2AArrivalPeers
        {
            uint32_t* p[FUSED_A2A_MAX_RANKS] = {};
        };

        // world == 1 leaves need == 0: nothing to add, nothing to wait for.
        static __global__ void a2aBoundaryBarrierKernel(A2AArrivalPeers peers,
                                                        int             myRank,
                                                        int             world)
        {
            const uint32_t need = (uint32_t)(world - 1);

            for(int j = 0; j < world; j++)
            {
                if(j != myRank)
                    __hip_atomic_fetch_add(
                        peers.p[j], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }

            while(__hip_atomic_load(peers.p[myRank], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)
                  < need)
                ;

            __hip_atomic_fetch_sub(
                peers.p[myRank], need, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    } // namespace Client
} // namespace TensileLite
