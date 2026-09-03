// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Builds the per-peer kernarg groups for a fused GEMM.A2A launch. Depends on the
// Tensile ABI header alone.

#include <Tensile/FusedA2AKernArg.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <vector>

namespace rocblaslt
{
    // The drain bits this layer asks for. IN_KERNEL maps to DRAIN_RECV; DRAIN_SEND has no
    // attribute at this layer.
    constexpr uint32_t fusedA2ADrainFor(hipblasLtA2ACompletionMode_t mode)
    {
        return mode == HIPBLASLT_A2A_COMPLETION_IN_KERNEL ? TensileLite::FUSED_A2A_DRAIN_RECV : 0u;
    }

    // One group per rank in FUSED_A2A_PEER_FIELDS order. Fills the flag, recv and SDMA
    // queue slots. Any of the three sources may be null, leaving its slots null. Returns
    // an empty list when world does not fit the segment. The flag slot is biased to the
    // caller's channel within each peer's region.
    inline std::vector<TensileLite::FusedA2APeerFields>
        buildFusedA2APeerFields(void* const*                peerFlag,
                                void* const*                recvPtrs,
                                uint32_t                    world,
                                uint32_t                    channel,
                                const hipblasLtSdmaQueue_t* queues)
    {
        if(!TensileLite::fusedA2AWorldSizeValid(static_cast<int>(world)))
            return {};

        const size_t flagBias = size_t(channel) * TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES;

        std::vector<TensileLite::FusedA2APeerFields> peers(world);
        for(uint32_t j = 0; j < world; ++j)
        {
            peers[j][TensileLite::FUSED_A2A_SLOT_FLAG_PTR]
                = peerFlag != nullptr && peerFlag[j] != nullptr
                      ? static_cast<char*>(peerFlag[j]) + flagBias
                      : nullptr;
            peers[j][TensileLite::FUSED_A2A_SLOT_RECV_PTR]
                = recvPtrs != nullptr ? recvPtrs[j] : nullptr;
            if(queues != nullptr)
            {
                peers[j][TensileLite::FUSED_A2A_SLOT_QUEUE_BUF] = queues[j].queueBuf;
                peers[j][TensileLite::FUSED_A2A_SLOT_RPTR]      = queues[j].rptr;
                peers[j][TensileLite::FUSED_A2A_SLOT_WPTR]      = queues[j].wptr;
                peers[j][TensileLite::FUSED_A2A_SLOT_DOORBELL]  = queues[j].doorbell;
            }
        }
        return peers;
    }
}
