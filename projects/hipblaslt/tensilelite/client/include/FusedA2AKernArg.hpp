// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host side of the fused GEMM.A2A kernarg segment ABI. Must stay
// byte-identical with the kernel side (Tensile/Components/Signature.py
// fusedA2AKernArgLayout + addArg).

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Tensile/KernelArguments.hpp>

namespace TensileLite
{
    namespace Client
    {
        // Compile-time slot count for the fused-A2A kernarg segment; must match
        // FUSED_A2A_MAX_RANKS in Tensile/Components/Signature.py. The host
        // always appends 8 peer_ptr slots (unused slots j>=W filled with
        // nullptr), regardless of the runtime world size. 8 is the world size
        // this ABI is built for, not a placeholder: raising it grows the segment
        // and deepens the unrolled per-rank scans in GlobalWriteBatch.py.
        constexpr int FUSED_A2A_MAX_RANKS = 8;

        // Byte offset of recv inside a peer block. Mirrored in Tensile/Components/Signature.py.
        constexpr size_t FUSED_A2A_PEER_RECV_OFFSET = 4096;
        static_assert(FUSED_A2A_MAX_RANKS * sizeof(uint32_t) <= FUSED_A2A_PEER_RECV_OFFSET,
                      "flag array must fit below the recv offset inside a peer block");

        // FUSED_A2A_MAX_RANKS must fit within what the DRAIN barrier's EXEC
        // mask (Tensile/Components/GlobalWriteBatch.py _emitFusedA2AHandshake)
        // can encode; twin-checked at Signature.py's FUSED_A2A_MAX_RANKS.
        static_assert(FUSED_A2A_MAX_RANKS <= 31,
                      "FUSED_A2A_MAX_RANKS exceeds the 31 the DRAIN EXEC mask can encode: "
                      "the S_BFM width operand is 5 bits on the wave32 arm and 6 on the "
                      "wave64 arm, so a world size of 32 (resp. 64) wraps the width to 0, "
                      "EXEC becomes empty, the DRAIN poll never issues, and the barrier is "
                      "silently skipped -- the epilogue then reads peer tiles that have not "
                      "arrived. That is a wrong answer, not a hang. Re-derive the mask "
                      "before raising this.");

        // Expected byte growth of args after appending the fused segment:
        //   (MAX_RANKS peer + 1 counter + 1 FusedSdmaQueues) pointers * 8B
        //   + 4 scalars * 4B = 96B.
        constexpr size_t FUSED_A2A_SEGMENT_BYTES = (FUSED_A2A_MAX_RANKS + 2) * 8 + 4 * 4;

        // Whether worldSize fits the fixed segment above: ranks >=
        // FUSED_A2A_MAX_RANKS have no peer_ptr slot, and worldSize <= 0 would
        // later be used as a divisor.
        constexpr bool fusedA2AWorldSizeValid(int worldSize)
        {
            return worldSize >= 1 && worldSize <= FUSED_A2A_MAX_RANKS;
        }

        // Append the fixed-size fused-A2A kernarg segment to `args`, in the
        // exact emission order of Signature.py fusedA2AKernArgLayout().
        // peer_ptr_0 is appendAligned<void*> to land on an 8-byte boundary,
        // mirroring the kernel metadata; peerPtrs may be shorter than
        // FUSED_A2A_MAX_RANKS, with the remaining slots filled with nullptr.
        inline void appendFusedSegment(
            KernelArguments&          args,
            std::vector<void*> const& peerPtrs, // size W (device d's per-peer block bases)
            void*                     counterPtr,
            void*                     sdmaQueues, // W-element SdmaQueueDeviceHandle array
            uint32_t                  myRank,
            uint32_t                  worldSize,
            uint32_t                  drain,
            uint32_t                  am)
        {
            size_t before = args.size();

            for(int j = 0; j < FUSED_A2A_MAX_RANKS; j++)
            {
                void* p = (j < (int)peerPtrs.size()) ? peerPtrs[j] : nullptr;
                if(j == 0)
                    args.appendAligned<void*>("peer_ptr_0", p);
                else
                    args.append<void*>("peer_ptr_" + std::to_string(j), p);
            }
            args.append<void*>("counter_ptr", counterPtr);
            args.append<void*>("FusedSdmaQueues", sdmaQueues);
            args.append<uint32_t>("FusedMyRank", myRank);
            args.append<uint32_t>("FusedW", worldSize);
            args.append<uint32_t>("FusedDrain", drain);
            args.append<uint32_t>("FusedAM", am);

            size_t grew = args.size() - before;
            if(grew != FUSED_A2A_SEGMENT_BYTES)
            {
                throw std::runtime_error(
                    "[fused-a2a] fused segment grew args by " + std::to_string(grew)
                    + " bytes, expected " + std::to_string(FUSED_A2A_SEGMENT_BYTES)
                    + " (alignment/padding mismatch; the epilogue would read wrong offsets)");
            }
        }
    } // namespace Client
} // namespace TensileLite
