// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host side of the fused GEMM.A2A kernarg segment ABI. Must stay
// byte-identical with the kernel side (Tensile/Components/Signature.py
// fusedA2AKernArgLayout + addArg).

#include <array>
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
        // always appends 8 peer groups (unused groups j>=W filled with
        // nullptr), regardless of the runtime world size. 8 is the world size
        // this ABI is built for, not a placeholder: raising it grows the segment
        // and deepens the unrolled per-rank scans in GlobalWriteBatch.py.
        constexpr int FUSED_A2A_MAX_RANKS = 8;

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

        constexpr size_t FUSED_A2A_LINE_BYTES = 64;

        constexpr size_t fusedA2AAlignLine(size_t nbytes)
        {
            return (nbytes + FUSED_A2A_LINE_BYTES - 1) / FUSED_A2A_LINE_BYTES
                   * FUSED_A2A_LINE_BYTES;
        }

        // Twinned with their namesakes in Tensile/Components/Signature.py.
        constexpr size_t FUSED_A2A_OUTBOUND_OFFSET
            = fusedA2AAlignLine((size_t)FUSED_A2A_MAX_RANKS * sizeof(uint32_t));
        constexpr size_t FUSED_A2A_FLAG_BLOCK_BYTES
            = fusedA2AAlignLine(FUSED_A2A_OUTBOUND_OFFSET + sizeof(uint32_t));
        constexpr uint32_t FUSED_A2A_DRAIN_RECV = 1u;
        constexpr uint32_t FUSED_A2A_DRAIN_SEND = 2u;

        // One group per peer: flag base, recv base, then the queue pointers.
        // Order is the contract with SdmaRingEmitter.py FUSED_A2A_PEER_FIELDS.
        constexpr size_t FUSED_A2A_PEER_QUEUE_FIELDS = 4;
        constexpr size_t FUSED_A2A_PEER_FIELDS_PER_RANK = 2 + FUSED_A2A_PEER_QUEUE_FIELDS;

        // (MAX_RANKS groups + 1 counter) pointers * 8B + 4 scalars * 4B.
        constexpr size_t FUSED_A2A_SEGMENT_BYTES
            = (FUSED_A2A_MAX_RANKS * FUSED_A2A_PEER_FIELDS_PER_RANK + 1) * 8 + 4 * 4;

        // Whether worldSize fits the fixed segment above: ranks >=
        // FUSED_A2A_MAX_RANKS have no peer group, and worldSize <= 0 would
        // later be used as a divisor.
        constexpr bool fusedA2AWorldSizeValid(int worldSize)
        {
            return worldSize >= 1 && worldSize <= FUSED_A2A_MAX_RANKS;
        }

        // One peer's kernarg group, in FUSED_A2A_PEER_FIELDS order:
        // flagPtr, recvPtr, queueBuf, rptr, wptr, doorbell.
        using FusedA2APeerFields = std::array<void*, FUSED_A2A_PEER_FIELDS_PER_RANK>;

        // Append the fixed-size fused-A2A kernarg segment to `args`, in the
        // exact emission order of Signature.py fusedA2AKernArgLayout().
        // peer_0_flagPtr is appendAligned<void*> to land on an 8-byte boundary,
        // mirroring the kernel metadata; `peers` may be shorter than
        // FUSED_A2A_MAX_RANKS, with the remaining groups filled with nullptr.
        // flag and recv carry no fixed offset from each other.
        inline void appendFusedSegment(
            KernelArguments&                      args,
            std::vector<FusedA2APeerFields> const& peers, // size W, in rank order
            void*                                 counterPtr,
            uint32_t                              myRank,
            uint32_t                              worldSize,
            uint32_t                              drain,
            uint32_t                              am)
        {
            size_t before = args.size();

            static constexpr const char* kField[FUSED_A2A_PEER_FIELDS_PER_RANK]
                = {"flagPtr", "recvPtr", "queueBuf", "rptr", "wptr", "doorbell"};
            static const FusedA2APeerFields kAbsent{};
            for(int j = 0; j < FUSED_A2A_MAX_RANKS; j++)
            {
                const std::string        g = "peer_" + std::to_string(j) + "_";
                FusedA2APeerFields const& f = j < (int)peers.size() ? peers[j] : kAbsent;
                if(j == 0)
                    args.appendAligned<void*>(g + kField[0], f[0]);
                else
                    args.append<void*>(g + kField[0], f[0]);
                for(size_t k = 1; k < FUSED_A2A_PEER_FIELDS_PER_RANK; k++)
                    args.append<void*>(g + kField[k], f[k]);
            }
            args.append<void*>("counter_ptr", counterPtr);
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
