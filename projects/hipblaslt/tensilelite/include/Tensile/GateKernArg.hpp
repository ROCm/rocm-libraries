// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host side of the triggered-gate kernarg tail (TriggeredGateEpilogue). Must stay
// byte-identical with tensilelite/Components/Signature.py: GateView (8-byte
// global pointer to the gate view) at +0, GateTileBase (u32) at +8.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace TensileLite
{
    constexpr size_t GATE_KERNARG_VIEW_OFFSET      = 0;
    constexpr size_t GATE_KERNARG_TILE_BASE_OFFSET = 8;

    // Gate table at GateView (gate-epilogue ABI, version 2). The kernel checks
    // abiVersion, reserved, tileCount, tileId and the PUBLISH bit, then adds 1 at
    // heapBase + readyOffset of record GateTileBase + WorkGroup1.
    constexpr uint32_t GATE_ABI_VERSION  = 2;
    constexpr uint32_t GATE_ROLE_PUBLISH = 1u;

    struct GateTableHeader
    {
        uint32_t abiVersion;
        uint32_t structSize;
        uint32_t tileRecordSize;
        uint32_t tileCount;
        uint32_t rank;
        uint32_t reserved;
        uint64_t epoch;
        uint64_t heapBase;
        uint64_t tiles; // device address of the first record
    };

    struct GateTileRecord
    {
        uint32_t tileId;
        uint32_t channel;
        uint32_t roleFlags;
        uint32_t publisherCount;
        uint64_t readyOffset;
        uint64_t arrivalOffset;
        uint64_t completionOffset;
        uint64_t errorOffset;
    };

    static_assert(sizeof(GateTableHeader) == 48 && offsetof(GateTableHeader, reserved) == 20
                      && offsetof(GateTableHeader, heapBase) == 32
                      && offsetof(GateTableHeader, tiles) == 40,
                  "gate table header layout");
    static_assert(sizeof(GateTileRecord) == 48 && offsetof(GateTileRecord, roleFlags) == 8
                      && offsetof(GateTileRecord, readyOffset) == 16,
                  "gate tile record layout");

    template <typename KA>
    inline void appendGateSegment(KA& args, void const* gateView, uint32_t gateTileBase)
    {
        args.template appendAligned<void const*>("GateView", gateView);
        size_t base = args.size() - sizeof(void const*);
        args.template append<uint32_t>("GateTileBase", gateTileBase);
        if(args.size() - base != GATE_KERNARG_TILE_BASE_OFFSET + sizeof(uint32_t))
            throw std::runtime_error("[gate] GateTileBase is not at GateView + 8");
    }
} // namespace TensileLite
