// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef STINKYTOFU_GFX1250_HAZARD_PROFILE
#define STINKYTOFU_GFX1250_HAZARD_PROFILE 0
#endif

#if STINKYTOFU_GFX1250_HAZARD_PROFILE
#include <cstdint>
#include <iostream>
#include <string>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/IRBase.hpp"
#endif

namespace stinkytofu {

class IRBase;
class StinkyAsmModule;

enum class XcntDrainReason {
    AtomicRule4a,
    SmemRule3,
    FlatRule2,
    ForeverSleep,
    ScalarPrefetch,
    VgprMsb,
};

#if STINKYTOFU_GFX1250_HAZARD_PROFILE
class XcntDrainProfile {
   public:
    explicit XcntDrainProfile(const StinkyAsmModule* module) : module(module) {}

    void noteTensorLoad() {
        usesTensorLoad = true;
    }

    void record(XcntDrainReason reason, const IRBase* anchor) {
        ++total;
        switch (reason) {
            case XcntDrainReason::AtomicRule4a:
                ++atomicRule4a;
                break;
            case XcntDrainReason::SmemRule3:
                ++smemRule3;
                break;
            case XcntDrainReason::FlatRule2:
                ++flatRule2;
                break;
            case XcntDrainReason::ForeverSleep:
                ++foreverSleep;
                break;
            case XcntDrainReason::ScalarPrefetch:
                ++scalarPrefetch;
                break;
            case XcntDrainReason::VgprMsb:
                ++vgprMsb;
                break;
        }

        if (isInGroup("loopWithPrefetch", anchor)) {
            ++loopWithPrefetch;
        } else if (isInGroup("noLoadLoopBody", anchor)) {
            ++noLoadLoopBody;
        } else {
            ++outsideRegions;
        }
    }

    void print() const {
        std::cerr << "[Gfx1250HazardPass] xcnt drains: total=" << total
                  << ", loopWithPrefetch=" << loopWithPrefetch
                  << ", noLoadLoopBody=" << noLoadLoopBody << ", outsideRegions=" << outsideRegions
                  << "\n";
        std::cerr << "[Gfx1250HazardPass] xcnt drain rules: atomic=" << atomicRule4a
                  << ", smem=" << smemRule3 << ", flat=" << flatRule2
                  << ", foreverSleep=" << foreverSleep << ", scalarPrefetch=" << scalarPrefetch
                  << ", vgprMsb=" << vgprMsb << "\n";
        std::cerr << "[Gfx1250HazardPass] tensor_load_to_lds: "
                  << (usesTensorLoad ? "used" : "not used") << "\n";
    }

   private:
    bool isInGroup(const std::string& name, const IRBase* anchor) const {
        if (module == nullptr) return false;
        auto range = module->findGroupRange(name);
        if (!range) return false;
        for (auto it = range->first; it != range->second; ++it)
            if (it.getNodePtr() == anchor) return true;
        return false;
    }

    const StinkyAsmModule* module = nullptr;
    uint64_t total = 0;
    uint64_t atomicRule4a = 0;
    uint64_t smemRule3 = 0;
    uint64_t flatRule2 = 0;
    uint64_t foreverSleep = 0;
    uint64_t scalarPrefetch = 0;
    uint64_t vgprMsb = 0;
    uint64_t loopWithPrefetch = 0;
    uint64_t noLoadLoopBody = 0;
    uint64_t outsideRegions = 0;
    bool usesTensorLoad = false;
};
#else
class XcntDrainProfile {
   public:
    explicit XcntDrainProfile(const StinkyAsmModule* /*module*/) {}
    void noteTensorLoad() {}
    void record(XcntDrainReason /*reason*/, const IRBase* /*anchor*/) {}
    void print() const {}
};
#endif

}  // namespace stinkytofu
