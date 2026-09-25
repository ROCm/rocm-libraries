// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CDNA5 hardware hazard rules shared between CDNA5ReadyQueue (DAG scheduler)
// and HazardGapAnalysisPass. Keep this header free of CDNA5ReadyQueue internals.

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {

struct HazardRule {
    const char* name;
    bool (*isProducer)(const StinkyInstruction&);
    bool (*isConsumer)(const StinkyInstruction&);
    RegType regType;
    int cycles;
};

// SALU sgpr -> any SMEM/tensor_load/VMEM address consumer.
inline bool isSaluHazardConsumer(const StinkyInstruction& inst) {
    return isGlobalMemLoad(inst) || isTensorLoad(inst);
}

// VALU vgpr -> VMEM address consumer (global_read / MUBUF / FLAT / global_prefetch).
inline bool isVmemAddrHazardConsumer(const StinkyInstruction& inst) {
    return isBufferMemLoad(inst) || isGlobalPrefetch(inst);
}

inline constexpr HazardRule kCdna5HazardRules[] = {
    {"SaluSgprToMemAddr", isScalarALU, isSaluHazardConsumer, RegType::S, 8},
    {"ValuVgprToVmemAddr", isVectorALU, isVmemAddrHazardConsumer, RegType::V, 32},
};
inline constexpr int kNumCdna5HazardRules =
    static_cast<int>(sizeof(kCdna5HazardRules) / sizeof(kCdna5HazardRules[0]));

}  // namespace stinkytofu
