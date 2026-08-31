// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CDNA5 hardware hazard rules shared between CDNA5ReadyQueue (DAG scheduler)
// and HazardGapAnalysisPass. Keep this header free of CDNA5ReadyQueue internals.

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {

// Which operand of the producer opens the gap, and which of the consumer closes it.
enum class HazardDir {
    WriteThenRead,  // producer writes a reg, consumer reads it (RAW-shaped)
    ReadThenWrite,  // producer reads a reg, consumer overwrites it (WAR-shaped)
};

// What the gap is counted in, which decides whether elapsing time can pay it.
enum class HazardUnit {
    Cycles,   // decays in advanceTime; reports a wait the scheduler may pay
    PipeOps,  // advances only when an isPipeOp instruction issues; not payable by time
};

struct HazardRule {
    const char* name;
    bool (*isProducer)(const StinkyInstruction&);
    bool (*isConsumer)(const StinkyInstruction&);
    RegType regType;
    // Cycles rules: the gap itself. PipeOps rules: 0 means the arch policy supplies it.
    int distance;
    HazardDir dir;
    HazardUnit unit;
    // PipeOps rules only: what advances the counter. Null for Cycles rules.
    bool (*isPipeOp)(const StinkyInstruction&);
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
    {"SaluSgprToMemAddr", isScalarALU, isSaluHazardConsumer, RegType::S, 8,
     HazardDir::WriteThenRead, HazardUnit::Cycles, nullptr},
    {"ValuVgprToVmemAddr", isVectorALU, isVmemAddrHazardConsumer, RegType::V, 32,
     HazardDir::WriteThenRead, HazardUnit::Cycles, nullptr},
    // mode2 WAR: a WMMA reads a vgpr, a later ds_load overwrites it. The gap is the
    // va_vdst follower count, which only grows when another matrix op issues -- elapsing
    // time cannot pay it, hence PipeOps. Distance 0: CDNA5Config supplies it per arch.
    {"WmmaVgprSrcToDsWrite", isMatrixInstruction, isDSRead, RegType::V, 0, HazardDir::ReadThenWrite,
     HazardUnit::PipeOps, isMatrixInstruction},
};
inline constexpr int kNumCdna5HazardRules =
    static_cast<int>(sizeof(kCdna5HazardRules) / sizeof(kCdna5HazardRules[0]));

}  // namespace stinkytofu
