/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/**
 * @file logical_count.cpp
 * @brief Python bindings for counting Logical IR instructions
 *
 * Ported from rocisa's count.cpp, adapted for StinkyTofu's enum-based design.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <unordered_map>

#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/ir/logical/LogicalOpcode.hpp"

namespace nb = nanobind;

namespace stinkytofu {
namespace logical {
namespace {

// ============================================================================
// Instruction category checks (using InstFlag - same as Assembly IR)
// ============================================================================
//
// Note: Now that LogicalInstruction has InstFlagSet, we can reuse the same
// flag-based categorization as Assembly IR. This provides a unified API across
// both IR layers and avoids manual opcode enumeration.
// ============================================================================

bool isGlobalRead(const std::shared_ptr<LogicalInstruction>& inst) {
    if (!inst) return false;
    // Global reads: MUBUF loads, FLAT loads, GLOBAL loads, scalar mem loads
    return inst->is(IF_MUBUFLoad) || inst->is(IF_FLATLoad) || inst->is(IF_GLOBALLoad) ||
           inst->is(IF_SMemLoad);
}

bool isSMemLoad(const std::shared_ptr<LogicalInstruction>& inst) {
    if (!inst) return false;
    return inst->is(IF_SMemLoad);
}

bool isLocalRead(const std::shared_ptr<LogicalInstruction>& inst) {
    if (!inst) return false;
    return inst->is(IF_DSRead);
}

bool isLocalWrite(const std::shared_ptr<LogicalInstruction>& inst) {
    if (!inst) return false;
    return inst->is(IF_DSStore);
}

bool isGlobalWrite(const std::shared_ptr<LogicalInstruction>& inst) {
    if (!inst) return false;
    // Global writes: MUBUF stores, FLAT stores, GLOBAL stores, scalar mem stores
    return inst->is(IF_MUBUFStore) || inst->is(IF_FLATStore) || inst->is(IF_GLOBALStore) ||
           inst->is(IF_SMemStore);
}

bool isMFMA(const std::shared_ptr<LogicalInstruction>& inst) {
    if (!inst) return false;
    return inst->is(IF_MFMA) || inst->is(IF_WMMA) || inst->is(IF_SMFMA) || inst->is(IF_SWMMA);
}

// ============================================================================
// Counting functions for PyLogicalModule
// ============================================================================

int countInstructions(const PyLogicalModule& module) {
    return static_cast<int>(module.getInstructions().size());
}

template <typename Predicate>
int countIf(const PyLogicalModule& module, Predicate predicate) {
    int count = 0;
    for (const auto& inst : module.getInstructions()) {
        if (predicate(inst)) {
            count++;
        }
    }
    return count;
}

int countGlobalRead(const PyLogicalModule& module) {
    return countIf(module, isGlobalRead);
}

int countSMemLoad(const PyLogicalModule& module) {
    return countIf(module, isSMemLoad);
}

int countLocalRead(const PyLogicalModule& module) {
    return countIf(module, isLocalRead);
}

int countLocalWrite(const PyLogicalModule& module) {
    return countIf(module, isLocalWrite);
}

int countGlobalWrite(const PyLogicalModule& module) {
    return countIf(module, isGlobalWrite);
}

int countMFMA(const PyLogicalModule& module) {
    return countIf(module, isMFMA);
}

// Weighted counting for composite instructions
int countWeightedLocalRead(const PyLogicalModule& module) {
    // Note: If certain LDS loads are composite (e.g., expand to 2 instructions),
    // add them to the weights map. Example:
    // std::unordered_map<logical::Opcode, int> weights = {
    //     {Opcode::SomeCompositeLoad, 2},
    // };

    // For now, just count normally (weight = 1 for all)
    return countLocalRead(module);
}

int countWeightedLocalWrite(const PyLogicalModule& module) {
    // Note: If certain LDS stores are composite (e.g., expand to 2 instructions),
    // add them to the weights map. Example:
    // std::unordered_map<logical::Opcode, int> weights = {
    //     {Opcode::SomeCompositeStore, 2},
    // };

    // For now, just count normally (weight = 1 for all)
    return countLocalWrite(module);
}

// Count specific opcode
int countOpcode(const PyLogicalModule& module, logical::Opcode opcode) {
    int count = 0;
    for (const auto& inst : module.getInstructions()) {
        if (inst->getOpcode() == opcode) {
            count++;
        }
    }
    return count;
}

// Convenience functions for specific opcodes
int countDSStoreB128(const PyLogicalModule& module) {
    return countOpcode(module, Opcode::DSStoreB128);
}

int countVMovB32(const PyLogicalModule& module) {
    return countOpcode(module, Opcode::VMovB32);
}

// Get all MFMA instructions
std::vector<std::shared_ptr<LogicalInstruction>> getMFMAs(const PyLogicalModule& module) {
    std::vector<std::shared_ptr<LogicalInstruction>> mfmas;
    for (const auto& inst : module.getInstructions()) {
        if (isMFMA(inst)) {
            mfmas.push_back(inst);
        }
    }
    return mfmas;
}

// Find the index of a target instruction
std::pair<int, bool> findInstIndex(const PyLogicalModule& module,
                                   const std::shared_ptr<LogicalInstruction>& targetInst) {
    int index = 0;
    for (const auto& inst : module.getInstructions()) {
        if (inst == targetInst) {
            return {index, true};
        }
        index++;
    }
    return {-1, false};
}

}  // namespace
}  // namespace logical
}  // namespace stinkytofu

// ============================================================================
// Python bindings
// ============================================================================

// NOLINTNEXTLINE(misc-use-internal-linkage)
void init_logical_count(nb::module_& m) {
    using namespace stinkytofu::logical;

    m.def("countInstructions", &countInstructions, "Count total instructions in module");

    m.def("countGlobalRead", &countGlobalRead, "Count global memory read instructions");

    m.def("countSMemLoad", &countSMemLoad, "Count scalar memory load instructions");

    m.def("countLocalRead", &countLocalRead, "Count LDS read instructions");

    m.def("countLocalWrite", &countLocalWrite, "Count LDS write instructions");

    m.def("countGlobalWrite", &countGlobalWrite, "Count global memory write instructions");

    m.def("countMFMA", &countMFMA, "Count MFMA/WMMA matrix instructions");

    m.def("countWeightedLocalRead", &countWeightedLocalRead,
          "Count LDS reads with weights for composite instructions");

    m.def("countWeightedLocalWrite", &countWeightedLocalWrite,
          "Count LDS writes with weights for composite instructions");

    m.def("countDSStoreB128", &countDSStoreB128, "Count ds_store_b128 instructions");

    m.def("countVMovB32", &countVMovB32, "Count v_mov_b32 instructions");

    m.def("getMFMAs", &getMFMAs, "Get all MFMA instructions in the module");

    m.def("findInstIndex", &findInstIndex,
          "Find the index of target instruction. Returns (index, found)");
}
