/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <array>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Function;

/// One printed node of a region, in the order of an IR checkpoint bundle.
struct DependencyInst {
    /// The `mod.ckpt` id the text carried, or a fresh one above every carried id.
    long id = -1;
    size_t block = 0;
    std::string text;
    std::string opcode;
    std::vector<std::string> reads;
    std::vector<std::string> writes;
    int issue = 0;
    int latency = 0;
    /// Never reordered: the scheduler splits its regions here, so nothing crosses it either.
    bool pinned = false;
    std::string pinReason;
};

/// `src` must stay before `dst`. Both are DependencyInst ids in one block.
///
/// `kind` is `raw`, `war` or `waw`, with `regClass` one of `vgpr`, `agpr`, `sgpr`, `mgpr`,
/// `scc`, `vcc`, `exec` or `lds`.
struct DependencyEdge {
    long src = -1;
    long dst = -1;
    std::string kind;
    std::string regClass;
    std::vector<std::string> regs;
};

struct RegionDependencies {
    std::vector<std::string> blocks;
    std::vector<DependencyInst> insts;
    std::vector<DependencyEdge> edges;
    /// Non-empty when the text could not be analyzed; nothing else is filled in then.
    std::string error;
};

/// The ordering constraints of every instruction of \p function, block by block: the register
/// edges the DAG scheduler builds (buildRegisterDependencyDAG), with their kind. GFX::PHI takes
/// no part in them.
STINKYTOFU_EXPORT RegionDependencies exportRegionDependencies(Function& function);

/// Parse \p stirText (a checkpoint bundle, ids optional) for \p arch and export its
/// dependencies.
STINKYTOFU_EXPORT RegionDependencies analyzeRegionText(const std::string& stirText,
                                                       const std::array<int, 3>& arch);

struct RegionOrderCheck {
    /// Base edges whose ends the edited text puts the other way round.
    std::vector<DependencyEdge> reversed;
    /// Moves the editor does not allow, and what the verifier says about the edited IR.
    std::vector<std::string> problems;
    /// Set when the base itself could not be analyzed.
    std::string error;
};

/// Compare \p editedText against \p baseText, both bundles with ids. Lines are matched by id;
/// a line without one is new and constrains nothing until the next analysis.
STINKYTOFU_EXPORT RegionOrderCheck checkRegionOrder(const std::string& baseText,
                                                    const std::string& editedText,
                                                    const std::array<int, 3>& arch);

}  // namespace stinkytofu
