/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#include "stinkytofu/analysis/asm/DependencyExport.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/pipeline/IRCheckpoint.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

/// The most problems one order check lists before it summarizes the rest.
constexpr size_t kMaxProblems = 40;

std::string trim(std::string_view text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    const size_t last = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(first, last - first + 1));
}

/// One line of a bundle: a block header, a printed IR node, or neither.
struct BundleLine {
    enum Kind { Other, Block, Node } kind = Other;
    std::string text;
    long id = -1;
};

/// The same line classes the checkpoint loader reads.
BundleLine classifyLine(const std::string& raw) {
    BundleLine out;
    std::string line = trim(raw);
    if (line.empty() || line.starts_with("//") || line.starts_with("#") ||
        line.starts_with("st.func") || line == "}" || line.starts_with("Successors:"))
        return out;
    if (line.front() == '^' && line.back() == ':') {
        out.kind = BundleLine::Block;
        out.text = line.substr(1, line.size() - 2);
        return out;
    }
    out.kind = BundleLine::Node;
    out.id = takeCheckpointId(line);
    out.text = trim(line);
    return out;
}

/// Register units the dependency walk keys on. The halves of VCC and EXEC alias the whole, so a
/// write of vcc_lo orders against a read of vcc.
struct RegUnit {
    RegType type;
    uint32_t idx;
    auto operator<=>(const RegUnit&) const = default;
};

RegUnit unitOf(RegType type, uint32_t idx) {
    switch (type) {
        case RegType::VCC:
        case RegType::VCC_LO:
        case RegType::VCC_HI:
            return {RegType::VCC, 0};
        case RegType::EXEC:
        case RegType::EXEC_LO:
        case RegType::EXEC_HI:
            return {RegType::EXEC, 0};
        case RegType::ACC:
        case RegType::AGPR:
            return {RegType::A, idx};
        default:
            return {type, idx};
    }
}

const char* regClassOf(RegType type) {
    switch (type) {
        case RegType::V:
            return "vgpr";
        case RegType::A:
            return "agpr";
        case RegType::S:
            return "sgpr";
        case RegType::M:
            return "mgpr";
        case RegType::SCC:
            return "scc";
        case RegType::VCC:
            return "vcc";
        case RegType::EXEC:
            return "exec";
        case RegType::LDS:
            return "lds";
        default:
            return "other";
    }
}

std::string unitName(const RegUnit& unit) {
    switch (unit.type) {
        case RegType::V:
            return "v" + std::to_string(unit.idx);
        case RegType::A:
            return "a" + std::to_string(unit.idx);
        case RegType::S:
            return "s" + std::to_string(unit.idx);
        case RegType::M:
            return "m" + std::to_string(unit.idx);
        case RegType::SCC:
            return "scc";
        case RegType::VCC:
            return "vcc";
        case RegType::EXEC:
            return "exec";
        case RegType::LDS:
            return "LDS" + std::to_string(unit.idx);
        default:
            return "?" + std::to_string(unit.idx);
    }
}

std::string operandName(const StinkyRegister& reg) {
    const RegUnit first = unitOf(reg.reg.type, reg.reg.idx);
    if (reg.reg.num <= 1) return unitName(first);
    std::string prefix = unitName(first);
    prefix.erase(std::find_if(prefix.begin(), prefix.end(), ::isdigit), prefix.end());
    return prefix + "[" + std::to_string(reg.reg.idx) + ":" +
           std::to_string(reg.reg.idx + reg.reg.num - 1) + "]";
}

std::string opcodeOf(const std::string& text) {
    const size_t open = text.find("\"st.");
    if (open != std::string::npos) {
        const size_t close = text.find('"', open + 4);
        if (close != std::string::npos) return text.substr(open + 4, close - open - 4);
    }
    return text.substr(0, text.find_first_of(" ({"));
}

bool writesExec(const StinkyInstruction& inst) {
    return std::any_of(
        inst.getDestRegs().begin(), inst.getDestRegs().end(), [](const StinkyRegister& r) {
            return r.isRegister() && unitOf(r.reg.type, r.reg.idx).type == RegType::EXEC;
        });
}

/// Edges keyed so a pair that conflicts on several registers is reported once per kind.
class EdgeSet {
   public:
    void add(size_t src, size_t dst, const std::string& kind, const std::string& regClass,
             const std::string& reg) {
        if (src == dst) return;
        auto& regs = edges[{src, dst, kind, regClass}];
        if (!reg.empty() && std::find(regs.begin(), regs.end(), reg) == regs.end())
            regs.push_back(reg);
    }

    std::vector<DependencyEdge> take(const std::vector<DependencyInst>& insts) const {
        std::vector<DependencyEdge> out;
        out.reserve(edges.size());
        for (const auto& [key, regs] : edges) {
            const auto& [src, dst, kind, regClass] = key;
            out.push_back({insts[src].id, insts[dst].id, kind, regClass, regs});
        }
        return out;
    }

   private:
    std::map<std::tuple<size_t, size_t, std::string, std::string>, std::vector<std::string>> edges;
};

/// RAW/WAR/WAW over one block, in program order: buildRegisterDependencyDAG with the kind of
/// every edge kept.
void addRegisterEdges(const std::vector<size_t>& order,
                      const std::vector<const StinkyInstruction*>& instOf, EdgeSet& edges) {
    std::map<RegUnit, std::set<size_t>> lastRead;
    std::map<RegUnit, size_t> lastWrite;
    for (size_t node : order) {
        const StinkyInstruction* inst = instOf[node];
        if (!inst || inst->getUnifiedOpcode() == GFX::PHI) continue;

        for (const StinkyRegister& src : inst->getSrcRegs()) {
            if (!src.isRegister()) continue;
            for (unsigned off = 0; off < src.reg.num; ++off) {
                const RegUnit unit = unitOf(src.reg.type, src.reg.idx + off);
                if (auto w = lastWrite.find(unit); w != lastWrite.end())
                    edges.add(w->second, node, "raw", regClassOf(unit.type), unitName(unit));
                lastRead[unit].insert(node);
            }
        }
        for (const StinkyRegister& dst : inst->getDestRegs()) {
            if (!dst.isRegister()) continue;
            for (unsigned off = 0; off < dst.reg.num; ++off) {
                const RegUnit unit = unitOf(dst.reg.type, dst.reg.idx + off);
                if (auto w = lastWrite.find(unit); w != lastWrite.end())
                    edges.add(w->second, node, "waw", regClassOf(unit.type), unitName(unit));
                if (auto r = lastRead.find(unit); r != lastRead.end()) {
                    for (size_t reader : r->second)
                        edges.add(reader, node, "war", regClassOf(unit.type), unitName(unit));
                    lastRead.erase(r);
                }
                lastWrite[unit] = node;
            }
        }
    }
}

struct ParsedRegion {
    std::unique_ptr<Function> function;
    std::vector<BundleLine> nodes;
    std::vector<std::string> blocks;
    std::string error;
};

ParsedRegion parseRegion(const std::string& stirText, const std::array<int, 3>& arch) {
    ParsedRegion out;
    if (!ArchHelper::getInstance().getArchInfo(arch[0], arch[1], arch[2])) {
        out.error = "unsupported architecture gfx" + std::to_string(arch[0]) +
                    std::to_string(arch[1]) + std::to_string(arch[2]);
        return out;
    }

    // Ids are stripped line by line, so a diagnostic's line number is the input's.
    std::string source;
    std::istringstream in(stirText);
    for (std::string raw; std::getline(in, raw);) {
        BundleLine line = classifyLine(raw);
        if (line.kind == BundleLine::Node) {
            std::string stripped = raw;
            (void)takeCheckpointId(stripped);
            source += stripped + "\n";
            out.nodes.push_back(std::move(line));
            continue;
        }
        if (line.kind == BundleLine::Block) out.blocks.push_back(line.text);
        source += raw + "\n";
    }

    const auto parsed = parseSourceStringWithDiagnostics(source);
    if (parsed.hasErrors()) {
        out.error = "the IR does not parse:";
        for (const auto& diag : parsed.diagnostics)
            if (diag.getLevel() == Diagnostic::Level::Error)
                out.error +=
                    "\n  line " + std::to_string(diag.getLine()) + ": " + diag.getMessage();
        return out;
    }

    out.function = std::make_unique<Function>("ckpt");
    PassContext ctx;
    if (StinkyIRConverter::populateFunctionFromString(source, *out.function, ctx,
                                                      getGfxArchID(arch[0], arch[1], arch[2])) !=
        StinkyErrorCode::SUCCESS) {
        out.error = "the IR does not convert to Stinky IR";
        return out;
    }
    return out;
}

/// A line that must keep its place: the scheduler cuts its regions at these.
std::string pinReasonOf(const IRBase* node) {
    const auto* inst = dyn_cast<StinkyInstruction>(node);
    if (!inst) return "directive";
    if (inst->getUnifiedOpcode() == GFX::PHI) return "phi";
    if (isBranch(*inst) || isEndOfFunction(*inst)) return "terminator";
    if (hasSideEffect(*inst)) return "side effect";
    if (writesExec(*inst)) return "writes exec";
    return {};
}

}  // namespace

RegionDependencies exportRegionDependencies(Function& function) {
    RegionDependencies out;
    const IRCheckpointRendering rendering = renderCheckpointRegion(function, /*withIds=*/false);
    out.blocks = rendering.blockLabels;

    const size_t n = rendering.nodes.size();
    std::vector<const StinkyInstruction*> instOf(n, nullptr);
    std::vector<std::vector<size_t>> blockOrder(out.blocks.size());
    out.insts.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const IRBase* node = rendering.nodes[i];
        blockOrder[rendering.blockOf[i]].push_back(i);
        DependencyInst& row = out.insts[i];
        row.id = static_cast<long>(i);
        row.block = rendering.blockOf[i];
        row.text = rendering.nodeText[i];
        row.opcode = opcodeOf(row.text);
        row.pinReason = pinReasonOf(node);
        row.pinned = !row.pinReason.empty();

        const auto* inst = dyn_cast<StinkyInstruction>(node);
        if (!inst) continue;
        instOf[i] = inst;
        row.issue = inst->issueCycles;
        row.latency = inst->latencyCycles;
        for (const StinkyRegister& r : inst->getSrcRegs())
            if (r.isRegister()) row.reads.push_back(operandName(r));
        for (const StinkyRegister& r : inst->getDestRegs())
            if (r.isRegister()) row.writes.push_back(operandName(r));
    }

    EdgeSet edges;
    for (const auto& order : blockOrder) addRegisterEdges(order, instOf, edges);
    out.edges = edges.take(out.insts);
    return out;
}

RegionDependencies analyzeRegionText(const std::string& stirText, const std::array<int, 3>& arch) {
    ParsedRegion parsed = parseRegion(stirText, arch);
    RegionDependencies out;
    if (!parsed.error.empty()) {
        out.error = parsed.error;
        return out;
    }
    out = exportRegionDependencies(*parsed.function);
    if (out.insts.size() != parsed.nodes.size()) {
        RegionDependencies failed;
        failed.error = std::to_string(parsed.nodes.size()) + " IR lines became " +
                       std::to_string(out.insts.size()) +
                       " IR nodes; an unknown mnemonic or a label line is the usual cause";
        return failed;
    }

    // Keep the ids the text carried; a line without one gets an id above all of them.
    long next = 0;
    for (const BundleLine& line : parsed.nodes) next = std::max(next, line.id + 1);
    std::unordered_map<long, long> renamed;
    std::unordered_set<long> seen;
    for (size_t i = 0; i < out.insts.size(); ++i) {
        long id = parsed.nodes[i].id;
        if (id < 0 || !seen.insert(id).second) id = next++;
        renamed[out.insts[i].id] = id;
        out.insts[i].id = id;
    }
    for (DependencyEdge& edge : out.edges) {
        edge.src = renamed.at(edge.src);
        edge.dst = renamed.at(edge.dst);
    }
    return out;
}

RegionOrderCheck checkRegionOrder(const std::string& baseText, const std::string& editedText,
                                  const std::array<int, 3>& arch) {
    RegionOrderCheck out;
    const RegionDependencies base = analyzeRegionText(baseText, arch);
    if (!base.error.empty()) {
        out.error = "base: " + base.error;
        return out;
    }
    size_t dropped = 0;
    auto problem = [&](const std::string& text) {
        if (out.problems.size() < kMaxProblems)
            out.problems.push_back(text);
        else
            ++dropped;
    };

    std::unordered_map<long, const DependencyInst*> baseById;
    for (const DependencyInst& inst : base.insts) baseById[inst.id] = &inst;

    // Where each id sits in the edited text.
    struct Place {
        size_t block;
        size_t position;
    };
    std::unordered_map<long, Place> edited;
    std::vector<std::string> editedBlocks;
    std::vector<std::vector<long>> editedOrder;
    {
        std::istringstream in(editedText);
        unsigned lineNumber = 0;
        for (std::string raw; std::getline(in, raw);) {
            ++lineNumber;
            const BundleLine line = classifyLine(raw);
            if (line.kind == BundleLine::Block) {
                editedBlocks.push_back(line.text);
                editedOrder.emplace_back();
                continue;
            }
            if (line.kind != BundleLine::Node) continue;
            if (editedOrder.empty()) {
                problem("line " + std::to_string(lineNumber) +
                        ": instruction before the first block");
                continue;
            }
            if (line.id < 0) continue;
            if (!baseById.contains(line.id)) {
                problem("line " + std::to_string(lineNumber) + ": unknown id " +
                        std::to_string(line.id));
                continue;
            }
            const Place place{editedOrder.size() - 1, editedOrder.back().size()};
            if (!edited.emplace(line.id, place).second) {
                problem("line " + std::to_string(lineNumber) + ": id " + std::to_string(line.id) +
                        " appears twice");
                continue;
            }
            editedOrder.back().push_back(line.id);
        }
    }

    if (editedBlocks != base.blocks) {
        problem(
            "the blocks changed: the editor reorders inside a block and never adds, removes "
            "or renames one");
    } else {
        for (const auto& [id, place] : edited) {
            const DependencyInst& inst = *baseById.at(id);
            if (place.block != inst.block)
                problem("id " + std::to_string(id) + " moved from ^" + base.blocks[inst.block] +
                        " to ^" + base.blocks[place.block] +
                        "; moves across blocks are not "
                        "supported");
        }
    }
    for (const DependencyInst& inst : base.insts)
        if (inst.pinReason == "terminator" && !edited.contains(inst.id))
            problem("id " + std::to_string(inst.id) + " (" + inst.opcode +
                    ") is a terminator and cannot be removed");

    auto inBlock = [&](long id, size_t block) {
        auto it = edited.find(id);
        return it != edited.end() && it->second.block == block;
    };
    for (const DependencyEdge& edge : base.edges) {
        const size_t block = baseById.at(edge.src)->block;
        if (!inBlock(edge.src, block) || !inBlock(edge.dst, block)) continue;
        if (edited.at(edge.dst).position < edited.at(edge.src).position)
            out.reversed.push_back(edge);
    }

    // A pinned line keeps the same lines on each side of it.
    std::unordered_map<long, size_t> basePosition;
    {
        std::vector<size_t> next(base.blocks.size(), 0);
        for (const DependencyInst& inst : base.insts) basePosition[inst.id] = next[inst.block]++;
    }
    for (const DependencyInst& pin : base.insts) {
        if (!pin.pinned || !inBlock(pin.id, pin.block)) continue;
        for (long other : editedOrder[pin.block]) {
            if (other == pin.id || baseById.at(other)->block != pin.block) continue;
            const bool wasBefore = basePosition.at(other) < basePosition.at(pin.id);
            const bool isBefore = edited.at(other).position < edited.at(pin.id).position;
            if (wasBefore != isBefore)
                problem("id " + std::to_string(other) + " (" + baseById.at(other)->opcode +
                        ") crosses id " + std::to_string(pin.id) + " (" + pin.opcode + ", " +
                        pin.pinReason + "), which nothing may cross");
        }
    }

    ParsedRegion parsed = parseRegion(editedText, arch);
    if (!parsed.error.empty()) {
        problem("edited: " + parsed.error);
    } else {
        AsmVerifierConfig verify;
        verify.abortOnError = false;
        if (const std::string found = validateStinkyIR(*parsed.function, verify); !found.empty())
            problem("verifier: " + found);
    }
    if (dropped) out.problems.push_back("... and " + std::to_string(dropped) + " more");
    return out;
}

}  // namespace stinkytofu
