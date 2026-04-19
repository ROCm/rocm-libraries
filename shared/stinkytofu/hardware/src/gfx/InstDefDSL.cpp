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
#include "gfx/InstDefDSL.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>  // todo: don't use iostream.

namespace stinkytofu {
std::string getReducedFilename(const char* filename, unsigned keepDepth) {
    // only preserve the filename and the last `keepDepth` directories
    namespace fs = std::filesystem;
    fs::path p(filename);

    // Always include the filename itself
    fs::path result = p.filename();

    // Append up to `keepDepth` last parent directories (from innermost outward)
    auto parent = p.parent_path();
    for (unsigned i = 0; i < keepDepth && !parent.empty(); ++i) {
        result = parent.filename() / result;
        parent = parent.parent_path();
    }

    return result.string();
}

bool GpuArch::add(std::unique_ptr<GfxInstDef> inst) {
    assert(!finalized && "GpuArch already finalized");
    if (added.find(inst->name) != added.end()) {
        std::cerr << "Error: Instruction " << inst->name << " already added!\n";
        error = true;
        return false;
    }
    added.insert(std::make_pair(inst->name, inst.get()));
    instructions.push_back(std::move(inst));
    return true;
}

bool GpuArch::erase(const std::string& name) {
    assert(!finalized && "GpuArch already finalized");
    auto it =
        std::remove_if(instructions.begin(), instructions.end(),
                       [&](const std::unique_ptr<GfxInstDef>& inst) { return inst->name == name; });

    if (it == instructions.end()) {
        std::cerr << "Error: Instruction " << name << " to erase is not found!\n";
        error = true;
        return false;
    }

    instructions.erase(it, instructions.end());
    added.erase(name);
    return true;
}

GfxInstDef* GpuArch::getInst(const std::string& name) {
    auto it = added.find(name);
    if (it != added.end()) return it->second;

    std::cerr << "Error: Instruction " << name << " not found!\n";
    error = true;
    return nullptr;
}

void GpuArch::setDefaultCosts(uint16_t cycle, uint16_t latency) {
    if (cycle == 0 || latency == 0) {
        std::cerr << "ERROR: Default costs cannot be 0!\n"
                  << "       Architecture: " << name << "\n"
                  << "       Provided: cycle=" << cycle << ", latency=" << latency << "\n";
        error = true;
        return;
    }
    defaultCycle_ = cycle;
    defaultLatency_ = latency;
}

void GpuArch::setInstructionCost(const std::string& opcode, uint16_t cycle, uint16_t latency) {
    instructionCosts_[opcode] = {cycle, latency};
}

bool GpuArch::applyInstructionCosts() {
    bool success = true;

    // Validation 1: Defaults must be set
    if (defaultCycle_ == 0 || defaultLatency_ == 0) {
        std::cerr << "ERROR: Default costs not set for architecture " << name << "!\n"
                  << "       Call setDefaultCosts() before applyInstructionCosts().\n"
                  << "       Current: defaultCycle_=" << defaultCycle_
                  << ", defaultLatency_=" << defaultLatency_ << "\n";
        error = true;
        return false;
    }

    // Apply costs to all instructions
    for (auto& inst : instructions) {
        auto it = instructionCosts_.find(inst->name);
        if (it != instructionCosts_.end()) {
            // Use instruction-specific cost
            inst->hwInstDesc.issue = it->second.first;
            inst->hwInstDesc.latency = it->second.second;
        } else {
            // Use default cost
            inst->hwInstDesc.issue = defaultCycle_;
            inst->hwInstDesc.latency = defaultLatency_;
        }
    }

    // Validation 2: NO instruction can have 0 cycle/latency (CRITICAL!)
    std::vector<std::string> invalidInstructions;
    for (const auto& inst : instructions) {
        if (inst->hwInstDesc.issue == 0 || inst->hwInstDesc.latency == 0) {
            invalidInstructions.push_back(inst->name);
        }
    }

    if (!invalidInstructions.empty()) {
        std::cerr << "ERROR: The following instructions have invalid costs (0 cycle/latency):\n";
        for (const auto& instName : invalidInstructions) {
            std::cerr << "       - " << instName << "\n";
        }
        std::cerr << "       This is a fatal error. Please check cost tables.\n";
        error = true;
        success = false;
    }

    return success;
}

}  // namespace stinkytofu
