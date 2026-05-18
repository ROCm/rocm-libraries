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
 * THE SOFTWARE IS PROVIDED "AS IS") WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "gfx/GpuArchManager.hpp"

#include <algorithm>
#include <iostream>  // fix me: don't use iostream.
#include <unordered_map>
#include <unordered_set>

namespace stinkytofu {
// declare all define##archName##Insts functions
#define STINKYTOFU_ARCH(archName) void define##archName##Insts(GpuArch& registry);
#include "Config/Archs.def"

// declare all set##archName##LogicalToArchMap functions
#define STINKYTOFU_ARCH(archName) void set##archName##LogicalToArchMap(GpuArch& registry);
#include "Config/Archs.def"

// declare all set##archName##RocisaToArchMap functions
#define STINKYTOFU_ARCH(archName) void set##archName##RocisaToArchMap(GpuArch& registry);
#include "Config/Archs.def"

// declare all set##archName##ConversionMap functions
#define STINKYTOFU_ARCH(archName) void set##archName##ConversionMap(GpuArch& registry);
#include "Config/Archs.def"

void GpuArchManager::addArch(const std::string& arch,
                             const std::function<void(GpuArch&)>& defineInsts,
                             const std::function<void(GpuArch&)>& setLogicalToArchMap,
                             const std::function<void(GpuArch&)>& setRocisaToArchMap,
                             const std::function<void(GpuArch&)>& setRocisaConversionMap) {
    instructionsByArch.push_back(std::make_unique<GpuArch>(arch));
    defineInsts(*instructionsByArch.back());
    setLogicalToArchMap(*instructionsByArch.back());
    setRocisaToArchMap(*instructionsByArch.back());
    setRocisaConversionMap(*instructionsByArch.back());

    // Check for errors immediately after initialization
    if (instructionsByArch.back()->hasError()) {
        std::cerr << "FATAL: Failed to initialize architecture " << arch << "\n";
        error = true;
        return;
    }

    instructionsByArch.back()->finalize();
}

void GpuArchManager::enumAllOpcodes() {
    std::unordered_set<std::string> seen;

    for (const auto& arch : getRegisteredArchs()) {
        for (const auto& inst : arch->getInstructions()) {
            if (seen.find(inst->name) != seen.end()) continue;

            seen.emplace(inst->name);
            allOpcodes.push_back(inst->name);
        }
    }

    if (allOpcodes.size() > UINT16_MAX) {
        error = true;
        std::cerr << "Error: Running out of opcodes! Please expand the opcode type to uint32_t!\n";
        return;
    }

    std::sort(allOpcodes.begin(), allOpcodes.end());

    // The pseudo opcodes for all architectures.
    allOpcodes.push_back("LABEL");
    allOpcodes.push_back("PHI");

    // Add invalid unified opcode to the end of the list.
    allOpcodes.push_back("INVALID");

    std::unordered_map<std::string, uint16_t> opcodeMap;
    opcodeMap.reserve(allOpcodes.size());

    unsigned curOpcode = 0;

    for (const auto& opcode : allOpcodes) opcodeMap.emplace(opcode, curOpcode++);

    // update all arch
    for (const auto& arch : getRegisteredArchs())
        for (const auto& inst : arch->getInstructions())
            inst->hwInstDesc.unifiedOpcode = opcodeMap[inst->name];
}

bool GpuArchManager::initAllArchs(GpuArchManager& manager) {
#define STINKYTOFU_ARCH(archName)                                                        \
    manager.addArch(#archName, define##archName##Insts, set##archName##LogicalToArchMap, \
                    set##archName##RocisaToArchMap, set##archName##ConversionMap);
#include "Config/Archs.def"

    manager.enumAllOpcodes();

    // Final validation - stop build if any errors
    bool hasAnyError = false;
    for (const auto& arch : manager.getRegisteredArchs()) {
        if (arch->hasError()) {
            std::cerr << "FATAL: Architecture " << arch->getName()
                      << " has errors. Cannot continue.\n";
            hasAnyError = true;
        }
    }

    if (hasAnyError || manager.hasError()) {
        std::cerr << "FATAL: Failed to initialize GPU architectures.\n";
        return false;
    }

    return true;
}
}  // namespace stinkytofu
