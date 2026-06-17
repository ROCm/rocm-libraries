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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "Utility.hpp"
#include "gfx/GpuArchManager.hpp"
#include "gfx/InstDefDSL.hpp"

using namespace stinkytofu;

namespace {
using Map = std::unordered_map<std::string, std::string>;

bool genSimpleOneToOneMapping(const GpuArch& arch, const Map& rocisaToHwInstMap,
                              std::ofstream& os) {
    const std::string& archName = arch.getName();

    bool success = true;

    EmitMacroGuard emitMacro(os, "GET_ROCISA_HW_MAPPING_TABLE");

    os << "using namespace rocisa;\n"
       << "// rocisaType, " << archName << " opcode\n"
       << "static const std::unordered_map<std::type_index, uint16_t> rocisaToHwInstMap = "
          "{\n";
    for (const auto& [rocisaInstName, hwInstName] : rocisaToHwInstMap) {
        if (!arch.has(hwInstName)) {
            std::cerr << "RocisaHwInstMappings.hpp: error: hardware instruction '" << hwInstName
                      << "' not found in " << archName << "\n"
                      << "  mapping: rocisa::" << rocisaInstName << " -> " << hwInstName
                      << "\n  source: getGfx" << archName << "RocisaSimpleMappings()\n"
                      << "  fix: add DEF_T(..., \"" << hwInstName << "\", ...) to hardware/src/gfx/"
                      << archName << "/" << archName
                      << "Instructions.def, or correct the mapping\n";
            success = false;
            continue;
        }
        os << "    {typeid(" << rocisaInstName
           << "), "
           //<< archName << "::" << hwInstName
           << arch.getInst(hwInstName)->hwInstDesc.isaOpcode << "}, // " << hwInstName << "\n";
    }
    os << "};\n\n";

    return success;
}

bool genConvertRocisaToHwInstFunc(const Map& convertRocisaToHwInstFunc, std::ofstream& os) {
    EmitMacroGuard emitMacro(os, "GET_ROCISA_TO_HW_CONVERSION_TABLE");

    // The callback function type:
    //   std::vector<StinkyInstruction*> (convertFn*)(rocisa::Instruction&, AsmIRBuilder&)

    os << "using namespace rocisa;\n";
    os << "static const std::unordered_map<std::type_index, ConvertRocisaToHwInstFunc>\n";
    os << "convertRocisaToHwInstFunc = {\n";
    for (const auto& [rocisaInstName, funcName] : convertRocisaToHwInstFunc) {
        os << "    {typeid(" << rocisaInstName << "), ";
        if (funcName.empty())
            os << "nullptr";
        else
            os << funcName;
        os << "},\n";
    }
    os << "};\n\n";
    return true;
}

void emitBanner(std::ofstream& os, const std::string& arch) {
    os << "//===----------------------------------------------------------------------===//\n"
       << "// Auto-generated Rocisa <--> StinkyAsm mappings for " << arch << "\n"
       << "//\n"
       << "// DO NOT EDIT MANUALLY!\n"
       << "// DO NOT USE #paragma once IN THIS FILE!\n"
       << "//===----------------------------------------------------------------------===//\n"
       << "\n"
       << "// This file defines the following tables:\n"
       << "// - rocisaToHwInstMap: Rocisa <--> StinkyAsm mappings\n"
       << "//   * Returns the StinkyAsm instruction for the Rocisa instruction.\n"
       << "//   * Use GET_ROCISA_HW_MAPPING_TABLE to access the table.\n"
       << "//\n"
       << "// - convertRocisaToHwInstFunc: Rocisa --> StinkyAsm conversion functions\n"
       << "//   * Returns the conversion function for the Rocisa instruction.\n"
       << "//   * Use GET_ROCISA_TO_HW_CONVERSION_TABLE to access the table.\n"
       << "//\n\n";
}

bool genRocisaMappings(const GpuArchManager& manager, const std::string& archName,
                       const std::string& outdir, const Map& rocisaToHwInstMap,
                       const Map& convertRocisaToHwInstFunc) {
    const GpuArch& arch = *manager.getArch(archName);

    std::filesystem::path path =
        std::filesystem::path(outdir) / ("stinkytofu/ir/rocisa/Rocisa" + archName + "Mappings.inc");

    std::ofstream os(path);
    if (!os) {
        std::cerr << "Cannot write " << path << "\n";
        return false;
    }
    std::cerr << "Generating Rocisa mappings for " << archName << " in " << path << "\n";

    bool success = true;

    emitBanner(os, archName);

    success &= genSimpleOneToOneMapping(arch, rocisaToHwInstMap, os);
    success &= genConvertRocisaToHwInstFunc(convertRocisaToHwInstFunc, os);

    return success;
}
}  // namespace

namespace stinkytofu {
// NOLINTNEXTLINE(misc-use-internal-linkage)
bool genAllArchRocisaMappings(GpuArchManager& manager, const std::string& outdir) {
    bool success = true;

    for (const auto& archName : manager.getRegisteredArchNames()) {
        const GpuArch* arch = manager.getArch(archName);
        const auto& rocisaMap = arch->getRocisaToArchMap();
        const auto& mapForRocisaInc = rocisaMap.empty() ? arch->getLogicalToArchMap() : rocisaMap;
        success &= genRocisaMappings(manager, archName, outdir, mapForRocisaInc,
                                     arch->getRocisaConversionMap());
    }

    return success;
}
}  // namespace stinkytofu
