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
 * @file GenLogicalToAsmMapping.cpp
 * @brief Generates Logical IR -> ASM mnemonic mapping (per-arch) for ToStinkyAsmPass.
 *
 * Uses the same per-arch data as the Rocisa LogicalToArchMap (logical name -> hw mnemonic)
 * so that logical IR lowering emits the correct mnemonic per architecture
 * (e.g. ds_read_* on CDNA, ds_load_* on RDNA).
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "gfx/GpuArchManager.hpp"
#include "gfx/InstDefDSL.hpp"

using namespace stinkytofu;

namespace {
bool genLogicalToAsmMappingsImpl(GpuArchManager& manager, const std::string& outdir) {
    std::filesystem::path path =
        std::filesystem::path(outdir) / "stinkytofu/ir/LogicalToAsmMappings_generated.inc";

    std::ofstream os(path);
    if (!os) {
        std::cerr << "Cannot write " << path << "\n";
        return false;
    }

    os << "//===----------------------------------------------------------------------===//\n"
       << "// Auto-generated Logical IR -> ASM mnemonic mapping (per-arch)\n"
       << "// Used by ToStinkyAsmPass for correct lowering (e.g. ds_read vs ds_load).\n"
       << "// DO NOT EDIT MANUALLY!\n"
       << "//===----------------------------------------------------------------------===//\n\n"
       << "#include <cstring>\n\n"
       << "namespace stinkytofu {\n"
       << "inline const char* getMnemonicForLogicalOnArch(const char* logicalName, GfxArchID "
          "arch)\n"
       << "{\n"
       << "    switch(arch)\n"
       << "    {\n";

    for (const std::string& archName : manager.getRegisteredArchNames()) {
        const GpuArch* arch = manager.getArch(archName);
        const auto& map = arch->getLogicalToArchMap();

        os << "        case GfxArchID::" << archName << ":\n";
        for (const auto& [logicalName, mnemonic] : map) {
            os << "            if(std::strcmp(logicalName, \"" << logicalName
               << "\") == 0) return \"" << mnemonic << "\";\n";
        }
        os << "            break;\n";
    }

    os << "        default:\n"
       << "            return nullptr;\n"
       << "    }\n"
       << "    return nullptr;\n"
       << "}\n"
       << "} // namespace stinkytofu\n";

    std::cerr << "Generating Logical IR -> ASM mappings in " << path << "\n";
    return true;
}
}  // namespace

namespace stinkytofu {
bool genLogicalToAsmMappings(GpuArchManager& manager, const std::string& outdir) {
    return genLogicalToAsmMappingsImpl(manager, outdir);
}
}  // namespace stinkytofu
