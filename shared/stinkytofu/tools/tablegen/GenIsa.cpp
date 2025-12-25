/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream> // fixme: don't use iostream.
#include <string>
#include <vector>

#include "Utility.hpp"
#include "gfx/GpuArchManager.hpp"
#include "gfx/InstDefDSL.hpp"

using namespace stinkytofu;
namespace
{
    class GenIsa
    {
        std::string arch;
        std::string outdir;

        // For aligning mnemonic strings in the table.
        uint16_t fixedMnemonicLen = 0;

        const GpuArch::List& insts;

        void emitBanner(std::ofstream& os);

        // Emit Mnemonic string table.
        void emitMnemonics(std::ofstream& os);

        // Emit Opcode enumerations.
        void emitEnumOpcodes(std::ofstream& os);

        // Emit HwInstDesc (Hardware Instruction Description) table.
        void emitInsts(std::ofstream& os);

        void emitUnifiedOpcodeMappings(std::ofstream& os);

        void emitMnemonicToIsaOpcodeMappings(std::ofstream& os);

    public:
        GenIsa(const std::string& arch, const GpuArch::List& insts, const std::string& outdir)
            : arch(arch)
            , outdir(outdir)
            , insts(insts)
        {
        }

        bool emit();
    };

    void emitSortedOpcodes(const GpuArch::List& insts, std::ofstream& os)
    {
        // sort the instructions by name
        std::vector<std::string> sortedInsts;
        for(auto& inst : insts)
        {
            sortedInsts.push_back(inst->name);
        }
        std::sort(sortedInsts.begin(), sortedInsts.end());
        for(auto& name : sortedInsts)
        {
            os << name << "\n";
        }
    }

    bool GenIsa::emit()
    {
        std::string outdirPath = outdir + "/hardware/" + arch;

        std::string   path = outdirPath + "Isa.inc";
        std::ofstream out(path);
        if(!out)
        {
            std::cerr << "Error: Cannot write " << path << "\n";
            return false;
        }

        size_t maxMnemonicLen = 0;
        for(auto& inst : insts)
            maxMnemonicLen = std::max(maxMnemonicLen, inst->name.size());

        assert(maxMnemonicLen <= UINT16_MAX && "mnemonic length exceeds uint16_t limit");

        fixedMnemonicLen = static_cast<uint16_t>(maxMnemonicLen);

        emitBanner(out);
        emitEnumOpcodes(out);
        emitInsts(out);
        emitUnifiedOpcodeMappings(out);
        emitMnemonicToIsaOpcodeMappings(out);

        path = outdirPath + "SortedOpcodes.txt";
        std::ofstream outSortedOpcodes(path);
        if(!outSortedOpcodes)
        {
            std::cerr << "Warning:Cannot write " << path << "\n";
            return false;
        }
        emitSortedOpcodes(insts, outSortedOpcodes);

        return true;
    }

    void GenIsa::emitBanner(std::ofstream& os)
    {
        os << "//===----------------------------------------------------------------------===//\n"
           << "// Auto-generated ISA header for " << arch << "\n"
           << "//\n"
           << "// DO NOT EDIT MANUALLY!\n"
           << "// DO NOT USE #paragma once IN THIS FILE!\n"
           << "//===----------------------------------------------------------------------===//\n"
           << "\n"
           << "// - Opcode: Opcode enumeration\n"
           << "//   * Defines the Opcode enum for " << arch << " ISA.\n"
           << "//   * Use GET_ISAINFO_OPCODE_ENUMERATION to access the table.\n"
           << "//\n"
           << "// - MCIDTable[opcode]: Machine Code Instruction Descriptor Table\n"
           << "//   * Returns the machine code instruction descriptor for the opcode.\n"
           << "//   * Use GET_ISAINFO_HWINSTDESC_TABLE to access the table.\n"
           << "//\n"
           << "// - get${arch}Opcode: Unified opcode mappings table\n"
           << "//   * Returns the isa opcode for the unified opcode.\n"
           << "//   * Use GET_ISAINFO_UOP_MAPPINGS to access the table.\n"
           << "//\n"
           << "// - MnemonicToIsaOpcodeMap: Mnemonic to isa opcode mappings table\n"
           << "//   * Returns the isa opcode for the mnemonic.\n"
           << "//   * Use GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS to access the table.\n"
           << "//\n\n";
    }

    // output format:
    //   HwInstDesc { isaOpcode, unifiedOpcode,
    //                issue, latency,
    //                mnemonic,
    //                makeFlagSet({flags}) }, // defined at file:line
    std::string hwInstDescToString(const GfxInstDef& inst, size_t maxMnemonicLen)
    {
        std::stringstream ss;

        std::string mnemonic = "\"" + inst.name + "\",";

        ss << "  { "
           // instruction's ISA-specific opcode
           << std::setw(3) << inst.hwInstDesc.isaOpcode
           << ","
           // instruction's unified opcode
           << std::setw(5) << inst.hwInstDesc.unifiedOpcode
           << ","
           // issue and latency cycles
           << std::setw(3) << inst.hwInstDesc.issue << "," << std::setw(4)
           << inst.hwInstDesc.latency
           << ","
           // mnemonic string
           << " " << std::left << std::setw(maxMnemonicLen + 3)
           << mnemonic
           // flags
           << "makeFlagSet({" << inst.getFlagsStr() << "}) }," << " // " << inst.definedFile << ":"
           << inst.definedLine << "\n";
        return ss.str();
    }

    void GenIsa::emitInsts(std::ofstream& os)
    {
        EmitMacroGuard emitMacro(os, "GET_ISAINFO_HWINSTDESC_TABLE");

        os << "// MCIDTable: Machine Code Instruction Descriptor Table\n"
           << "static constexpr HwInstDesc MCIDTable[] = {\n";

        for(auto& inst : insts)
            os << hwInstDescToString(*inst, fixedMnemonicLen);

        os << "};\n\n";
    }

    void GenIsa::emitEnumOpcodes(std::ofstream& os)
    {
        EmitMacroGuard emitMacro(os, "GET_ISAINFO_OPCODE_ENUMERATION");

        os << "enum " << arch << " : uint16_t {\n";
        for(auto& inst : insts)
        {
            os << "  " << std::left << std::setw(fixedMnemonicLen) << inst->name << ", // "
               << inst->definedFile << ":" << inst->definedLine << "\n";
        }
        os << "};\n\n";
    }

    // emit:
    //
    // uint16_t getArchOpcode(uint16_t unifiedOpcode) {
    //     using namespace AMDGPU;
    //     static constexpr uint16_t Table[][2] = {
    //       { GFX::global op, arch::local op },
    //       ...
    //     };
    //     unsigned mid;
    //     unsigned low = 0;
    //     unsigned high = ..;
    //     while(low <= high) {
    //       mid = (low + high) / 2;
    //       if(Table[mid][0] == unifiedOpcode)
    //         return Table[mid][1];
    //       else if(Table[mid][0] < unifiedOpcode)
    //         low = mid + 1;
    //       else
    //         high = mid - 1;
    //     }
    //     return UINT16_MAX;
    // }
    //
    void GenIsa::emitUnifiedOpcodeMappings(std::ofstream& os)
    {
        EmitMacroGuard emitMacro(os, "GET_ISAINFO_UOP_MAPPINGS");

        os << "uint16_t get" << arch << "Opcode(uint16_t unifiedOpcode) {\n";
        os << "    // unifiedOpcode, " << arch << " opcode\n";
        os << "    static constexpr uint16_t Table[][2] = {\n";

        // sort the instructions by unified opcode
        std::vector<GfxInstDef*> sortedInsts;
        for(auto& inst : insts)
        {
            sortedInsts.push_back(inst.get());
        }

        std::sort(
            sortedInsts.begin(), sortedInsts.end(), [](const GfxInstDef* a, const GfxInstDef* b) {
                return a->hwInstDesc.unifiedOpcode < b->hwInstDesc.unifiedOpcode;
            });

        for(auto& inst : sortedInsts)
        {
            os << "      { ";
            os << inst->hwInstDesc.unifiedOpcode << ", ";
            os << inst->hwInstDesc.isaOpcode;
            os << " }, // " << inst->name << "\n";
        }
        os << "    };\n";
        os << "    unsigned mid;\n";
        os << "    unsigned low = 0;\n";
        os << "    unsigned high = " << insts.size() - 1 << ";\n";
        os << "    while(low <= high) {\n";
        os << "      mid = low + (high - low) / 2; // Avoid overflow\n";
        os << "      if(Table[mid][0] == unifiedOpcode) {\n";
        os << "        return Table[mid][1];\n";
        os << "      } else if(Table[mid][0] < unifiedOpcode) {\n";
        os << "        low = mid + 1;\n";
        os << "      } else {\n";
        os << "        high = mid - 1;\n";
        os << "      }\n";
        os << "    }\n";
        os << "    return GFX::INVALID;\n";
        os << "}\n";
    }

    void GenIsa::emitMnemonicToIsaOpcodeMappings(std::ofstream& os)
    {
        EmitMacroGuard emitMacro(os, "GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS");
        os << "// Mnemonic, " << arch << " isaOpcode\n";
        os << "static const std::unordered_map<std::string, IsaOpcode> MnemonicToIsaOpcodeMap = "
              "{\n";
        for(auto& inst : insts)
        {
            os << "  {\"" << inst->name << "\", " << inst->hwInstDesc.isaOpcode << "},\n";
        }
        os << "};\n\n";
    }

    bool genIsaDefinitions(const GpuArch& arch, const std::string& outdir)
    {
        GenIsa generator(arch.getName(), arch.getInstructions(), outdir);

        bool success = generator.emit();
        if(!success)
            std::cerr << "Failed to generate ISA definitions for " << arch.getName() << "\n";

        return success;
    }

    class GenAllArchIsa
    {
        const GpuArchManager& manager;
        const std::string&    outdir;

    public:
        GenAllArchIsa(const GpuArchManager& manager, const std::string& outdir)
            : manager(manager)
            , outdir(outdir)
        {
        }

        bool emit() const;
        void emitBanner(std::ofstream& os) const;
        bool emitUnifiedOpcodes(std::ofstream& os) const;
    };

    void GenAllArchIsa::emitBanner(std::ofstream& os) const
    {
        os << "//===----------------------------------------------------------------------===//\n"
           << "// Auto-generated ISA header for all architectures\n"
           << "//\n"
           << "// DO NOT EDIT MANUALLY!\n"
           << "// DO NOT USE #paragma once IN THIS FILE!\n"
           << "//===----------------------------------------------------------------------===//\n"
           << "\n";
    }

    bool GenAllArchIsa::emit() const
    {
        bool                  success = true;
        std::filesystem::path path    = std::filesystem::path(outdir) / ("hardware/gfxIsa.inc");

        std::ofstream os(path);
        if(!os)
        {
            std::cerr << "Error:Cannot write " << path << "\n";
            return false;
        }

        emitBanner(os);
        success &= emitUnifiedOpcodes(os);
        return success;
    }

    bool GenAllArchIsa::emitUnifiedOpcodes(std::ofstream& os) const
    {
        EmitMacroGuard emitMacro(os, "GET_ISAINFO_UNIFIED_OPCODES");

        size_t maxMnemonicLen = 0;
        for(const auto& opcode : manager.getAllOpcodes())
            maxMnemonicLen = std::max(maxMnemonicLen, opcode.size());

        size_t maxArchLen = 0;
        for(const auto& arch : manager.getRegisteredArchs())
            maxArchLen = std::max(maxArchLen, arch->getName().size());

        os << "enum GFX : uint16_t {\n";
        for(const auto& opcode : manager.getAllOpcodes())
        {
            os << "  " << std::left << std::setw(maxMnemonicLen) << opcode << ", // ";

            // list all archs that have this opcode
            for(const auto& arch : manager.getRegisteredArchs())
            {
                if(arch->has(opcode))
                    os << std::left << std::setw(maxArchLen) << arch->getName() << " ";
                else
                    os << std::left << std::setw(maxArchLen + 1) << " ";
            }

            os << "\n";
        }
        os << "};\n\n";

        return true;
    }
} // namespace anonymous

namespace stinkytofu
{
    bool genAllArchDefinitions(GpuArchManager& manager, const std::string& outdir)
    {
        bool success = true;
        for(const auto& arch : manager.getRegisteredArchs())
            success &= genIsaDefinitions(*arch, outdir);

        GenAllArchIsa generator(manager, outdir);
        success &= generator.emit();

        return success;
    }
} // namespace stinkytofu
