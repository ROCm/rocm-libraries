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
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "ir/asm/IRParser.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmPrinter.hpp"
#include "isa/gfx/GfxIsa.hpp"
#include "stinkytofu.hpp"

namespace stinkytofu
{
    void IRBase::dump()
    {
        dump(std::cerr);
    }

    AnalysisManager::~AnalysisManager()
    {
        for(auto& entry : analysisPasses)
        {
            delete entry.second.first;
        }
    }

    void PassContext::cleanup()
    {
        // Use Function's deleteAllBasicBlocks method to clean up
        // This will clear all IR in each BasicBlock and delete all BasicBlocks
        if(function)
        {
            function->deleteAllBasicBlocks();
        }
    }

    void BasicBlock::dump(std::ostream& out) const
    {
        if(!label.empty())
        {
            out << "BasicBlock: " << label << "\n";
        }
        else
        {
            out << "BasicBlock (unlabeled)\n";
        }

        out << "  Number of instructions: " << ir.size() << "\n";

        for(const IRBase& irNode : ir)
        {
            out << "  ";
            irNode.dump(out);
        }

        if(!successors.empty())
        {
            out << "  Successors: ";
            for(size_t i = 0; i < successors.size(); ++i)
            {
                if(i > 0)
                    out << ", ";
                if(!successors[i]->getLabel().empty())
                    out << successors[i]->getLabel();
                else
                    out << "<unlabeled>";
            }
            out << "\n";
        }

        out.flush();
    }

    void Function::dump(std::ostream& out) const
    {
        out << "Function: " << name << "\n";
        out << "BasicBlocks: " << size() << "\n";
        out << "---\n";

        for(const BasicBlock& bb : basicBlocks)
        {
            bb.dump(out);
            out << "\n";
        }

        out.flush();
    }

    //----------------------------------------------------------------------
    // PassManager optional config implementation
    //----------------------------------------------------------------------
    static bool                            DebugFlag = false;
    static std::unordered_set<std::string> DebugTypes;

    bool isDebugOnlyEnabled(const char* TYPE)
    {
        return DebugFlag && DebugTypes.count(TYPE);
    }

    void PassManagerDebugConfig::addDebugOnly(const std::string& passName)
    {
        DebugFlag = true;
        DebugTypes.insert(passName);
    }

    void PassManagerDebugConfig::clearDebugOnly()
    {
        DebugFlag = false;
        DebugTypes.clear();
    }

    PassManagerDebugConfig::PassManagerDebugConfig()
        : printAfterAll(false)
        , printBeforeAll(false)
    {
    }

    PassManagerDebugConfig::~PassManagerDebugConfig() {}

    void PassManagerDebugConfig::setPrintAfterAll(bool v)
    {
        printAfterAll = v;
    }

    void PassManagerDebugConfig::setPrintBeforeAll(bool v)
    {
        printBeforeAll = v;
    }

    void PassManagerDebugConfig::addOnlyPrintBefore(const std::string& passName)
    {
        onlyPrintBefore.insert(passName);
    }

    void PassManagerDebugConfig::addOnlyPrintAfter(const std::string& passName)
    {
        onlyPrintAfter.insert(passName);
    }

    void PassManagerDebugConfig::setDumpToFileInBefore(const std::string& filename)
    {
        dumpStreamBefore = std::make_unique<std::ofstream>(filename, std::ofstream::out);
        if(static_cast<std::ofstream*>(dumpStreamBefore.get())->fail())
        {
            std::cerr << "Error: Unable to open file " << filename << " for writing.\n";
        }
    }

    void PassManagerDebugConfig::setDumpToFileInAfter(const std::string& filename)
    {
        dumpStreamAfter = std::make_unique<std::ofstream>(filename, std::ofstream::out);
        if(static_cast<std::ofstream*>(dumpStreamAfter.get())->fail())
        {
            std::cerr << "Error: Unable to open file " << filename << " for writing.\n";
        }
    }

    bool PassManagerDebugConfig::shouldPrintBefore(const std::string& passName) const
    {
        return printBeforeAll || onlyPrintBefore.count(passName);
    }

    bool PassManagerDebugConfig::shouldPrintAfter(const std::string& passName) const
    {
        return printAfterAll || onlyPrintAfter.count(passName);
    }

    std::ostream& PassManagerDebugConfig::getOutputStreamInBefore() const
    {
        if(dumpStreamBefore)
        {
            return *dumpStreamBefore.get();
        }
        return std::cout;
    }

    std::ostream& PassManagerDebugConfig::getOutputStreamInAfter() const
    {
        if(dumpStreamAfter)
        {
            return *dumpStreamAfter.get();
        }
        return std::cout;
    }

    //----------------------------------------------------------------------
    // PassManager implementation
    //----------------------------------------------------------------------
    // Run the passes on the Function.
    // Passes can operate on BasicBlocks and their instruction lists.
    void PassManager::run()
    {
        Function& func = passCtx.getFunction();

        for(const auto& pass : passes)
        {
            if(dbgCfg && dbgCfg->shouldPrintBefore(pass->getName()))
            {
                dbgCfg->getOutputStreamInBefore()
                    << "\n*** Before Pass: " << pass->getName() << " ***\n";
                func.dump(dbgCfg->getOutputStreamInBefore());
            }

            pass->run(func, passCtx);

            if(dbgCfg && dbgCfg->shouldPrintAfter(pass->getName()))
            {
                dbgCfg->getOutputStreamInAfter()
                    << "\n*** After Pass: " << pass->getName() << " ***\n";
                func.dump(dbgCfg->getOutputStreamInAfter());
            }
        }
    }

    void PassManager::setDebugConfig(std::unique_ptr<PassManagerDebugConfig> cfg)
    {
        dbgCfg = std::move(cfg);
    }

    void PassManager::setKernelConfig(std::array<int, 3> arch,
                                      uint32_t           ta0,
                                      uint32_t           tb0,
                                      uint32_t           tm0,
                                      uint32_t           nGRA,
                                      uint32_t           nGRB,
                                      uint32_t           nGRM,
                                      uint32_t           numWaves)
    {
        StinkyKernelInfo kr;
        kr.arch   = arch;
        kr.TileA0 = ta0;
        kr.TileB0 = tb0;
        kr.TileM0 = tm0;
        kr.NumGRA = nGRA;
        kr.NumGRB = nGRB;
        kr.NumGRM = nGRM;
        // Automatically determine wavefront size based on architecture
        kr.WavefrontSize = getWaveFrontSize(arch[0], arch[1], arch[2]);
        kr.NumWaves      = numWaves;
        passCtx.addKernelInfo(kr);
    }

    void PassManager::setOptConfig(const StinkyOptInfo& opt)
    {
        passCtx.setOptInfo(opt);
    }

    //----------------------------------------------------------------------
    // StinkyIRConverter implementation
    //----------------------------------------------------------------------

    StinkyIRConverter::StinkyIRConverter()
        : arch({9, 4, 2})
    {
    }

    StinkyIRConverter::StinkyIRConverter(const std::array<int, 3>& targetArch)
        : arch(targetArch)
    {
    }

    StinkyErrorCode StinkyIRConverter::populateFunctionFromString(const std::string& irText,
                                                                  Function&          func,
                                                                  PassContext&       passCtx,
                                                                  GfxArchID          arch)
    {
        // Parse the raw instruction string
        auto parsedInstructions = parseSourceString(irText);

        // Create an entry BasicBlock to hold all instructions
        BasicBlock* entryBB = func.createBasicBlock("entry");
        func.setEntryBlock(entryBB);
        IRList& irlist = entryBB->getIR();

        // Create the IR builder
        StinkyInstIRBuilder irBuilder = passCtx.getIRBuilder<StinkyInstIRBuilder>(irlist, arch);

        // Convert parsed instructions to StinkyInstruction objects
        for(const auto& inst : parsedInstructions)
        {
            // Check if it's a label
            if(inst->isLabel)
            {
                irBuilder.createStinkyLabel(irlist.end(), inst->opcodeStr);
                continue;
            }

            // Get the opcode and hardware instruction descriptor
            auto              opcode     = getMnemonicToIsaOpcode(inst->opcodeStr, arch);
            const HwInstDesc* hwInstDesc = getMCIDByIsaOp(opcode, arch);

            if(hwInstDesc == nullptr)
            {
                std::cerr << "Warning: No hardware instruction descriptor found for opcode "
                          << opcode << " in arch gfx" << static_cast<int>(arch) << "\n";
            }
            else
            {
                StinkyInstruction* stinkyInst
                    = irBuilder.createStinkyInstBefore(irlist.end(), hwInstDesc);

                // Move destination and source registers
                stinkyInst->destRegs = inst->destRegs;
                stinkyInst->srcRegs  = inst->srcRegs;

                // Overwrite cycles when valid (> 0), otherwise use default from HwInstDesc
                if(inst->issueCycles > 0)
                {
                    stinkyInst->issueCycles = inst->issueCycles;
                }

                if(inst->latencyCycles > 0)
                {
                    stinkyInst->latencyCycles = inst->latencyCycles;
                }
            }
        }

        return StinkyErrorCode::SUCCESS;
    }

    Function* StinkyIRConverter::convertToFunction(const std::string& rawInstructions)
    {
        // Create a fresh PassContext for this conversion
        passCtx = std::make_unique<PassContext>();

        // Set up kernel configuration
        StinkyKernelInfo kernelInfo;
        kernelInfo.arch   = arch;
        kernelInfo.TileA0 = 0;
        kernelInfo.TileB0 = 0;
        kernelInfo.TileM0 = 0;
        kernelInfo.NumGRA = 0;
        kernelInfo.NumGRB = 0;
        kernelInfo.NumGRM = 0;
        // Get WavefrontSize from hardware configuration based on architecture
        kernelInfo.WavefrontSize = getWaveFrontSize(arch[0], arch[1], arch[2]);
        kernelInfo.NumWaves      = 0;

        passCtx->addKernelInfo(kernelInfo);

        // Get the Function from PassContext
        Function& func = passCtx->getFunction();

        // Get the architecture ID
        GfxArchID archID = getGfxArchID(arch[0], arch[1], arch[2]);

        // Use the shared conversion logic
        StinkyErrorCode result
            = populateFunctionFromString(rawInstructions, func, *passCtx, archID);
        if(result != StinkyErrorCode::SUCCESS)
        {
            // Conversion failed, cleanup and return nullptr
            passCtx.reset();
            return nullptr;
        }

        return &func;
    }

    PassContext* StinkyIRConverter::getPassContext()
    {
        return passCtx.get();
    }

    void StinkyIRConverter::cleanup()
    {
        passCtx.reset();
    }

    StinkyIRConverter::~StinkyIRConverter()
    {
        cleanup();
    }

} // namespace stinkytofu
