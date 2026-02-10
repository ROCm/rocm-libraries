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

#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/ir/asm/DefUseChain.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/core/stinkytofu.hpp"

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

    void PassContext::setGemmTileConfig(const GemmTileConfig& config)
    {
        gemmConfig = config;
        // Automatically compute WavefrontSize from architecture
        // WavefrontSize is stored separately as it's derived, not configured
        if(gemmConfig.arch[0] != 0)
        {
            wavefrontSize
                = getWaveFrontSize(gemmConfig.arch[0], gemmConfig.arch[1], gemmConfig.arch[2]);
        }
        else
        {
            STINKY_UNREACHABLE("Invalid architecture, unable to compute wavefront size");
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

        // Build use-def chains once before running passes
        // TODO: Remove this once all IR generation uses automatic builder setters
        buildUseDefChain(func);

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

    void PassManager::setGemmTileConfig(const GemmTileConfig& config)
    {
        passCtx.setGemmTileConfig(config);
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
        GemmTileConfig config;
        config.arch     = arch;
        config.TileA0   = ta0;
        config.TileB0   = tb0;
        config.TileM0   = tm0;
        config.NumGRA   = nGRA;
        config.NumGRB   = nGRB;
        config.NumGRM   = nGRM;
        config.NumWaves = numWaves;
        // WavefrontSize is automatically computed by setGemmTileConfig from arch
        passCtx.setGemmTileConfig(config);
    }

    void PassManager::setPassFeatureConfig(const PassFeatureConfig& config)
    {
        passCtx.setPassFeatureConfig(config);
    }

    void PassManager::setFunction(Function& externalFunc)
    {
        // Transfer BasicBlocks from external Function to PassContext's internal Function
        Function& internalFunc = passCtx.getFunction();

        // Clear any existing BasicBlocks in internal Function
        internalFunc.deleteAllBasicBlocks();

        // Copy function name
        internalFunc.setName(externalFunc.getName());

        // Transfer BasicBlocks by moving them from external to internal
        // We need to detach from external and attach to internal
        while(!externalFunc.getBasicBlocks().empty())
        {
            BasicBlock* bb = &externalFunc.getBasicBlocks().front();
            externalFunc.getBasicBlocks().remove(bb);
            bb->setParent(&internalFunc);
            internalFunc.getBasicBlocks().push_back(bb);
        }

        // Transfer entry block pointer
        if(externalFunc.getEntryBlock())
        {
            internalFunc.setEntryBlock(externalFunc.getEntryBlock());
        }

        // Copy GEMM configuration if available
        internalFunc.setGemmTileConfig(externalFunc.getGemmTileConfig());
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
                stinkyInst->setDestRegs(inst->destRegs);
                stinkyInst->setSrcRegs(inst->srcRegs);

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
        GemmTileConfig config;
        config.arch     = arch;
        config.TileA0   = 0;
        config.TileB0   = 0;
        config.TileM0   = 0;
        config.NumGRA   = 0;
        config.NumGRB   = 0;
        config.NumGRM   = 0;
        config.NumWaves = 0;
        // WavefrontSize is automatically computed by setGemmTileConfig from arch
        passCtx->setGemmTileConfig(config);

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
