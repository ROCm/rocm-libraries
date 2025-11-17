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
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "ir/asm/StinkyAsmPrinter.hpp"
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
        while(!irlist.empty())
        {
            IRBase* ir = irlist.begin().getNodePtr();
            irlist.erase(irlist.begin());
            delete ir;
        }
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
    static void dumpStinkyInstList(const IRList& irlist, std::ostream& out)
    {
        const auto irlistString = toString(irlist);
        out << irlistString;

        out.flush();
    }

    // Run the scheduling pass on the instructions in the module.
    // This will build the use-def chain and then schedule the instructions in a DAG.
    void PassManager::run()
    {
        IRList& irlist = passCtx.getIRList();

        for(const auto& pass : passes)
        {
            if(dbgCfg && dbgCfg->shouldPrintBefore(pass->getName()))
            {
                dbgCfg->getOutputStreamInBefore()
                    << "\n*** Before Pass: " << pass->getName() << " ***\n";
                dumpStinkyInstList(irlist, dbgCfg->getOutputStreamInBefore());
            }

            pass->run(irlist, passCtx);

            if(dbgCfg && dbgCfg->shouldPrintAfter(pass->getName()))
            {
                dbgCfg->getOutputStreamInAfter()
                    << "\n*** After Pass: " << pass->getName() << " ***\n";
                dumpStinkyInstList(irlist, dbgCfg->getOutputStreamInAfter());
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
                                      uint32_t           wavefrontSz,
                                      uint32_t           numWaves)
    {
        StinkyKernelInfo kr;
        kr.arch          = arch;
        kr.TileA0        = ta0;
        kr.TileB0        = tb0;
        kr.TileM0        = tm0;
        kr.NumGRA        = nGRA;
        kr.NumGRB        = nGRB;
        kr.NumGRM        = nGRM;
        kr.WavefrontSize = wavefrontSz;
        kr.NumWaves      = wavefrontSz;
        passCtx.addKernelInfo(kr);
    }

    void PassManager::setOptConfig(const StinkyOptInfo& opt)
    {
        passCtx.setOptInfo(opt);
    }
} // namespace stinkytofu
