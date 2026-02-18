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
#include <fstream>
#include <iostream>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/DefUseChain.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu
{
    AnalysisManager::~AnalysisManager()
    {
        for(auto& entry : analysisPasses)
        {
            delete entry.second.first;
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
    void PassManager::run(Function& F)
    {
        // Build use-def chains once before running passes
        // TODO: Remove this once all IR generation uses automatic builder setters
        buildUseDefChain(F);

        for(const auto& pass : passes)
        {
            if(dbgCfg && dbgCfg->shouldPrintBefore(pass->getName()))
            {
                dbgCfg->getOutputStreamInBefore()
                    << "\n*** Before Pass: " << pass->getName() << " ***\n";
                F.dump(dbgCfg->getOutputStreamInBefore());
            }

            pass->run(F, passCtx);

            if(dbgCfg && dbgCfg->shouldPrintAfter(pass->getName()))
            {
                dbgCfg->getOutputStreamInAfter()
                    << "\n*** After Pass: " << pass->getName() << " ***\n";
                F.dump(dbgCfg->getOutputStreamInAfter());
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
}
