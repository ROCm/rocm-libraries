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
#include "stinkytofu/pipeline/Backend.hpp"

#include <string>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/ModulePassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
Backend::Backend(StinkyAsmModule& module) : module(module) {}

std::array<int, 3> Backend::getArch() const {
    return module.getArch();
}

bool Backend::runOptimization() {
    auto* pipeline = BackendRegistry::getArchPipeline(module.getArch());
    if (!pipeline || !pipeline->builder) return true;

    ModulePassManager mpm;
    if (!pipeline->builder(mpm, module, module.getPassBuilder())) return true;

    configurePassManager(mpm);
    mpm.run(module);
    return true;
}

void Backend::configurePassManager(ModulePassManager& pm) {
    const auto& opts = module.getModuleOptions();

    GemmTileConfig gemmTileConfig;
    gemmTileConfig.arch = module.getArch();
    gemmTileConfig.TileA0 = opts.TileA0;
    gemmTileConfig.TileB0 = opts.TileB0;
    gemmTileConfig.TileM0 = opts.TileM0;
    gemmTileConfig.NumGRA = opts.NumGRA;
    gemmTileConfig.NumGRB = opts.NumGRB;
    gemmTileConfig.NumGRM = opts.NumGRM;
    gemmTileConfig.NumWaves = opts.WaveGroup0 * opts.WaveGroup1;

    // Entry-point validation. This is the GEMM backend: a kernel arriving here
    // without a tile configuration is misconfigured, not merely unusual. 0 is
    // not a valid tile size and 0 waves is not an occupancy anything runs at,
    // so both mean the module options never carried the values. Fail loudly
    // instead of letting every downstream pass silently work from defaults and
    // schedule for a kernel shape that does not exist.
    //
    // Deliberately NOT in PassContext::setGemmTileConfig: that is the generic
    // config setter and has legitimate non-GEMM callers, notably
    // StinkyIRConverter::convertToFunction, which parses arbitrary asm text and
    // has no tile config to give.
    // Every tile dimension, not just the first: a config carrying TileA0 but
    // leaving TileB0 or TileM0 at 0 would pass a TileA0-only gate and schedule
    // for a zero-width tile, which is the same silent default this check exists
    // to stop.
    //
    // Scoped to OptLevel > O0, because that is exactly when the config is
    // consumed: Gfx1250Backend gates the DAG scheduler on
    // `runScheduler = optLevel != O0`, and the scheduler and the cycle
    // estimators are what read the tile shape. At O0 the backend is a
    // legalization/emission path, and it has real non-GEMM callers that have no
    // tile shape to give and are not wrong for that -- rocisa drives bare
    // instruction modules through it (rocisa/test/test_mubuf.py,
    // test_streamk_fences.py, test_pass_plugin.py all use OptLevel 0), as does
    // stinkytofu-opt on raw asm. Aborting there would fail a caller for not
    // supplying something nothing downstream is going to look at.
    const bool tileConfigIsUsed = opts.OptLevel > 0;
    const auto rejectUnsetTile = [tileConfigIsUsed](const char* name, uint32_t value) {
        if (!tileConfigIsUsed || value != 0) return;
        report_fatal_error(std::string("GemmTileConfig::") + name +
                           " is 0 at the backend entry, so the tile configuration was never "
                           "set. Set TileA0, TileB0 and TileM0 in the module options before "
                           "running the backend.");
    };
    rejectUnsetTile("TileA0", gemmTileConfig.TileA0);
    rejectUnsetTile("TileB0", gemmTileConfig.TileB0);
    rejectUnsetTile("TileM0", gemmTileConfig.TileM0);
    if (tileConfigIsUsed && gemmTileConfig.NumWaves == 0) {
        report_fatal_error(
            "GemmTileConfig::NumWaves is 0 at the backend entry (WaveGroup0 * WaveGroup1). Set "
            "WaveGroup0 and WaveGroup1 in the module options before running the backend.");
    }

    pm.setGemmTileConfig(gemmTileConfig);

    AsmCapsConfig asmCapsConfig;
    auto msbVal = opts.VgprMsbMode;
    if (msbVal < 0 || msbVal > static_cast<int>(VgprMsbMode::Msb16)) msbVal = 0;
    asmCapsConfig.vgprMsbMode = static_cast<VgprMsbMode>(msbVal);

    // When VgprMsbMode was not set explicitly (standalone path without rocisa),
    // auto-probe using comgr if available.
    if (asmCapsConfig.vgprMsbMode == VgprMsbMode::None) {
        auto arch = module.getArch();
        GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        asmCapsConfig = ToolchainCaps::probe(archId);
    }

    // After the probe above, which replaces the whole struct.
    asmCapsConfig.requiresXCntForVolatileVMEM = opts.RequiresXCntForVolatileVMEM;
    asmCapsConfig.enableXnackReplay = opts.EnableXnackReplay;

    pm.setAsmCapsConfig(asmCapsConfig);

    if (opts.EnableRemarks) {
        pm.getPassContext().setRemarksEnabled(true);
    }
}

}  // namespace stinkytofu
