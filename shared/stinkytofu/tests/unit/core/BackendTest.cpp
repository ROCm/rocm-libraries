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
#include <gtest/gtest.h>

#include <array>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/pipeline/Backend.hpp"

using namespace stinkytofu;

namespace {

constexpr std::array<int, 3> kArch{12, 5, 0};

// Dummy but VALID tile options. The backend entry rejects a zeroed tile config
// (see configurePassManager), so a test that only cares about something else
// still has to look like a configured kernel. The values are arbitrary; only
// "not zero" is load-bearing.
std::unique_ptr<StinkyAsmModule> makeModule(const std::array<int, 3>& arch = kArch) {
    StinkyAsmModule::ModuleOptions opts{};
    opts.OptLevel = 0;
    opts.TileA0 = 128;
    opts.TileB0 = 128;
    opts.TileM0 = 32;
    opts.WaveGroup0 = 2;
    opts.WaveGroup1 = 2;
    return std::make_unique<StinkyAsmModule>("test", arch, opts);
}

// A module whose tile options were never filled in, i.e. what a caller that
// forgot to configure the backend actually produces.
std::unique_ptr<StinkyAsmModule> makeUnconfiguredModule() {
    StinkyAsmModule::ModuleOptions opts{};
    opts.OptLevel = 0;
    return std::make_unique<StinkyAsmModule>("test", kArch, opts);
}

}  // namespace

TEST(BackendTest, GetArchMatchesModuleArch) {
    auto module = makeModule();
    Backend backend(*module);
    EXPECT_EQ(backend.getArch(), kArch);
}

TEST(BackendTest, GetArchDifferentStepping) {
    std::array<int, 3> arch{9, 0, 10};
    auto module = makeModule(arch);
    Backend backend(*module);
    EXPECT_EQ(backend.getArch(), arch);
}

// runOptimization with no registered pipeline builder returns true (no-op success).
TEST(BackendTest, RunOptimizationWithNoPipelineSucceeds) {
    // Use an arch that has no registered PipelineBuilder so the early-exit
    // branch (BackendRegistry returns nullptr) is exercised.
    std::array<int, 3> arch{0, 0, 0};
    auto module = makeModule(arch);
    Backend backend(*module);
    EXPECT_TRUE(backend.runOptimization());
}

// ---------------------------------------------------------------------------
// Tile-config validation at the backend entry.
//
// A kernel reaching the GEMM backend without a tile configuration is
// misconfigured: 0 is not a valid tile size, and 0 waves is not an occupancy
// anything runs at. Both used to sail through and leave every downstream pass
// scheduling for a kernel shape that does not exist, so both now abort.
// ---------------------------------------------------------------------------

TEST(BackendTileConfigDeathTest, UnsetTileSizeAborts) {
    EXPECT_DEATH(
        {
            auto module = makeUnconfiguredModule();
            Backend backend(*module);
            backend.runOptimization();
        },
        "TileA0 is 0");
}

TEST(BackendTileConfigDeathTest, UnsetWaveGroupsAbort) {
    EXPECT_DEATH(
        {
            StinkyAsmModule::ModuleOptions opts{};
            opts.OptLevel = 0;
            opts.TileA0 = 128;
            opts.TileB0 = 128;
            opts.TileM0 = 32;
            // WaveGroup0/1 left at 0, so NumWaves comes out 0.
            StinkyAsmModule module("test", kArch, opts);
            Backend backend(module);
            backend.runOptimization();
        },
        "NumWaves is 0");
}

TEST(BackendTest, ConfiguredModuleRunsOptimization) {
    auto module = makeModule();
    Backend backend(*module);
    EXPECT_TRUE(backend.runOptimization());
}
