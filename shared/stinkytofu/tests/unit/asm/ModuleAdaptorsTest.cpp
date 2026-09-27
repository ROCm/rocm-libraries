// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the module adaptors (ModuleAdaptors.hpp):
//   - FunctionToModuleAdaptor: runs its inner PassManager on EVERY function
//     (entry + callables), each exactly once.
//   - MainOnlyAdaptor: runs its inner PassManager on ONLY the entry function.
//
// A tiny counting Function pass records which functions it was run on, so the
// tests assert per-function visit counts and config propagation.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/ModulePassManager.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/pipeline/ModuleAdaptors.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;
constexpr std::array<int, 3> kArchTriple = {12, 5, 0};

// Records the name of every Function it is run on, and the GEMM arch it observed
// via the (propagated) PassContext — proving config reached the inner PM.
class RecordingPass : public Pass {
   public:
    static char ID;
    explicit RecordingPass(std::vector<std::string>* visited, std::vector<int>* archMajor)
        : visited(visited), archMajor(archMajor) {}

    const char* getName() const override {
        return "RecordingPass";
    }
    Pass::ID getPassID() const override {
        return &ID;
    }

    PreservedAnalyses run(Function& func, PassContext& ctx, AnalysisManager&) override {
        visited->push_back(func.getName());
        archMajor->push_back(ctx.getGemmTileConfig().arch[0]);
        return PreservedAnalyses::none();
    }

   private:
    std::vector<std::string>* visited;
    std::vector<int>* archMajor;
};
char RecordingPass::ID = 0;

class ModuleAdaptorsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        module = std::make_unique<StinkyAsmModule>("adaptor_test", kArchTriple,
                                                   StinkyAsmModule::ModuleOptions{});
        Function& entry = module->getFunction();
        entry.setName("entry_func");
        setFunctionArch(entry, kArch);

        for (const char* name : {"callee_a", "callee_b"}) {
            Function& callable = module->createFunction(name);
            setFunctionArch(callable, kArch);
        }
    }

    // A ModulePassManager configured with a real GEMM arch, so we can verify the
    // adaptors propagate config down to the inner PM.
    ModulePassManager makeConfiguredMPM() {
        ModulePassManager mpm;
        GemmTileConfig cfg;
        cfg.arch = kArchTriple;
        mpm.setGemmTileConfig(cfg);
        return mpm;
    }

    std::unique_ptr<StinkyAsmModule> module;
    std::vector<std::string> visited;
    std::vector<int> archMajor;
};

TEST_F(ModuleAdaptorsTest, FunctionToModuleAdaptorVisitsEveryFunctionOnce) {
    PassManager fpm;
    fpm.addPass(std::make_unique<RecordingPass>(&visited, &archMajor));

    ModulePassManager mpm = makeConfiguredMPM();
    mpm.addPass(createFunctionToModuleAdaptor(std::move(fpm)));
    mpm.run(*module);

    // Entry first (getFunctions() returns [entry, ...callables]), then callables,
    // each exactly once.
    EXPECT_EQ(visited, (std::vector<std::string>{"entry_func", "callee_a", "callee_b"}));
}

TEST_F(ModuleAdaptorsTest, MainOnlyAdaptorVisitsEntryOnly) {
    PassManager fpm;
    fpm.addPass(std::make_unique<RecordingPass>(&visited, &archMajor));

    ModulePassManager mpm = makeConfiguredMPM();
    mpm.addPass(createMainOnlyAdaptor(std::move(fpm)));
    mpm.run(*module);

    EXPECT_EQ(visited, (std::vector<std::string>{"entry_func"}));
}

TEST_F(ModuleAdaptorsTest, AdaptorPropagatesConfigToInnerPM) {
    PassManager fpm;
    fpm.addPass(std::make_unique<RecordingPass>(&visited, &archMajor));

    ModulePassManager mpm = makeConfiguredMPM();
    mpm.addPass(createFunctionToModuleAdaptor(std::move(fpm)));
    mpm.run(*module);

    // Every run saw the configured arch (12), i.e. config reached the inner PM.
    ASSERT_EQ(archMajor.size(), 3u);
    for (int major : archMajor) EXPECT_EQ(major, 12);
}

}  // namespace
