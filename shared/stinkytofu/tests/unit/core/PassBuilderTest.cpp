/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/pipeline/PassBuilder.hpp"

// Static build of the example links the OBJECT lib and registers the pass directly,
// so it needs the header; the shared build dlopens the MODULE instead (path define).
#if defined(STINKYTOFU_BUILD_EXAMPLES) && !defined(STINKYTOFU_PLUGIN_HELLOWORLD_PATH)
#include "HelloWorldPass.hpp"
#endif

using namespace stinkytofu;

// A trivial pass that increments a counter for testing.
namespace {
class CounterPass : public Pass {
    static char ID;
    int& counter_;

   public:
    explicit CounterPass(int& counter) : counter_(counter) {}
    Pass::ID getPassID() const override {
        return &ID;
    }
    const char* getName() const override {
        return "CounterPass";
    }
    PreservedAnalyses run(Function& func, PassContext& ctx, AnalysisManager& AM) override {
        (void)func;
        (void)ctx;
        (void)AM;
        ++counter_;
        return PreservedAnalyses::all();
    }
};
char CounterPass::ID = 0;
}  // namespace

TEST(PassBuilderTest, NoCallbacksApplyIsNoop) {
    PassBuilder PB;
    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);
    PassManager pm;
    PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, pm, module);
    // No crash, no passes added — just a no-op.
}

TEST(PassBuilderTest, RegisterAndApplyCallback) {
    PassBuilder PB;
    int counter = 0;
    PB.registerAtExtensionPoint(PipelineExtensionPoint::InnerRegionEnd,
                                [&counter](PassManager& PM, StinkyAsmModule&) {
                                    PM.addPass(std::make_unique<CounterPass>(counter));
                                });

    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);
    PassManager pm;
    PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, pm, module);

    pm.run(module.getFunction());
    EXPECT_EQ(counter, 1);
}

TEST(PassBuilderTest, MultipleCallbacksSameExtensionPoint) {
    PassBuilder PB;
    int counter = 0;
    auto makeCB = [&counter](PassManager& PM, StinkyAsmModule&) {
        PM.addPass(std::make_unique<CounterPass>(counter));
    };
    PB.registerAtExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, makeCB);
    PB.registerAtExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, makeCB);

    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);
    PassManager pm;
    PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, pm, module);

    pm.run(module.getFunction());
    EXPECT_EQ(counter, 2);
}

TEST(PassBuilderTest, DifferentExtensionPointsAreIndependent) {
    PassBuilder PB;
    int innerCounter = 0;
    int outerCounter = 0;
    PB.registerAtExtensionPoint(PipelineExtensionPoint::InnerRegionEnd,
                                [&innerCounter](PassManager& PM, StinkyAsmModule&) {
                                    PM.addPass(std::make_unique<CounterPass>(innerCounter));
                                });
    PB.registerAtExtensionPoint(PipelineExtensionPoint::AfterRegionPasses,
                                [&outerCounter](PassManager& PM, StinkyAsmModule&) {
                                    PM.addPass(std::make_unique<CounterPass>(outerCounter));
                                });

    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);

    PassManager pm1;
    PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, pm1, module);
    pm1.run(module.getFunction());
    EXPECT_EQ(innerCounter, 1);
    EXPECT_EQ(outerCounter, 0);

    PassManager pm2;
    PB.applyExtensionPoint(PipelineExtensionPoint::AfterRegionPasses, pm2, module);
    pm2.run(module.getFunction());
    EXPECT_EQ(innerCounter, 1);
    EXPECT_EQ(outerCounter, 1);
}

TEST(PassBuilderTest, NamedPassFactoryRegisterAndCreate) {
    int counter = 0;
    PassBuilder::registerNamedPassFactory("TestCounterPass", [&counter](StinkyAsmModule&) {
        return std::make_unique<CounterPass>(counter);
    });

    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);
    auto pass = PassBuilder::createPassByName("TestCounterPass", module);
    ASSERT_NE(pass, nullptr);
    EXPECT_STREQ(pass->getName(), "CounterPass");
}

TEST(PassBuilderTest, CreateUnknownPassReturnsNull) {
    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);
    auto pass = PassBuilder::createPassByName("NonExistentPass", module);
    EXPECT_EQ(pass, nullptr);
}

// --- PluginData on StinkyAsmModule ---

TEST(PluginDataTest, SetAndGetI64) {
    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);

    module.setPluginDataI64("enableTDMA", 1);
    module.setPluginDataI64("enableTDMB", 0);

    EXPECT_EQ(module.getPluginDataI64("enableTDMA"), 1);
    EXPECT_EQ(module.getPluginDataI64("enableTDMB"), 0);
    EXPECT_EQ(module.getPluginDataI64("missing", -1), -1);
}

TEST(PluginDataTest, SetAndGetStr) {
    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);

    module.setPluginDataStr("kernelName", "my_kernel");

    EXPECT_EQ(module.getPluginDataStr("kernelName"), "my_kernel");
    EXPECT_EQ(module.getPluginDataStr("missing", "default"), "default");
}

TEST(PluginDataTest, OverwriteValue) {
    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);

    module.setPluginDataI64("key", 42);
    EXPECT_EQ(module.getPluginDataI64("key"), 42);

    module.setPluginDataI64("key", 99);
    EXPECT_EQ(module.getPluginDataI64("key"), 99);
}

TEST(PluginDataTest, PassBuilderAccessFromModule) {
    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);

    int counter = 0;
    module.getPassBuilder().registerAtExtensionPoint(
        PipelineExtensionPoint::InnerRegionEnd, [&counter](PassManager& PM, StinkyAsmModule&) {
            PM.addPass(std::make_unique<CounterPass>(counter));
        });

    PassManager pm;
    module.getPassBuilder().applyExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, pm, module);
    pm.run(module.getFunction());
    EXPECT_EQ(counter, 1);
}

// loadPlugin() must fail softly on a path that does not exist — return false,
// not crash — so a caller that probes for an optional plugin stays safe.
// (This is the load API's robustness; the libstinkytofu-must-not-DT_NEEDED-a-
// plugin invariant behind #8342 is a static ELF property and is asserted by the
// LibStinkytofuHasNoPluginDependency CTest, not from inside this binary — which
// links libstinkytofu and so could not even start if that invariant were broken.)
TEST(PluginLoadingTest, LoadMissingPluginReturnsFalse) {
    EXPECT_FALSE(PassBuilder::loadPlugin("/nonexistent/libstinkytofu-plugin-does-not-exist.so"));
}

// Note: examplePluginPath() resolves the plugin relative to the *installed*
// libstinkytofu layout (<libdir>/stinkytofu/plugins), which does not exist in the
// build tree, so it is not asserted here. Its end-to-end resolution is covered by
// rocisa's test_pass_plugin.py running against the installed wheel.

// --- HelloWorldPass example plugin integration test ---
// Gated on STINKYTOFU_BUILD_EXAMPLES: only compiled when the example plugins are built.
// LLVM-style: shared builds load the plugin dynamically via loadPlugin(),
// static builds link the OBJECT lib and call registerHelloWorldPassPlugin() directly.
#ifdef STINKYTOFU_BUILD_EXAMPLES
TEST(PluginIntegrationTest, HelloWorldPassReadsAndWritesPluginData) {
#ifdef STINKYTOFU_PLUGIN_HELLOWORLD_PATH
    ASSERT_TRUE(PassBuilder::loadPlugin(STINKYTOFU_PLUGIN_HELLOWORLD_PATH));
#else
    registerHelloWorldPassPlugin();
#endif

    StinkyAsmModule::ModuleOptions opts{};
    StinkyAsmModule module("test", {12, 5, 0}, opts);

    module.setPluginDataStr("greeting", "Hello from test!");
    module.getPassBuilder().registerAtExtensionPoint(
        PipelineExtensionPoint::AfterRegionPasses, [](PassManager& PM, StinkyAsmModule& mod) {
            auto pass = PassBuilder::createPassByName("HelloWorldPass", mod);
            if (pass) PM.addPass(std::move(pass));
        });

    {
        PassManager pm;
        module.getPassBuilder().applyExtensionPoint(PipelineExtensionPoint::AfterRegionPasses, pm,
                                                    module);
        pm.run(module.getFunction());
    }

    EXPECT_EQ(module.getPluginDataI64("pass_executed"), 1);
    EXPECT_EQ(module.getPluginDataStr("greeting_result"), "executed: Hello from test!");

    PassBuilder::unloadPlugins();
}
#endif  // STINKYTOFU_BUILD_EXAMPLES
