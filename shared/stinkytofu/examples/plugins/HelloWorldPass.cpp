// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HelloWorldPass.hpp"

#include "stinkytofu/bindings/python/Module.hpp"

namespace stinkytofu {
namespace {

class HelloWorldPassImpl : public Pass {
    static char ID;
    StinkyAsmModule& module_;

   public:
    explicit HelloWorldPassImpl(StinkyAsmModule& module) : module_(module) {}

    Pass::ID getPassID() const override {
        return &ID;
    }
    const char* getName() const override {
        return "HelloWorldPass";
    }

    PreservedAnalyses run(Function& func, PassContext& ctx, AnalysisManager& AM) override {
        (void)func;
        (void)ctx;
        (void)AM;
        std::string greeting = module_.getPluginDataStr("greeting", "Hello from plugin!");
        module_.setPluginDataStr("greeting_result", "executed: " + greeting);
        module_.setPluginDataI64("pass_executed", 1);
        return PreservedAnalyses::all();
    }
};

char HelloWorldPassImpl::ID = 0;

}  // namespace

void registerHelloWorldPassPlugin() {
    PassBuilder::registerNamedPassFactory("HelloWorldPass",
                                          [](StinkyAsmModule& module) -> std::unique_ptr<Pass> {
                                              return std::make_unique<HelloWorldPassImpl>(module);
                                          });
}

}  // namespace stinkytofu

#ifdef _WIN32
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
PLUGIN_EXPORT void registerPlugin() {
    stinkytofu::registerHelloWorldPassPlugin();
}
}
