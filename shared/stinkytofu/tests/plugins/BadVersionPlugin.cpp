// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test-only fixture: a plugin that deliberately reports a stinkytofu version
// that cannot match whatever this test suite is actually built against.
// PassBuilder::loadPlugin() must reject it before ever calling registerPlugin()
// — see PluginLoadingTest.RejectsVersionMismatch in PassBuilderTest.cpp.

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/pipeline/PassBuilder.hpp"

#ifdef _WIN32
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
PLUGIN_EXPORT const char* stinkytofuPluginVersion() {
    return "0.0.0-not-a-real-build";
}

PLUGIN_EXPORT void registerPlugin() {
    // Intentionally minimal — loadPlugin() must reject this plugin on the
    // version check alone, before this function ever runs.
    stinkytofu::PassBuilder::registerNamedPassFactory(
        "BadVersionPluginShouldNeverRegister",
        [](stinkytofu::StinkyAsmModule&) -> std::unique_ptr<stinkytofu::Pass> { return nullptr; });
}
}
