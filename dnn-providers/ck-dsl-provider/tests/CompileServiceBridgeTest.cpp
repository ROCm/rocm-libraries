// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>

#include "CkDslContainer.hpp"
#include "python/CompilePayload.hpp"
#include "python/CompileServiceBridge.hpp"

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::PayloadDict;

// compile_smoke drives the full frozen DSL flow (lower -> native comgr) without
// a GPU: comgr cross-compiles, so we get a real HSACO targeting any arch. This
// also exercises the bridge's mp_call -> KernelArtifact translation end to end.
TEST(TestCompileServiceBridge, CompileSmokeProducesHsaco) {
    CkDslContainer container;
    auto artifact = container.compileServiceBridge().compileSmoke("gfx950");

    EXPECT_FALSE(artifact.hsaco.empty()) << "compile_smoke must return a non-empty HSACO";
    EXPECT_FALSE(artifact.kernelName.empty());
    EXPECT_NE(artifact.isa.find("gfx950"), std::string::npos)
        << "compiled ISA '" << artifact.isa << "' does not target gfx950";
}

// An unknown op_kind raises ValueError in compile_service.compile; the bridge
// must surface that as a HipdnnPluginException across the nlr/C++ boundary
// (not longjmp past the C++ frames).
TEST(TestCompileServiceBridge, RaisesOnUnknownOpKind) {
    CkDslContainer container;
    auto& bridge = container.compileServiceBridge();
    const PayloadDict empty;

    try {
        bridge.compile("totally_unknown_op", empty, "gfx950");
        FAIL() << "expected compile() of an unknown op to throw";
    } catch (const hipdnn_plugin_sdk::HipdnnPluginException& error) {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
        const std::string message = error.getMessage();
        EXPECT_NE(message.find("unsupported op_kind"), std::string::npos)
            << "translated message missing Python detail: " << message;
    }
}

// is_applicable likewise raises on an unknown op_kind.
TEST(TestCompileServiceBridge, IsApplicableRaisesOnUnknownOpKind) {
    CkDslContainer container;
    auto& bridge = container.compileServiceBridge();
    const PayloadDict empty;

    EXPECT_THROW(bridge.isApplicable("totally_unknown_op", empty, "gfx950"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
