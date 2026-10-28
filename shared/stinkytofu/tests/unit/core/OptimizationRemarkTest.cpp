// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

using namespace stinkytofu;

TEST(OptimizationRemarkTest, HandlerReceivesKindAndRemarkName) {
    PassContext ctx;
    ctx.setRemarksEnabled(true);
    std::vector<OptimizationRemark> received;
    ctx.setRemarkHandler([&](const OptimizationRemark& remark) { received.push_back(remark); });

    std::ostringstream captured;
    std::streambuf* original = std::cerr.rdbuf(captured.rdbuf());
    emitRemark(ctx, {OptimizationRemark::Kind::Missed, "SomePass", "NotDone", "why not"});
    emitRemark(ctx, {OptimizationRemark::Kind::Passed, "SomePass", "Done", "what changed"});
    std::cerr.rdbuf(original);

    ASSERT_EQ(received.size(), 2u);
    EXPECT_EQ(received[0].kind, OptimizationRemark::Kind::Missed);
    EXPECT_EQ(received[0].passName, "SomePass");
    EXPECT_EQ(received[0].remarkName, "NotDone");
    EXPECT_EQ(received[0].message, "why not");
    EXPECT_EQ(received[1].kind, OptimizationRemark::Kind::Passed);
    EXPECT_EQ(received[1].remarkName, "Done");
    EXPECT_EQ(received[1].message, "what changed");
    EXPECT_EQ(captured.str(), "missed: SomePass: why not\nremark: SomePass: what changed\n");
}

TEST(OptimizationRemarkTest, DisabledRemarksReachNoHandler) {
    PassContext ctx;
    bool called = false;
    ctx.setRemarkHandler([&](const OptimizationRemark&) { called = true; });
    emitRemark(ctx, {OptimizationRemark::Kind::Analysis, "SomePass", "Info", "text"});
    EXPECT_FALSE(called);
}
