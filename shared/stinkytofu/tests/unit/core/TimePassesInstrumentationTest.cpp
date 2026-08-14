// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <string>
#include <thread>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/TimePassesInstrumentation.hpp"

using namespace stinkytofu;

namespace {

// TimePassesInstrumentation attributes pipeline wall time to individual passes.
// The tests drive its callbacks directly: the Function and PassContext are only
// there to satisfy the PassInstrumentation interface.
constexpr auto kMeasurableWork = std::chrono::milliseconds(20);

// A pass that spends `kMeasurableWork` inside another pass, so the outer pass's
// self time must come out near zero while its total covers both.
std::string reportOfNestedRun() {
    Function func("entry");
    PassContext ctx;
    TimePassesInstrumentation timer;

    timer.beforePass("OuterPass", func, ctx);
    timer.beforePass("InnerPass", func, ctx);
    std::this_thread::sleep_for(kMeasurableWork);
    timer.afterPass("InnerPass", func, ctx);
    timer.afterPass("OuterPass", func, ctx);

    std::ostringstream report;
    timer.report(report, "kernelLabel");
    return report.str();
}

TEST(TimePassesInstrumentationTest, ReportsEveryPassWithItsLabel) {
    const std::string report = reportOfNestedRun();

    EXPECT_NE(report.find("kernelLabel"), std::string::npos) << report;
    EXPECT_NE(report.find("OuterPass"), std::string::npos) << report;
    EXPECT_NE(report.find("InnerPass"), std::string::npos) << report;
    EXPECT_NE(report.find("total"), std::string::npos) << report;
}

TEST(TimePassesInstrumentationTest, SelfTimeExcludesNestedPasses) {
    // Rows are ordered by descending self time, so the pass that did the work
    // must be listed ahead of the one that merely wrapped it.
    const std::string report = reportOfNestedRun();

    EXPECT_LT(report.find("InnerPass"), report.find("OuterPass")) << report;
}

TEST(TimePassesInstrumentationTest, UnbalancedCallbacksDoNotCrash) {
    Function func("entry");
    PassContext ctx;
    TimePassesInstrumentation timer;

    timer.afterPass("NeverEntered", func, ctx);

    std::ostringstream report;
    timer.report(report, "");
    EXPECT_EQ(report.str().find("NeverEntered"), std::string::npos) << report.str();
}

TEST(TimePassesSessionTest, EnabledSessionPublishesObserverAndPrintsOnExit) {
    std::ostringstream out;
    ASSERT_EQ(getActiveTimePasses(), nullptr);
    {
        TimePassesSession session(/*enable=*/true, "kernelLabel", out);
        EXPECT_NE(getActiveTimePasses(), nullptr);
        EXPECT_TRUE(out.str().empty());
    }
    EXPECT_EQ(getActiveTimePasses(), nullptr);
    EXPECT_NE(out.str().find("kernelLabel"), std::string::npos) << out.str();
}

TEST(TimePassesSessionTest, DisabledSessionIsInert) {
    std::ostringstream out;
    {
        TimePassesSession session(/*enable=*/false, "kernelLabel", out);
        EXPECT_EQ(getActiveTimePasses(), nullptr);
    }
    EXPECT_TRUE(out.str().empty()) << out.str();
}

TEST(TimePassesSessionTest, NestedSessionJoinsTheOpenOne) {
    // Backend opens a session per module; a driver that already opened one for
    // the whole run must keep ownership so only one report is printed.
    std::ostringstream outer;
    std::ostringstream inner;
    {
        TimePassesSession outerSession(/*enable=*/true, "run", outer);
        auto observer = getActiveTimePasses();
        {
            TimePassesSession innerSession(/*enable=*/true, "module", inner);
            EXPECT_EQ(getActiveTimePasses(), observer);
        }
        EXPECT_EQ(getActiveTimePasses(), observer);
        EXPECT_TRUE(inner.str().empty()) << inner.str();
    }
    EXPECT_NE(outer.str().find("run"), std::string::npos) << outer.str();
}

}  // namespace
