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

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"

using namespace stinkytofu;

namespace {

constexpr std::array<int, 3> kTestArch{99, 9, 0};
constexpr std::array<int, 3> kOtherArch{98, 8, 1};

/// One rule whose status follows a capability, which is the shape every real
/// per-arch table has.
AllocationRules probeTable(bool gateOpen) {
    AllocationRule rule;
    rule.name = "ProbeRule";
    rule.description = "a rule that exists to be found";
    rule.status = gateOpen ? RuleStatus::Active : RuleStatus::Off;
    rule.forbidsBase = [](RegType, uint32_t, uint32_t) { return false; };
    return AllocationRules({rule});
}

/// Registrations are process-global, so each test cleans up after itself.
class AllocationRulesRegistryTest : public ::testing::Test {
   protected:
    void TearDown() override {
        AllocationRulesRegistry::clearArch(kTestArch);
        AllocationRulesRegistry::clearArch(kOtherArch);
    }
};

}  // namespace

TEST_F(AllocationRulesRegistryTest, AnUnregisteredTripleYieldsAnEmptyTable) {
    // The state the tree ships in, so the case that matters most: every caller
    // gets a table, and that table forbids nothing. No null anywhere.
    const AllocationRules rules = AllocationRulesRegistry::forArch(kTestArch, AsmCapsConfig{});
    EXPECT_TRUE(rules.empty());
    EXPECT_EQ(rules.forbidsBase(RegType::V, 0, 1), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::S, 7, 4), nullptr);
    EXPECT_FALSE(rules.prices());
    EXPECT_FALSE(AllocationRulesRegistry::hasArch(kTestArch));
}

TEST_F(AllocationRulesRegistryTest, ARegisteredTripleYieldsItsTable) {
    AllocationRulesRegistry::setArch(kTestArch,
                                     [](const AsmCapsConfig&) { return probeTable(true); });

    EXPECT_TRUE(AllocationRulesRegistry::hasArch(kTestArch));
    const AllocationRules rules = AllocationRulesRegistry::forArch(kTestArch, AsmCapsConfig{});
    ASSERT_EQ(rules.all().size(), 1u);
    EXPECT_EQ(rules.all()[0].name, "ProbeRule");
    EXPECT_EQ(rules.all()[0].status, RuleStatus::Active);
}

TEST_F(AllocationRulesRegistryTest, ARegistrationDoesNotLeakToAnotherTriple) {
    AllocationRulesRegistry::setArch(kTestArch,
                                     [](const AsmCapsConfig&) { return probeTable(true); });
    EXPECT_TRUE(AllocationRulesRegistry::forArch(kOtherArch, AsmCapsConfig{}).empty());
}

TEST_F(AllocationRulesRegistryTest, CapsParameterizeRatherThanSelect) {
    // The triple decides which rules exist; capabilities decide whether a
    // conditional one is in force. Same key, two answers.
    bool gateOpen = false;
    AllocationRulesRegistry::setArch(
        kTestArch, [&gateOpen](const AsmCapsConfig&) { return probeTable(gateOpen); });

    EXPECT_EQ(AllocationRulesRegistry::forArch(kTestArch, AsmCapsConfig{}).all()[0].status,
              RuleStatus::Off);
    gateOpen = true;
    EXPECT_EQ(AllocationRulesRegistry::forArch(kTestArch, AsmCapsConfig{}).all()[0].status,
              RuleStatus::Active);
}

TEST_F(AllocationRulesRegistryTest, AGatedOffRuleIsStillListed) {
    // "Rule present, capability unset" and "no rule" colour identically and want
    // different fixes, so the table must not hide the inert one.
    AllocationRulesRegistry::setArch(kTestArch,
                                     [](const AsmCapsConfig&) { return probeTable(false); });

    const AllocationRules rules = AllocationRulesRegistry::forArch(kTestArch, AsmCapsConfig{});
    ASSERT_EQ(rules.all().size(), 1u);
    EXPECT_EQ(rules.all()[0].status, RuleStatus::Off);
    EXPECT_TRUE(rules.toString().find("ProbeRule") != std::string::npos) << rules.toString();
}

TEST_F(AllocationRulesRegistryTest, ASecondRegistrationReplacesTheFirst) {
    AllocationRulesRegistry::setArch(kTestArch,
                                     [](const AsmCapsConfig&) { return probeTable(true); });
    AllocationRulesRegistry::setArch(kTestArch,
                                     [](const AsmCapsConfig&) { return probeTable(false); });

    EXPECT_EQ(AllocationRulesRegistry::forArch(kTestArch, AsmCapsConfig{}).all()[0].status,
              RuleStatus::Off);
}

TEST_F(AllocationRulesRegistryTest, AnEmptyFactoryYieldsAnEmptyTable) {
    AllocationRulesRegistry::setArch(kTestArch, nullptr);
    EXPECT_TRUE(AllocationRulesRegistry::forArch(kTestArch, AsmCapsConfig{}).empty());
}

TEST_F(AllocationRulesRegistryTest, ClearArchRemovesOnlyThatTriple) {
    AllocationRulesRegistry::setArch(kTestArch,
                                     [](const AsmCapsConfig&) { return probeTable(true); });
    AllocationRulesRegistry::setArch(kOtherArch,
                                     [](const AsmCapsConfig&) { return probeTable(true); });

    AllocationRulesRegistry::clearArch(kTestArch);
    EXPECT_FALSE(AllocationRulesRegistry::hasArch(kTestArch));
    EXPECT_TRUE(AllocationRulesRegistry::hasArch(kOtherArch));
}

TEST_F(AllocationRulesRegistryTest, RegisteredKeysAreSortedAndSpellTheTriple) {
    AllocationRulesRegistry::setArch(kTestArch,
                                     [](const AsmCapsConfig&) { return probeTable(true); });
    AllocationRulesRegistry::setArch(kOtherArch,
                                     [](const AsmCapsConfig&) { return probeTable(true); });

    const std::vector<std::string> keys = AllocationRulesRegistry::archKeys();
    EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
    const std::set<std::string> unique(keys.begin(), keys.end());
    EXPECT_EQ(keys.size(), unique.size());
    // "gfx" + major + minor + stepping-as-hex-digit, as BackendRegistry spells it.
    EXPECT_NE(unique.find("gfx9990"), unique.end());
    EXPECT_NE(unique.find("gfx9881"), unique.end());
}

TEST_F(AllocationRulesRegistryTest, RegisterAllLinksTheSelfRegisteringTables) {
    // The anchors are what keep those TUs in a static archive, so this asserts
    // linkage as much as content: a dropped anchor takes its registrar with it
    // and the table silently disappears.
    AllocationRulesRegistry::registerAll();
    const std::vector<std::string> keys = AllocationRulesRegistry::archKeys();
    EXPECT_FALSE(keys.empty()) << "registerAll linked no rules TU";
}

TEST_F(AllocationRulesRegistryTest, EveryRegisteredArchDeclaresUsableRules) {
    // It exists so a registered table is held to these invariants rather than
    // after someone remembers to write this test.
    AllocationRulesRegistry::registerAll();
    for (const std::string& key : AllocationRulesRegistry::archKeys()) {
        SCOPED_TRACE(key);
        std::array<int, 3> arch{};
        ASSERT_TRUE(BackendRegistry::parseArchKey(key, arch));

        const AllocationRules rules = AllocationRulesRegistry::forArch(arch, AsmCapsConfig{});
        EXPECT_TRUE(rules.problems().empty()) << rules.toString();

        std::set<std::string_view> names;
        for (const AllocationRule& rule : rules.all()) {
            EXPECT_FALSE(rule.name.empty());
            EXPECT_FALSE(rule.description.empty());
            EXPECT_NE(rule.kind(), RuleKind::Empty) << rule.name;
            EXPECT_TRUE(names.insert(rule.name).second) << "duplicate rule name " << rule.name;
        }
        // Not asserted: that every rule is inert with capabilities unset. A rule
        // may be an unconditional property of the ISA -- scalar tuple alignment
        // is one -- and gating those on a capability would be wrong.
    }
}
