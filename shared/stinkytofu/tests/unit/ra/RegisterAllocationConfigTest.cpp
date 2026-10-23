// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationConfig.hpp"

using namespace stinkytofu;

namespace {

using Dialect = RegisterAllocationDialect;
using Mode = RegisterAllocationMode;
using Unbankable = RegisterAllocationOptions::UnbankableOperands;
using Quadrant = RegisterAllocationOptions::Quadrant;

RegisterAllocationConfig parsed(std::string_view text, Dialect dialect = Dialect::Kernel) {
    Expected<RegisterAllocationConfig> config = parseRegisterAllocationConfig(text, dialect);
    EXPECT_FALSE(config.hasError()) << text << ": " << config.getError();
    return config.hasError() ? RegisterAllocationConfig{} : *config;
}

std::string errorOf(std::string_view text, Dialect dialect = Dialect::Kernel) {
    Expected<RegisterAllocationConfig> config = parseRegisterAllocationConfig(text, dialect);
    EXPECT_TRUE(config.hasError()) << text << " should not parse";
    return config.hasError() ? config.getError() : std::string{};
}

std::string canonical(std::string_view text) {
    return canonicalRegisterAllocationConfig(parsed(text));
}

}  // namespace

TEST(RegisterAllocationConfigTest, AKernelDefaultsToOffWithBothClassesAndTheChain) {
    const RegisterAllocationConfig config = parsed("");
    EXPECT_EQ(config.mode, Mode::Off);
    EXPECT_EQ(config.options.allocate, RegClassSet::all());
    EXPECT_EQ(config.options.allocator,
              "greedy-compact+greedy-compact-largest+greedy-compact-program");
    EXPECT_EQ(config.options.unbankableOperands, Unbankable::Auto);
    EXPECT_EQ(config.options.quadrant, Quadrant::Any);
    EXPECT_FALSE(config.options.applyToOperands);
    EXPECT_TRUE(config.options.verify);
    EXPECT_FALSE(config.options.report);
}

TEST(RegisterAllocationConfigTest, TheToolDefaultsToShadowingVgprsWithGreedy) {
    const RegisterAllocationConfig config = parsed("", Dialect::Tool);
    EXPECT_EQ(config.mode, Mode::Shadow);
    EXPECT_EQ(config.options.allocate, RegClassSet::only(RegType::V));
    EXPECT_EQ(config.options.allocator, "greedy");
    EXPECT_EQ(config.options.unbankableOperands, Unbankable::Auto);
    EXPECT_FALSE(config.options.applyToOperands);
}

TEST(RegisterAllocationConfigTest, AModeIsAWordADigitOrTheModeKey) {
    for (const char* text : {"apply", "2", "mode=apply", "mode=2"}) {
        const RegisterAllocationConfig config = parsed(text);
        EXPECT_EQ(config.mode, Mode::Apply) << text;
        EXPECT_TRUE(config.options.applyToOperands) << text;
    }
    EXPECT_EQ(parsed("shadow").mode, Mode::Shadow);
    EXPECT_FALSE(parsed("shadow").options.applyToOperands);
    EXPECT_EQ(parsed("force").mode, Mode::Force);
    EXPECT_TRUE(parsed("force").options.applyToOperands);
    EXPECT_EQ(parsed("classes=vs,3").mode, Mode::Force);
}

TEST(RegisterAllocationConfigTest, TwoDifferentModesAreAnError) {
    EXPECT_NE(errorOf("apply,shadow").find("two modes"), std::string::npos);
    EXPECT_EQ(parsed("apply,2").mode, Mode::Apply);
}

TEST(RegisterAllocationConfigTest, EveryKeyReachesTheOptions) {
    const RegisterAllocationConfig config = parsed(
        "apply,classes=vs,allocator=greedy-compact-freedom+greedy,unbankable=hold,"
        "quadrant=prefer,rules=A+B,rules=C,noRule=D,ruleAudit,noRules,pinReg=s0:3+v5,"
        "pinReg=s9,regionEnd=^loop,noVerify");
    const RegisterAllocationOptions& options = config.options;
    EXPECT_EQ(options.allocate, RegClassSet::all());
    EXPECT_EQ(options.allocator, "greedy-compact-freedom+greedy");
    EXPECT_EQ(options.unbankableOperands, Unbankable::Hold);
    EXPECT_EQ(options.quadrant, Quadrant::Prefer);
    EXPECT_EQ(options.rules.activate, (std::vector<std::string>{"A", "B", "C"}));
    EXPECT_EQ(options.rules.disable, (std::vector<std::string>{"D"}));
    EXPECT_TRUE(options.rules.auditAll);
    EXPECT_TRUE(options.rules.disableAll);
    ASSERT_EQ(options.pinRegisters.size(), 3u);
    EXPECT_EQ(options.pinRegisters[0].regClass, RegType::S);
    EXPECT_EQ(options.pinRegisters[0].end, 3u);
    EXPECT_EQ(options.pinRegisters[1].regClass, RegType::V);
    EXPECT_EQ(options.pinRegisters[1].start, 5u);
    EXPECT_EQ(options.regionEnd, "^loop");
    EXPECT_FALSE(options.verify);
    EXPECT_TRUE(parsed("rules=all").options.rules.activateAll);
}

TEST(RegisterAllocationConfigTest, BlanksAndEmptyWordsAreIgnored) {
    const RegisterAllocationConfig config = parsed(" apply , classes = vs ,");
    EXPECT_EQ(config.mode, Mode::Apply);
    EXPECT_EQ(config.options.allocate, RegClassSet::all());
}

TEST(RegisterAllocationConfigTest, AnUnknownWordIsAnErrorThatNamesIt) {
    EXPECT_NE(errorOf("apply,clases=vs").find("'clases=vs'"), std::string::npos);
    EXPECT_NE(errorOf("hybrid").find("'hybrid'"), std::string::npos);
}

TEST(RegisterAllocationConfigTest, ABadValueSaysWhatTheKeyTakes) {
    EXPECT_NE(errorOf("unbankable=freeze").find("hold, allocate or auto"), std::string::npos);
    EXPECT_NE(errorOf("classes=x").find("s, v or vs"), std::string::npos);
    EXPECT_NE(errorOf("quadrant=strict").find("any or prefer"), std::string::npos);
    EXPECT_NE(errorOf("mode=7").find("0..3"), std::string::npos);
    EXPECT_NE(errorOf("pinReg=x3").find("'x3'"), std::string::npos);
    EXPECT_NE(errorOf("regionEnd=").find("label"), std::string::npos);
}

TEST(RegisterAllocationConfigTest, AnUnregisteredAllocatorIsAnError) {
    const std::string error = errorOf("allocator=greedy+nope");
    EXPECT_NE(error.find("'nope'"), std::string::npos);
    EXPECT_NE(error.find("greedy-compact"), std::string::npos) << "lists what is registered";
}

TEST(RegisterAllocationConfigTest, DiagnosticFlagsBelongToTheToolOnly) {
    for (const char* flag : {"report", "emitRegisterMap", "emitSymbolBreadcrumbs"}) {
        EXPECT_NE(errorOf(std::string("apply,") + flag).find("stinkytofu-opt"), std::string::npos)
            << flag;
    }
    EXPECT_NE(errorOf("apply,dumpBlocks=out/blocks.").find("StinkyTofuRegisterAllocationDump"),
              std::string::npos);
    const RegisterAllocationConfig config = parsed(
        "apply,report,emitRegisterMap,emitSymbolBreadcrumbs,dumpBlocks=out/blocks.", Dialect::Tool);
    EXPECT_TRUE(config.options.report);
    EXPECT_TRUE(config.options.emitRegisterMap);
    EXPECT_TRUE(config.options.emitSymbolBreadcrumbs);
    EXPECT_EQ(config.options.blockDumpPrefix, "out/blocks.");
    EXPECT_NE(errorOf("dumpBlocks=", Dialect::Tool).find("a path prefix"), std::string::npos);
}

TEST(RegisterAllocationConfigTest, TheToolRefusesModeOff) {
    EXPECT_NE(errorOf("off", Dialect::Tool).find("leave the pass out"), std::string::npos);
}

TEST(RegisterAllocationConfigTest, CanonicalSpellingDropsDefaultsAndUsesTheDigit) {
    EXPECT_EQ(canonical("apply"), "2");
    EXPECT_EQ(canonical("2,classes=vs,allocator=greedy-compact+greedy-compact-largest+"
                        "greedy-compact-program"),
              "2");
    EXPECT_EQ(canonical("2,classes=s"), "2,classes=s");
    EXPECT_EQ(canonical("mode=apply,allocator=greedy-compact,unbankable=auto,quadrant=any"),
              "2,allocator=greedy-compact");
    EXPECT_EQ(canonical("shadow"), "1");
    EXPECT_EQ(canonical("force"), "3");
}

TEST(RegisterAllocationConfigTest, CanonicalSpellingOfModeOffIsZeroWhateverElseIsSet) {
    EXPECT_EQ(canonical(""), "0");
    EXPECT_EQ(canonical("off,classes=vs,unbankable=hold"), "0");
}

TEST(RegisterAllocationConfigTest, CanonicalSpellingHasAFixedOrderAndSortedLists) {
    EXPECT_EQ(canonical("noVerify,quadrant=prefer,classes=s,3,rules=B+A,pinReg=v5+s0:3,"
                        "unbankable=hold,noRule=Z,regionEnd=end,rules=A"),
              "3,classes=s,unbankable=hold,quadrant=prefer,rules=A+B,noRule=Z,pinReg=s0:3+v5,"
              "regionEnd=end,noVerify");
    EXPECT_EQ(canonical("2,rules=B,rules=all"), "2,rules=all+B");
}

TEST(RegisterAllocationConfigTest, CanonicalSpellingKeepsTheAllocatorChainInOrder) {
    EXPECT_EQ(canonical("2,allocator=greedy+greedy-compact"), "2,allocator=greedy+greedy-compact");
}

TEST(RegisterAllocationConfigTest, CanonicalSpellingParsesBackToItself) {
    for (const char* text : {"apply", "shadow,classes=vs", "force,unbankable=allocate,noVerify",
                             "2,quadrant=prefer,pinReg=s4+s0:1,regionEnd=^loop",
                             "1,allocator=greedy-compact-freedom+greedy,rules=all,ruleAudit"}) {
        const std::string once = canonical(text);
        EXPECT_EQ(canonical(once), once) << text;
    }
}

TEST(RegisterAllocationConfigTest, AKernelTakesItsModeFromTheNumber) {
    Expected<RegisterAllocationConfig> config =
        kernelRegisterAllocationConfig(2, "shadow,classes=vs");
    ASSERT_FALSE(config.hasError()) << config.getError();
    EXPECT_EQ(config->mode, Mode::Apply);
    EXPECT_TRUE(config->options.applyToOperands);
    EXPECT_EQ(config->options.allocate, RegClassSet::all());

    config = kernelRegisterAllocationConfig(0, "apply");
    ASSERT_FALSE(config.hasError());
    EXPECT_EQ(config->mode, Mode::Off);
    EXPECT_FALSE(config->options.applyToOperands);

    config = kernelRegisterAllocationConfig(2, "");
    ASSERT_FALSE(config.hasError());
    EXPECT_EQ(config->options.allocate, RegClassSet::all());
    EXPECT_EQ(config->options.allocator,
              "greedy-compact+greedy-compact-largest+greedy-compact-program");

    EXPECT_TRUE(kernelRegisterAllocationConfig(2, "bogus").hasError());
}

TEST(RegisterAllocationConfigTest, OnlyAnApplyingModeRewritesAnAllocatedClass) {
    EXPECT_TRUE(kernelRewritesRegisters(2, "", RegType::S));
    EXPECT_TRUE(kernelRewritesRegisters(3, "", RegType::S));
    EXPECT_TRUE(kernelRewritesRegisters(2, "", RegType::V));
    EXPECT_TRUE(kernelRewritesRegisters(2, "2,classes=vs", RegType::V));
    EXPECT_FALSE(kernelRewritesRegisters(2, "2,classes=v", RegType::S));
    EXPECT_FALSE(kernelRewritesRegisters(1, "1,classes=vs", RegType::S));
    EXPECT_FALSE(kernelRewritesRegisters(2, "bogus", RegType::S)) << "nothing runs";
}

TEST(RegisterAllocationConfigTest, AKernelsSgprsAreCompactedOnlyWhenScalarsAreAllocatedAndApplied) {
    StinkyAsmModule::ModuleOptions options;
    EXPECT_FALSE(options.sgprsCompacted());
    options.RegisterAllocation = 2;
    EXPECT_TRUE(options.sgprsCompacted()) << "a mode alone allocates scalars";
    options.RegisterAllocationConfig = "2,classes=v";
    EXPECT_FALSE(options.sgprsCompacted());
    options.RegisterAllocationConfig = "2,classes=vs";
    EXPECT_TRUE(options.sgprsCompacted());
    options.RegisterAllocation = 1;
    EXPECT_FALSE(options.sgprsCompacted()) << "shadow renumbers nothing";
}

TEST(RegisterAllocationConfigTest, ClassListsAndHeldRangesRefuseTypos) {
    EXPECT_EQ(parseRegClasses("vs"), RegClassSet::all());
    EXPECT_EQ(parseRegClasses("s"), RegClassSet::only(RegType::S));
    EXPECT_FALSE(parseRegClasses("").has_value());
    EXPECT_FALSE(parseRegClasses("vx").has_value());

    const std::optional<AllocationScope::HeldRange> one = parseHeldRange("s7");
    ASSERT_TRUE(one.has_value());
    EXPECT_EQ(one->start, 7u);
    EXPECT_EQ(one->end, 7u);
    const std::optional<AllocationScope::HeldRange> run = parseHeldRange("v3:9");
    ASSERT_TRUE(run.has_value());
    EXPECT_EQ(run->regClass, RegType::V);
    EXPECT_EQ(run->end, 9u);
    EXPECT_FALSE(parseHeldRange("s9:3").has_value());
    EXPECT_FALSE(parseHeldRange("x1").has_value());
    EXPECT_FALSE(parseHeldRange("s").has_value());
}
