// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <set>
#include <string>

#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/support/Diagnostic.hpp"

using namespace stinkytofu;

TEST(DiagnosticTest, TextLocatedFormatEndsWithTheId) {
    const Diagnostic diagnostic(Diagnostic::Level::Error, DiagId::UnknownMnemonic,
                                "Unknown mnemonic 's_nopp' for gfx1250", 3, 3);
    EXPECT_EQ(diagnostic.format(),
              "3:3: error: Unknown mnemonic 's_nopp' for gfx1250 [unknown-mnemonic]");
    EXPECT_EQ(diagnostic.getId(), DiagId::UnknownMnemonic);
    EXPECT_EQ(diagnostic.getIRLocation(), nullptr);
}

TEST(DiagnosticTest, IRLocatedFormatNamesFunctionBlockAndIndex) {
    const Diagnostic inst(Diagnostic::Level::Error, DiagId::RegisterType, "type is wrong",
                          IRLocation{"k", "entry", 1});
    EXPECT_EQ(inst.format(), "@k ^entry #1: error: type is wrong [register-type]");
    ASSERT_NE(inst.getIRLocation(), nullptr);
    EXPECT_EQ(inst.getIRLocation()->block, "entry");
    EXPECT_EQ(inst.getLine(), 0u);

    const Diagnostic block(Diagnostic::Level::Warning, DiagId::EmptyFunction, "m",
                           IRLocation{"k", "entry", -1});
    EXPECT_EQ(block.format(), "@k ^entry: warning: m [empty-function]");
    const Diagnostic function(Diagnostic::Level::Error, DiagId::InvalidSSA, "m", IRLocation{"k"});
    EXPECT_EQ(function.format(), "@k: error: m [invalid-ssa]");
    const Diagnostic unnamed(Diagnostic::Level::Error, DiagId::RegisterWidth, "m",
                             IRLocation{"", "entry", 0});
    EXPECT_EQ(unnamed.format(), "^entry #0: error: m [register-width]");
}

TEST(DiagnosticTest, EverySpellingIsDistinct) {
    std::set<std::string> spellings;
    size_t count = 0;
#define DIAGNOSTIC_ID(ENUM, STR, DESC) \
    ++count;                           \
    EXPECT_TRUE(spellings.insert(diagIdName(DiagId::ENUM)).second) << STR;
#include "stinkytofu/support/DiagnosticIds.def"
    EXPECT_EQ(spellings.size(), count);
}

TEST(DiagnosticTest, ParserErrorsCarryTheirKind) {
    auto idOf = [](const std::string& text) {
        const ParseResult result = parseSourceStringWithDiagnostics(text);
        EXPECT_FALSE(result.diagnostics.empty()) << text;
        return result.diagnostics.empty() ? DiagId::Syntax : result.diagnostics.front().getId();
    };
    EXPECT_EQ(idOf("v[0] = \"st.v_mov_b32\"(v[1] { issueCycles = 1, latencyCycles = 1 }\n"),
              DiagId::Syntax);
    EXPECT_EQ(idOf("q[0] = \"st.v_mov_b32\"(v1) { issueCycles = 1, latencyCycles = 1 }\n"),
              DiagId::UnknownRegisterType);
    EXPECT_EQ(idOf("v[3:1] = \"st.v_mov_b32\"(v1) { issueCycles = 1, latencyCycles = 1 }\n"),
              DiagId::OutOfRange);
    EXPECT_EQ(idOf("\"st.s_nop\"(0)\n"), DiagId::MissingAttributes);
    EXPECT_EQ(idOf("\"st.s_nop\"(0) { issueCycles = 1, latencyCycles = 1, colour = 2 }\n"),
              DiagId::UnknownAttribute);
}
