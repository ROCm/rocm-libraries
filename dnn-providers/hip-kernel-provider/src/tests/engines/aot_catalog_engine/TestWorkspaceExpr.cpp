// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// CPU-only unit tests for data-driven workspace sizing: the evalWorkspace
// evaluator (arithmetic/clamp node set + fail-closed error paths), the
// elementSizeBytes dtype helper, and the family.json parser (int back-compat,
// JSON-AST expressions, and fail-closed rejection of malformed expressions,
// exercised through the public Catalog::loadForDevice loader). No GPU required.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "catalog/Catalog.hpp"
#include "catalog/CatalogTypes.hpp"
#include "launch/LaunchAbi.hpp"

namespace
{

using aot_catalog_engine::catalog::Catalog;
using aot_catalog_engine::catalog::elementSizeBytes;
using aot_catalog_engine::catalog::ProblemShape;
using aot_catalog_engine::catalog::ShapeValue;
using aot_catalog_engine::catalog::WorkspaceExpr;
using aot_catalog_engine::catalog::WsOp;
using aot_catalog_engine::launch::evalWorkspace;
using aot_catalog_engine::launch::SymbolTable;
using PluginException = hipdnn_plugin_sdk::HipdnnPluginException;
namespace fs = std::filesystem;

// --- Small builders for hand-authored expression trees -----------------------

WorkspaceExpr lit(int64_t value)
{
    WorkspaceExpr e;
    e.op = WsOp::LITERAL;
    e.literal = value;
    return e;
}

WorkspaceExpr sym(std::string name)
{
    WorkspaceExpr e;
    e.op = WsOp::SYMBOL;
    e.symbol = std::move(name);
    return e;
}

WorkspaceExpr node(WsOp op, std::vector<WorkspaceExpr> args)
{
    WorkspaceExpr e;
    e.op = op;
    e.args = std::move(args);
    return e;
}

} // namespace

// -----------------------------------------------------------------------------
// evalWorkspace: node semantics
// -----------------------------------------------------------------------------

TEST(TestAotWorkspaceEval, LiteralAndSymbol)
{
    const SymbolTable syms = {{"M", 128}};
    EXPECT_EQ(evalWorkspace(lit(0), syms), 0);
    EXPECT_EQ(evalWorkspace(lit(4096), syms), 4096);
    EXPECT_EQ(evalWorkspace(sym("M"), syms), 128);
}

TEST(TestAotWorkspaceEval, Arithmetic)
{
    const SymbolTable syms = {{"M", 100}, {"N", 7}, {"elem_size", 2}};

    // Variadic ops.
    EXPECT_EQ(evalWorkspace(node(WsOp::MUL, {sym("M"), sym("N"), sym("elem_size")}), syms), 1400);
    EXPECT_EQ(evalWorkspace(node(WsOp::ADD, {sym("M"), sym("N"), lit(1)}), syms), 108);
    EXPECT_EQ(evalWorkspace(node(WsOp::MIN, {sym("M"), sym("N")}), syms), 7);
    EXPECT_EQ(evalWorkspace(node(WsOp::MAX, {sym("M"), sym("N")}), syms), 100);

    // Single-operand variadic degenerates to that operand.
    EXPECT_EQ(evalWorkspace(node(WsOp::MUL, {sym("M")}), syms), 100);

    // Binary ops.
    EXPECT_EQ(evalWorkspace(node(WsOp::SUB, {sym("M"), sym("N")}), syms), 93);
    EXPECT_EQ(evalWorkspace(node(WsOp::CEIL_DIV, {sym("M"), sym("N")}), syms), 15); // ceil(100/7)
    EXPECT_EQ(evalWorkspace(node(WsOp::FLOOR_DIV, {sym("M"), sym("N")}), syms), 14); // floor(100/7)
}

TEST(TestAotWorkspaceEval, AlignUp)
{
    const SymbolTable syms = {{"n", 100}};
    EXPECT_EQ(evalWorkspace(node(WsOp::ALIGN_UP, {sym("n"), lit(256)}), syms), 256);
    EXPECT_EQ(evalWorkspace(node(WsOp::ALIGN_UP, {lit(256), lit(256)}), syms), 256);
    EXPECT_EQ(evalWorkspace(node(WsOp::ALIGN_UP, {lit(257), lit(256)}), syms), 512);
    EXPECT_EQ(evalWorkspace(node(WsOp::ALIGN_UP, {lit(0), lit(256)}), syms), 0);
}

// A realistic im2col-shaped nesting: align_up(M * N * elem_size, 256).
TEST(TestAotWorkspaceEval, NestedExpression)
{
    const SymbolTable syms = {{"M", 128}, {"N", 130}, {"elem_size", 2}};
    const WorkspaceExpr expr
        = node(WsOp::ALIGN_UP, {node(WsOp::MUL, {sym("M"), sym("N"), sym("elem_size")}), lit(256)});
    // 128*130*2 = 33280; align_up(33280, 256) = 33280 (already a multiple).
    EXPECT_EQ(evalWorkspace(expr, syms), 33280);
}

// -----------------------------------------------------------------------------
// evalWorkspace: fail-closed error paths
// -----------------------------------------------------------------------------

TEST(TestAotWorkspaceEval, UndefinedSymbolThrows)
{
    const SymbolTable syms = {{"M", 8}};
    EXPECT_THROW(evalWorkspace(sym("elem_size"), syms), PluginException);
    EXPECT_THROW(evalWorkspace(node(WsOp::MUL, {sym("M"), sym("K")}), syms), PluginException);
}

TEST(TestAotWorkspaceEval, DivisionByZeroThrows)
{
    const SymbolTable syms = {{"M", 8}};
    EXPECT_THROW(evalWorkspace(node(WsOp::CEIL_DIV, {sym("M"), lit(0)}), syms), PluginException);
    EXPECT_THROW(evalWorkspace(node(WsOp::FLOOR_DIV, {sym("M"), lit(0)}), syms), PluginException);
    EXPECT_THROW(evalWorkspace(node(WsOp::ALIGN_UP, {sym("M"), lit(0)}), syms), PluginException);
}

TEST(TestAotWorkspaceEval, NegativeSubThrows)
{
    const SymbolTable syms = {{"M", 3}, {"N", 8}};
    EXPECT_THROW(evalWorkspace(node(WsOp::SUB, {sym("M"), sym("N")}), syms), PluginException);
}

// -----------------------------------------------------------------------------
// elementSizeBytes: dtype -> element width
// -----------------------------------------------------------------------------

TEST(TestAotWorkspaceElemSize, MapsKnownDtypes)
{
    auto with = [](const char* dtype) {
        ProblemShape p;
        p.emplace("dtype", ShapeValue{std::string(dtype)});
        return elementSizeBytes(p);
    };
    EXPECT_EQ(with("f16"), 2);
    EXPECT_EQ(with("bf16"), 2);
    EXPECT_EQ(with("f8"), 1);
    EXPECT_EQ(with("bf8fnuz"), 1);
    EXPECT_EQ(with("f32"), 4);
}

TEST(TestAotWorkspaceElemSize, UnknownOrMissingIsNullopt)
{
    ProblemShape unknown;
    unknown.emplace("dtype", ShapeValue{std::string("i4")});
    EXPECT_FALSE(elementSizeBytes(unknown).has_value());

    ProblemShape missing;
    missing.emplace("M", ShapeValue{static_cast<int64_t>(8)});
    EXPECT_FALSE(elementSizeBytes(missing).has_value());

    // A non-string dtype value is treated as absent (fails closed).
    ProblemShape wrongType;
    wrongType.emplace("dtype", ShapeValue{static_cast<int64_t>(2)});
    EXPECT_FALSE(elementSizeBytes(wrongType).has_value());
}

// -----------------------------------------------------------------------------
// Parser (via Catalog::loadForDevice): back-compat, expressions, fail-closed
// -----------------------------------------------------------------------------

namespace
{

// Writes a minimal single-kernel family.json (plus the dummy .co it references)
// into a temp arch dir and returns the catalog root. `workspaceJson` is spliced
// in verbatim as the "workspace_bytes" value (a number, string, or object).
class TestAotWorkspaceFamily : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _root = fs::temp_directory_path()
                / ("hipdnn_aot_ws_test_"
                   + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        std::error_code ec;
        fs::remove_all(_root, ec);
        _familyDir = _root / K_ARCH / "ws_family";
        fs::create_directories(_familyDir);
        std::ofstream(_familyDir / "dummy.co") << "x"; // existence is all the loader checks
    }
    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(_root, ec);
    }

    Catalog loadWith(const std::string& workspaceJson)
    {
        const std::string json = R"({
            "family": "ws_family",
            "op_kind": "matmul",
            "kernels": [
                {
                    "symbol": "k",
                    "co_file": "dummy.co",
                    "constraints": { "dtype": { "equals": "f16" } },
                    "workspace_bytes": )"
                                 + workspaceJson + R"(,
                    "grid": { "x": 1, "y": 1, "z": 1 },
                    "block": [1, 1, 1],
                    "args_signature": [ { "name": "out", "type": "ptr" } ]
                }
            ]
        })";
        std::ofstream(_familyDir / "family.json") << json;
        return Catalog::loadForDevice(_root.string(), K_ARCH);
    }

    static constexpr const char* K_ARCH = "gfxTEST";
    fs::path _root;
    fs::path _familyDir;
};

const WorkspaceExpr& onlyKernelWorkspace(const Catalog& catalog)
{
    return catalog.families().front().kernels.front().workspace;
}

} // namespace

// A bare integer parses as today's static constant (back-compat) and evaluates
// to itself regardless of the symbol table.
TEST_F(TestAotWorkspaceFamily, IntegerLiteralBackCompat)
{
    const Catalog catalog = loadWith("0");
    ASSERT_EQ(catalog.families().size(), 1u);
    EXPECT_EQ(evalWorkspace(onlyKernelWorkspace(catalog), SymbolTable{}), 0);

    const Catalog nonZero = loadWith("4096");
    ASSERT_EQ(nonZero.families().size(), 1u);
    EXPECT_EQ(evalWorkspace(onlyKernelWorkspace(nonZero), SymbolTable{}), 4096);
}

// A JSON-AST expression parses into a tree that evaluates against grid symbols.
TEST_F(TestAotWorkspaceFamily, ExpressionParsesAndEvaluates)
{
    const Catalog catalog
        = loadWith(R"({ "align_up": [ { "mul": ["M", "N", "elem_size"] }, 256 ] })");
    ASSERT_EQ(catalog.families().size(), 1u);
    const SymbolTable syms = {{"M", 128}, {"N", 130}, {"elem_size", 2}};
    EXPECT_EQ(evalWorkspace(onlyKernelWorkspace(catalog), syms), 33280);
}

// A bare symbol string is a SYMBOL reference.
TEST_F(TestAotWorkspaceFamily, SymbolStringParses)
{
    const Catalog catalog = loadWith("\"K\"");
    ASSERT_EQ(catalog.families().size(), 1u);
    EXPECT_EQ(evalWorkspace(onlyKernelWorkspace(catalog), SymbolTable{{"K", 64}}), 64);
}

// Fail-closed: an unknown operator key skips the whole family (no-throw load
// contract), so the catalog loads empty rather than accepting a bad expression.
TEST_F(TestAotWorkspaceFamily, UnknownOperatorKeyIsRejected)
{
    const Catalog catalog = loadWith(R"({ "pow": ["M", 2] })");
    EXPECT_TRUE(catalog.empty());
}

// Fail-closed: wrong arity (sub needs exactly two operands) is rejected.
TEST_F(TestAotWorkspaceFamily, WrongArityIsRejected)
{
    EXPECT_TRUE(loadWith(R"({ "sub": ["M"] })").empty());
    EXPECT_TRUE(loadWith(R"({ "ceil_div": ["M", "N", 1] })").empty());
}

// Fail-closed: an object with two operator keys is ambiguous and rejected.
TEST_F(TestAotWorkspaceFamily, MultipleOperatorKeysAreRejected)
{
    const Catalog catalog = loadWith(R"({ "mul": ["M", 2], "add": ["M", 1] })");
    EXPECT_TRUE(catalog.empty());
}

// Fail-closed: a negative integer literal is rejected at parse time.
TEST_F(TestAotWorkspaceFamily, NegativeLiteralIsRejected)
{
    EXPECT_TRUE(loadWith("-8").empty());
}
