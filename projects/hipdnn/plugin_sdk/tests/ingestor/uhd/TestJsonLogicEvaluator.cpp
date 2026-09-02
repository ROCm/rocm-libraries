// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/uhd/JsonLogicEvaluator.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>

/// @file TestJsonLogicEvaluator.cpp
/// @brief RFC 0019 §6.2's derived-feature operators.
///
/// The evaluator decides the value of every computed feature in a signature, so a defect here
/// does not throw or crash -- it produces a number, the model scores it, and a plausible kernel
/// is selected for the wrong reason. Nothing downstream can see that. The cases below are
/// therefore weighted toward the arithmetic edges where a wrong answer stays in range, and
/// toward the guards that decide between throwing and returning a value.
namespace hipdnn_plugin_sdk::ingestor::uhd
{
namespace
{

/// Evaluates @p expr, which is written as JSON text so each case reads the way a signature
/// author would write it rather than as a hand-built object tree.
double eval(const std::string& expr, const VariableContext& ctx = {})
{
    const JsonLogicEvaluator evaluator;
    return evaluator.evaluateDouble(JsonLogicEvaluator::parse(expr), ctx);
}

VariableContext withQuery()
{
    VariableContext ctx;
    ctx.bindNamespace("q", {{"N", int64_t{64}}, {"C", int64_t{3}}, {"H", int64_t{224}}});
    ctx.bindNamespace("kernel", {{"tile_m", int64_t{128}}, {"name", std::string{"conv_fwd"}}});
    return ctx;
}

TEST(TestIngestorJsonLogic, CeilDivRoundsUpRatherThanTruncating)
{
    // The tiling operator, and the reason it exists as an operator instead of `/`. A grid
    // covering 224 rows with 128-row tiles needs two tiles, not one: truncation loses the
    // partial tile, which is exactly the case where occupancy and tail effects live. Both
    // answers are small positive integers, so nothing downstream can tell them apart.
    EXPECT_DOUBLE_EQ(eval(R"({"ceil_div": [224, 128]})"), 2.0);
    EXPECT_DOUBLE_EQ(eval(R"({"ceil_div": [256, 128]})"), 2.0) << "an exact multiple gained a tile";
    EXPECT_DOUBLE_EQ(eval(R"({"ceil_div": [1, 128]})"), 1.0);
}

TEST(TestIngestorJsonLogic, ArithmeticGuardsThrowRatherThanProducingNonFinite)
{
    // Every one of these would otherwise yield inf or NaN, and a GBDT reads NaN as *missing*
    // rather than as an error -- it takes the default branch and scores a different problem
    // than the one asked about, silently. Throwing degrades the whole ranking instead, which
    // RFC 0019 §5 step 7 defines and which shows up in the trace.
    EXPECT_THROW(eval(R"({"/": [1, 0]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"%": [1, 0]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"ceil_div": [1, 0]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"log2": [0]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"log2": [-1]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"rsqrt": [0]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"rsqrt": [-4]})"), JsonLogicError);
}

TEST(TestIngestorJsonLogic, ArithmeticGuardsRejectTheWrongArgumentCount)
{
    // A signature is hand-written JSON with no schema behind it. An operator quietly ignoring
    // a third argument would compute something the author did not write.
    EXPECT_THROW(eval(R"({"%": [1, 2, 3]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"ceil_div": [8]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"log2": [8, 2]})"), JsonLogicError);
    EXPECT_THROW(eval(R"({"/": [8]})"), JsonLogicError);
}

TEST(TestIngestorJsonLogic, ArithmeticComputesWhatItSays)
{
    EXPECT_DOUBLE_EQ(eval(R"({"+": [1, 2, 3]})"), 6.0);
    EXPECT_DOUBLE_EQ(eval(R"({"-": [10, 3]})"), 7.0);
    EXPECT_DOUBLE_EQ(eval(R"({"*": [2, 3, 4]})"), 24.0);
    EXPECT_DOUBLE_EQ(eval(R"({"/": [12, 3]})"), 4.0);
    EXPECT_DOUBLE_EQ(eval(R"({"%": [7, 3]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"abs": [-5]})"), 5.0);
    EXPECT_DOUBLE_EQ(eval(R"({"min": [3, 1, 2]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"max": [3, 1, 2]})"), 3.0);
    EXPECT_DOUBLE_EQ(eval(R"({"pow": [2, 10]})"), 1024.0);
    EXPECT_DOUBLE_EQ(eval(R"({"log2": [1024]})"), 10.0);
    EXPECT_DOUBLE_EQ(eval(R"({"rsqrt": [4]})"), 0.5);
}

TEST(TestIngestorJsonLogic, DivisibleAnswersTheAlignmentQuestion)
{
    // Alignment predicates gate whole families of kernels. Getting the sense inverted selects
    // precisely the kernels that cannot run the shape.
    EXPECT_DOUBLE_EQ(eval(R"({"divisible": [256, 64]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"divisible": [255, 64]})"), 0.0);
}

TEST(TestIngestorJsonLogic, AVariableResolvesFromItsNamespace)
{
    const auto ctx = withQuery();
    EXPECT_DOUBLE_EQ(eval(R"("$q.N")", ctx), 64.0);
    EXPECT_DOUBLE_EQ(eval(R"("$kernel.tile_m")", ctx), 128.0);
    EXPECT_DOUBLE_EQ(eval(R"({"ceil_div": ["$q.H", "$kernel.tile_m"]})", ctx), 2.0);
}

TEST(TestIngestorJsonLogic, AnUndefinedVariableThrowsRatherThanReadingAsZero)
{
    // Zero is a legal value for most features, so substituting it for an absent one produces a
    // score with no indication anything was missing. The distinct exception type is what lets
    // the caller tell a misspelled signature from a genuinely broken expression.
    const auto ctx = withQuery();
    EXPECT_THROW(eval(R"("$q.does_not_exist")", ctx), UndefinedVariableError);
    EXPECT_THROW(eval(R"("$q.does_not_exist")", ctx), JsonLogicError)
        << "UndefinedVariableError must remain a JsonLogicError, which callers catch";
}

TEST(TestIngestorJsonLogic, ValueOrDefaultIsHowAnAbsentVariableIsTolerated)
{
    // The supported way to write an optional feature. Its existence is why `var` can afford to
    // throw: an author who means "0 when absent" has a way to say so.
    const auto ctx = withQuery();
    EXPECT_DOUBLE_EQ(eval(R"({"value_or_default": ["$q.absent", 7]})", ctx), 7.0);
    EXPECT_DOUBLE_EQ(eval(R"({"value_or_default": ["$q.N", 7]})", ctx), 64.0)
        << "the default displaced a value that was present";
}

TEST(TestIngestorJsonLogic, StringsAreNotSilentlyNumeric)
{
    // A categorical value reaching a numeric operator is an authoring error. Returning NaN
    // would hand a GBDT a *missing* marker; the fix that established this was a real defect,
    // where a string feature scored as absent instead of failing.
    const auto ctx = withQuery();
    EXPECT_THROW(eval(R"({"+": ["$kernel.name", 1]})", ctx), JsonLogicError);
}

TEST(TestIngestorJsonLogic, ComparisonAndLogicYieldOneAndZero)
{
    // Features are doubles, so a predicate has to arrive as 1.0 or 0.0 -- a bool that converted
    // some other way would still train, on an axis with the wrong scale.
    EXPECT_DOUBLE_EQ(eval(R"({"<": [1, 2]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({">=": [1, 2]})"), 0.0);
    EXPECT_DOUBLE_EQ(eval(R"({"==": [2, 2]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"!=": [2, 2]})"), 0.0);
    EXPECT_DOUBLE_EQ(eval(R"({"and": [true, true]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"or": [false, true]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"!": [false]})"), 1.0);
}

TEST(TestIngestorJsonLogic, IfSelectsABranchAndDoesNotEvaluateTheOther)
{
    // The untaken branch divides by zero. If `if` evaluated both eagerly the expression would
    // throw, and every guarded feature -- "use this ratio unless the denominator is 0" -- would
    // be unwritable.
    EXPECT_DOUBLE_EQ(eval(R"({"if": [true, 1, {"/": [1, 0]}]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"if": [false, {"/": [1, 0]}, 2]})"), 2.0);
}

TEST(TestIngestorJsonLogic, InAndAllReadMembershipOverALiteralSet)
{
    EXPECT_DOUBLE_EQ(eval(R"({"in": [2, [1, 2, 3]]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"in": [9, [1, 2, 3]]})"), 0.0);
    // `all` binds each element to $current for the predicate, so the predicate is the second
    // argument rather than the array being truthy by itself.
    EXPECT_DOUBLE_EQ(eval(R"({"all": [[2, 4, 6], {">": ["$current", 0]}]})"), 1.0);
    EXPECT_DOUBLE_EQ(eval(R"({"all": [[2, -4, 6], {">": ["$current", 0]}]})"), 0.0);
    EXPECT_DOUBLE_EQ(eval(R"({"all": [[], {">": ["$current", 0]}]})"), 1.0)
        << "an empty set vacuously satisfies the predicate";
}

TEST(TestIngestorJsonLogic, ARunawayExpressionIsRejectedRatherThanRecursedInto)
{
    // A UHD is drop-in data, possibly third-party. Without the depth bound a nested expression
    // recurses until the stack runs out, which is a crash inside the caller's process rather
    // than a descriptor that fails to load.
    std::string nested = "1";
    for(size_t i = 0; i < JsonLogicEvaluator::MAX_EXPRESSION_DEPTH + 8; ++i)
    {
        nested.insert(0, R"({"abs": [)");
        nested.append("]}");
    }
    EXPECT_THROW(eval(nested), JsonLogicError);

    // Just under the bound still evaluates, so the guard is a bound and not a blanket refusal.
    std::string shallow = "-1";
    for(size_t i = 0; i < 8; ++i)
    {
        shallow.insert(0, R"({"abs": [)");
        shallow.append("]}");
    }
    EXPECT_DOUBLE_EQ(eval(shallow), 1.0);
}

TEST(TestIngestorJsonLogic, ExtractVariablesFindsEveryVariableAnExpressionReads)
{
    // RFC 0019 §6.3's contract checks are built on this: check 2 compares the model's `$kernel.*`
    // axes against the UED's declared knobs. A variable this misses is a knob the check cannot
    // know is required, so the mismatch it exists to catch passes.
    const auto expr = JsonLogicEvaluator::parse(
        R"({"ceil_div": ["$q.H", {"if": [{">": ["$kernel.tile_m", 0]},
                                         "$kernel.tile_m",
                                         "$device.cu_count"]}]})");
    const auto found = JsonLogicEvaluator::extractVariables(expr);

    EXPECT_EQ(found.size(), 3U);
    EXPECT_TRUE(found.count("$q.H") != 0);
    EXPECT_TRUE(found.count("$kernel.tile_m") != 0) << "a variable inside a branch was missed";
    EXPECT_TRUE(found.count("$device.cu_count") != 0);

    // $current is bound by `all` for the duration of its own predicate, never by the caller.
    // Reporting it would make the §6.3 coverage check demand a knob that cannot be supplied.
    const auto withCurrent = JsonLogicEvaluator::parse(
        R"({"all": [[1, 2], {">": ["$current", "$q.N"]}]})");
    const auto currentVars = JsonLogicEvaluator::extractVariables(withCurrent);
    EXPECT_TRUE(currentVars.count("$current") == 0) << "$current reported as a caller binding";
    EXPECT_TRUE(currentVars.count("$q.N") != 0);
}

TEST(TestIngestorJsonLogic, ClearingANamespaceLeavesTheOthersBound)
{
    // Ranking rebinds only `kernel.*` per candidate and reuses the shared problem and device
    // row (§6 step 2). If clearing took the others with it, every candidate after the first
    // would score against an empty problem.
    auto ctx = withQuery();
    ctx.clearNamespace("kernel");

    EXPECT_FALSE(ctx.has("$kernel.tile_m"));
    EXPECT_TRUE(ctx.has("$q.N"));
    EXPECT_DOUBLE_EQ(eval(R"("$q.N")", ctx), 64.0);
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
