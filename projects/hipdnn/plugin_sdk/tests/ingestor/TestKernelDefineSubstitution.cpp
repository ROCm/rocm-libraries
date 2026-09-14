// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelDefineSubstitution.hpp>

/**
 * @file TestKernelDefineSubstitution.cpp
 * @brief Tests for KernelDefineSubstitution.hpp: the rendering table for `$kernel.<field>`
 *        bound into a `hiprtc_file` define, and everything the substituter refuses.
 *
 * The rendering table is the contract two kernel variants rely on: if a bool rendered as
 * `true` in one build and `1` in another, or if a variant's flag text collided with
 * another's, the two would compile to one binary and silently become one kernel. The
 * refusals matter as much -- each one turns an authoring mistake into a message naming the
 * field, instead of a hipRTC error inside someone else's source file.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;

/// The schema every case below validates against: one field per KMD type, plus a field the
/// kernel omits and the schema defaults.
MetadataSchema testSchema()
{
    MetadataSchema schema;
    schema.name = "test_schema";
    schema.fields = {
        MetadataField{"flag", MetadataType::BOOL, {}},
        MetadataField{"block_size", MetadataType::INT, {}},
        MetadataField{"dtype", MetadataType::STRING, {}},
        MetadataField{"alpha", MetadataType::FLOAT, {}},
        MetadataField{"tile", MetadataType::INT_LIST, {}},
        MetadataField{"vector_width", MetadataType::INT, MetadataValue{int64_t{4}}},
    };
    return schema;
}

/// A completed metadata tuple: what `completeMetadata` hands `prepare()`, with the
/// defaulted field already filled in.
MetadataValues completedMetadata()
{
    return MetadataValues{
        {"flag", MetadataValue{true}},
        {"block_size", MetadataValue{int64_t{128}}},
        {"dtype", MetadataValue{std::string("bfloat16")}},
        {"alpha", MetadataValue{1.0}},
        {"tile", MetadataValue{std::vector<int64_t>{64, 32}}},
        {"vector_width", MetadataValue{int64_t{4}}},
    };
}

std::string substituted(const std::string& templateText)
{
    std::string out;
    std::string error;
    EXPECT_TRUE(substituteKernelDefine(templateText, completedMetadata(), out, error)) << error;
    return out;
}

/// Both entry points must reject the same template, or a define that passes load-time
/// validation fails at prepare() -- after the engine has already advertised itself.
void expectRejectedByBoth(const std::string& templateText, const std::string& expectedFragment)
{
    std::string out;
    std::string substituteError;
    EXPECT_FALSE(substituteKernelDefine(templateText, completedMetadata(), out, substituteError));
    EXPECT_NE(substituteError.find(expectedFragment), std::string::npos) << substituteError;

    std::string validateError;
    EXPECT_FALSE(validateKernelDefineTemplate(templateText, testSchema(), validateError));
    EXPECT_NE(validateError.find(expectedFragment), std::string::npos) << validateError;
}

TEST(TestKernelDefineSubstitution, BoolRendersAsOneOrZero)
{
    MetadataValues metadata = completedMetadata();
    std::string out;
    std::string error;

    ASSERT_TRUE(substituteKernelDefine("$kernel.flag", metadata, out, error)) << error;
    EXPECT_EQ(out, "1");

    metadata["flag"] = MetadataValue{false};
    ASSERT_TRUE(substituteKernelDefine("$kernel.flag", metadata, out, error)) << error;
    EXPECT_EQ(out, "0");
}

TEST(TestKernelDefineSubstitution, IntRendersAsDecimal)
{
    EXPECT_EQ(substituted("$kernel.block_size"), "128");

    MetadataValues metadata = completedMetadata();
    metadata["block_size"] = MetadataValue{int64_t{-7}};
    std::string out;
    std::string error;
    ASSERT_TRUE(substituteKernelDefine("$kernel.block_size", metadata, out, error)) << error;
    EXPECT_EQ(out, "-7");
}

TEST(TestKernelDefineSubstitution, StringRendersVerbatim)
{
    EXPECT_EQ(substituted("$kernel.dtype"), "bfloat16");
}

TEST(TestKernelDefineSubstitution, TwoVariantsOfOneFieldRenderDistinctText)
{
    MetadataValues metadata = completedMetadata();
    std::string first;
    std::string second;
    std::string error;

    ASSERT_TRUE(substituteKernelDefine("$kernel.dtype", metadata, first, error)) << error;
    metadata["dtype"] = MetadataValue{std::string("float16")};
    ASSERT_TRUE(substituteKernelDefine("$kernel.dtype", metadata, second, error)) << error;

    EXPECT_EQ(first, "bfloat16");
    EXPECT_EQ(second, "float16");
}

TEST(TestKernelDefineSubstitution, DefaultedFieldResolvesFromCompletedMetadata)
{
    // The kernel descriptor omits `vector_width`; the KMD default filled it before
    // prepare(). Substituting against any earlier snapshot would reject a legal kernel.
    EXPECT_EQ(substituted("$kernel.vector_width"), "4");

    std::string error;
    EXPECT_TRUE(validateKernelDefineTemplate("$kernel.vector_width", testSchema(), error)) << error;
}

TEST(TestKernelDefineSubstitution, LiteralTextAroundAndBetweenTokensIsPreserved)
{
    // A token runs to the first character that cannot continue an identifier, so a
    // trailing `_t` would be read as part of the field name, not as a suffix.
    EXPECT_EQ(substituted("hip_$kernel.dtype"), "hip_bfloat16");
    EXPECT_EQ(substituted("$kernel.dtype,$kernel.block_size"), "bfloat16,128");
    EXPECT_EQ(substituted("x$kernel.flag"), "x1");
}

TEST(TestKernelDefineSubstitution, TemplateWithoutTokenPassesThroughByteIdentical)
{
    // Includes characters the substituter refuses next to a token: with no token there is
    // nothing to mistake for an expression, so an ordinary literal define still works.
    const std::string literal = "float4(-1) ? a : b";
    std::string out;
    std::string error;

    ASSERT_TRUE(substituteKernelDefine(literal, completedMetadata(), out, error)) << error;
    EXPECT_EQ(out, literal);

    ASSERT_TRUE(validateKernelDefineTemplate(literal, testSchema(), error)) << error;
}

TEST(TestKernelDefineSubstitution, UndeclaredFieldIsRejected)
{
    std::string out;
    std::string substituteError;
    EXPECT_FALSE(
        substituteKernelDefine("$kernel.missing", completedMetadata(), out, substituteError));
    EXPECT_NE(substituteError.find("missing"), std::string::npos) << substituteError;

    std::string validateError;
    EXPECT_FALSE(validateKernelDefineTemplate("$kernel.missing", testSchema(), validateError));
    EXPECT_NE(validateError.find("does not declare"), std::string::npos) << validateError;
}

TEST(TestKernelDefineSubstitution, FloatIsRejected)
{
    expectRejectedByBoth("$kernel.alpha", "float");
}

TEST(TestKernelDefineSubstitution, IntListIsRejected)
{
    expectRejectedByBoth("$kernel.tile", "int_list");
}

TEST(TestKernelDefineSubstitution, NonKernelTokenIsRejected)
{
    expectRejectedByBoth("$graph.batch", "$kernel.");
    expectRejectedByBoth("$dtype", "$kernel.");
    expectRejectedByBoth("$KERNEL.dtype", "$kernel.");
}

TEST(TestKernelDefineSubstitution, TokenWithoutFieldNameIsRejected)
{
    expectRejectedByBoth("$kernel.", "no field name");
    expectRejectedByBoth("$kernel.9x", "no field name");
}

TEST(TestKernelDefineSubstitution, ExpressionSyntaxBesideATokenIsRejected)
{
    // Each of these is an attempt at a language this substituter does not have. Rendering
    // them literally would put `128 * 2` or `bfloat16 == float16` into a -D flag, which is
    // exactly the half-working outcome the refusal exists to prevent.
    expectRejectedByBoth("$kernel.block_size * 2", "literal token replacement");
    expectRejectedByBoth("$kernel.block_size + 1", "literal token replacement");
    expectRejectedByBoth("$kernel.dtype == float16", "literal token replacement");
    expectRejectedByBoth("$kernel.flag ? 1 : 0", "literal token replacement");
    expectRejectedByBoth("max($kernel.block_size)", "literal token replacement");
    expectRejectedByBoth("$kernel.dtype|float16", "literal token replacement");
}

TEST(TestKernelDefineSubstitution, ReplacementTextIsNotRescanned)
{
    // A string field whose value contains a '$' is data, not a template: rescanning it
    // would let metadata reach into the substituter's own grammar.
    MetadataValues metadata = completedMetadata();
    metadata["dtype"] = MetadataValue{std::string("$kernel.block_size")};
    std::string out;
    std::string error;

    ASSERT_TRUE(substituteKernelDefine("$kernel.dtype", metadata, out, error)) << error;
    EXPECT_EQ(out, "$kernel.block_size");
}

TEST(TestKernelDefineSubstitution, ValidationAcceptsEveryRenderableDeclaredField)
{
    const auto schema = testSchema();
    std::string error;

    EXPECT_TRUE(validateKernelDefineTemplate("$kernel.flag", schema, error)) << error;
    EXPECT_TRUE(validateKernelDefineTemplate("$kernel.block_size", schema, error)) << error;
    EXPECT_TRUE(validateKernelDefineTemplate("$kernel.dtype", schema, error)) << error;
    EXPECT_TRUE(validateKernelDefineTemplate("", schema, error)) << error;
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
