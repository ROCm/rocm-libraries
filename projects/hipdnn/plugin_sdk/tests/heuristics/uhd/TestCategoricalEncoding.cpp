// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>

/// Categorical vocabularies belong to the model. A bare reference to a declared field reads
/// its code; inside an expression the value keeps its string identity.

namespace
{

using hipdnn_plugin_sdk::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::uhd::JsonLogicError;

using Encoding = std::map<std::string, std::map<std::string, int32_t>>;

/// Extract a one-entry signature with `$kernel.<field>` bound to `value` under @p encoding.
double extractKernelFeature(const std::string& field,
                            const std::string& value,
                            const Encoding& encoding)
{
    const FeatureExtractor extractor({"$kernel." + field}, encoding);
    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{field, value}});
    return extractor.extract(ctx).at(0);
}

// ---- A declared field reaches the model as its declared code -------------------

TEST(TestCategoricalEncoding, AStringFeatureReachesTheModelAsTheCodeItsDescriptorDeclares)
{
    // Without an encoding this signature throws.
    const Encoding encoding{{"$kernel.dtype", {{"bf16", 0}, {"fp16", 1}}},
                            {"$kernel.layout", {{"BSHD", 0}, {"NCHW", 1}}}};

    EXPECT_DOUBLE_EQ(extractKernelFeature("dtype", "fp16", encoding), 1.0);
    EXPECT_DOUBLE_EQ(extractKernelFeature("layout", "BSHD", encoding), 0.0);
}

TEST(TestCategoricalEncoding, TwoReferencesSharingATrailingNameDoNotMerge)
{
    // Vocabularies are keyed by the whole reference, not the trailing field name, so the
    // KMD's "BF16" and the runtime's "bf16" never share one.
    const Encoding encoding{{"$kernel.dtype", {{"BF16", 7}}},
                            {"$q.attention_dense.dtype", {{"bf16", 3}}}};

    const FeatureExtractor extractor({"$kernel.dtype", "$q.attention_dense.dtype"}, encoding);
    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{"dtype", std::string("BF16")}});
    ctx.bindQueryVars({{"q.attention_dense.dtype", std::string("bf16")}});

    const auto row = extractor.extract(ctx);
    EXPECT_DOUBLE_EQ(row.at(0), 7.0);
    EXPECT_DOUBLE_EQ(row.at(1), 3.0) << "two references sharing a trailing name must keep "
                                        "separate vocabularies";
}

TEST(TestCategoricalEncoding, ACaseVariantIsADistinctValue)
{
    // No case folding: a model fitted on one spelling has never seen the other.
    const Encoding encoding{{"$kernel.dtype", {{"BF16", 4}}}};

    EXPECT_DOUBLE_EQ(extractKernelFeature("dtype", "BF16", encoding), 4.0);
    EXPECT_THROW(extractKernelFeature("dtype", "bf16", encoding), JsonLogicError);
}

TEST(TestCategoricalEncoding, TwoDescriptorsMayEncodeTheSameFieldDifferently)
{
    // Deliberate: comparability between engines lives in the score's units (§11.3), not in
    // the feature codes.
    const FeatureExtractor first({"$kernel.dtype"}, Encoding{{"$kernel.dtype", {{"bf16", 0}}}});
    const FeatureExtractor second({"$kernel.dtype"}, Encoding{{"$kernel.dtype", {{"bf16", 9}}}});

    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{"dtype", std::string("bf16")}});

    EXPECT_DOUBLE_EQ(first.extract(ctx).at(0), 0.0);
    EXPECT_DOUBLE_EQ(second.extract(ctx).at(0), 9.0);
}

TEST(TestCategoricalEncoding, TheEncodingTravelsIntoTheFeaturesHash)
{
    // RFC 0019 §6.5: a changed map leaves the signature text identical, so the hash must
    // cover the codes too.
    const std::vector<nlohmann::json> signature{"$kernel.dtype"};

    EXPECT_NE(FeatureExtractor::computeHash(signature, Encoding{{"$kernel.dtype", {{"bf16", 0}}}}),
              FeatureExtractor::computeHash(signature, Encoding{{"$kernel.dtype", {{"bf16", 9}}}}));

    // An empty encoding hashes as no encoding, so existing models keep their contracts.
    EXPECT_EQ(FeatureExtractor::computeHash({"$kernel.tile_m"}, Encoding{}),
              FeatureExtractor::computeHash({"$kernel.tile_m"}));
}

// ---- An unencodable string still fails loudly ----------------------------------

TEST(TestCategoricalEncoding, StringOutsideAnyDeclaredFieldStillThrows)
{
    // `pipeline` is not declared categorical, so the string has no numeric meaning.
    const Encoding encoding{{"$kernel.dtype", {{"bf16", 0}}}};

    EXPECT_THROW(extractKernelFeature("pipeline", "intrawave", encoding), JsonLogicError);
}

TEST(TestCategoricalEncoding, ValueOutsideADeclaredFieldStillThrows)
{
    // A value the model was never trained on must surface rather than score.
    const Encoding encoding{{"$kernel.dtype", {{"bf16", 0}, {"fp16", 1}}}};

    EXPECT_THROW(extractKernelFeature("dtype", "float16", encoding), JsonLogicError);
}

TEST(TestCategoricalEncoding, EncodedReferencesKeepStringIdentityInsideExpressions)
{
    const Encoding encoding{{"$kernel.dtype", {{"fp16", 1}}}};
    const FeatureExtractor extractor(
        {"$kernel.dtype", nlohmann::json::parse(R"({"==": ["$kernel.dtype", "fp16"]})")}, encoding);
    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{"dtype", std::string("fp16")}});
    EXPECT_EQ(extractor.extract(ctx), (std::vector<double>{1, 1}));

    // The code is the model's reading of the field, not the field's value, so no arithmetic.
    const FeatureExtractor arithmetic({nlohmann::json::parse(R"({"+": ["$kernel.dtype", 1]})")},
                                      encoding);
    EXPECT_THROW(arithmetic.extract(ctx), JsonLogicError);
}

TEST(TestCategoricalEncoding, AStringLiteralIsNotACategory)
{
    const FeatureExtractor extractor({nlohmann::json::parse(R"({"+": ["fp16", 1]})")},
                                     Encoding{{"$kernel.dtype", {{"fp16", 1}}}});
    const FeatureExtractionContext ctx;

    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
}

} // namespace
