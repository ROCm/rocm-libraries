// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>

/// Categorical vocabularies belong to the model. Numeric expression contexts read
/// the code of an encoded reference; equality and membership retain string identity.

namespace
{

using hipdnn_plugin_sdk::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::uhd::JsonLogicError;

using Encoding = std::map<std::string, std::map<std::string, int32_t>>;

/// Extract a one-entry signature with `$kernel.<field>` bound to `value`, under the
/// encoding @p encoding -- which is what a descriptor declaring that field looks like.
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
    // The end-to-end point: without an encoding this signature throws and the engine
    // degrades to declared order. The numbers are the descriptor's own, not a table's.
    const Encoding encoding{{"$kernel.dtype", {{"bf16", 0}, {"fp16", 1}}},
                            {"$kernel.layout", {{"BSHD", 0}, {"NCHW", 1}}}};

    EXPECT_DOUBLE_EQ(extractKernelFeature("dtype", "fp16", encoding), 1.0);
    EXPECT_DOUBLE_EQ(extractKernelFeature("layout", "BSHD", encoding), 0.0);
}

TEST(TestCategoricalEncoding, TwoReferencesSharingATrailingNameDoNotMerge)
{
    // The bug this keying designs out. `$kernel.dtype` carries the rocKE KMD's "BF16";
    // `$q.attention_dense.dtype` carries the runtime's "bf16". Keyed by the trailing
    // field name both are "dtype", one vocabulary, and whichever spelling was seen first
    // decides -- which is how a gfx942 sweep died at training on a value the other half
    // of the same corpus had produced. Keyed by the whole reference they cannot meet.
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
    // No folding. The frozen table folded ASCII case because it had to reconcile two
    // vocabularies it could not see; a generated map records the spellings its corpus
    // actually held, so "BF16" is encodable exactly when the corpus contained "BF16".
    // A model fitted on one spelling has never been shown the other.
    const Encoding encoding{{"$kernel.dtype", {{"BF16", 4}}}};

    EXPECT_DOUBLE_EQ(extractKernelFeature("dtype", "BF16", encoding), 4.0);
    EXPECT_THROW(extractKernelFeature("dtype", "bf16", encoding), JsonLogicError);
}

TEST(TestCategoricalEncoding, TwoDescriptorsMayEncodeTheSameFieldDifferently)
{
    // The property the frozen table existed to provide, deliberately given up. Each
    // model reads its own codes; comparability between engines lives in the score's
    // declared units (§11.3), not in the feature vectors that produced it.
    const FeatureExtractor first({"$kernel.dtype"}, Encoding{{"$kernel.dtype", {{"bf16", 0}}}});
    const FeatureExtractor second({"$kernel.dtype"}, Encoding{{"$kernel.dtype", {{"bf16", 9}}}});

    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{"dtype", std::string("bf16")}});

    EXPECT_DOUBLE_EQ(first.extract(ctx).at(0), 0.0);
    EXPECT_DOUBLE_EQ(second.extract(ctx).at(0), 9.0);
}

TEST(TestCategoricalEncoding, TheEncodingTravelsIntoTheFeaturesHash)
{
    // RFC 0019 §6.5: a changed map changes what the model reads while leaving the
    // signature text identical, so the signature alone cannot guard the contract. Two
    // descriptors differing only in their codes must not share a features_hash.
    const std::vector<nlohmann::json> signature{"$kernel.dtype"};

    EXPECT_NE(FeatureExtractor::computeHash(signature, Encoding{{"$kernel.dtype", {{"bf16", 0}}}}),
              FeatureExtractor::computeHash(signature, Encoding{{"$kernel.dtype", {{"bf16", 9}}}}));

    // And a signature reading no string field hashes as it did before the field existed,
    // so models that never carried an encoding keep their contracts intact.
    EXPECT_EQ(FeatureExtractor::computeHash({"$kernel.tile_m"}, Encoding{}),
              FeatureExtractor::computeHash({"$kernel.tile_m"}));
}

// ---- An unencodable string still fails loudly ----------------------------------

TEST(TestCategoricalEncoding, StringOutsideAnyDeclaredFieldStillThrows)
{
    // The descriptor does not declare `pipeline` categorical, so this string means
    // nothing numerically. Scoring it as data is exactly the failure the throw exists
    // to prevent.
    const Encoding encoding{{"$kernel.dtype", {{"bf16", 0}}}};

    EXPECT_THROW(extractKernelFeature("pipeline", "intrawave", encoding), JsonLogicError);
}

TEST(TestCategoricalEncoding, ValueOutsideADeclaredFieldStillThrows)
{
    // "float16" is a plausible spelling this codebase never produces. The field IS
    // declared categorical, so this is not a type error -- it is a catalog that moved
    // past what the model was trained on, and it must surface rather than score.
    const Encoding encoding{{"$kernel.dtype", {{"bf16", 0}, {"fp16", 1}}}};

    EXPECT_THROW(extractKernelFeature("dtype", "float16", encoding), JsonLogicError);
}

TEST(TestCategoricalEncoding, EncodedReferencesWorkInInlineNumericAndStringContexts)
{
    const FeatureExtractor extractor(
        {nlohmann::json::parse(R"({"+": ["$kernel.dtype", 1]})"),
         nlohmann::json::parse(R"({"==": ["$kernel.dtype", "fp16"]})")},
        Encoding{{"$kernel.dtype", {{"fp16", 1}}}});
    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{"dtype", std::string("fp16")}});
    EXPECT_EQ(extractor.extract(ctx), (std::vector<double>{2, 1}));
}

TEST(TestCategoricalEncoding, AStringLiteralIsNotACategory)
{
    const FeatureExtractor extractor({nlohmann::json::parse(R"({"+": ["fp16", 1]})")},
                                     Encoding{{"$kernel.dtype", {{"fp16", 1}}}});
    const FeatureExtractionContext ctx;

    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
}

} // namespace
