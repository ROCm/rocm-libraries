// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/uhd/CategoricalEncoding.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/FeatureExtractor.hpp>

/**
 * @file TestCategoricalEncoding.cpp
 * @brief The fixed string -> number table and the one place it applies
 * (RFC 0019 §6.5).
 *
 * Two things are being defended, and they pull in opposite directions:
 *
 *  - a categorical field has to reach the model as a number, or `dtype` and `layout`
 *    cannot be features at all;
 *  - a string that is *not* an encodable category has to keep failing loudly, because
 *    the alternative already happened: a NaN went down a GBDT's `default_left` branch
 *    and scored as ordinary data with nothing in the log.
 *
 * And one thing that outlives both: the numbers themselves. A trained model.bin has
 * them baked into its split thresholds, so renumbering an existing assignment breaks
 * every model in the field without breaking a load, a features_hash, or any test that
 * only checks the runtime agrees with itself. FrozenPrefixIsPinned is that test.
 */

namespace
{

using hipdnn_plugin_sdk::ingestor::uhd::CATEGORICAL_ENCODING_FROZEN_DIGEST;
using hipdnn_plugin_sdk::ingestor::uhd::CATEGORICAL_ENCODING_FROZEN_ENTRIES;
using hipdnn_plugin_sdk::ingestor::uhd::CATEGORICAL_ENCODING_TABLE;
using hipdnn_plugin_sdk::ingestor::uhd::categoricalEncodingDigest;
using hipdnn_plugin_sdk::ingestor::uhd::encodeCategorical;
using hipdnn_plugin_sdk::ingestor::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::ingestor::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::ingestor::uhd::JsonLogicError;

/// Extract a one-entry signature with `$kernel.<field>` bound to `value`.
double extractKernelFeature(const std::string& field, const std::string& value)
{
    const FeatureExtractor extractor({"$kernel." + field});
    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{field, value}});
    return extractor.extract(ctx).at(0);
}

// ---- The numbers are frozen ----------------------------------------------------
//
// The only mechanical guard that survives a maintainer who has never read this file.

TEST(TestCategoricalEncoding, FrozenPrefixIsPinned)
{
    // Editing, reordering or deleting any frozen assignment lands here. Appending past
    // the frozen prefix does not, which is the difference that makes the pin usable.
    EXPECT_EQ(categoricalEncodingDigest(), std::string(CATEGORICAL_ENCODING_FROZEN_DIGEST))
        << "An existing categorical assignment changed. Every model.bin already trained "
           "has the old numbers baked into its split thresholds, so this is a silent "
           "re-pointing of every threshold in the field, not a refactor. Restore the "
           "assignment and append instead.";

    // A deletion shortens the table without touching the frozen prefix's contents, so
    // the digest alone would not see it.
    EXPECT_GE(CATEGORICAL_ENCODING_TABLE.size(), CATEGORICAL_ENCODING_FROZEN_ENTRIES);
}

// ---- A known category encodes to its number ------------------------------------

TEST(TestCategoricalEncoding, KnownCategoryEncodesToItsNumber)
{
    // Spot-checked against the table's stated ordering rule (byte width ascending):
    // bf16 and fp16 are the two-byte pair, fp32 sits in the four-byte block.
    EXPECT_EQ(encodeCategorical("dtype", "bf16"), 12.0);
    EXPECT_EQ(encodeCategorical("dtype", "fp16"), 13.0);
    EXPECT_EQ(encodeCategorical("dtype", "fp32"), 15.0);
    EXPECT_EQ(encodeCategorical("layout", "NCHW"), 2.0);
}

TEST(TestCategoricalEncoding, AStringFeatureReachesTheModelAsItsCode)
{
    // The end-to-end point of the table: before it, this signature threw and the
    // engine degraded to static order.
    EXPECT_DOUBLE_EQ(extractKernelFeature("dtype", "fp16"), 13.0);
    EXPECT_DOUBLE_EQ(extractKernelFeature("layout", "BSHD"), 7.0);
}

// ---- Case is a spelling, not a category ----------------------------------------

TEST(TestCategoricalEncoding, LetterCaseDoesNotChangeTheCode)
{
    // The rocKE KMDs declare `"BF16"`; to_string(DataType) produces `"bf16"`. One value,
    // two shift keys. Rejecting either one stopped a gfx942 sweep at training and bought
    // nothing, so the lookup folds ASCII case -- and folds it to the *same* code, not to
    // a second row that happens to carry the same number.
    EXPECT_EQ(encodeCategorical("dtype", "BF16"), 12.0);
    EXPECT_EQ(encodeCategorical("dtype", "bf16"), 12.0);
    EXPECT_EQ(encodeCategorical("dtype", "Bf16"), 12.0);
    EXPECT_EQ(encodeCategorical("dtype", "FP16"), encodeCategorical("dtype", "fp16"));

    // Both directions: the table spells layouts upper case, so the fold has to work from
    // an upper-case row towards a lower-case query too.
    EXPECT_EQ(encodeCategorical("layout", "nchw"), 2.0);
    EXPECT_EQ(encodeCategorical("LAYOUT", "NCHW"), 2.0);
}

TEST(TestCategoricalEncoding, ACaseVariantReachesTheModelAsTheSameCode)
{
    // End to end, through the extractor the engine actually runs: the KMD's spelling and
    // the runtime's spelling land on one number, which is the only reason a model trained
    // on one is meaningful against the other.
    EXPECT_DOUBLE_EQ(extractKernelFeature("dtype", "BF16"),
                     extractKernelFeature("dtype", "bf16"));
    EXPECT_DOUBLE_EQ(extractKernelFeature("dtype", "BF16"), 12.0);
}

TEST(TestCategoricalEncoding, FoldingCaseDoesNotInventACategory)
{
    // The fold is over letter case only. `pipeline` has no table in any casing, so this
    // still means nothing numerically and still has to throw rather than reach a row.
    EXPECT_THROW(extractKernelFeature("PIPELINE", "intrawave"), JsonLogicError);
}

// ---- An unencodable string still fails loudly ----------------------------------

TEST(TestCategoricalEncoding, StringOutsideAnyCategoryStillThrows)
{
    // `pipeline` has no table, so this string means nothing numerically. Scoring it as
    // data is exactly the failure the throw in toDouble exists to prevent.
    EXPECT_THROW(extractKernelFeature("pipeline", "intrawave"), JsonLogicError);
}

TEST(TestCategoricalEncoding, ValueOutsideAKnownCategoryStillThrows)
{
    // "float16" is a plausible spelling that this codebase never produces --
    // to_string(DataType) emits "fp16". Accepting near-misses is how a training corpus
    // and a runtime end up on two different axes.
    //
    // This is the line the case fold above must not cross. Folding case makes `BF16`
    // and `bf16` one value, because they are; widening it into anything that also
    // reconciles two vocabularies' spellings lands here, and that is the whole reason
    // this test outranks the convenience of accepting `float16`.
    EXPECT_THROW(extractKernelFeature("dtype", "float16"), JsonLogicError);
    EXPECT_EQ(encodeCategorical("dtype", "FLOAT16"), std::nullopt);
}

TEST(TestCategoricalEncoding, ArithmeticOnACategoryStillThrows)
{
    // The encoding applies at the signature entry, not inside every numeric context.
    // `dtype + 1` is arithmetic on a category: it has no meaning whether or not the
    // category is encodable, so it must keep failing.
    const FeatureExtractor extractor({R"({"+": ["$kernel.dtype", 1]})"});
    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{"dtype", std::string("fp16")}});

    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
}

TEST(TestCategoricalEncoding, AStringLiteralIsNotACategory)
{
    // A signature entry that is a bare literal, not a reference. If literals encoded,
    // "fp16" would become 13 wherever it appeared, including inside a comparison.
    const FeatureExtractor extractor({"\"fp16\""});
    const FeatureExtractionContext ctx;

    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
}

// ---- The number does not depend on who asked -----------------------------------

TEST(TestCategoricalEncoding, SameCategoryEncodesIdenticallyWhoeverAsked)
{
    // The reason the table is global rather than generated per descriptor
    // (RFC 0019 §11.3): two engines' scores are only comparable if their feature
    // vectors mean the same thing. Here the same category arrives through a different
    // namespace, a different field position and a separately constructed extractor --
    // every axis a per-descriptor map would have varied along.
    const FeatureExtractor kernelSide({"$kernel.dtype"});
    FeatureExtractionContext kernelCtx;
    kernelCtx.bindKernelVars({{"dtype", std::string("bf16")}});

    const FeatureExtractor querySide({"$q.batch", "$q.dtype"});
    FeatureExtractionContext queryCtx;
    queryCtx.bindQueryVars({{"batch", int64_t{8}}, {"dtype", std::string("bf16")}});

    EXPECT_DOUBLE_EQ(kernelSide.extract(kernelCtx).at(0), querySide.extract(queryCtx).at(1));
    EXPECT_DOUBLE_EQ(kernelSide.extract(kernelCtx).at(0), 12.0);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
