// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestFeatureExtractor.cpp
 * @brief Tests for the UHD feature extraction system.
 */

#include "heuristics/uhd/FeatureExtractor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_set>

using hipdnn_backend::heuristics::uhd::FeatureExtractionContext;
using hipdnn_backend::heuristics::uhd::FeatureExtractor;
using hipdnn_backend::heuristics::uhd::JsonLogicError;

namespace
{

class TestFeatureExtractor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set up a basic device context
        const FeatureExtractionContext::ValueMap deviceVars = {
            {"cu_count", 120.0},
            {"warp_size", int64_t{64}},
            {"total_global_mem", int64_t{68719476736}}, // 64 GB
        };
        _ctx.bindDeviceVars(deviceVars);

        // Set up kernel metadata
        const FeatureExtractionContext::ValueMap kernelVars = {
            {"tile_m", 64.0},
            {"tile_n", 64.0},
            {"tile_k", 16.0},
            {"split_k", 1.0},
        };
        _ctx.bindKernelVars(kernelVars);

        // Set up query properties
        const FeatureExtractionContext::ValueMap queryVars = {
            {"batch", 32.0},
            {"seqlen", 512.0},
            {"heads", 8.0},
        };
        _ctx.bindQueryVars(queryVars);
    }

    FeatureExtractionContext _ctx;
};

// ========== Basic feature extraction ==========

TEST_F(TestFeatureExtractor, ExtractsSingleFeature)
{
    const std::vector<std::string> signature = {"\"$device.cu_count\""};
    const FeatureExtractor extractor(signature);

    auto features = extractor.extract(_ctx);
    ASSERT_EQ(features.size(), 1u);
    EXPECT_DOUBLE_EQ(features[0], 120.0);
}

TEST_F(TestFeatureExtractor, ExtractsMultipleFeatures)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
        "\"$q.batch\"",
    };
    const FeatureExtractor extractor(signature);

    auto features = extractor.extract(_ctx);
    ASSERT_EQ(features.size(), 3u);
    EXPECT_DOUBLE_EQ(features[0], 120.0);
    EXPECT_DOUBLE_EQ(features[1], 64.0);
    EXPECT_DOUBLE_EQ(features[2], 32.0);
}

TEST_F(TestFeatureExtractor, ExtractsComputedFeatures)
{
    const std::vector<std::string> signature = {
        R"({"+": ["$kernel.tile_m", "$kernel.tile_n"]})", // 64 + 64 = 128
        R"({"*": ["$q.batch", "$q.seqlen"]})", // 32 * 512 = 16384
        R"({"ceil_div": ["$device.cu_count", "$kernel.tile_m"]})", // ceil(120/64) = 2
    };
    const FeatureExtractor extractor(signature);

    auto features = extractor.extract(_ctx);
    ASSERT_EQ(features.size(), 3u);
    EXPECT_DOUBLE_EQ(features[0], 128.0);
    EXPECT_DOUBLE_EQ(features[1], 16384.0);
    EXPECT_DOUBLE_EQ(features[2], 2.0);
}

// ========== Feature count ==========

TEST_F(TestFeatureExtractor, ReportsCorrectFeatureCount)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
        "\"$kernel.tile_n\"",
        "\"$q.batch\"",
    };
    const FeatureExtractor extractor(signature);
    EXPECT_EQ(extractor.featureCount(), 4u);
}

// ========== Variable references ==========

TEST_F(TestFeatureExtractor, CollectsVariableReferences)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        R"({"*": ["$kernel.tile_m", "$q.batch"]})",
    };
    const FeatureExtractor extractor(signature);

    const auto& refs = extractor.getVariableRefs();
    EXPECT_EQ(refs.size(), 3u);
    EXPECT_TRUE(refs.count("$device.cu_count") > 0);
    EXPECT_TRUE(refs.count("$kernel.tile_m") > 0);
    EXPECT_TRUE(refs.count("$q.batch") > 0);
}

// ========== Context validation ==========

TEST_F(TestFeatureExtractor, ValidatesCompleteContext)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
    };
    const FeatureExtractor extractor(signature);
    EXPECT_TRUE(extractor.validateContext(_ctx));
}

TEST_F(TestFeatureExtractor, DetectsIncompleteContext)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.missing_field\"",
    };
    const FeatureExtractor extractor(signature);
    EXPECT_FALSE(extractor.validateContext(_ctx));
}

TEST_F(TestFeatureExtractor, ReportsMissingVariables)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.missing_field\"",
        "\"$q.unknown\"",
    };
    const FeatureExtractor extractor(signature);

    const auto missing = extractor.getMissingVariables(_ctx);
    EXPECT_EQ(missing.size(), 2u);

    const std::unordered_set<std::string> missingSet(missing.begin(), missing.end());
    EXPECT_TRUE(missingSet.count("$kernel.missing_field") > 0);
    EXPECT_TRUE(missingSet.count("$q.unknown") > 0);
}

// ========== Signature hash ==========

TEST_F(TestFeatureExtractor, ComputesConsistentHash)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
    };
    const FeatureExtractor extractor1(signature);
    const FeatureExtractor extractor2(signature);

    EXPECT_EQ(extractor1.getSignatureHash(), extractor2.getSignatureHash());
}

TEST_F(TestFeatureExtractor, DifferentSignaturesDifferentHash)
{
    const std::vector<std::string> sig1 = {"\"$device.cu_count\""};
    const std::vector<std::string> sig2 = {"\"$kernel.tile_m\""};

    const FeatureExtractor extractor1(sig1);
    const FeatureExtractor extractor2(sig2);

    EXPECT_NE(extractor1.getSignatureHash(), extractor2.getSignatureHash());
}

TEST_F(TestFeatureExtractor, HashLengthIsConsistent)
{
    const std::vector<std::string> signature = {"\"$device.cu_count\""};
    const FeatureExtractor extractor(signature);

    // Hash format: "sha256:" (7 chars) + 16-char truncated hex = 23 chars
    // This matches the Python uhd_gen tool's format for cross-language consistency.
    EXPECT_EQ(extractor.getSignatureHash().length(), 23u);
    EXPECT_TRUE(extractor.getSignatureHash().rfind("sha256:", 0) == 0);
}

// ========== KMD field validation ==========

TEST_F(TestFeatureExtractor, ValidatesKnownKmdFields)
{
    const std::vector<std::string> signature = {
        "\"$kernel.tile_m\"",
        "\"$kernel.tile_n\"",
    };
    const FeatureExtractor extractor(signature);

    const std::unordered_set<std::string> kmdFields = {"tile_m", "tile_n", "tile_k", "split_k"};
    EXPECT_TRUE(extractor.validateAgainstKmdFields(kmdFields));
}

TEST_F(TestFeatureExtractor, DetectsMissingKmdFields)
{
    const std::vector<std::string> signature = {
        "\"$kernel.tile_m\"",
        "\"$kernel.unknown_field\"",
    };
    const FeatureExtractor extractor(signature);

    const std::unordered_set<std::string> kmdFields = {"tile_m", "tile_n"};
    EXPECT_FALSE(extractor.validateAgainstKmdFields(kmdFields));
}

TEST_F(TestFeatureExtractor, ReportsMissingKmdFields)
{
    const std::vector<std::string> signature = {
        "\"$kernel.tile_m\"",
        "\"$kernel.unknown1\"",
        "\"$kernel.unknown2\"",
    };
    const FeatureExtractor extractor(signature);

    const std::unordered_set<std::string> kmdFields = {"tile_m"};
    const auto missing = extractor.getMissingKmdFields(kmdFields);

    EXPECT_EQ(missing.size(), 2u);
    const std::unordered_set<std::string> missingSet(missing.begin(), missing.end());
    EXPECT_TRUE(missingSet.count("unknown1") > 0);
    EXPECT_TRUE(missingSet.count("unknown2") > 0);
}

TEST_F(TestFeatureExtractor, NonKernelVarsIgnoredInKmdValidation)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$q.batch\"",
    };
    const FeatureExtractor extractor(signature);

    // Empty KMD fields should still pass because no $kernel.* refs exist
    const std::unordered_set<std::string> emptyKmdFields;
    EXPECT_TRUE(extractor.validateAgainstKmdFields(emptyKmdFields));
}

// ========== Context binding ==========

TEST_F(TestFeatureExtractor, ClearResetsContext)
{
    _ctx.clear();

    const std::vector<std::string> signature = {"\"$device.cu_count\""};
    const FeatureExtractor extractor(signature);
    EXPECT_FALSE(extractor.validateContext(_ctx));
}

TEST_F(TestFeatureExtractor, SingleBindAddsVariable)
{
    FeatureExtractionContext ctx;
    ctx.bind("$custom.value", 42.0);

    const std::vector<std::string> signature = {"\"$custom.value\""};
    const FeatureExtractor extractor(signature);

    const auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 1u);
    EXPECT_DOUBLE_EQ(features[0], 42.0);
}

// ========== Consistency and Edge Case Tests ==========

TEST_F(TestFeatureExtractor, ExtractorProducesSameResultsOnMultipleCalls)
{
    const std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
        "\"$q.batch\"",
    };
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({{"cu_count", 120.0}});
    ctx.bindKernelVars({{"tile_m", 256.0}});
    ctx.bindQueryVars({{"batch", 32.0}});

    // Extract multiple times and verify same results
    const auto features1 = extractor.extract(ctx);
    const auto features2 = extractor.extract(ctx);
    const auto features3 = extractor.extract(ctx);

    ASSERT_EQ(features1.size(), 3u);
    ASSERT_EQ(features2.size(), 3u);
    ASSERT_EQ(features3.size(), 3u);

    for(size_t i = 0; i < features1.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(features1[i], features2[i]) << "Mismatch at index " << i;
        EXPECT_DOUBLE_EQ(features2[i], features3[i]) << "Mismatch at index " << i;
    }
}

TEST_F(TestFeatureExtractor, EmptySignatureProducesEmptyVector)
{
    const std::vector<std::string> emptySignature;
    const FeatureExtractor extractor(emptySignature);

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({{"cu_count", 120.0}});

    const auto features = extractor.extract(ctx);
    EXPECT_TRUE(features.empty());
}

TEST_F(TestFeatureExtractor, EmptySignatureHasEmptyHash)
{
    const std::vector<std::string> emptySignature;
    const FeatureExtractor extractor(emptySignature);

    // Empty signature should produce a consistent (possibly empty) hash
    const auto& hash = extractor.getSignatureHash();
    // Hash of empty content is still a valid SHA-256 hash
    EXPECT_FALSE(hash.empty());
}

// ========== Signature wire format (RFC §7.2) ==========

TEST_F(TestFeatureExtractor, AcceptsBareFieldReference)
{
    // RFC §7.2 canonical spelling, and what tools/uhd_gen emits. A bare reference is
    // not valid JSON, so it must be lifted rather than parsed.
    const std::vector<std::string> signature = {"$device.cu_count", "$kernel.tile_m"};
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({{"cu_count", 120.0}});
    ctx.bindKernelVars({{"tile_m", 256.0}});

    const auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 2u);
    EXPECT_DOUBLE_EQ(features[0], 120.0);
    EXPECT_DOUBLE_EQ(features[1], 256.0);
}

TEST_F(TestFeatureExtractor, BareAndQuotedReferencesAreEquivalent)
{
    const std::vector<std::string> bare = {"$device.cu_count", "$kernel.tile_m"};
    const std::vector<std::string> quoted = {"\"$device.cu_count\"", "\"$kernel.tile_m\""};

    const FeatureExtractor bareExtractor(bare);
    const FeatureExtractor quotedExtractor(quoted);

    EXPECT_EQ(bareExtractor.getSignatureHash(), quotedExtractor.getSignatureHash());

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({{"cu_count", 120.0}});
    ctx.bindKernelVars({{"tile_m", 256.0}});

    EXPECT_EQ(bareExtractor.extract(ctx), quotedExtractor.extract(ctx));
}

TEST_F(TestFeatureExtractor, MixesBareReferencesWithDerivedExpressions)
{
    const std::vector<std::string> signature = {
        "$q.batch",
        R"({"*": ["$q.batch", "$q.num_heads"]})",
    };
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"batch", 32.0}, {"num_heads", 8.0}});

    const auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 2u);
    EXPECT_DOUBLE_EQ(features[0], 32.0);
    EXPECT_DOUBLE_EQ(features[1], 256.0);
}

TEST_F(TestFeatureExtractor, StringValuedBindingIsATypeError)
{
    // RFC 0019 §7.2 requires failing closed on a type error, not only on an unknown
    // symbol. Yielding NaN here would be silently wrong: a GBDT treats NaN as a
    // missing value, routes it down default_left, and returns an ordinary leaf — so
    // the garbage would be scored as data and never surface.
    //
    // NOTE: $device.arch is no longer a valid device feature per RFC 0019 §6.1
    // (architecture is a KDP property, not a runtime device feature). This test
    // uses a hypothetical string-valued field to verify type-error handling.
    const std::vector<std::string> signature = {"$device.string_field"};
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({{"string_field", std::string("test_value")}});

    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
}

TEST_F(TestFeatureExtractor, NumericBindingsOfEveryTypeStillResolve)
{
    // The type check must reject only strings — double, int64 and bool all convert.
    const std::vector<std::string> signature
        = {"$device.as_double", "$device.as_int", "$device.as_bool"};
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({
        {"as_double", 1.5},
        {"as_int", int64_t{7}},
        {"as_bool", true},
    });

    const auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 3u);
    EXPECT_DOUBLE_EQ(features[0], 1.5);
    EXPECT_DOUBLE_EQ(features[1], 7.0);
    EXPECT_DOUBLE_EQ(features[2], 1.0);
}

TEST_F(TestFeatureExtractor, UnknownSymbolStillReportsAsUndefined)
{
    // The type error must not swallow the pre-existing unknown-symbol diagnostic.
    const FeatureExtractor extractor(std::vector<std::string>{"$device.not_bound"});

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({{"cu_count", 120.0}});

    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
}

TEST_F(TestFeatureExtractor, MalformedNonReferenceEntryThrows)
{
    const std::vector<std::string> signature = {"{not valid json"};
    EXPECT_THROW(FeatureExtractor{signature}, JsonLogicError);
}

// ========== Hash is order-sensitive (RFC §7.2) ==========

TEST_F(TestFeatureExtractor, PermutedSignatureProducesDifferentHash)
{
    // RFC §7.2 requires the signature to match training exactly. Hashing a sorted copy
    // would make a permuted signature — a real feature-contract break — invisible.
    const std::vector<std::string> signature = {"$q.batch", "$kernel.tile_m", "$device.cu_count"};
    const std::vector<std::string> permuted = {"$device.cu_count", "$kernel.tile_m", "$q.batch"};

    EXPECT_NE(FeatureExtractor::computeHash(signature), FeatureExtractor::computeHash(permuted));
}

// Pinned against tools/uhd_gen/features.py compute_features_hash. Canonicalization is
// structural, not textual: each entry is parsed, then the array is dumped compact. The
// mirror of every case below lives in
// tools/uhd_gen/tests/test_features_hash.py::test_canonical_form_matches_runtime, and
// both must be updated together — editing one alone silently breaks loading for every
// descriptor the generator emits.

TEST_F(TestFeatureExtractor, HashMatchesGeneratorForBareReferences)
{
    const std::vector<std::string> signature = {"$q.batch", "$kernel.tile_m", "$device.cu_count"};
    EXPECT_EQ(FeatureExtractor::computeHash(signature), "sha256:fe9d0487031089e0");
}

TEST_F(TestFeatureExtractor, HashMatchesGeneratorForPrequotedReference)
{
    EXPECT_EQ(FeatureExtractor::computeHash({"\"$q.batch\""}), "sha256:611513da8e8614b2");
}

TEST_F(TestFeatureExtractor, HashMatchesGeneratorForDerivedExpression)
{
    // The case that exposed the original divergence: hashing raw entry strings makes
    // this an opaque escaped string on the Python side and a parsed node here.
    const std::vector<std::string> signature = {
        "$q.batch",
        R"({"*": ["$q.batch", "$q.num_heads"]})",
    };
    EXPECT_EQ(FeatureExtractor::computeHash(signature), "sha256:d5ae6976facefe74");
}

TEST_F(TestFeatureExtractor, HashMatchesGeneratorForNestedExpression)
{
    const std::vector<std::string> signature
        = {R"({"log2": [{"*": ["$q.batch", "$q.num_heads"]}]})"};
    EXPECT_EQ(FeatureExtractor::computeHash(signature), "sha256:8f014cf81bab5f8c");
}

TEST_F(TestFeatureExtractor, HashMatchesGeneratorForEmptySignature)
{
    EXPECT_EQ(FeatureExtractor::computeHash({}), "sha256:4f53cda18c2baa0c");
}

// ========== Numeric literals the two languages render differently ==========
//
// Strings, keys, escaping and unicode canonicalize identically. Numbers are the one
// axis where nlohmann and Python disagree, so literals near or past the divergence
// are rejected rather than hashed into a digest the other side cannot reproduce.
// Mirrored by tools/uhd_gen/tests/test_features_hash.py.

TEST_F(TestFeatureExtractor, RejectsFloatAtScientificNotationThreshold)
{
    // nlohmann renders this "1e+15", Python "1000000000000000.0".
    EXPECT_THROW(FeatureExtractor::computeHash({R"({">": ["$q.batch", 1e15]})"}), JsonLogicError);
}

TEST_F(TestFeatureExtractor, RejectsIntegerBeyondInt64)
{
    // nlohmann degrades this to double, which both diverges from Python's arbitrary
    // precision *and* collides with neighbouring values.
    EXPECT_THROW(FeatureExtractor::computeHash({"18446744073709551616"}), JsonLogicError);
    EXPECT_THROW(FeatureExtractor::computeHash({"-9223372036854775809"}), JsonLogicError);
}

TEST_F(TestFeatureExtractor, RejectsNonFiniteLiteral)
{
    // Python's json accepts these as extensions; nlohmann does not.
    EXPECT_THROW(FeatureExtractor::computeHash({"NaN"}), JsonLogicError);
    EXPECT_THROW(FeatureExtractor::computeHash({"Infinity"}), JsonLogicError);
    EXPECT_THROW(FeatureExtractor::computeHash({"1e400"}), JsonLogicError);
}

TEST_F(TestFeatureExtractor, AcceptsLiteralJustBelowThreshold)
{
    // Boundary check, and a cross-language pin: the generator produces the same digest
    // for this input.
    EXPECT_EQ(FeatureExtractor::computeHash({"999999999999999.0"}), "sha256:1449061ef40ea91e");
}

TEST_F(TestFeatureExtractor, AcceptsRealisticFeatureLiterals)
{
    // Tile sizes, dimensions and thresholds are orders of magnitude clear of the bound.
    EXPECT_NO_THROW(FeatureExtractor::computeHash({R"({">": ["$q.seqlen_q", 4096]})"}));
    EXPECT_NO_THROW(FeatureExtractor::computeHash({R"({"*": ["$kernel.tile_m", 0.5]})"}));
    EXPECT_NO_THROW(FeatureExtractor::computeHash({"$q.batch", "1e14", "-1e14"}));
}

TEST_F(TestFeatureExtractor, RejectionAppliesToNestedLiterals)
{
    // The walk has to reach literals buried in operator trees, not just top level.
    EXPECT_THROW(FeatureExtractor::computeHash({R"({"log2": [{"+": ["$q.batch", 1e16]}]})"}),
                 JsonLogicError);
}

TEST_F(TestFeatureExtractor, ConstructorRejectsUnsafeLiteralToo)
{
    // computeHash runs from the ctor, so an unsafe literal cannot slip in by building
    // an extractor directly instead of hashing.
    EXPECT_THROW(FeatureExtractor{std::vector<std::string>{"1e15"}}, JsonLogicError);
}

// ========== Shared vs. per-candidate partitioning (RFC §6 step 2) ==========

TEST_F(TestFeatureExtractor, PartitionsSignatureByKernelDependence)
{
    const std::vector<std::string> signature = {
        "$device.cu_count", // shared
        "$q.batch", // shared
        "$kernel.tile_m", // per-candidate
        R"({"*": ["$q.batch", "$kernel.tile_m"]})", // per-candidate (mixed)
        R"({"*": ["$q.batch", "$device.cu_count"]})", // shared (derived, no kernel ref)
    };
    const FeatureExtractor extractor(signature);

    EXPECT_EQ(extractor.featureCount(), 5u);
    EXPECT_EQ(extractor.kernelDependentCount(), 2u);
}

TEST_F(TestFeatureExtractor, BareKernelReferenceUnderShapeIsKernelDependent)
{
    // extractVariables reports the syntactic reference ($kernel), but the shape
    // operator resolves a synthesized name ($kernel.shape_0). Matching only on the
    // "$kernel." prefix would file this as shared, and it would then be evaluated in
    // the shared pass before any kernel metadata is bound.
    const std::vector<std::string> signature = {R"({"shape": ["$kernel", 0]})"};
    const FeatureExtractor extractor(signature);

    EXPECT_EQ(extractor.kernelDependentCount(), 1u);

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({{"cu_count", 120.0}});

    // Shared pass must not touch it — no kernel vars are bound yet.
    auto row = extractor.extractSharedRow(ctx);
    ASSERT_EQ(row.size(), 1u);

    ctx.bindKernelVars({{"shape_0", 128.0}});
    extractor.extractKernelInto(ctx, row);
    EXPECT_DOUBLE_EQ(row[0], 128.0);
}

TEST_F(TestFeatureExtractor, NamespacedTensorReferenceUnderShapeIsKernelDependent)
{
    const std::vector<std::string> signature = {R"({"shape": ["$kernel.tensor", 0]})"};
    const FeatureExtractor extractor(signature);

    EXPECT_EQ(extractor.kernelDependentCount(), 1u);
}

TEST_F(TestFeatureExtractor, SharedPlusKernelExtractionMatchesFullExtraction)
{
    const std::vector<std::string> signature = {
        "$device.cu_count",
        "$kernel.tile_m",
        "$q.batch",
        R"({"*": ["$q.batch", "$kernel.tile_m"]})",
    };
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindDeviceVars({{"cu_count", 120.0}});
    ctx.bindQueryVars({{"batch", 32.0}});
    ctx.bindKernelVars({{"tile_m", 256.0}});

    auto split = extractor.extractSharedRow(ctx);
    extractor.extractKernelInto(ctx, split);

    EXPECT_EQ(split, extractor.extract(ctx));
}

TEST_F(TestFeatureExtractor, SharedRowIsReusableAcrossCandidates)
{
    const std::vector<std::string> signature = {"$q.batch", "$kernel.tile_m"};
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"batch", 32.0}});

    // Shared slots are evaluated once; only kernel slots change per candidate.
    const auto sharedRow = extractor.extractSharedRow(ctx);

    ctx.bindKernelVars({{"tile_m", 64.0}});
    auto rowA = sharedRow;
    extractor.extractKernelInto(ctx, rowA);

    ctx.clearKernelVars();
    ctx.bindKernelVars({{"tile_m", 128.0}});
    auto rowB = sharedRow;
    extractor.extractKernelInto(ctx, rowB);

    EXPECT_DOUBLE_EQ(rowA[0], 32.0);
    EXPECT_DOUBLE_EQ(rowB[0], 32.0);
    EXPECT_DOUBLE_EQ(rowA[1], 64.0);
    EXPECT_DOUBLE_EQ(rowB[1], 128.0);
}

TEST_F(TestFeatureExtractor, ExtractKernelIntoRejectsWrongWidthRow)
{
    const std::vector<std::string> signature = {"$q.batch", "$kernel.tile_m"};
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"batch", 32.0}});
    ctx.bindKernelVars({{"tile_m", 64.0}});

    std::vector<double> tooNarrow(1, 0.0);
    EXPECT_THROW(extractor.extractKernelInto(ctx, tooNarrow), JsonLogicError);
}

TEST_F(TestFeatureExtractor, ClearKernelVarsDropsOnlyKernelBindings)
{
    // Reusing a context across candidates is only safe if a candidate that omits a
    // field cannot inherit the previous candidate's value.
    const std::vector<std::string> signature = {"$q.batch", "$kernel.tile_m"};
    const FeatureExtractor extractor(signature);

    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"batch", 32.0}});
    ctx.bindKernelVars({{"tile_m", 64.0}});
    EXPECT_NO_THROW(extractor.extract(ctx));

    ctx.clearKernelVars();

    // $q.* survives, $kernel.* is gone — so the omission surfaces instead of going stale.
    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
    const FeatureExtractor queryOnly(std::vector<std::string>{"$q.batch"});
    EXPECT_NO_THROW(queryOnly.extract(ctx));
}

} // namespace
