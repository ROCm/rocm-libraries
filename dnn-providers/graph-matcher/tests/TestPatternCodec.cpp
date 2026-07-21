// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Phase-6 PatternCodec tests: the flat, versioned, little-endian wire form of a
// CompiledPattern (PatternCodec::serialize / deserialize / emitEmbeddedArray).
// These defend the codec's contracts only -- the builder/compiler/matcher are
// covered by earlier phases and are used here purely to produce patterns to
// serialize and to prove a *restored* pattern still matches a real graph
// identically. deserialize treats its input as untrusted and never throws; it
// returns DeserializeResult{ok,pattern,error}, so rejections are asserted via
// ok==false + a non-empty error string (never EXPECT_THROW).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_graph_matcher/PatternCodec.hpp>
#include <hipdnn_graph_matcher/PatternCompiler.hpp>
#include <hipdnn_graph_matcher/Predicate.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <random>
#include <string>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;
namespace data = hipdnn_flatbuffers_sdk::data_objects;

// --- Helpers -------------------------------------------------------------

gm::VarId varOf(const gm::CompiledPattern& p, std::string_view name) {
    for (gm::VarId v = 0; v < p.varCount(); ++v) {
        if (p.varName(v) == name) {
            return v;
        }
    }
    return UINT32_MAX;
}

gm::SymId symIdOf(const gm::CompiledPattern& p, std::string_view name) {
    for (gm::SymId s = 0; s < p.symCount(); ++s) {
        if (p.symName(s) == name) {
            return s;
        }
    }
    return UINT32_MAX;
}

// A rich pattern exercising, on a matmul->bias(pointwise) chain that matches
// createValidMatmulBiasActivGraph:
//   * two nodes with operand + result edges,
//   * an OPTIONAL operand edge ($opt = pointwise in_2, absent in the graph),
//   * three symbols bound through shape constraints (m,k,n) + an explicit
//     dimBinding,
//   * every constraint kind the vocabulary offers (Dtype/Rank/Shape/Layout
//     [Contiguous + PackedOrder]/Attr/UseCount/ConsumerCount/NoConsumerOutside/
//     SameDtype/SameDim),
//   * a native predicate (hipdnn.same_dtype, resolved from builtin()).
// Every constraint genuinely holds for the graph so the pattern MATCHES; that is
// what lets the round-trip test compare the restored match to the original.
gm::CompiledPattern buildRichPattern() {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    pb.addNode("pointwise", {{"in_0", "$h"}, {"in_1", "$bias"}, {"in_2", "$opt", true}},
               {{"out_0", "$y"}});

    pb.constrainDtype("$a", {static_cast<int32_t>(data::DataType::FLOAT)});
    pb.constrainRank("$a", 2);
    pb.constrainShape("$a", {gm::DimSpec::of("m"), gm::DimSpec::of("k")});  // m:=4, k:=8
    pb.constrainShape("$b", {gm::DimSpec::of("k"), gm::DimSpec::of("n")});  // k==8, n:=5
    pb.bindDim("$h", 0, "m");                                               // 4 == m
    pb.constrainContiguous("$a");                                           // Layout/Contiguous
    pb.constrainLayout("$b", {0, 1});                                       // Layout/PackedOrder
    pb.constrainAttr(1, "operation", gm::Cmp::Eq,
                     {static_cast<int64_t>(data::PointwiseMode::ADD)});  // bias is ADD
    pb.constrainUseCount("$h", gm::Cmp::Eq, 1);
    pb.constrainConsumerCount("$h", gm::Cmp::Eq, 1);
    pb.constrainNoConsumerOutside("$h");
    pb.constrainSameDtype("$a", "$b");
    pb.constrainSameDim("$a", 1, "$b", 0);  // 8 == 8
    pb.addPredicate("hipdnn.same_dtype", {{gm::PredicateArg::Source::Var, "$a", 0},
                                          {gm::PredicateArg::Source::Var, "$b", 0}});
    return pb.build();
}

// Matches the rich pattern against createValidMatmulBiasActivGraph. The graph
// builder is returned via keepAlive so its buffer outlives the view.

gm::MatchResult matchAgainstMatmulBias(const gm::CompiledPattern& p,
                                       flatbuffers::FlatBufferBuilder& keepAlive) {
    keepAlive = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(keepAlive.GetBufferPointer(), keepAlive.GetSize());
    gm::GraphView view(graph);
    return gm::Matcher::match(p, view);
}

// A single pointwise(ADD) node with in_0=uid1 -> out_0=uid2 and NO in_2, so an
// optional in_2 operand slot is genuinely absent (its var stays unbound).
flatbuffers::FlatBufferBuilder makeSinglePointwiseGraph() {
    flatbuffers::FlatBufferBuilder b;
    const std::vector<int64_t> dims{4, 4};
    const std::vector<int64_t> strides{4, 1};
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        data::CreateTensorAttributesDirect(b, 1, "x", data::DataType::FLOAT, &strides, &dims),
        data::CreateTensorAttributesDirect(b, 2, "y", data::DataType::FLOAT, &strides, &dims)};
    auto attrs = data::CreatePointwiseAttributes(b, data::PointwiseMode::ADD, flatbuffers::nullopt,
                                                 flatbuffers::nullopt, flatbuffers::nullopt,
                                                 flatbuffers::nullopt, /*in_0=*/1,
                                                 flatbuffers::nullopt,  // in_1 absent
                                                 flatbuffers::nullopt, /*out_0=*/2);
    std::vector<flatbuffers::Offset<data::Node>> n{data::CreateNodeDirect(
        b, "pw", data::DataType::FLOAT, data::NodeAttributes::PointwiseAttributes, attrs.Union())};
    b.Finish(data::CreateGraphDirect(b, "test", data::DataType::FLOAT, data::DataType::FLOAT,
                                     data::DataType::FLOAT, &t, &n));
    return b;
}

// Convergence / determinism fixture JSON: a matmul->pointwise chain with shape
// symbols, a use-count constraint and a cross-node same_dim constraint.
const char* const kMatmulBiasJson = R"({
  "schema": "hipdnn.criteria/v1",
  "nodes": [
    {"id":"mm","op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$h"}},
    {"id":"pw","op":"pointwise","operands":{"in_0":"$h"},"results":{"out_0":"$y"}}
  ],
  "constraints": [
    {"on":"$a","dtype":"FLOAT","shape":["m","k"]},
    {"on":"$b","shape":["k","n"]},
    {"on":"$h","use":"exactly_once"},
    {"kind":"same_dim","args":["$a",1,"$b",0]}
  ]
})";

// =========================================================================
// Round-trip fidelity
// =========================================================================

// The load-bearing test: a rich pattern serialized then deserialized restores to
// a pattern that (a) preserves the summary shape (node/var/sym counts + anchor)
// and (b) MATCHES the same graph with byte-for-byte identical bindings
// (varUids/symVals/nodeMap) as the original. Teeth: it compares the restored
// match result to the ORIGINAL's, so any dropped edge/constraint/symbol that
// altered the match would redden this.
TEST(PatternCodec, RichPatternRoundTripPreservesShapeAndMatch) {
    const gm::CompiledPattern original = buildRichPattern();

    const std::vector<uint8_t> bytes = gm::PatternCodec::serialize(original);
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(bytes);
    ASSERT_TRUE(dr.ok) << dr.error;
    const gm::CompiledPattern& restored = dr.pattern;

    // (a) summary shape preserved.
    EXPECT_EQ(restored.nodeCount(), original.nodeCount());
    EXPECT_EQ(restored.varCount(), original.varCount());
    EXPECT_EQ(restored.symCount(), original.symCount());
    EXPECT_EQ(restored.anchor(), original.anchor());
    ASSERT_EQ(original.nodeCount(), 2u);
    ASSERT_EQ(original.varCount(), 6u);  // $a $b $h $bias $opt $y
    ASSERT_EQ(original.symCount(), 3u);  // m k n

    // (b) restored pattern matches the same graph identically to the original.
    flatbuffers::FlatBufferBuilder ba, bb;
    const gm::MatchResult ro = matchAgainstMatmulBias(original, ba);
    const gm::MatchResult rr = matchAgainstMatmulBias(restored, bb);
    ASSERT_TRUE(ro.matched);
    ASSERT_TRUE(rr.matched);
    EXPECT_EQ(rr.varUids, ro.varUids);
    EXPECT_EQ(rr.symVals, ro.symVals);
    EXPECT_EQ(rr.nodeMap, ro.nodeMap);

    // Spot-check the concrete facts too, so a same-but-wrong pair of results
    // cannot slip through vector equality.
    EXPECT_EQ(rr.uidOf(varOf(restored, "$a")), 1);
    EXPECT_EQ(rr.uidOf(varOf(restored, "$b")), 2);
    EXPECT_EQ(rr.uidOf(varOf(restored, "$h")), 3);
    EXPECT_EQ(rr.uidOf(varOf(restored, "$y")), 5);
    EXPECT_EQ(rr.uidOf(varOf(restored, "$opt")), -1);  // absent optional stays unbound
    EXPECT_EQ(rr.symOf(symIdOf(restored, "k")), 8);
    EXPECT_EQ(rr.symOf(symIdOf(restored, "m")), 4);
    EXPECT_EQ(rr.symOf(symIdOf(restored, "n")), 5);
}

// =========================================================================
// Determinism / round-trip stability / convergence
// =========================================================================

// serialize is pure: the same pattern yields the exact same bytes every call.
TEST(PatternCodec, SerializeIsDeterministic) {
    const gm::CompiledPattern p = buildRichPattern();
    EXPECT_EQ(gm::PatternCodec::serialize(p), gm::PatternCodec::serialize(p));
}

// Round-trip is a fixed point on bytes: serialize(deserialize(bytes)) == bytes.
TEST(PatternCodec, RoundTripReproducesIdenticalBytes) {
    const gm::CompiledPattern p = buildRichPattern();
    const std::vector<uint8_t> bytes = gm::PatternCodec::serialize(p);
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(bytes);
    ASSERT_TRUE(dr.ok) << dr.error;
    EXPECT_EQ(gm::PatternCodec::serialize(dr.pattern), bytes);
}

// Convergence: two independent compiles of the same JSON serialize to identical
// bytes -- the property AOT golden-bytes embedding relies on.
TEST(PatternCodec, IndependentCompilesConvergeToSameBytes) {
    const gm::CompileResult a = gm::PatternCompiler::fromJson(kMatmulBiasJson);
    const gm::CompileResult b = gm::PatternCompiler::fromJson(kMatmulBiasJson);
    ASSERT_TRUE(a.ok) << a.error;
    ASSERT_TRUE(b.ok) << b.error;
    EXPECT_EQ(gm::PatternCodec::serialize(a.pattern), gm::PatternCodec::serialize(b.pattern));
}

// =========================================================================
// Header / version validation (fail-closed, never throws)
// =========================================================================

TEST(PatternCodec, RejectsNullBuffer) {
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(nullptr, 0);
    EXPECT_FALSE(dr.ok);
    EXPECT_FALSE(dr.error.empty());
}

TEST(PatternCodec, RejectsEmptyBuffer) {
    const std::vector<uint8_t> empty;
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(empty);
    EXPECT_FALSE(dr.ok);
    EXPECT_FALSE(dr.error.empty());
}

TEST(PatternCodec, RejectsBadMagic) {
    std::vector<uint8_t> bytes = gm::PatternCodec::serialize(buildRichPattern());
    ASSERT_GE(bytes.size(), 4u);
    bytes[0] ^= 0xFF;  // corrupt the 'H' of "HDGM"
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(bytes);
    EXPECT_FALSE(dr.ok);
    EXPECT_FALSE(dr.error.empty());
}

TEST(PatternCodec, RejectsUnsupportedWireVersion) {
    std::vector<uint8_t> bytes = gm::PatternCodec::serialize(buildRichPattern());
    ASSERT_GE(bytes.size(), 6u);
    // Version is a u16 LE at offset 4; bump it far past what we understand.
    bytes[4] = 0xFF;
    bytes[5] = 0xFF;
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(bytes);
    EXPECT_FALSE(dr.ok);
    EXPECT_FALSE(dr.error.empty());
}

TEST(PatternCodec, RejectsBadEndianByte) {
    std::vector<uint8_t> bytes = gm::PatternCodec::serialize(buildRichPattern());
    ASSERT_GE(bytes.size(), 7u);
    bytes[6] = 0x01;  // endian byte lives just after the u16 version; 0 is the only valid value
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(bytes);
    EXPECT_FALSE(dr.ok);
    EXPECT_FALSE(dr.error.empty());
}

// =========================================================================
// Adversarial input (no crash; every call returns)
// =========================================================================

// Every strict prefix of a valid blob is missing bytes the reader needs, so it
// must fail closed -- never crash, never spuriously succeed.
TEST(PatternCodec, EveryTruncationFailsClosed) {
    const std::vector<uint8_t> bytes = gm::PatternCodec::serialize(buildRichPattern());
    ASSERT_GT(bytes.size(), 1u);
    // The full buffer deserializes; every shorter prefix does not.
    ASSERT_TRUE(gm::PatternCodec::deserialize(bytes).ok);
    for (size_t cut = 1; cut < bytes.size(); ++cut) {
        const gm::DeserializeResult dr = gm::PatternCodec::deserialize(bytes.data(), cut);
        EXPECT_FALSE(dr.ok) << "prefix of length " << cut << " unexpectedly deserialized";
        EXPECT_FALSE(dr.error.empty()) << "prefix of length " << cut;
    }
}

// A flood of random byte vectors must never crash the fail-closed reader; it may
// return ok==true (a random blob happening to be well-formed) or ok==false, but
// it always returns. Deterministic: the PRNG is fixed-seeded and the seed is
// reported so a failure reproduces exactly.
TEST(PatternCodec, RandomBytesNeverCrash) {
    constexpr uint32_t kSeed = 0xC0DEC0DEu;
    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<int> byteDist(0, 255);
    std::uniform_int_distribution<size_t> lenDist(0, 512);

    size_t okCount = 0;
    for (int i = 0; i < 8000; ++i) {
        std::vector<uint8_t> buf(lenDist(rng));
        for (auto& b : buf) {
            b = static_cast<uint8_t>(byteDist(rng));
        }
        const gm::DeserializeResult dr = gm::PatternCodec::deserialize(buf);
        if (dr.ok) {
            ++okCount;
        } else {
            EXPECT_FALSE(dr.error.empty()) << "seed=" << kSeed << " iter=" << i;
        }
    }
    // We reached here without crashing -- the only real contract. okCount is
    // reported (not asserted) for visibility.
    SCOPED_TRACE("random-blob ok count (informational): " + std::to_string(okCount));
    SUCCEED() << "seed=" << kSeed << " ok=" << okCount;
}

// A valid header followed by a wildly oversized count field must be rejected on
// the bound check, not honored into an over-read.
TEST(PatternCodec, RejectsOversizedPoolCount) {
    std::vector<uint8_t> bytes = gm::PatternCodec::serialize(buildRichPattern());
    // Layout: magic(4) ver(2) endian(1) anchor(4) then u32 pool count at offset 11.
    ASSERT_GE(bytes.size(), 15u);
    bytes[11] = 0xFF;
    bytes[12] = 0xFF;
    bytes[13] = 0xFF;
    bytes[14] = 0xFF;
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(bytes);
    EXPECT_FALSE(dr.ok);
    EXPECT_FALSE(dr.error.empty());
}

// =========================================================================
// Field-level round-trip: optional edge, packed-order layout
// =========================================================================

// The optional-operand flag survives the wire: after round-trip the pattern
// still matches a graph lacking that operand and leaves the var unbound (uidOf
// == -1). Teeth: if the optional flag were dropped, in_2 would be required and
// the match would fail outright.
TEST(PatternCodec, OptionalEdgeFlagRoundTrips) {
    gm::PatternBuilder pb;
    pb.addNode("pointwise", {{"in_0", "$x"}, {"in_2", "$opt", true}}, {{"out_0", "$y"}});
    const gm::CompiledPattern original = pb.build();

    const gm::DeserializeResult dr =
        gm::PatternCodec::deserialize(gm::PatternCodec::serialize(original));
    ASSERT_TRUE(dr.ok) << dr.error;

    auto builder = makeSinglePointwiseGraph();  // no in_2
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const gm::MatchResult r = gm::Matcher::match(dr.pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(varOf(dr.pattern, "$x")), 1);
    EXPECT_EQ(r.uidOf(varOf(dr.pattern, "$opt")), -1);  // absent optional stays unbound
}

// A PackedOrder layout constraint (which carries an axisOrder vector) round-trips
// with the same accept/reject verdict as before serialization. Teeth: a correct
// order matches and a transposed order is rejected -- both preserved.
TEST(PatternCodec, PackedOrderLayoutRoundTrips) {
    auto builder = util::createValidMatmulGraph();  // a[4,8] strides[8,1]
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    auto buildLayout = [](std::vector<uint32_t> order) {
        gm::PatternBuilder pb;
        pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
        pb.constrainLayout("$a", order);
        return pb.build();
    };

    // {0,1} == row-major -> matches; verdict must survive the round-trip.
    const gm::CompiledPattern okPat = buildLayout({0, 1});
    const gm::DeserializeResult okDr =
        gm::PatternCodec::deserialize(gm::PatternCodec::serialize(okPat));
    ASSERT_TRUE(okDr.ok) << okDr.error;
    EXPECT_TRUE(gm::Matcher::match(okPat, view).matched);
    EXPECT_TRUE(gm::Matcher::match(okDr.pattern, view).matched);

    // {1,0} == transposed -> rejects; verdict must survive too.
    const gm::CompiledPattern noPat = buildLayout({1, 0});
    const gm::DeserializeResult noDr =
        gm::PatternCodec::deserialize(gm::PatternCodec::serialize(noPat));
    ASSERT_TRUE(noDr.ok) << noDr.error;
    EXPECT_FALSE(gm::Matcher::match(noPat, view).matched);
    EXPECT_FALSE(gm::Matcher::match(noDr.pattern, view).matched);

    // The axisOrder vector itself round-trips byte-for-byte.
    EXPECT_EQ(gm::PatternCodec::serialize(noDr.pattern), gm::PatternCodec::serialize(noPat));
}

// =========================================================================
// emitEmbeddedArray
// =========================================================================

// Counts non-overlapping occurrences of `needle` in `hay`.
size_t countOccurrences(const std::string& hay, const std::string& needle) {
    size_t count = 0;
    for (size_t pos = hay.find(needle); pos != std::string::npos;
         pos = hay.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

// The emitted C++ text declares the array + a matching `_size`, has exactly one
// 0x.. token per byte, and -- the real contract -- the bytes it encodes, parsed
// back out of the text, deserialize successfully and reproduce the source blob.
TEST(PatternCodec, EmitEmbeddedArrayShapeAndBytesRoundTrip) {
    const gm::CompiledPattern p = buildRichPattern();
    const std::vector<uint8_t> bytes = gm::PatternCodec::serialize(p);
    ASSERT_FALSE(bytes.empty());

    const std::string text = gm::PatternCodec::emitEmbeddedArray("kEmbeddedCriteria", bytes);

    // Declares the array and a matching size constant.
    EXPECT_NE(text.find("kEmbeddedCriteria[]"), std::string::npos);
    EXPECT_NE(text.find("kEmbeddedCriteria_size = " + std::to_string(bytes.size())),
              std::string::npos);
    // Exactly one hex byte token per source byte.
    EXPECT_EQ(countOccurrences(text, "0x"), bytes.size());

    // Parse every 0x.. token back out and confirm it equals the source blob...
    std::vector<uint8_t> parsed;
    for (size_t pos = text.find("0x"); pos != std::string::npos; pos = text.find("0x", pos + 2)) {
        parsed.push_back(static_cast<uint8_t>(std::stoul(text.substr(pos + 2, 2), nullptr, 16)));
    }
    EXPECT_EQ(parsed, bytes);

    // ...and that those exact embedded bytes deserialize into a live pattern.
    const gm::DeserializeResult dr = gm::PatternCodec::deserialize(parsed);
    ASSERT_TRUE(dr.ok) << dr.error;
    EXPECT_EQ(gm::PatternCodec::serialize(dr.pattern), bytes);
}

}  // namespace
