// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestArch.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestGraphs.hpp"

/**
 * @file TestAttentionDenseDispatch.cpp
 * @brief The hipkernel:Gfx942AttentionDense and hipkernel:Gfx950AttentionDense packs at
 *        the bind and prepare layer: what the graph match writes into the bound tokens,
 *        and what prepare() does with them before it reaches a loader.
 *
 * Every failure this file is after is a SILENT one. A token carrying the wrong uid
 * launches the kernel against the wrong buffers; a scale bound as an integer rather than
 * as its bit pattern reads on the device as a denormal near zero. Neither faults, neither
 * sets a status, and the kernel-matching tests cannot see either -- kernelMatches() reads
 * only the causal and sliding_window tokens, so the four operand uids and the scale could
 * hold anything and every matcher case would still pass.
 *
 * The launch geometry is not reachable here. prepare() calls the arch's
 * gfx94xAttentionDenseGeometry() after buildIngestorKernelCode has loaded the code object,
 * and applies the result to the kernel code it keeps. No accessor exposes that geometry.
 * TestAttentionDenseGeometry.cpp pins what the functions compute; what prepare() passes
 * them needs a real archive and a real device. The launch values a save stores are tested
 * through gfx94xAttentionDenseLaunchValues() in TestPackLaunchValues.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using attention_dense_graphs::dispatchHandler;
using attention_dense_graphs::Graph;
using attention_dense_graphs::GraphSpec;
using attention_dense_graphs::ieee754Bits;
using attention_dense_graphs::kpackKernel;
using attention_dense_graphs::prepareFailure;
using attention_dense_graphs::pythonAbi;
using attention_dense_graphs::pythonLseAbi;
using attention_dense_graphs::TensorSpec;
using attention_dense_graphs::withStats;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

// ---------------------------------------------------------------------------
// The token names, restated
// ---------------------------------------------------------------------------
//
// Written out again rather than shared with the pack: these strings are the contract
// between the matcher that writes a token and the dispatch that reads it, and a test
// sharing the constant would keep passing through a rename that broke nothing but also
// pinned nothing. Each is the arch's token prefix followed by one of these.

constexpr std::string_view Q_TOKEN = ".q.uid";
constexpr std::string_view K_TOKEN = ".k.uid";
constexpr std::string_view V_TOKEN = ".v.uid";
constexpr std::string_view O_TOKEN = ".o.uid";
constexpr std::string_view CAUSAL_TOKEN = ".causal";
constexpr std::string_view SLIDING_WINDOW_TOKEN = ".sliding_window";
constexpr std::string_view SCALE_BITS_TOKEN = ".scale_bits";
constexpr std::string_view STATS_TOKEN = ".stats.uid";

/// @p arch's spelling of @p suffix.
std::string tokenName(const AttentionDenseTestArch& arch, std::string_view suffix)
{
    return std::string(arch.tokenPrefix) + std::string(suffix);
}

/// Every token @p arch's dispatch reads, in one place, so the case that removes them one at
/// a time cannot quietly stop covering one.
std::vector<std::string> allTokens(const AttentionDenseTestArch& arch)
{
    std::vector<std::string> tokens;
    for(const std::string_view suffix :
        {Q_TOKEN, K_TOKEN, V_TOKEN, O_TOKEN, CAUSAL_TOKEN, SLIDING_WINDOW_TOKEN, SCALE_BITS_TOKEN})
    {
        tokens.push_back(tokenName(arch, suffix));
    }
    if(arch.bindsStats)
    {
        tokens.push_back(tokenName(arch, STATS_TOKEN));
    }
    return tokens;
}

// ---------------------------------------------------------------------------
// A graph the engine serves
// ---------------------------------------------------------------------------

// Mutually distinct, and none of them a small ordinal: a uid that happened to equal an
// axis index or a loop counter could read as correct after a transposition.
constexpr int64_t Q_UID = 11;
constexpr int64_t K_UID = 22;
constexpr int64_t V_UID = 33;
constexpr int64_t O_UID = 44;

/// A fifth tensor, BHSD rather than BSHD, that no node references. It exists only so a
/// bound O token can be made to name a real but wrongly-strided tensor.
constexpr int64_t BHSD_UID = 55;

/// The stats output of a graph that binds one: FLOAT, [B, H, S, 1], packed.
constexpr int64_t STATS_UID = 66;

/// A FLOAT [B, H, S, 1] tensor no node references, strided token-major rather than packed,
/// so a bound stats token can be made to name a real tensor the kernel does not write.
constexpr int64_t TOKEN_MAJOR_STATS_UID = 77;

/// The stats token of a graph that binds no stats output.
constexpr int64_t NO_STATS = -1;

constexpr int64_t ABSENT_UID = 99;

constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 4;
constexpr int64_t SEQ = 512;
constexpr int64_t HEAD_SIZE = 128;

/// The graph every case binds: the default shape at Sq = Skv = 512 under the uids above,
/// top-left causal. It only has to be one the engine matches, so the tokens it binds can be
/// read; the applicability rules are the matcher suites' subject.
GraphSpec servedGraph(float scale = 0.125F)
{
    GraphSpec spec;
    spec.uids = {Q_UID, K_UID, V_UID, O_UID};
    spec.seqLenQ = SEQ;
    spec.seqLenKv = SEQ;
    spec.attnScaleValue = scale;
    return spec;
}

/// @p spec with the tensors no node names that a tampered token can point at: a BHSD one of
/// the operands' dims and, where @p arch binds stats, a token-major FLOAT [B, H, S, 1] one.
GraphSpec withUnnamedTensors(const AttentionDenseTestArch& arch, GraphSpec spec)
{
    TensorSpec bhsd;
    bhsd.uid = BHSD_UID;
    bhsd.dims = {BATCH, HEADS, SEQ, HEAD_SIZE};
    bhsd.strides = sdpaStrides(SdpaLayout::BHSD, HEADS, SEQ, HEAD_SIZE);
    bhsd.dataType = data_objects::DataType::BFLOAT16;
    spec.unnamedTensors.push_back(bhsd);
    if(arch.bindsStats)
    {
        TensorSpec tokenMajor;
        tokenMajor.uid = TOKEN_MAJOR_STATS_UID;
        tokenMajor.dims = {BATCH, HEADS, SEQ, 1};
        tokenMajor.strides = {SEQ * HEADS, 1, HEADS, 1};
        spec.unnamedTensors.push_back(tokenMajor);
    }
    return spec;
}

// ---------------------------------------------------------------------------
// The graph match seam
// ---------------------------------------------------------------------------

/// Runs the pack's graph match over @p graph. An empty result means the engine declined,
/// which no case here expects -- every graph built above is one it serves.
BoundTokens bindingsFor(const Graph& graph)
{
    auto bound = graph.match();
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the fixture graph, so there are no "
                                      "tokens to read and no bindings to hand prepare()";
    if(!bound.has_value())
    {
        return {};
    }
    return *bound;
}

/// The integer @p token carries. The sentinel is deliberately not zero: zero is a real
/// value for the causal and sliding-window tokens, so a missing token answering zero
/// would read as a correct binding.
constexpr int64_t NOT_BOUND = -1000;

int64_t tokenValue(const BoundTokens& bound, const std::string& token)
{
    return tryGetBoundInt(bound, token).value_or(NOT_BOUND);
}

// ---------------------------------------------------------------------------
// A descriptor prepare() can be driven with
// ---------------------------------------------------------------------------

/// A KPACK descriptor for @p arch's engine at the (@p blockM, @p blockN) tile, recording the
/// ABI the rocKE builder declares so the signature comparison agrees and a failure is the
/// archive's rather than a mismatch; TestAttentionDenseSignature.cpp owns that comparison.
/// A gfx942 emit_lse kernel takes lse_ptr last.
KernelDefinition makeKernel(const AttentionDenseTestArch& arch,
                            int64_t blockM = 256,
                            int64_t blockN = 64,
                            bool writesLse = false)
{
    return kpackKernel(arch, writesLse ? pythonLseAbi() : pythonAbi(), blockM, blockN);
}

std::string prepareFailure(const Graph& graph, const BoundTokens& bound)
{
    return prepareFailure(graph, bound, makeKernel(graph.arch()));
}

/// The fragment only the missing-token refusal carries.
constexpr const char* MISSING_TOKEN_MARKER = "missing bound token";

/// The fragment only the output-layout refusal carries.
constexpr const char* OUTPUT_LAYOUT_MARKER = "not dense BSHD";

/// The fragment only the tile refusal carries.
constexpr const char* TILE_MARKER = "no supported block_m/block_n tile";

/// The fragment only the stats-tensor refusal carries.
constexpr const char* STATS_MARKER = "the stats tensor is not the packed";

/// The fragment only the signature refusal carries.
constexpr const char* SIGNATURE_MARKER = "is packaged with arguments";

using Tiles = std::vector<std::pair<int64_t, int64_t>>;

/// An arch and the D128 tiles its prepare() must take and refuse.
struct DenseDispatchCase
{
    const AttentionDenseTestArch* arch;
    /// Every tile a D128 catalog candidate can carry.
    Tiles buildableTiles;
    /// Zero, negative, and each tile rule's nearest violators.
    Tiles unsupportedTiles;
};

/// gfx942: off the 32-row quantum (48), past the 1024-thread CTA (1024), and block_n not
/// dividing block_m (64/128, 256/48).
const DenseDispatchCase GFX942_DISPATCH{
    &GFX942_ATTENTION_DENSE,
    {{64, 64}, {128, 64}, {256, 32}, {256, 64}, {512, 64}},
    {{0, 64}, {256, 0}, {-256, 64}, {48, 64}, {1024, 64}, {64, 128}, {256, 48}}};

/// gfx950: block_m it does not build (64, 512) and D128 block_n 256, past the LDS budget.
const DenseDispatchCase GFX950_DISPATCH{
    &GFX950_ATTENTION_DENSE,
    {{128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}},
    {{0, 64}, {256, 0}, {-256, 64}, {64, 64}, {512, 64}, {128, 256}, {256, 256}}};

class TestAttentionDenseDispatch : public ::testing::TestWithParam<const DenseDispatchCase*>
{
protected:
    static const AttentionDenseTestArch& arch()
    {
        return *GetParam()->arch;
    }
};

} // namespace

// =============================================================================
// Registration
// =============================================================================

/// If the arch's register...AttentionDenseSymbols stops adding the dispatch symbol, or
/// IngestorPacks drops this engine's row, every descriptor naming the symbol is dropped at
/// load: the pack disappears, then the engine, and at the default log level nothing says
/// so. The graph would simply be served by some other provider, or by none.
TEST_P(TestAttentionDenseDispatch, DispatchSymbolResolves)
{
    registerNativeIngestorSymbols();

    EXPECT_NE(hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
                  std::string(arch().dispatchSymbol)),
              nullptr);
}

// =============================================================================
// Workspace
// =============================================================================

/// This kernel's only scratch is LDS and registers, and the shipped 10-argument ABI has no
/// workspace pointer to hand global scratch through, so there is nowhere for a non-zero
/// answer to be used. This is the only place the zero is pinned.
TEST_P(TestAttentionDenseDispatch, WorkspaceBytesIsAlwaysZero)
{
    const Graph graph(arch(), servedGraph());
    const auto bound = bindingsFor(graph);
    const auto* handler = dispatchHandler(arch());
    ASSERT_NE(handler, nullptr);

    EXPECT_EQ(handler->workspaceBytes(graph.context(), bound, makeKernel(arch())), 0U);
}

// =============================================================================
// What the match binds
// =============================================================================

/// The four uids the launch resolves device buffers by. launch() looks each one up with
/// findDeviceBuffer and passes the results positionally, so a Q/K transposition here is a
/// kernel reading keys as queries: full-rate arithmetic, no fault, wrong numbers. The
/// uids are mutually distinct, so any permutation of the four fails this.
TEST_P(TestAttentionDenseDispatch, BindsEachOperandUidToItsOwnToken)
{
    const Graph graph(arch(), servedGraph());
    const auto bound = bindingsFor(graph);

    EXPECT_EQ(tokenValue(bound, tokenName(arch(), Q_TOKEN)), Q_UID);
    EXPECT_EQ(tokenValue(bound, tokenName(arch(), K_TOKEN)), K_UID);
    EXPECT_EQ(tokenValue(bound, tokenName(arch(), V_TOKEN)), V_UID);
    EXPECT_EQ(tokenValue(bound, tokenName(arch(), O_TOKEN)), O_UID);
}

/// The softmax scale is an f32 kernarg, and a BoundTokens value is an int64_t, so the
/// matcher carries the float as its IEEE-754 storage and the dispatch memcpys it back.
///
/// Expectations written as literals read off the format rather than computed the way the
/// pack computes them. Both mutations this exists for are integer-valued and would pass
/// against any derived expectation: binding zero, and binding the float converted to an
/// integer -- which is zero for every scale a real attention graph uses, since they are
/// all below one. A zero scale is not an error on the device; it flattens every softmax
/// to a uniform distribution and returns success.
TEST_P(TestAttentionDenseDispatch, BindsTheSoftmaxScaleAsItsIeeeBitPattern)
{
    const std::string scaleBits = tokenName(arch(), SCALE_BITS_TOKEN);

    // 0.1875 = 1.5 * 2^-3. Sign 0, biased exponent 124 (0x7C), mantissa 0x400000.
    // Chosen with a non-zero mantissa: a power of two alone would not catch an encoding
    // that dropped the fraction field.
    const Graph fraction(arch(), servedGraph(0.1875F));
    EXPECT_EQ(tokenValue(bindingsFor(fraction), scaleBits), 0x3E400000);

    // 0.125 = 1.0 * 2^-3. Same exponent, zero mantissa -- the neighbouring encoding, so
    // the two together pin the exponent and the fraction independently.
    const Graph power(arch(), servedGraph(0.125F));
    EXPECT_EQ(tokenValue(bindingsFor(power), scaleBits), 0x3E000000);

    // The scale a D128 graph actually carries, 1/sqrt(128). Its pattern is impractical to
    // write by hand, so this case only adds that a realistic value survives the trip.
    constexpr float REALISTIC_SCALE = 0.08838834764831843F;
    const Graph realistic(arch(), servedGraph(REALISTIC_SCALE));
    EXPECT_EQ(tokenValue(bindingsFor(realistic), scaleBits), ieee754Bits(REALISTIC_SCALE));
}

/// The two tokens kernelMatches() compares against a candidate's baked metadata. The
/// causal flag is DERIVED -- hipDNN has no such boolean -- so it is a value this pack
/// invents. The sliding window is zero for both graphs here, which carry no window, so a
/// variant built with a window cannot match them. Both mask shapes are read because a
/// flag stuck at one value matches half the catalog wrongly.
TEST_P(TestAttentionDenseDispatch, BindsTheDerivedMaskAndAZeroSlidingWindow)
{
    const std::string causalToken = tokenName(arch(), CAUSAL_TOKEN);
    const std::string windowToken = tokenName(arch(), SLIDING_WINDOW_TOKEN);

    const Graph causal(arch(), servedGraph());
    const auto causalBound = bindingsFor(causal);
    EXPECT_EQ(tokenValue(causalBound, causalToken), 1);
    EXPECT_EQ(tokenValue(causalBound, windowToken), 0);

    auto unmaskedSpec = servedGraph();
    unmaskedSpec.rightBound.reset();
    const Graph unmasked(arch(), unmaskedSpec);
    const auto unmaskedBound = bindingsFor(unmasked);
    EXPECT_EQ(tokenValue(unmaskedBound, causalToken), 0);
    EXPECT_EQ(tokenValue(unmaskedBound, windowToken), 0);
}

// =============================================================================
// What prepare() does with them
// =============================================================================

/// The positive control the refusals below depend on. A prepare() that threw for every
/// input would satisfy all of them and dispatch nothing.
///
/// A failure is still expected -- the archive is absent -- but it must be the loader's,
/// which means the bindings were read and the output tensor was accepted first.
TEST_P(TestAttentionDenseDispatch, ReachesTheLoaderOnceTheBindingsAndOutputAreGood)
{
    const Graph graph(arch(), servedGraph());
    const std::string failure = prepareFailure(graph, bindingsFor(graph));

    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    EXPECT_EQ(failure.find(MISSING_TOKEN_MARKER), std::string::npos) << failure;
    EXPECT_EQ(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;
    EXPECT_NE(failure.find("does not exist"), std::string::npos) << failure;
}

/// Empty BoundTokens is what a mismatched catalog entry would hand prepare(). Without the
/// read guard, the bindings are default-constructed zeros: four uids of zero that resolve
/// to whatever buffer the caller happened to label zero, and a scale of positive zero.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const Graph graph(arch(), servedGraph());
    const auto* handler = dispatchHandler(arch());
    ASSERT_NE(handler, nullptr);

    EXPECT_THROW(handler->prepare(graph.context(), BoundTokens{}, makeKernel(arch())),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// Each token removed on its own, which is the shape a rename takes: the matcher writes a
/// new spelling, the dispatch still reads the old one, and every other token is present.
///
/// The failure is matched on its text rather than only on its type, because the archive is
/// absent too -- a dispatch that stopped reading a token would still throw here, just from
/// the loader instead. Naming the token is what distinguishes the two.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareWhenAnyOneBoundTokenIsMissing)
{
    const Graph graph(arch(), servedGraph());
    const auto bound = bindingsFor(graph);
    const auto tokens = allTokens(arch());
    ASSERT_EQ(bound.size(), tokens.size())
        << "the match binds a token this case does not know about, or has stopped binding one";

    for(const std::string& token : tokens)
    {
        SCOPED_TRACE(token);

        BoundTokens tampered = bound;
        tampered.erase(token);

        const std::string failure = prepareFailure(graph, tampered);
        ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
        EXPECT_NE(failure.find(MISSING_TOKEN_MARKER), std::string::npos) << failure;
        EXPECT_NE(failure.find(token), std::string::npos) << failure;
    }
}

/// The O token naming no tensor at all. prepare() looks the output up to re-check its
/// layout, and a null result has to be refused by name: the checks that follow index the
/// tensor's dims, so a dropped null test is a dereference of nothing.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareWhenTheOutputTokenNamesNoTensor)
{
    const Graph graph(arch(), servedGraph());
    auto tampered = bindingsFor(graph);
    tampered[tokenName(arch(), O_TOKEN)] = ABSENT_UID;

    const std::string failure = prepareFailure(graph, tampered);
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;
}

/// The O token naming a real tensor that is head-major rather than token-major. The
/// kernel bakes BSHD for the epilogue and takes no stride arguments, so a BHSD output is
/// written in bounds at the wrong addresses -- every element lands somewhere inside the
/// buffer, and the result is a transposed tensor reported as a success.
///
/// graph_match is the gate for this and declines such a graph outright, so this case
/// drives the defence-in-depth check in prepare() directly, which nothing else reaches.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareWhenTheOutputTokenNamesANonBshdTensor)
{
    const Graph graph(arch(), withUnnamedTensors(arch(), servedGraph()));
    auto tampered = bindingsFor(graph);
    tampered[tokenName(arch(), O_TOKEN)] = BHSD_UID;

    const std::string failure = prepareFailure(graph, tampered);
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;
}

/// Every tile a D128 catalog candidate can carry reaches the loader: the tile check is not
/// what stops a legal alternative.
TEST_P(TestAttentionDenseDispatch, ReachesTheLoaderForEveryBuildableTile)
{
    const Graph graph(arch(), servedGraph());
    const auto bound = bindingsFor(graph);
    for(const auto& [blockM, blockN] : GetParam()->buildableTiles)
    {
        SCOPED_TRACE(std::to_string(blockM) + "/" + std::to_string(blockN));
        const std::string failure
            = prepareFailure(graph, bound, makeKernel(arch(), blockM, blockN));
        ASSERT_FALSE(failure.empty()) << "prepare() cannot succeed without the archive";
        EXPECT_EQ(failure.find(TILE_MARKER), std::string::npos) << failure;
    }
}

/// A candidate whose tile is missing, mistyped, zero, negative, or breaks one of the
/// arch's tile rules. The grid and CTA come from block_m, so prepare() refuses by name
/// before the loader -- a zero would otherwise be a division by zero, and any substitute
/// tile launches this binary with another binary's geometry.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareACandidateWithoutASupportedTile)
{
    const Graph graph(arch(), servedGraph());
    const auto bound = bindingsFor(graph);

    std::vector<KernelDefinition> malformed;
    for(const auto& [blockM, blockN] : GetParam()->unsupportedTiles)
    {
        malformed.push_back(makeKernel(arch(), blockM, blockN));
    }
    for(const char* field : {"block_m", "block_n"})
    {
        auto kernel = makeKernel(arch());
        kernel.metadata.erase(field);
        malformed.push_back(kernel);
    }
    auto mistyped = makeKernel(arch());
    mistyped.metadata[std::string("block_m")] = std::string("256");
    malformed.push_back(mistyped);

    for(std::size_t i = 0; i < malformed.size(); ++i)
    {
        SCOPED_TRACE("malformed candidate " + std::to_string(i));
        const std::string failure = prepareFailure(graph, bound, malformed.at(i));
        ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
        EXPECT_NE(failure.find(TILE_MARKER), std::string::npos) << failure;
    }
}

/// This pack supplies nothing to compile from source with, so a candidate that would be
/// compiled from source is refused before anything is built, and the refusal names its kind.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareACandidateThatIsNotKpack)
{
    const Graph graph(arch(), servedGraph());
    auto kernel = makeKernel(arch());
    kernel.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    kernel.source.sourceFile = "kernels/AttentionDense.cpp";
    kernel.source.entryPoint = "attention_dense";

    const std::string failure = prepareFailure(graph, bindingsFor(graph), kernel);
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find("'embedded_source'"), std::string::npos) << failure;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseDispatch,
                         ::testing::Values(&GFX942_DISPATCH, &GFX950_DISPATCH),
                         [](const ::testing::TestParamInfo<const DenseDispatchCase*>& info) {
                             return std::string(info.param->arch->label);
                         });

// =============================================================================
// gfx942: the stats output and the emit_lse form
// =============================================================================

/// The stats output the launch writes the LSE to, or NO_STATS: kernelMatches compares it
/// against the candidate's emit_lse, and launch() resolves it with findDeviceBuffer as the
/// trailing lse_ptr. NO_STATS is distinct from NOT_BOUND, so a dropped token cannot read
/// as a graph without stats.
TEST(TestGfx942AttentionDenseDispatch, BindsTheStatsOutputOrNoStats)
{
    const std::string statsToken = tokenName(GFX942_ATTENTION_DENSE, STATS_TOKEN);
    const Graph withoutStats(GFX942_ATTENTION_DENSE, servedGraph());
    EXPECT_EQ(tokenValue(bindingsFor(withoutStats), statsToken), NO_STATS);

    const Graph bindsStats(GFX942_ATTENTION_DENSE, withStats(servedGraph(), STATS_UID));
    EXPECT_EQ(tokenValue(bindingsFor(bindsStats), statsToken), STATS_UID);
}

/// A graph that binds stats prepares the 11-argument form: an emit_lse candidate's
/// recorded signature passes the comparison and only the absent archive stops it, while
/// the 10-argument form is refused for it, and the 11-argument form for a graph without
/// stats.
TEST(TestGfx942AttentionDenseDispatch, PreparesTheLseFormExactlyForAGraphThatBindsStats)
{
    const auto& arch = GFX942_ATTENTION_DENSE;
    const Graph bindsStats(arch, withStats(servedGraph(), STATS_UID));
    const auto statsBound = bindingsFor(bindsStats);

    const std::string lseFailure
        = prepareFailure(bindsStats, statsBound, makeKernel(arch, 256, 64, /*writesLse=*/true));
    ASSERT_FALSE(lseFailure.empty()) << "prepare() cannot succeed without the archive";
    EXPECT_NE(lseFailure.find("does not exist"), std::string::npos) << lseFailure;

    const std::string plainFailure = prepareFailure(bindsStats, statsBound, makeKernel(arch));
    EXPECT_NE(plainFailure.find(SIGNATURE_MARKER), std::string::npos) << plainFailure;

    const Graph withoutStats(arch, servedGraph());
    const std::string extraFailure = prepareFailure(
        withoutStats, bindingsFor(withoutStats), makeKernel(arch, 256, 64, /*writesLse=*/true));
    EXPECT_NE(extraFailure.find(SIGNATURE_MARKER), std::string::npos) << extraFailure;
}

/// graph_match binds only a stats tensor the kernel can write, so prepare() re-checks the
/// one the token names before it builds anything: a uid with no tensor, a token-major
/// FLOAT tensor, and a tensor that is not FLOAT (Q) are each refused by name.
TEST(TestGfx942AttentionDenseDispatch, RefusesToPrepareWhenTheStatsTokenNamesATensorItCannotWrite)
{
    const auto& arch = GFX942_ATTENTION_DENSE;
    const Graph graph(arch, withUnnamedTensors(arch, servedGraph()));
    for(const int64_t uid : {ABSENT_UID, TOKEN_MAJOR_STATS_UID, Q_UID})
    {
        SCOPED_TRACE(uid);
        auto tampered = bindingsFor(graph);
        tampered[tokenName(arch, STATS_TOKEN)] = uid;

        const std::string failure
            = prepareFailure(graph, tampered, makeKernel(arch, 256, 64, /*writesLse=*/true));
        ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
        EXPECT_NE(failure.find(STATS_MARKER), std::string::npos) << failure;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
