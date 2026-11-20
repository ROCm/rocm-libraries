// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
using attention_dense_graphs::BIAS_UID;
using attention_dense_graphs::dispatchHandler;
using attention_dense_graphs::Graph;
using attention_dense_graphs::GraphSpec;
using attention_dense_graphs::ieee754Bits;
using attention_dense_graphs::kpackKernel;
using attention_dense_graphs::prepareFailure;
using attention_dense_graphs::pythonAbi;
using attention_dense_graphs::pythonBiasAbi;
using attention_dense_graphs::pythonLseAbi;
using attention_dense_graphs::pythonLseBiasAbi;
using attention_dense_graphs::TensorSpec;
using attention_dense_graphs::withBias;
using attention_dense_graphs::withStats;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
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
/// An arch that serves a bias: the flag every graph binds (1 with a bias, else 0), and the
/// four a graph with a bias binds besides.
constexpr std::string_view BIAS_TOKEN = ".bias";
constexpr std::string_view BIAS_UID_TOKEN = ".bias.uid";
constexpr std::string_view BIAS_STRIDE_BATCH_TOKEN = ".bias.stride_batch";
constexpr std::string_view BIAS_STRIDE_HEAD_TOKEN = ".bias.stride_head";
constexpr std::string_view BIAS_STRIDE_QUERY_TOKEN = ".bias.stride_query";

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
    if(arch.servesBias)
    {
        tokens.push_back(tokenName(arch, BIAS_TOKEN));
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

/// @p kernel as a body that does not read the strides: gfx942's V-prefetch pipeline, gfx950's
/// persistent gqa_pair body (fields its KMD declares only once such a kernel ships).
void makePackedOnlyBody(const AttentionDenseTestArch& arch, KernelDefinition& kernel)
{
    if(arch.completesGridOrder)
    {
        kernel.metadata[std::string("v_reg_prefetch")] = int64_t{1};
    }
    else
    {
        kernel.metadata[std::string("persistent")] = int64_t{1};
        kernel.metadata[std::string("persist_decode")] = std::string("gqa_pair");
    }
}

std::string prepareFailure(const Graph& graph, const BoundTokens& bound)
{
    return prepareFailure(graph, bound, makeKernel(graph.arch()));
}

/// The fragment only the missing-token refusal carries.
constexpr const char* MISSING_TOKEN_MARKER = "missing bound token";

/// The fragment only the packed-only refusal carries: the operands are not dense BSHD and the
/// kernel reads packed storage only.
constexpr const char* OUTPUT_LAYOUT_MARKER = "not dense BSHD";

/// The fragment only the refusal of a missing or ill-formed operand carries.
constexpr const char* OPERAND_MARKER = "is missing or is not a rank-4 tensor";

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

/// A token naming no tensor at all, for each operand in turn. prepare() looks all four up to
/// re-check their layouts, and a null result has to be refused by name: the checks that
/// follow index the tensor's dims, so a dropped null test is a dereference of nothing.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareWhenAnOperandTokenNamesNoTensor)
{
    const Graph graph(arch(), servedGraph());
    for(const auto& [token, role] : {std::pair{Q_TOKEN, "query"},
                                     std::pair{K_TOKEN, "key"},
                                     std::pair{V_TOKEN, "value"},
                                     std::pair{O_TOKEN, "output"}})
    {
        SCOPED_TRACE(role);
        auto tampered = bindingsFor(graph);
        tampered[tokenName(arch(), token)] = ABSENT_UID;

        const std::string failure = prepareFailure(graph, tampered);
        ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
        EXPECT_NE(failure.find(std::string("the ") + role + " tensor " + OPERAND_MARKER),
                  std::string::npos)
            << failure;
    }
}

/// The O token naming a real tensor that is head-major rather than token-major. Where the
/// kernels address packed storage only, a head-major output is written in bounds at the wrong
/// addresses -- every element lands somewhere inside the buffer, and the result is a
/// transposed tensor reported as a success -- so prepare() refuses it. Where the kernels read
/// the strides it is a layout they serve: the baseline body gets past the layout check, and
/// only a body that ignores the strides (the V-prefetch pipeline) refuses it.
///
/// graph_match is the gate for this and declines such a graph outright where it is not
/// served, so this case drives the defence-in-depth check in prepare() directly, which nothing
/// else reaches.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareWhenTheOutputTokenNamesANonBshdTensor)
{
    const Graph graph(arch(), withUnnamedTensors(arch(), servedGraph()));
    auto tampered = bindingsFor(graph);
    tampered[tokenName(arch(), O_TOKEN)] = BHSD_UID;

    auto packedOnly = makeKernel(arch());
    const std::string baseline = prepareFailure(graph, tampered);
    ASSERT_FALSE(baseline.empty()) << "prepare() cannot succeed without the archive";
    EXPECT_EQ(baseline.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << baseline;
    makePackedOnlyBody(arch(), packedOnly);
    const std::string failure = prepareFailure(graph, tampered, packedOnly);
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;
}

/// prepare() holds all four operands to the stride predicate, not the output alone: the graph
/// is bound from a served graph, then prepared over a graph whose one operand breaks a launch
/// rule the match would have declined. Each refusal names the predicate's reason.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareWhenAnyOperandBreaksALaunchRule)
{
    const Graph good(arch(), servedGraph());
    const auto bound = bindingsFor(good);

    struct Defect
    {
        const char* reason;
        std::function<void(GraphSpec&, int64_t)> apply;
    };
    const std::vector<Defect> defects{
        {"pointer_alignment",
         [](GraphSpec& spec, int64_t uid) { spec.operand(uid).alignment = 8; }},
        {"stride_alignment",
         [](GraphSpec& spec, int64_t uid) {
             auto strides = spec.stridesOf(uid);
             strides[2] += 4;
             spec.operand(uid).strides = strides;
         }},
        {"overlap",
         [](GraphSpec& spec, int64_t uid) {
             auto strides = spec.stridesOf(uid);
             strides[1] = HEAD_SIZE / 2;
             spec.operand(uid).strides = strides;
         }},
        {"head_dim_stride",
         [](GraphSpec& spec, int64_t uid) {
             auto strides = spec.stridesOf(uid);
             strides[3] = 2;
             spec.operand(uid).strides = strides;
         }},
        {"size_limit", [](GraphSpec& spec, int64_t uid) {
             auto strides = spec.stridesOf(uid);
             strides[0] = int64_t{1} << 31;
             spec.operand(uid).strides = strides;
         }}};
    const std::array<std::pair<int64_t, const char*>, 4> operands{
        {{Q_UID, "q"}, {K_UID, "k"}, {V_UID, "v"}, {O_UID, "o"}}};
    for(const auto& [uid, role] : operands)
    {
        for(const auto& defect : defects)
        {
            SCOPED_TRACE("uid " + std::to_string(uid) + " " + defect.reason);
            auto spec = servedGraph();
            defect.apply(spec, uid);
            const Graph bad(arch(), spec);
            const std::string failure = prepareFailure(bad, bound);
            ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
            // The refusal names the operand it found broken: "(<reason>): <role> ...".
            EXPECT_NE(failure.find(std::string("(") + defect.reason + "): " + role + " "),
                      std::string::npos)
                << failure;
        }
    }
}

/// The K/V slab rule runs in prepare() with the candidate's own block_n, so a tile whose
/// reach passes 2^31 bytes is refused even when a smaller tile of the same graph is served.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareATileWhoseKeyValueReachPassesTheLimit)
{
    GraphSpec spec = servedGraph();
    spec.batch = 1;
    spec.numQueryHeads = 1;
    spec.numKvHeads = 1;
    // 2^23 - 128 rows of D128: a reach of exactly 2^31 bytes at a 64-row tile.
    spec.seqLenKv = (int64_t{1} << 23) - 128;
    const Graph graph(arch(), spec);
    const auto bound = bindingsFor(graph);

    EXPECT_EQ(prepareFailure(graph, bound, makeKernel(arch(), 256, 64)).find("size_limit"),
              std::string::npos);
    const std::string failure = prepareFailure(graph, bound, makeKernel(arch(), 256, 128));
    EXPECT_NE(failure.find("(size_limit)"), std::string::npos) << failure;
}

/// The whole-tensor preflight runs in prepare() too: operands whose slabs and strides the
/// predicate admits, but whose tensors hold 2^31 elements or more, are refused. 8192 batches
/// of the default 512 x 4 x 128 are 2^31 elements of Q and O and 2^32 bytes of K and V.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareOperandsPastTheWholeTensorPreflight)
{
    const Graph served(arch(), servedGraph());
    const auto bound = bindingsFor(served);

    auto spec = servedGraph();
    spec.batch = int64_t{1} << 13;
    const Graph large(arch(), spec);
    const std::string failure = prepareFailure(large, bound);
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find("past the kernel's 32-bit offset limits"), std::string::npos) << failure;
}

// =============================================================================
// The launch state prepare() builds: the canonical strides in the binding
// =============================================================================

/// The strides the kernel is launched with, for a packed graph: rocKE's `packed_launch_strides`
/// of the same spec. The graph is B2, 4 heads, 512 tokens of 128: batch 512 * 4 * 128, head
/// 128, token 4 * 128, the same for Q, K, V and O.
TEST_P(TestAttentionDenseDispatch, BindsThePackedStridesOfAPackedGraph)
{
    const Graph graph(arch(), servedGraph());
    const auto launch
        = attention_dense_graphs::resolveLaunchOf(graph, bindingsFor(graph), makeKernel(arch()));

    const AttentionDenseStrides expected{
        262144, 128, 512, 262144, 128, 512, 262144, 128, 512, 262144, 128, 512};
    EXPECT_EQ(launch.binding.strides, expected);
}

/// An axis of extent 1 launches canonical: batch and head 0, token the head size, whatever
/// the graph declared there. One batch, one KV head (4 query heads over it).
TEST_P(TestAttentionDenseDispatch, BindsCanonicalStridesForAxesOfExtentOne)
{
    GraphSpec spec = servedGraph();
    spec.batch = 1;
    spec.numKvHeads = 1;
    // Garbage in every exempt slot: stride 0 and a negative one on the batch axes, an odd head
    // stride on K and V.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        auto strides = spec.stridesOf(uid);
        strides[0] = uid == K_UID ? int64_t{-5} : int64_t{0};
        if(uid == K_UID || uid == V_UID)
        {
            strides[1] = 3;
        }
        spec.operand(uid).strides = strides;
    }
    const Graph graph(arch(), spec);
    auto kernel = makeKernel(arch());
    kernel.metadata[std::string("num_query_heads")] = spec.numQueryHeads;
    kernel.metadata[std::string("num_kv_heads")] = spec.numKvHeads;
    const auto launch = attention_dense_graphs::resolveLaunchOf(graph, bindingsFor(graph), kernel);

    // Q and O: batch 0, head 128, token 4 * 128. K and V (one head): batch 0, head 0, token 128.
    const AttentionDenseStrides expected{0, 128, 512, 0, 0, 128, 0, 0, 128, 0, 128, 512};
    EXPECT_EQ(launch.binding.strides, expected);
}

/// Layouts other than packed BSHD, which the grid bodies of both arches read: the binding
/// carries each operand's own strides.
TEST_P(TestAttentionDenseDispatch, BindsTheStridesOfEveryAdmittedLayoutWhereKernelsReadThem)
{
    struct Case
    {
        const char* name;
        GraphSpec spec;
        AttentionDenseStrides expected;
    };
    std::vector<Case> cases;

    // Head-major: batch H * S * D, head S * D, token D (B2, 4 heads, 512 tokens of 128).
    auto bhsd = servedGraph();
    bhsd.setEveryLayout(SdpaLayout::BHSD);
    cases.push_back(
        {"head-major",
         bhsd,
         {262144, 65536, 128, 262144, 65536, 128, 262144, 65536, 128, 262144, 65536, 128}});

    // Q, K and V cut out of one [B, S, 3 * 4 heads, D] buffer; O packed on its own.
    auto qkv = servedGraph();
    for(const int64_t uid : {Q_UID, K_UID, V_UID})
    {
        qkv.operand(uid).strides = std::vector<int64_t>{
            SEQ * 3 * HEADS * HEAD_SIZE, HEAD_SIZE, 3 * HEADS * HEAD_SIZE, 1};
    }
    cases.push_back({"packed QKV",
                     qkv,
                     {786432, 128, 1536, 786432, 128, 1536, 786432, 128, 1536, 262144, 128, 512}});

    // Eight query heads over two KV heads, head-major: each operand has its own head count.
    auto grouped = servedGraph();
    grouped.numQueryHeads = 8;
    grouped.numKvHeads = 2;
    grouped.setEveryLayout(SdpaLayout::BHSD);
    cases.push_back(
        {"grouped head-major",
         grouped,
         {524288, 65536, 128, 131072, 65536, 128, 131072, 65536, 128, 524288, 65536, 128}});

    // Rows padded by 8 elements: token 4 * 128 + 8, batch 512 rows of that.
    auto padded = servedGraph();
    padded.setEveryLayout(SdpaLayout::PADDED_BSHD);
    cases.push_back({"padded rows",
                     padded,
                     {266240, 128, 520, 266240, 128, 520, 266240, 128, 520, 266240, 128, 520}});

    // Every operand its own layout, so no two groups of three in the binding are equal and
    // each can only come from its own operand: any operand mix-up in the launch strides fails
    // here (the cases above all have K = V, so this is the one that catches a K/V swap). Q
    // packed, K head-major, V rows padded by 8 (token 520), O rows padded by 16 (token 528).
    auto distinct = servedGraph();
    distinct.operand(K_UID).layout = SdpaLayout::BHSD;
    distinct.operand(V_UID).layout = SdpaLayout::PADDED_BSHD;
    // A pad of 16, not PADDED_BSHD's 8, so O's group differs from V's.
    constexpr int64_t O_ROW = HEADS * HEAD_SIZE + 16;
    distinct.operand(O_UID).strides = std::vector<int64_t>{SEQ * O_ROW, HEAD_SIZE, O_ROW, 1};
    cases.push_back({"every operand its own layout",
                     distinct,
                     {262144, 128, 512, 262144, 65536, 128, 266240, 128, 520, 270336, 128, 528}});

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        const Graph graph(arch(), testCase.spec);
        const auto launch = attention_dense_graphs::resolveLaunchOf(
            graph, bindingsFor(graph), makeKernel(arch()));
        EXPECT_EQ(launch.binding.strides, testCase.expected);
    }
}

/// A body that does not read the strides cannot be handed a strided graph: prepare() refuses,
/// by name, where the kernel match would not have matched.
TEST_P(TestAttentionDenseDispatch, RefusesToPrepareAStridedGraphForABodyThatIgnoresTheStrides)
{
    auto spec = servedGraph();
    spec.setEveryLayout(SdpaLayout::BHSD);
    const Graph strided(arch(), spec);
    const auto bound = bindingsFor(strided);

    auto kernel = makeKernel(arch());
    makePackedOnlyBody(arch(), kernel);
    const std::string failure = prepareFailure(strided, bound, kernel);
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find("reads packed operands only and does not read the stride arguments"),
              std::string::npos)
        << failure;
    EXPECT_NE(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;

    // The same body over a packed graph gets past the layout check.
    const Graph packed(arch(), servedGraph());
    const std::string packedFailure = prepareFailure(packed, bindingsFor(packed), kernel);
    ASSERT_FALSE(packedFailure.empty());
    EXPECT_EQ(packedFailure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << packedFailure;
}

/// What prepare() saves with the plan: the candidate's own block_n and whether its body reads the
/// strides, through the values the engine's strided contract stores. The grid bodies read them;
/// gfx942's V-prefetch pipeline and gfx950's persistent gqa_pair body do not.
TEST_P(TestAttentionDenseDispatch, SavesTheKernelsBlockNAndWhetherItReadsTheStrides)
{
    const Graph graph(arch(), servedGraph());
    const auto bound = bindingsFor(graph);
    const auto& engineArch = arch().label == GFX942_ATTENTION_DENSE.label
                                 ? gfx942AttentionDenseArch()
                                 : gfx950AttentionDenseArch();
    const auto savedValues = [&](const KernelDefinition& kernel) {
        const auto launch = attention_dense_graphs::resolveLaunchOf(graph, bound, kernel);
        return attention_dense::launchValues(launch.binding,
                                             launch.problem,
                                             launch.blockM,
                                             engineArch,
                                             engineArch.currentContract,
                                             launch.grid);
    };

    // The tile of the candidate, whatever it is: block_n is not the baseline's 64 here.
    for(const int64_t blockN : {int64_t{32}, int64_t{64}})
    {
        SCOPED_TRACE(blockN);
        const auto values = savedValues(makeKernel(arch(), 256, blockN));
        EXPECT_EQ(std::get<int64_t>(values.at("block_n")), blockN);
        EXPECT_EQ(std::get<int64_t>(values.at("block_m")), 256);
        EXPECT_EQ(std::get<int64_t>(values.at("reads_strides")), 1);
    }
    // A body that ignores the strides saves 0: gfx942's V-prefetch pipeline, gfx950's persistent
    // gqa_pair body.
    auto packedOnly = makeKernel(arch());
    makePackedOnlyBody(arch(), packedOnly);
    EXPECT_EQ(std::get<int64_t>(savedValues(packedOnly).at("reads_strides")), 0);
    if(arch().completesGridOrder)
    {
        auto kdb = makeKernel(arch());
        kdb.metadata[std::string("lds_num_buffers")] = int64_t{2};
        EXPECT_EQ(std::get<int64_t>(savedValues(kdb).at("reads_strides")), 0);
    }
}

/// A gfx950 persistent body that decodes through the shape parameters reads the strides: prepare()
/// admits a strided graph for it, with the strides in the binding and the flag saved as 1.
TEST(TestAttentionDenseDispatchGfx950, AStridedGraphIsPlannedForAPersistentBodyThatReadsTheShape)
{
    auto spec = servedGraph();
    spec.setEveryLayout(SdpaLayout::BHSD);
    const Graph graph(GFX950_ATTENTION_DENSE, spec);
    const auto bound = bindingsFor(graph);
    for(const char* decode : {"qb_major", "hkv_major", "bt_hkv_minor", "hq_minor_swz", "auto"})
    {
        SCOPED_TRACE(decode);
        auto kernel = makeKernel(GFX950_ATTENTION_DENSE);
        kernel.metadata[std::string("persistent")] = int64_t{1};
        kernel.metadata[std::string("persist_decode")] = std::string(decode);
        const auto launch = attention_dense_graphs::resolveLaunchOf(graph, bound, kernel);
        EXPECT_TRUE(launch.binding.readsStrides);
        const AttentionDenseStrides expected{
            262144, 65536, 128, 262144, 65536, 128, 262144, 65536, 128, 262144, 65536, 128};
        EXPECT_EQ(launch.binding.strides, expected);
    }
    // The gqa_pair orders keep packed addressing: the same graph is refused by name.
    for(const char* decode : {"gqa_pair", "gqa_pair_2phase"})
    {
        SCOPED_TRACE(decode);
        auto kernel = makeKernel(GFX950_ATTENTION_DENSE);
        kernel.metadata[std::string("persistent")] = int64_t{1};
        kernel.metadata[std::string("persist_decode")] = std::string(decode);
        const std::string failure = prepareFailure(graph, bound, kernel);
        EXPECT_NE(failure.find("reads packed operands only and does not read the stride arguments"),
                  std::string::npos)
            << failure;
    }
}

/// What a launch passes after the shape arguments is the binding's strides, converted as the
/// kernel takes them: the twelve i32 values, in kernarg order. Checked for a graph with
/// extent-1 axes (canonical values) and a head-major one.
TEST_P(TestAttentionDenseDispatch, LaunchesTheStridesTheBindingHolds)
{
    GraphSpec canonical = servedGraph();
    canonical.batch = 1;
    canonical.numKvHeads = 1;
    GraphSpec headMajor = servedGraph();
    headMajor.setEveryLayout(SdpaLayout::BHSD);

    for(const auto& spec : {canonical, headMajor})
    {
        const Graph graph(arch(), spec);
        auto kernel = makeKernel(arch());
        kernel.metadata[std::string("num_query_heads")] = spec.numQueryHeads;
        kernel.metadata[std::string("num_kv_heads")] = spec.numKvHeads;
        const auto launch
            = attention_dense_graphs::resolveLaunchOf(graph, bindingsFor(graph), kernel);

        const std::array<void*, 4> operands{nullptr, nullptr, nullptr, nullptr};
        const attention_dense::AttentionDenseKernelArguments arguments(
            launch, attention_dense::AttentionDenseAbi::RUNTIME_STRIDES, {}, operands, nullptr);
        // 4 pointers, scale, 5 shape values, then the strides last.
        ASSERT_EQ(arguments.params().size(), 4U + 1U + 5U + 12U);
        for(std::size_t index = 0; index < 12; ++index)
        {
            const auto* passed = static_cast<const int32_t*>(arguments.params().at(10 + index));
            EXPECT_EQ(static_cast<int64_t>(*passed), launch.binding.strides.at(index)) << index;
        }
        // Not just consistent: for the head-major graph these are the head-major values (Q's
        // head stride is S * D).
        if(spec.operand(Q_UID).layout == SdpaLayout::BHSD)
        {
            EXPECT_EQ(*static_cast<const int32_t*>(arguments.params().at(10 + 1)), 65536);
        }
    }
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

// =============================================================================
// The additive bias
// =============================================================================

namespace
{

/// The fragment only the graph/candidate bias disagreement carries.
constexpr const char* BIAS_MARKER = "disagree on the additive bias";

/// The fragment only the out-of-range bias stride refusal carries.
constexpr const char* BIAS_STRIDE_RANGE_MARKER = "is outside [0, 2^31)";

/// The fragment only the bias reach refusal carries.
constexpr const char* BIAS_REACH_MARKER = "reach element 2^31";

/// The four further tokens a graph with a bias binds.
std::vector<std::string> biasTokens(const AttentionDenseTestArch& arch)
{
    std::vector<std::string> tokens;
    for(const std::string_view suffix :
        {BIAS_UID_TOKEN, BIAS_STRIDE_BATCH_TOKEN, BIAS_STRIDE_HEAD_TOKEN, BIAS_STRIDE_QUERY_TOKEN})
    {
        tokens.push_back(tokenName(arch, suffix));
    }
    return tokens;
}

/// servedGraph with an fp32 [B, 1, Sq, Skv] attn_mask: the head axis broadcasts.
GraphSpec servedBiasGraph()
{
    auto spec = servedGraph();
    const auto bias = withBias({BATCH, 1, SEQ, SEQ});
    spec.unnamedTensors = bias.unnamedTensors;
    spec.attnMaskUid = bias.attnMaskUid;
    return spec;
}

/// A descriptor of @p arch's engine that records the ABI a bias (or plain) kernel declares,
/// with the LSE pointer before the bias tail when @p writesLse, and carries the
/// use_additive_bias field of a completed record.
KernelDefinition makeBiasKernel(const AttentionDenseTestArch& arch,
                                bool bias,
                                bool writesLse = false,
                                int64_t blockM = 256,
                                int64_t blockN = 64)
{
    auto abi = writesLse ? pythonLseAbi() : pythonAbi();
    if(bias)
    {
        abi = writesLse ? pythonLseBiasAbi() : pythonBiasAbi();
    }
    auto kernel = kpackKernel(arch, abi, blockM, blockN);
    kernel.metadata[std::string("use_additive_bias")] = int64_t{bias ? 1 : 0};
    return kernel;
}

/// The arches that serve an additive bias.
class TestAttentionDenseBiasDispatch
    : public ::testing::TestWithParam<const AttentionDenseTestArch*>
{
protected:
    static const AttentionDenseTestArch& arch()
    {
        return *GetParam();
    }
};

} // namespace

/// The bias uid launch() resolves the buffer by, and the strides it passes: the batch and
/// query-row pitches as the tensor carries them, and zero on the broadcast head axis
/// whatever stride the tensor writes there. A plain graph binds the flag at zero and
/// nothing else.
TEST_P(TestAttentionDenseBiasDispatch, BindsTheBiasUidAndItsKernelStrides)
{
    const Graph biased(arch(), servedBiasGraph());
    const auto bound = bindingsFor(biased);
    EXPECT_EQ(tokenValue(bound, tokenName(arch(), BIAS_TOKEN)), 1);
    EXPECT_EQ(tokenValue(bound, tokenName(arch(), BIAS_UID_TOKEN)), BIAS_UID);
    EXPECT_EQ(tokenValue(bound, tokenName(arch(), BIAS_STRIDE_BATCH_TOKEN)), SEQ * SEQ);
    EXPECT_EQ(tokenValue(bound, tokenName(arch(), BIAS_STRIDE_HEAD_TOKEN)), 0);
    EXPECT_EQ(tokenValue(bound, tokenName(arch(), BIAS_STRIDE_QUERY_TOKEN)), SEQ);

    const Graph plain(arch(), servedGraph());
    const auto plainBound = bindingsFor(plain);
    EXPECT_EQ(tokenValue(plainBound, tokenName(arch(), BIAS_TOKEN)), 0);
    for(const auto& token : biasTokens(arch()))
    {
        EXPECT_EQ(tokenValue(plainBound, token), NOT_BOUND) << token;
    }
}

/// A bias graph on a bias candidate reaches the loader: the binding is read in full and
/// the fourteen-argument signature is the one expected.
TEST_P(TestAttentionDenseBiasDispatch, ReachesTheLoaderForABiasGraphOnABiasCandidate)
{
    const Graph graph(arch(), servedBiasGraph());
    const std::string failure
        = prepareFailure(graph, bindingsFor(graph), makeBiasKernel(arch(), /*bias=*/true));
    ASSERT_FALSE(failure.empty()) << "prepare() cannot succeed without the archive";
    EXPECT_EQ(failure.find(BIAS_MARKER), std::string::npos) << failure;
    EXPECT_EQ(failure.find(SIGNATURE_MARKER), std::string::npos) << failure;
    EXPECT_NE(failure.find("does not exist"), std::string::npos) << failure;
}

/// kernel_match never pairs a bias graph with a plain candidate or the reverse; prepare()
/// refuses both by name, because either would launch with the wrong argument list.
TEST_P(TestAttentionDenseBiasDispatch, RefusesToPrepareWhenGraphAndCandidateDisagreeOnBias)
{
    const Graph biased(arch(), servedBiasGraph());
    const std::string onPlain
        = prepareFailure(biased, bindingsFor(biased), makeBiasKernel(arch(), /*bias=*/false));
    EXPECT_NE(onPlain.find(BIAS_MARKER), std::string::npos) << onPlain;

    const Graph plain(arch(), servedGraph());
    const std::string onBias
        = prepareFailure(plain, bindingsFor(plain), makeBiasKernel(arch(), /*bias=*/true));
    EXPECT_NE(onBias.find(BIAS_MARKER), std::string::npos) << onBias;

    // A record written before the field reads as a plain candidate (the KMD default 0).
    auto legacy = makeBiasKernel(arch(), /*bias=*/false);
    legacy.metadata.erase(std::string("use_additive_bias"));
    const std::string onLegacy = prepareFailure(biased, bindingsFor(biased), legacy);
    EXPECT_NE(onLegacy.find(BIAS_MARKER), std::string::npos) << onLegacy;
}

/// Each bias token removed on its own from a bias graph's binding is refused by name, as
/// the operand tokens are.
TEST_P(TestAttentionDenseBiasDispatch, RefusesToPrepareWhenAnyOneBiasTokenIsMissing)
{
    const Graph graph(arch(), servedBiasGraph());
    const auto bound = bindingsFor(graph);
    ASSERT_EQ(bound.size(), allTokens(arch()).size() + biasTokens(arch()).size());

    for(const auto& token : biasTokens(arch()))
    {
        SCOPED_TRACE(token);
        BoundTokens tampered = bound;
        tampered.erase(token);

        const std::string failure
            = prepareFailure(graph, tampered, makeBiasKernel(arch(), /*bias=*/true));
        EXPECT_NE(failure.find(MISSING_TOKEN_MARKER), std::string::npos) << failure;
        EXPECT_NE(failure.find(token), std::string::npos) << failure;
    }
}

/// graph_match binds every bias stride in [0, 2^31) and bounds the reach they give over the
/// graph's extents. prepare() checks both again, so a binding it did not produce cannot wrap
/// an i32 kernarg: each stride token at -1 or 2^31 is refused by name, and at 0 or the
/// largest stride whose reach still fits accepted; one more is refused for its reach.
TEST_P(TestAttentionDenseBiasDispatch, RefusesToPrepareABiasStrideOutsideTheI32Range)
{
    const Graph graph(arch(), servedBiasGraph());
    const auto bound = bindingsFor(graph);
    const std::vector<std::string> strideTokens{tokenName(arch(), BIAS_STRIDE_BATCH_TOKEN),
                                                tokenName(arch(), BIAS_STRIDE_HEAD_TOKEN),
                                                tokenName(arch(), BIAS_STRIDE_QUERY_TOKEN)};

    for(const auto& token : strideTokens)
    {
        for(const int64_t value : {int64_t{-1}, int64_t{1} << 31})
        {
            SCOPED_TRACE(token + " = " + std::to_string(value));
            BoundTokens tampered = bound;
            ASSERT_EQ(tampered.count(token), 1U);
            tampered[token] = value;

            const std::string failure
                = prepareFailure(graph, tampered, makeBiasKernel(arch(), /*bias=*/true));
            EXPECT_NE(failure.find(BIAS_STRIDE_RANGE_MARKER), std::string::npos) << failure;
            EXPECT_NE(failure.find(token), std::string::npos) << failure;
        }
        // The boundary values themselves are accepted: the binding is read and
        // prepare() goes on to the loader, which fails only for want of the archive. The
        // largest stride is the one whose reach, (Skv - 1) plus (extent - 1) * stride per
        // axis, is 2^31 - 1 with the other two strides as the graph binds them.
        const auto axis = static_cast<size_t>(&token - strideTokens.data());
        const std::array<int64_t, 3> extents{BATCH, HEADS, SEQ};
        std::array<int64_t, 3> others{SEQ * SEQ, 0, SEQ};
        others[axis] = 0;
        int64_t reachWithoutAxis = SEQ - 1;
        for(size_t i = 0; i < others.size(); ++i)
        {
            reachWithoutAxis += (extents[i] - 1) * others[i];
        }
        const int64_t largest
            = (std::numeric_limits<int32_t>::max() - reachWithoutAxis) / (extents[axis] - 1);
        for(const int64_t value : {int64_t{0}, largest})
        {
            SCOPED_TRACE(token + " = " + std::to_string(value));
            BoundTokens tampered = bound;
            tampered[token] = value;

            const std::string failure
                = prepareFailure(graph, tampered, makeBiasKernel(arch(), /*bias=*/true));
            EXPECT_EQ(failure.find(BIAS_STRIDE_RANGE_MARKER), std::string::npos) << failure;
            EXPECT_EQ(failure.find(BIAS_REACH_MARKER), std::string::npos) << failure;
            EXPECT_NE(failure.find("does not exist"), std::string::npos) << failure;
        }
        SCOPED_TRACE(token + " = " + std::to_string(largest + 1));
        BoundTokens tooFar = bound;
        tooFar[token] = largest + 1;
        const std::string pastIt
            = prepareFailure(graph, tooFar, makeBiasKernel(arch(), /*bias=*/true));
        EXPECT_NE(pastIt.find(BIAS_REACH_MARKER), std::string::npos) << pastIt;
    }
}

namespace
{

using attention_dense::AttentionDenseAbi;
using attention_dense::AttentionDenseContract;

/// What requireBiasBindingMatchesContract does with a binding and a contract: SUCCESS and an
/// empty message when it accepts them, else the status and text it throws.
struct ContractCheck
{
    hipdnnPluginStatus_t status = HIPDNN_PLUGIN_STATUS_SUCCESS;
    std::string message;
};

ContractCheck checkBiasAgainstContract(const AttentionDenseTestArch& arch,
                                       const AttentionDenseBinding& binding,
                                       const AttentionDenseContract& contract)
{
    try
    {
        attention_dense::requireBiasBindingMatchesContract(
            binding, contract, arch.label, "test_kernel");
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        return {error.getStatus(), error.getMessage()};
    }
    return {};
}

AttentionDenseBinding biasBinding()
{
    AttentionDenseBinding binding;
    binding.bias = AttentionDenseBias{BIAS_UID, {7, 3, 5}};
    return binding;
}

/// The contracts that pass no bias (the arch's plain one, and the baked-head-count form a
/// legacy handler launches) and the one that does, each under the arch's own name.
struct ArchContracts
{
    std::string legacySymbol;
    AttentionDenseContract plain;
    AttentionDenseContract legacy;
    AttentionDenseContract bias;
};

ArchContracts contractsOf(const AttentionDenseTestArch& arch)
{
    ArchContracts contracts;
    contracts.legacySymbol = std::string(arch.dispatchSymbol) + ".v1";
    contracts.plain = {arch.plainContractSymbol, AttentionDenseAbi::RUNTIME_STRIDES};
    contracts.legacy = {contracts.legacySymbol, AttentionDenseAbi::BAKED_HEAD_COUNTS};
    contracts.bias = {arch.biasContractSymbol, AttentionDenseAbi::RUNTIME_STRIDES_WITH_BIAS};
    return contracts;
}

} // namespace

/// The binding-versus-contract check is a pure function, because makePreparedAttentionDense,
/// the one place that calls it, runs only after a code object has loaded and prepare() cannot
/// get that far here. A plan is consistent exactly when its binding has a bias and its
/// contract passes the bias arguments, or neither does.
TEST_P(TestAttentionDenseBiasDispatch, AcceptsABindingThatAgreesWithItsContractOnTheBias)
{
    const auto contracts = contractsOf(arch());

    const auto biased = checkBiasAgainstContract(arch(), biasBinding(), contracts.bias);
    EXPECT_EQ(biased.status, HIPDNN_PLUGIN_STATUS_SUCCESS) << biased.message;

    for(const auto* plain : {&contracts.plain, &contracts.legacy})
    {
        const auto result = checkBiasAgainstContract(arch(), AttentionDenseBinding{}, *plain);
        EXPECT_EQ(result.status, HIPDNN_PLUGIN_STATUS_SUCCESS)
            << plain->symbol << ": " << result.message;
    }
}

/// A bias binding under a contract that passes no bias arguments would launch a kernel that
/// adds nothing: a wrong result with no fault. It is refused as an internal error (the
/// graph is fine; the code that paired the two is not), naming the contract and the bias.
TEST_P(TestAttentionDenseBiasDispatch, RefusesABiasBindingUnderAContractWithoutBiasArguments)
{
    const auto contracts = contractsOf(arch());
    for(const auto* plain : {&contracts.plain, &contracts.legacy})
    {
        SCOPED_TRACE(plain->symbol);
        const auto result = checkBiasAgainstContract(arch(), biasBinding(), *plain);
        EXPECT_EQ(result.status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) << result.message;
        EXPECT_NE(result.message.find(std::string(plain->symbol)), std::string::npos)
            << result.message;
        EXPECT_NE(result.message.find("passes no bias arguments"), std::string::npos)
            << result.message;
        EXPECT_NE(result.message.find("ignore"), std::string::npos) << result.message;
        EXPECT_NE(result.message.find("uid " + std::to_string(BIAS_UID)), std::string::npos)
            << result.message;
    }
}

/// The reverse: the contract passes a bias pointer and strides, and the plan has no bias to
/// pass them from (launch() would find no buffer for the uid).
TEST_P(TestAttentionDenseBiasDispatch, RefusesABiasContractWithoutABiasBinding)
{
    const auto contracts = contractsOf(arch());
    const auto result = checkBiasAgainstContract(arch(), AttentionDenseBinding{}, contracts.bias);
    EXPECT_EQ(result.status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) << result.message;
    EXPECT_NE(result.message.find(std::string(contracts.bias.symbol)), std::string::npos)
        << result.message;
    EXPECT_NE(result.message.find("binds no additive bias"), std::string::npos) << result.message;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseBiasDispatch,
                         ::testing::ValuesIn(biasServingArches()),
                         attentionDenseArchName);

/// A gfx942 graph can carry a bias and write the LSE at once, and the two tensors are
/// different uids the launch resolves apart: the bias flag and uid and the stats uid are
/// each bound, and the launch passes lse_ptr, then the bias tail, then the twelve strides, so a
/// kernel recording the LSE form in that order is prepared and one with lse_ptr last is refused.
TEST(TestGfx942AttentionDenseDispatch, BindsAndPreparesTheBiasAlongsideTheStatsOutput)
{
    const auto& arch = GFX942_ATTENTION_DENSE;
    ASSERT_NE(BIAS_UID, STATS_UID);
    const Graph graph(arch, withStats(servedBiasGraph(), STATS_UID));
    const auto bound = bindingsFor(graph);
    EXPECT_EQ(tokenValue(bound, tokenName(arch, BIAS_TOKEN)), 1);
    EXPECT_EQ(tokenValue(bound, tokenName(arch, BIAS_UID_TOKEN)), BIAS_UID);
    EXPECT_EQ(tokenValue(bound, tokenName(arch, STATS_TOKEN)), STATS_UID);

    const std::string accepted
        = prepareFailure(graph, bound, makeBiasKernel(arch, /*bias=*/true, /*writesLse=*/true));
    ASSERT_FALSE(accepted.empty()) << "prepare() cannot succeed without the archive";
    EXPECT_EQ(accepted.find(SIGNATURE_MARKER), std::string::npos) << accepted;
    EXPECT_NE(accepted.find("does not exist"), std::string::npos) << accepted;

    auto lseLast = makeBiasKernel(arch, /*bias=*/true);
    lseLast.source.signature.push_back(KernelArgument{
        attention_dense_graphs::BUFFER_KIND, attention_dense_graphs::POINTER_BYTES, 0, "lse_ptr"});
    const std::string refused = prepareFailure(graph, bound, lseLast);
    EXPECT_NE(refused.find(SIGNATURE_MARKER), std::string::npos) << refused;

    // The plain LSE form has no bias tail, which the graph's bias needs.
    auto noTail = makeBiasKernel(arch, /*bias=*/true, /*writesLse=*/true);
    noTail.source.signature = pythonLseAbi();
    const std::string tailless = prepareFailure(graph, bound, noTail);
    EXPECT_NE(tailless.find(SIGNATURE_MARKER), std::string::npos) << tailless;
}

/// The v1 handler is registered only for plans saved before the head counts became runtime
/// arguments, and has no bias form: a bias graph prepared on it is refused by name, before
/// any code object is loaded, rather than launched with the wrong argument list.
TEST(TestGfx950AttentionDenseDispatch, RefusesToPrepareABiasPlanOnTheLegacyHandler)
{
    const auto& arch = GFX950_ATTENTION_DENSE;
    registerNativeIngestorSymbols();
    const auto* legacy = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string("hipkernel.gfx950_attention_dense.dispatch.v1"));
    ASSERT_NE(legacy, nullptr);

    const Graph graph(arch, servedBiasGraph());
    std::string failure;
    try
    {
        legacy->prepare(graph.context(), bindingsFor(graph), makeBiasKernel(arch, /*bias=*/true));
    }
    catch(const std::exception& error)
    {
        failure = error.what();
    }
    ASSERT_FALSE(failure.empty()) << "prepare() cannot succeed without the archive";
    EXPECT_NE(failure.find("has no additive bias form"), std::string::npos) << failure;
    EXPECT_EQ(failure.find("does not exist"), std::string::npos) << failure;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
