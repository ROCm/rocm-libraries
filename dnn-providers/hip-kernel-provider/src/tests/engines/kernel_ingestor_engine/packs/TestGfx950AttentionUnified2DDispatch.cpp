// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>

#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DTestGraphs.hpp"

/**
 * @file TestGfx950AttentionUnified2DDispatch.cpp
 * @brief The hipkernel:Gfx950AttentionUnified2D pack at the bind and prepare layer: what the
 *        graph match writes into the bound tokens, how much workspace it asks for, and what
 *        prepare() refuses before it reaches a loader.
 *
 * Every failure this file is after is a SILENT one. A token carrying the wrong uid launches
 * the kernel against the wrong buffers; a scale bound as an integer reads on the device as a
 * denormal near zero; a workspace too small for the identity tables lets the kernel read
 * whatever follows it. None faults and none sets a status.
 *
 * The launch itself is not reachable here. prepare() loads a code object after these checks
 * and hands the grid to a PreparedDispatch subclass in the pack's anonymous namespace, so a
 * prepared dispatch is opaque to a test. The grid and the bytes launch() copies are pure
 * functions of the same inputs prepare() passes them, pinned exactly by
 * TestGfx950AttentionUnified2DGeometry.cpp (the identity block table, seq_lens and
 * query_start_len for a batch above one with seqlen_q != seqlen_kv); what prepare() passes
 * them needs a real archive and a real gfx950.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace unified2d;
using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

/// Every token the dispatch reads, in one place, so the case that removes them one at a time
/// cannot quietly stop covering one.
const std::vector<std::string_view>& allTokens()
{
    static const std::vector<std::string_view> s_tokens{
        Q_TOKEN, K_TOKEN, V_TOKEN, O_TOKEN, MASK_TOKEN, SLIDING_WINDOW_TOKEN, SCALE_BITS_TOKEN};
    return s_tokens;
}

const IKernelDispatchHandler<Handle>* dispatchHandler()
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
}

/// The sentinel is deliberately not zero: zero is a real value for the mask and sliding-window
/// tokens, so a missing token answering zero would read as a correct binding.
constexpr int64_t NOT_BOUND = -1000;

int64_t tokenValue(const BoundTokens& bound, std::string_view token)
{
    return tryGetBoundInt(bound, token).value_or(NOT_BOUND);
}

BoundTokens bindingsFor(const Graph& graph)
{
    const auto bound = matchGraph(graph);
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the fixture graph, so there are no "
                                      "tokens to read and no bindings to hand prepare()";
    return bound.value_or(BoundTokens{});
}

int64_t ieee754Bits(float value)
{
    int32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit to round-trip");
    std::memcpy(&bits, &value, sizeof(value));
    return static_cast<int64_t>(bits);
}

/// prepare()'s refusal text for the default graph and a kpack descriptor built from @p spec.
std::string prepareFailureFor(const KernelSpec& spec, const GraphSpec& graphSpec = GraphSpec{})
{
    const Graph graph(graphSpec);
    return prepareFailure(graph, bindingsFor(graph), makeKpackKernel(spec, pythonAbi()));
}

constexpr const char* MISSING_TOKEN_MARKER = "missing bound token";
constexpr const char* OUTPUT_LAYOUT_MARKER = "not dense BSHD";
constexpr const char* GEOMETRY_MARKER = "declares no supported geometry";
constexpr const char* CAPACITY_MARKER = "serves at most";
constexpr const char* BLOCK_MARKER = "does not divide seqlen_kv";

} // namespace

// =============================================================================
// Registration
// =============================================================================

/// If registerGfx950AttentionUnified2DSymbols stops adding DISPATCH_SYMBOL, or IngestorPacks
/// drops this engine's row, every descriptor naming the symbol is dropped at load: the pack
/// disappears, then the engine, and at the default log level nothing says so.
TEST(TestGfx950AttentionUnified2DDispatch, DispatchSymbolResolves)
{
    EXPECT_NE(dispatchHandler(), nullptr);
}

// =============================================================================
// Workspace
// =============================================================================

/// 768 bytes for the smallest graphs: three 256-byte sections. Worked out by hand for the
/// default graph (B2, Skv 256, block 16): 32 table entries are 128 bytes, seq_lens and
/// query_start_len a few bytes each.
TEST(TestGfx950AttentionUnified2DDispatch, WorkspaceIsThreeAlignedSections)
{
    const Graph graph;
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(
        handler->workspaceBytes(graph.context(), bindingsFor(graph), makeKernel(KernelSpec{})),
        768U);
}

/// The table is `batch * Skv / block_size` entries, so the candidate's block size and the
/// graph's batch and Skv all move the answer. B3, Skv 1024: block 16 is 192 entries (768
/// bytes -> three sections, 1280 total); block 32 is 96 entries (384 bytes -> two sections for
/// the table, 1024 total).
TEST(TestGfx950AttentionUnified2DDispatch, WorkspaceFollowsTheBlockSizeAndTheGraph)
{
    GraphSpec spec;
    spec.batch = 3;
    spec.seqLenKv = 1024;
    const Graph graph(spec);
    const auto bound = bindingsFor(graph);
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);

    const KernelSpec block16;
    EXPECT_EQ(handler->workspaceBytes(graph.context(), bound, makeKernel(block16)), 1280U);
    KernelSpec block32;
    block32.blockSize = 32;
    EXPECT_EQ(handler->workspaceBytes(graph.context(), bound, makeKernel(block32)), 1024U);
}

TEST(TestGfx950AttentionUnified2DDispatch, WorkspaceQueryRefusesACandidateItCannotLayOut)
{
    const Graph graph;
    const auto bound = bindingsFor(graph);
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);

    KernelSpec malformed;
    malformed.numWarps = 3;
    EXPECT_THROW(handler->workspaceBytes(graph.context(), bound, makeKernel(malformed)),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    // A block that does not divide Skv: 256 % 64 == 0 is fine, 96 % 64 is not.
    GraphSpec oddSkv;
    oddSkv.seqLenQ = 32;
    oddSkv.seqLenKv = 96;
    const Graph oddGraph(oddSkv);
    KernelSpec block64;
    block64.blockSize = 64;
    block64.tileSize = 256;
    EXPECT_THROW(
        handler->workspaceBytes(oddGraph.context(), bindingsFor(oddGraph), makeKernel(block64)),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

// =============================================================================
// What the match binds
// =============================================================================

/// The four uids the launch resolves device buffers by. launch() looks each one up with
/// findDeviceBuffer and passes the results positionally, so a Q/K transposition here is a
/// kernel reading keys as queries: full-rate arithmetic, no fault, wrong numbers. The uids
/// are mutually distinct, so any permutation of the four fails this.
TEST(TestGfx950AttentionUnified2DDispatch, BindsEachOperandUidToItsOwnToken)
{
    const auto bound = bindingsFor(Graph{});

    EXPECT_EQ(tokenValue(bound, Q_TOKEN), Q_UID);
    EXPECT_EQ(tokenValue(bound, K_TOKEN), K_UID);
    EXPECT_EQ(tokenValue(bound, V_TOKEN), V_UID);
    EXPECT_EQ(tokenValue(bound, O_TOKEN), O_UID);
}

/// The softmax scale is an f32 kernarg and a BoundTokens value an int64_t, so the matcher
/// carries the float as its IEEE-754 storage. Expectations written as literals read off the
/// format: both mutations this exists for are integer-valued -- binding zero, and binding the
/// float converted to an integer, which is zero for every scale a real graph uses.
TEST(TestGfx950AttentionUnified2DDispatch, BindsTheSoftmaxScaleAsItsIeeeBitPattern)
{
    const auto scaleBits = [](float scale) {
        GraphSpec spec;
        spec.attnScaleValue = scale;
        return tokenValue(bindingsFor(Graph(spec)), SCALE_BITS_TOKEN);
    };

    // 0.1875 = 1.5 * 2^-3: a non-zero mantissa, so an encoding that dropped the fraction field
    // would not pass.
    EXPECT_EQ(scaleBits(0.1875F), 0x3E400000);
    EXPECT_EQ(scaleBits(0.125F), 0x3E000000);
    EXPECT_EQ(scaleBits(SCALE), ieee754Bits(SCALE));
}

/// An absent attn_scale_value is cuDNN's default of 1.0, resolved once at match.
TEST(TestGfx950AttentionUnified2DDispatch, AnAbsentScaleBindsOne)
{
    GraphSpec spec;
    spec.attnScaleValue = std::nullopt;
    EXPECT_EQ(tokenValue(bindingsFor(Graph(spec)), SCALE_BITS_TOKEN), 0x3F800000);
}

/// All three mask kinds bind their own value, and the sliding window is a constant zero that
/// exists so a variant built with a window cannot be matched by a graph that never asked for
/// one. A mask stuck at one value matches a third of the catalog wrongly.
TEST(TestGfx950AttentionUnified2DDispatch, BindsEachMaskKindAndAZeroSlidingWindow)
{
    for(const int64_t mask : {NO_MASK, TOP_LEFT_CAUSAL, BOTTOM_RIGHT_CAUSAL})
    {
        SCOPED_TRACE("mask " + std::to_string(mask));
        GraphSpec spec;
        setMask(spec, mask);
        const auto bound = bindingsFor(Graph(spec));
        EXPECT_EQ(tokenValue(bound, MASK_TOKEN), mask);
        EXPECT_EQ(tokenValue(bound, SLIDING_WINDOW_TOKEN), 0);
    }
}

TEST(TestGfx950AttentionUnified2DDispatch, BindsExactlyTheTokensTheDispatchReads)
{
    EXPECT_EQ(bindingsFor(Graph{}).size(), allTokens().size());
}

// =============================================================================
// What prepare() does with them
// =============================================================================

/// The positive control the refusals below depend on: a prepare() that threw for every input
/// would satisfy all of them and dispatch nothing. A failure is still expected -- the archive
/// is absent -- but it must be the loader's, which means the bindings, the geometry, the
/// capacity and the identity tables were all accepted first.
TEST(TestGfx950AttentionUnified2DDispatch, ReachesTheLoaderOnceEverythingBeforeItIsGood)
{
    const std::string failure = prepareFailureFor(KernelSpec{});

    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    for(const char* marker : {MISSING_TOKEN_MARKER,
                              OUTPUT_LAYOUT_MARKER,
                              GEOMETRY_MARKER,
                              CAPACITY_MARKER,
                              BLOCK_MARKER})
    {
        EXPECT_EQ(failure.find(marker), std::string::npos) << failure;
    }
    EXPECT_NE(failure.find("does not exist"), std::string::npos) << failure;
}

/// Empty BoundTokens is what a mismatched catalog entry would hand prepare(). Without the read
/// guard the bindings are default-constructed zeros: four uids of zero that resolve to
/// whatever buffer the caller happened to label zero, and a scale of positive zero.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const Graph graph;
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);

    EXPECT_THROW(handler->prepare(graph.context(), BoundTokens{}, makeKernel(KernelSpec{})),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// Each token removed on its own, which is the shape a rename takes: the matcher writes a new
/// spelling, the dispatch still reads the old one, and every other token is present. Matched
/// on text rather than only type, because the archive is absent too -- a dispatch that stopped
/// reading a token would still throw here, just from the loader.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareWhenAnyOneBoundTokenIsMissing)
{
    const Graph graph;
    const auto bound = bindingsFor(graph);

    for(const std::string_view token : allTokens())
    {
        SCOPED_TRACE(std::string(token));

        BoundTokens tampered = bound;
        tampered.erase(std::string(token));

        const std::string failure
            = prepareFailure(graph, tampered, makeKpackKernel(KernelSpec{}, pythonAbi()));
        ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
        EXPECT_NE(failure.find(MISSING_TOKEN_MARKER), std::string::npos) << failure;
        EXPECT_NE(failure.find(std::string(token)), std::string::npos) << failure;
    }
}

/// The O token naming no tensor at all: the checks that follow index the tensor's dims, so a
/// dropped null test is a dereference of nothing.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareWhenTheOutputTokenNamesNoTensor)
{
    const Graph graph;
    auto tampered = bindingsFor(graph);
    tampered[std::string(O_TOKEN)] = EXTRA_UID;

    const std::string failure
        = prepareFailure(graph, tampered, makeKpackKernel(KernelSpec{}, pythonAbi()));
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;
}

/// The O token naming a real tensor that is head-major rather than token-major: the kernel
/// writes query rows back at `[token, head, d]`, so a BHSD output is written in bounds at the
/// wrong addresses and reported as a success. graph_match declines such a graph outright; this
/// drives the defence-in-depth check in prepare() directly.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareWhenTheOutputTokenNamesANonBshdTensor)
{
    GraphSpec spec;
    spec.oLayout = Layout::BHSD;
    const Graph graph(spec);
    // graph_match refuses this graph, so bind the tokens from the well-formed one and point
    // O at the same tensor in the BHSD graph.
    const auto good = matchGraph(Graph{});
    ASSERT_TRUE(good.has_value());

    const std::string failure
        = prepareFailure(graph, *good, makeKpackKernel(KernelSpec{}, pythonAbi()));
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;
}

/// A candidate whose geometry the Python builder would refuse. The grid and CTA come from it,
/// so prepare() refuses by name before the loader: any substitute would launch this binary with
/// another binary's geometry.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareACandidateWithoutASupportedGeometry)
{
    std::vector<KernelSpec> malformed;
    const auto with = [&malformed](auto mutate) {
        KernelSpec spec;
        mutate(spec);
        malformed.push_back(spec);
    };
    with([](KernelSpec& s) { s.numWarps = 3; });
    with([](KernelSpec& s) { s.numWarps = 0; });
    with([](KernelSpec& s) { s.blockMPerWarp = 24; });
    with([](KernelSpec& s) {
        s.blockMPerWarp = 32;
        s.numWarps = 8;
    });
    with([](KernelSpec& s) { s.tileSize = 70; });
    with([](KernelSpec& s) { s.tileSize = 0; });
    with([](KernelSpec& s) { s.blockSize = 24; });
    with([](KernelSpec& s) { s.useMfma32x32 = 1; }); // needs 32 rows per warp
    with([](KernelSpec& s) { s.useTransposedQk32x32 = 1; }); // needs the 32x32 path
    with([](KernelSpec& s) { s.numSeqs = 0; });
    with([](KernelSpec& s) { s.maskType = 3; });
    with([](KernelSpec& s) { s.headSize = 96; });
    with([](KernelSpec& s) { s.numQueryHeads = 9; });

    for(std::size_t i = 0; i < malformed.size(); ++i)
    {
        SCOPED_TRACE("malformed candidate " + std::to_string(i));
        const std::string failure = prepareFailureFor(malformed.at(i));
        ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
        EXPECT_NE(failure.find(GEOMETRY_MARKER), std::string::npos) << failure;
    }
}

/// A completed record omits no field, so one that does is uncompleted or malformed.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareACandidateMissingAField)
{
    const Graph graph;
    const auto bound = bindingsFor(graph);
    const auto complete = metadataOf(KernelSpec{});
    for(const auto& entry : complete)
    {
        const std::string& field = entry.first;
        if(field == "dtype")
        {
            continue; // read by the matcher, not by prepare()
        }
        SCOPED_TRACE(field);
        auto kernel = makeKpackKernel(KernelSpec{}, pythonAbi());
        kernel.metadata.erase(field);
        const std::string failure = prepareFailure(graph, bound, kernel);
        ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
        EXPECT_NE(failure.find(GEOMETRY_MARKER), std::string::npos) << failure;
    }

    auto mistyped = makeKpackKernel(KernelSpec{}, pythonAbi());
    mistyped.metadata[std::string("num_warps")] = std::string("4");
    const std::string failure = prepareFailure(graph, bound, mistyped);
    EXPECT_NE(failure.find(GEOMETRY_MARKER), std::string::npos) << failure;
}

/// One past the capacity the kernel was built for. The search would leave the last blocks
/// reading another sequence's lengths without a fault, so prepare() refuses by name.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareABatchAboveTheCandidatesCapacity)
{
    GraphSpec spec;
    spec.batch = 4;
    spec.seqLenQ = 16;
    spec.seqLenKv = 16;
    KernelSpec capacityThree;
    capacityThree.numSeqs = 3; // 2 iterations -> 3 sequences

    const std::string tooMany = prepareFailureFor(capacityThree, spec);
    ASSERT_FALSE(tooMany.empty());
    EXPECT_NE(tooMany.find(CAPACITY_MARKER), std::string::npos) << tooMany;

    spec.batch = 3;
    const std::string exactlyFull = prepareFailureFor(capacityThree, spec);
    ASSERT_FALSE(exactlyFull.empty());
    EXPECT_EQ(exactlyFull.find(CAPACITY_MARKER), std::string::npos) << exactlyFull;
}

/// A block size that does not divide the graph's Skv is not a whole-block cache.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareABlockSizeThatDoesNotDivideSeqLenKv)
{
    GraphSpec spec;
    spec.seqLenQ = 32;
    spec.seqLenKv = 96;
    KernelSpec block64;
    block64.blockSize = 64;
    block64.tileSize = 256;

    const std::string failure = prepareFailureFor(block64, spec);
    ASSERT_FALSE(failure.empty());
    EXPECT_NE(failure.find(BLOCK_MARKER), std::string::npos) << failure;
}

/// This pack supplies nothing to compile from source with, so a candidate that would be
/// compiled from source is refused before anything is built, and the refusal names its kind.
TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareACandidateThatIsNotKpack)
{
    const Graph graph;
    auto kernel = makeKernel(KernelSpec{});
    kernel.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    kernel.source.sourceFile = "kernels/AttentionUnified2D.cpp";
    kernel.source.entryPoint = "attention_unified_2d";

    const std::string failure = prepareFailure(graph, bindingsFor(graph), kernel);
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find("'embedded_source'"), std::string::npos) << failure;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
