// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestGfx950AttentionDenseDispatch.cpp
 * @brief The hipkernel:Gfx950AttentionDense pack at the bind and prepare layer: what the
 *        graph match writes into the bound tokens, and what prepare() does with them
 *        before it reaches a loader.
 *
 * Every failure this file is after is a SILENT one. A token carrying the wrong uid
 * launches the kernel against the wrong buffers; a scale bound as an integer rather than
 * as its bit pattern reads on the device as a denormal near zero. Neither faults, neither
 * sets a status, and the kernel-matching tests cannot see either -- kernelMatches() reads
 * only the causal and sliding_window tokens, so the four operand uids and the scale could
 * hold anything and every matcher case would still pass.
 *
 * ROUTE. Two seams, both device-free:
 *
 *  1. The graph matcher, resolved out of GraphMatchRegistry by the symbol the installed
 *     descriptors name. It is the sole producer of bound tokens, so reading its result is
 *     reading exactly what a plan build hands the dispatch handler.
 *  2. The dispatch handler's prepare(), reached through DispatchRegistry with a descriptor
 *     pointing at an archive that is not there -- the technique
 *     TestGfx950AttentionDenseSignature.cpp establishes. Everything prepare() checks
 *     before buildIngestorKernelCode is therefore reachable on any machine, and the
 *     archive's absence is the failure that arrives when those checks pass.
 *
 * WHAT IS NOT REACHABLE HERE. prepare() calls gfx950AttentionDenseGeometry() AFTER
 * buildIngestorKernelCode has loaded the code object, and applies the result through
 * IngestorKernelCode::setGridSize / setBlockSize onto an IRunnableKernel held inside a
 * PreparedDispatch subclass that lives in the pack's own anonymous namespace. The base
 * class hipdnn_plugin_sdk::ingestor::PreparedDispatch declares nothing but a destructor,
 * so a prepared dispatch is opaque to a test even where one can be produced. The call
 * site's ARGUMENTS -- (block_m, seqLenQ, numQueryHeads, batch) -- are consequently not
 * observable without a real archive and a real gfx950 device. The geometry file pins what
 * the function computes; nothing here pins what prepare() passes it. What IS reachable is
 * the tile check prepare() makes before loading: a candidate with no supported tile is
 * refused by name rather than launched with a substituted one.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.graph_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_dense.dispatch";

// ---------------------------------------------------------------------------
// The token names, restated
// ---------------------------------------------------------------------------
//
// The pack's own spellings live in the anonymous namespace of
// Gfx950AttentionDenseNative.cpp, so no test translation unit can name them. Written out
// again here deliberately: these strings are the contract between the matcher that writes
// a token and the dispatch that reads it, and a test sharing the constant would keep
// passing through a rename that broke nothing but also pinned nothing.

constexpr std::string_view Q_TOKEN = "gfx950_attention_dense.q.uid";
constexpr std::string_view K_TOKEN = "gfx950_attention_dense.k.uid";
constexpr std::string_view V_TOKEN = "gfx950_attention_dense.v.uid";
constexpr std::string_view O_TOKEN = "gfx950_attention_dense.o.uid";
constexpr std::string_view CAUSAL_TOKEN = "gfx950_attention_dense.causal";
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx950_attention_dense.sliding_window";
constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_dense.scale_bits";

/// Every token the dispatch reads, in one place, so the case that removes them one at a
/// time cannot quietly stop covering one.
const std::vector<std::string_view>& allTokens()
{
    static const std::vector<std::string_view> s_tokens{
        Q_TOKEN, K_TOKEN, V_TOKEN, O_TOKEN, CAUSAL_TOKEN, SLIDING_WINDOW_TOKEN, SCALE_BITS_TOKEN};
    return s_tokens;
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

/// A uid no tensor in the graph carries.
constexpr int64_t ABSENT_UID = 99;

constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 4;
constexpr int64_t SEQ = 512;
constexpr int64_t HEAD_SIZE = 128;

/// BSHD strides for (B, H, S, D) logical dims -- token-major, head varying fastest.
std::vector<int64_t> bshdStrides()
{
    return {SEQ * HEADS * HEAD_SIZE, HEAD_SIZE, HEADS * HEAD_SIZE, 1};
}

/// BHSD strides for the same dims: head-major. Agrees with BSHD on the batch axis and
/// on the element axis, and disagrees on the two in between, which is the whole class of
/// layout the kernel cannot read.
std::vector<int64_t> bhsdStrides()
{
    return {HEADS * SEQ * HEAD_SIZE, SEQ * HEAD_SIZE, HEAD_SIZE, 1};
}

/// One bf16 operand of the fixture shape. The fields no case here varies -- name, dtype
/// and the virtual flag -- are fixed.
flatbuffers::Offset<data_objects::TensorAttributes>
    addTensor(flatbuffers::FlatBufferBuilder& builder,
              int64_t uid,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& dims)
{
    return data_objects::CreateTensorAttributesDirect(
        builder, uid, nullptr, data_objects::DataType::BFLOAT16, &strides, &dims, false);
}

/// Which mask the built graph asks for.
enum class Mask
{
    /// right_bound 0 with TOP_LEFT alignment: the causal corner, causal token 1.
    TOP_LEFT_CAUSAL,
    /// right_bound left unset: an unmasked graph, causal token 0.
    UNMASKED
};

/// bf16, D128, B=2, Hq=Hkv=4, Sq=Skv=512, dense BSHD throughout -- a shape the engine
/// accepts. The applicability rules are TestGfx950AttentionDenseMatchers.cpp's subject;
/// here the graph only has to be one that matches, so the tokens it binds can be read.
flatbuffers::FlatBufferBuilder buildGraph(float scale, Mask mask, bool withBhsdTensor)
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> dims{BATCH, HEADS, SEQ, HEAD_SIZE};
    const std::vector<int64_t> strides = bshdStrides();
    const std::vector<int64_t> wrongStrides = bhsdStrides();

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        tensors.push_back(addTensor(builder, uid, strides, dims));
    }
    if(withBhsdTensor)
    {
        tensors.push_back(addTensor(builder, BHSD_UID, wrongStrides, dims));
    }

    data_objects::SdpaAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_q_tensor_uid(Q_UID);
    attributesBuilder.add_k_tensor_uid(K_UID);
    attributesBuilder.add_v_tensor_uid(V_UID);
    attributesBuilder.add_o_tensor_uid(O_UID);
    attributesBuilder.add_left_bound(-1);
    if(mask == Mask::TOP_LEFT_CAUSAL)
    {
        attributesBuilder.add_right_bound(0);
    }
    attributesBuilder.add_diagonal_alignment(data_objects::DiagonalAlignment::TOP_LEFT);
    attributesBuilder.add_causal_mask(false);
    attributesBuilder.add_causal_mask_bottom_right(false);
    attributesBuilder.add_attn_scale_value(scale);
    attributesBuilder.add_alibi_mask(false);
    attributesBuilder.add_padding_mask(false);
    attributesBuilder.add_mma_core_mode(data_objects::DataType::UNSET);
    attributesBuilder.add_implementation(data_objects::AttentionImplementation::AUTO);
    const auto attributes = attributesBuilder.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   attributes.Union()));

    auto name = builder.CreateString("gfx950_attention_dense_dispatch");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// Keeps a built graph alive for as long as a case reads it, and hands out the
/// MatchContext an engine sees. The context holds references into this object, so it must
/// outlive every use.
class AttentionGraph
{
public:
    explicit AttentionGraph(float scale = 0.125F,
                            Mask mask = Mask::TOP_LEFT_CAUSAL,
                            bool withBhsdTensor = false)
        : _builder(buildGraph(scale, mask, withBhsdTensor))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
        _properties.gcnArchName = "gfx950";
        _properties.warpSize = 64;
    }

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    DeviceProperties _properties;
};

// ---------------------------------------------------------------------------
// The two registry seams
// ---------------------------------------------------------------------------

/// Runs the pack's graph match over @p graph. An empty result means the engine declined,
/// which no case here expects -- every graph built above is one it serves.
BoundTokens bindingsFor(const AttentionGraph& graph)
{
    registerNativeIngestorSymbols();

    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    auto bound = matcher(graph.context());
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the fixture graph, so there are no "
                                      "tokens to read and no bindings to hand prepare()";
    if(!bound.has_value())
    {
        return {};
    }
    return *bound;
}

const IKernelDispatchHandler<Handle>* dispatchHandler()
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
}

/// The integer @p token carries. The sentinel is deliberately not zero: zero is a real
/// value for the causal and sliding-window tokens, so a missing token answering zero
/// would read as a correct binding.
constexpr int64_t NOT_BOUND = -1000;

int64_t tokenValue(const BoundTokens& bound, std::string_view token)
{
    return tryGetBoundInt(bound, token).value_or(NOT_BOUND);
}

/// @p value's IEEE-754 storage, the way BoundTokens must carry a float.
///
/// Used only for a scale whose bit pattern is impractical to write as a literal; the two
/// cases that pin the encoding use literals read off the format, since a helper that
/// repeats what the pack does would assert only that the two agree.
int64_t ieee754Bits(float value)
{
    int32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit to round-trip");
    std::memcpy(&bits, &value, sizeof(value));
    return static_cast<int64_t>(bits);
}

// ---------------------------------------------------------------------------
// A descriptor prepare() can be driven with
// ---------------------------------------------------------------------------

/// The ABI the rocKE builder declares: four `ptr<dtype, global>`, then `scale` as f32,
/// then `batch`, `seqlen_q`, `seqlen_kv` as i32. Recorded so the signature comparison
/// agrees and the failure below is the archive's, not a signature mismatch;
/// TestGfx950AttentionDenseSignature.cpp owns that comparison.
std::vector<KernelArgument> recordedSignature()
{
    constexpr uint32_t POINTER_BYTES = 8;
    constexpr uint32_t SCALAR_BYTES = 4;

    const KernelArgument buffer{"global_buffer", POINTER_BYTES, 0, ""};
    const KernelArgument scalar{"by_value", SCALAR_BYTES, 0, ""};
    return {buffer, buffer, buffer, buffer, scalar, scalar, scalar, scalar};
}

/// A KPACK descriptor for this engine naming an archive that is not there, carrying the
/// completed metadata prepare() reads: head_size and the (block_m, block_n) tile.
///
/// That absence is the point: every check prepare() performs BEFORE the loader runs is
/// reached, and when they all pass the loader reports the missing file. No device, no
/// compile, no launch.
KernelDefinition makeKernel(int64_t blockM = 256, int64_t blockN = 64)
{
    KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051b1");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051b2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051b3");
    kernel.name = "attention_dense.bf16_d128_hq4_kv4_ca.gfx950";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = "attention_dense";
    kernel.source.signature = recordedSignature();
    kernel.originDirectory = "/nonexistent";
    kernel.treeRoot = "/nonexistent";
    kernel.metadata = {
        {std::string("head_size"), HEAD_SIZE},
        {std::string("block_m"), blockM},
        {std::string("block_n"), blockN},
    };
    return kernel;
}

/// The message prepare() throws for @p bound and @p kernel, or empty if it returned a
/// prepared dispatch -- which it cannot, since the archive is absent.
std::string prepareFailure(const AttentionGraph& graph,
                           const BoundTokens& bound,
                           const KernelDefinition& kernel = makeKernel())
{
    const auto* handler = dispatchHandler();
    EXPECT_NE(handler, nullptr);
    if(handler == nullptr)
    {
        return {};
    }

    try
    {
        handler->prepare(graph.context(), bound, kernel);
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

/// The fragment only the missing-token refusal carries.
constexpr const char* MISSING_TOKEN_MARKER = "missing bound token";

/// The fragment only the output-layout refusal carries.
constexpr const char* OUTPUT_LAYOUT_MARKER = "not dense BSHD";

/// The fragment only the tile refusal carries.
constexpr const char* TILE_MARKER = "no supported block_m/block_n tile";

} // namespace

// =============================================================================
// Registration
// =============================================================================

/// If registerGfx950AttentionDenseSymbols stops adding DISPATCH_SYMBOL, or IngestorPacks
/// drops this engine's row, every descriptor naming the symbol is dropped at load: the
/// pack disappears, then the engine, and at the default log level nothing says so. The
/// graph would simply be served by some other provider, or by none. Nothing else in the
/// fast suite asks the registry for this symbol.
TEST(TestGfx950AttentionDenseDispatch, DispatchSymbolResolves)
{
    registerNativeIngestorSymbols();

    EXPECT_NE(hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
                  std::string(DISPATCH_SYMBOL)),
              nullptr);
}

// =============================================================================
// Workspace
// =============================================================================

/// This kernel's only scratch is LDS and registers, and the shipped 8-argument ABI has no
/// workspace pointer to hand global scratch through, so there is nowhere for a non-zero
/// answer to be used. The one caller -- execution-plan build -- never checks the value,
/// which makes this the only place the zero is pinned.
TEST(TestGfx950AttentionDenseDispatch, WorkspaceBytesIsAlwaysZero)
{
    const AttentionGraph graph;
    const auto bound = bindingsFor(graph);
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);

    EXPECT_EQ(handler->workspaceBytes(graph.context(), bound, makeKernel()), 0U);
}

// =============================================================================
// What the match binds
// =============================================================================

/// The four uids the launch resolves device buffers by. launch() looks each one up with
/// findDeviceBuffer and passes the results positionally, so a Q/K transposition here is a
/// kernel reading keys as queries: full-rate arithmetic, no fault, wrong numbers. The
/// uids are mutually distinct, so any permutation of the four fails this.
TEST(TestGfx950AttentionDenseDispatch, BindsEachOperandUidToItsOwnToken)
{
    const AttentionGraph graph;
    const auto bound = bindingsFor(graph);

    EXPECT_EQ(tokenValue(bound, Q_TOKEN), Q_UID);
    EXPECT_EQ(tokenValue(bound, K_TOKEN), K_UID);
    EXPECT_EQ(tokenValue(bound, V_TOKEN), V_UID);
    EXPECT_EQ(tokenValue(bound, O_TOKEN), O_UID);
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
TEST(TestGfx950AttentionDenseDispatch, BindsTheSoftmaxScaleAsItsIeeeBitPattern)
{
    // 0.1875 = 1.5 * 2^-3. Sign 0, biased exponent 124 (0x7C), mantissa 0x400000.
    // Chosen with a non-zero mantissa: a power of two alone would not catch an encoding
    // that dropped the fraction field.
    const AttentionGraph fraction(0.1875F);
    EXPECT_EQ(tokenValue(bindingsFor(fraction), SCALE_BITS_TOKEN), 0x3E400000);

    // 0.125 = 1.0 * 2^-3. Same exponent, zero mantissa -- the neighbouring encoding, so
    // the two together pin the exponent and the fraction independently.
    const AttentionGraph power(0.125F);
    EXPECT_EQ(tokenValue(bindingsFor(power), SCALE_BITS_TOKEN), 0x3E000000);

    // The scale a D128 graph actually carries, 1/sqrt(128). Its pattern is impractical to
    // write by hand, so this case only adds that a realistic value survives the trip.
    constexpr float REALISTIC_SCALE = 0.08838834764831843F;
    const AttentionGraph realistic(REALISTIC_SCALE);
    EXPECT_EQ(tokenValue(bindingsFor(realistic), SCALE_BITS_TOKEN), ieee754Bits(REALISTIC_SCALE));
}

/// The two tokens kernelMatches() compares against a candidate's baked metadata. The
/// causal flag is DERIVED -- hipDNN has no such boolean -- so it is a value this pack
/// invents, and the sliding window is a constant zero that exists so a variant built with
/// a window cannot be matched by a graph that never asked for one. Both mask shapes are
/// read here because a flag stuck at one value matches half the catalog wrongly.
TEST(TestGfx950AttentionDenseDispatch, BindsTheDerivedMaskAndAZeroSlidingWindow)
{
    const AttentionGraph causal(0.125F, Mask::TOP_LEFT_CAUSAL);
    const auto causalBound = bindingsFor(causal);
    EXPECT_EQ(tokenValue(causalBound, CAUSAL_TOKEN), 1);
    EXPECT_EQ(tokenValue(causalBound, SLIDING_WINDOW_TOKEN), 0);

    const AttentionGraph unmasked(0.125F, Mask::UNMASKED);
    const auto unmaskedBound = bindingsFor(unmasked);
    EXPECT_EQ(tokenValue(unmaskedBound, CAUSAL_TOKEN), 0);
    EXPECT_EQ(tokenValue(unmaskedBound, SLIDING_WINDOW_TOKEN), 0);
}

// =============================================================================
// What prepare() does with them
// =============================================================================

/// The positive control the refusals below depend on. A prepare() that threw for every
/// input would satisfy all of them and dispatch nothing.
///
/// A failure is still expected -- the archive is absent -- but it must be the loader's,
/// which means the bindings were read and the output tensor was accepted first.
TEST(TestGfx950AttentionDenseDispatch, ReachesTheLoaderOnceTheBindingsAndOutputAreGood)
{
    const AttentionGraph graph;
    const std::string failure = prepareFailure(graph, bindingsFor(graph));

    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    EXPECT_EQ(failure.find(MISSING_TOKEN_MARKER), std::string::npos) << failure;
    EXPECT_EQ(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;
}

/// Empty BoundTokens is what a mismatched catalog entry would hand prepare(). Without the
/// read guard, the bindings are default-constructed zeros: four uids of zero that resolve
/// to whatever buffer the caller happened to label zero, and a scale of positive zero.
TEST(TestGfx950AttentionDenseDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const AttentionGraph graph;
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);

    EXPECT_THROW(handler->prepare(graph.context(), BoundTokens{}, makeKernel()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// Each token removed on its own, which is the shape a rename takes: the matcher writes a
/// new spelling, the dispatch still reads the old one, and every other token is present.
///
/// The failure is matched on its text rather than only on its type, because the archive is
/// absent too -- a dispatch that stopped reading a token would still throw here, just from
/// the loader instead. Naming the token is what distinguishes the two.
TEST(TestGfx950AttentionDenseDispatch, RefusesToPrepareWhenAnyOneBoundTokenIsMissing)
{
    const AttentionGraph graph;
    const auto bound = bindingsFor(graph);
    ASSERT_EQ(bound.size(), allTokens().size())
        << "the match binds a token this case does not know about, or has stopped binding one";

    for(const std::string_view token : allTokens())
    {
        SCOPED_TRACE(std::string(token));

        BoundTokens tampered = bound;
        tampered.erase(std::string(token));

        const std::string failure = prepareFailure(graph, tampered);
        ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
        EXPECT_NE(failure.find(MISSING_TOKEN_MARKER), std::string::npos) << failure;
        EXPECT_NE(failure.find(std::string(token)), std::string::npos) << failure;
    }
}

/// The O token naming no tensor at all. prepare() looks the output up to re-check its
/// layout, and a null result has to be refused by name: the checks that follow index the
/// tensor's dims, so a dropped null test is a dereference of nothing.
TEST(TestGfx950AttentionDenseDispatch, RefusesToPrepareWhenTheOutputTokenNamesNoTensor)
{
    const AttentionGraph graph;
    auto tampered = bindingsFor(graph);
    tampered[std::string(O_TOKEN)] = ABSENT_UID;

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
TEST(TestGfx950AttentionDenseDispatch, RefusesToPrepareWhenTheOutputTokenNamesANonBshdTensor)
{
    const AttentionGraph graph(0.125F, Mask::TOP_LEFT_CAUSAL, /*withBhsdTensor=*/true);
    auto tampered = bindingsFor(graph);
    tampered[std::string(O_TOKEN)] = BHSD_UID;

    const std::string failure = prepareFailure(graph, tampered);
    ASSERT_FALSE(failure.empty()) << "prepare() did not fail at all";
    EXPECT_NE(failure.find(OUTPUT_LAYOUT_MARKER), std::string::npos) << failure;
}

/// Every tile a catalog candidate can carry, at both block_m, reaches the loader: the tile
/// check is not what stops a legal alternative.
TEST(TestGfx950AttentionDenseDispatch, ReachesTheLoaderForEveryBuildableTile)
{
    const AttentionGraph graph;
    const auto bound = bindingsFor(graph);
    for(const auto& [blockM, blockN] : std::vector<std::pair<int64_t, int64_t>>{
            {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}})
    {
        SCOPED_TRACE(std::to_string(blockM) + "/" + std::to_string(blockN));
        const std::string failure = prepareFailure(graph, bound, makeKernel(blockM, blockN));
        ASSERT_FALSE(failure.empty()) << "prepare() cannot succeed without the archive";
        EXPECT_EQ(failure.find(TILE_MARKER), std::string::npos) << failure;
    }
}

/// A candidate whose tile is missing, zero, negative, unbuilt or unbuildable at its head
/// size (D128 block_n 256 exceeds LDS). The grid and CTA come from block_m, so prepare()
/// refuses by name before the loader -- a zero would otherwise be a division by zero, and
/// any substitute tile launches this binary with another binary's geometry.
TEST(TestGfx950AttentionDenseDispatch, RefusesToPrepareACandidateWithoutASupportedTile)
{
    const AttentionGraph graph;
    const auto bound = bindingsFor(graph);

    std::vector<KernelDefinition> malformed;
    for(const auto& [blockM, blockN] : std::vector<std::pair<int64_t, int64_t>>{
            {0, 64}, {256, 0}, {-256, 64}, {64, 64}, {512, 64}, {128, 256}, {256, 256}})
    {
        malformed.push_back(makeKernel(blockM, blockN));
    }
    for(const char* field : {"block_m", "block_n"})
    {
        auto kernel = makeKernel();
        kernel.metadata.erase(field);
        malformed.push_back(kernel);
    }
    auto mistyped = makeKernel();
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

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
