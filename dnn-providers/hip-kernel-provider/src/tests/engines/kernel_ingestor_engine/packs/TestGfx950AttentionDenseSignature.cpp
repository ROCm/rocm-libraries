// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestGfx950AttentionDenseSignature.cpp
 * @brief Pins the kernarg ABI hipkernel:Gfx950AttentionDense marshals against the ABI
 *        the rocKE builder declares.
 *
 * A wrong argument count, order, kind or width is not a launch error. The driver copies
 * whatever the host hands it into the kernarg segment and the kernel reads it as the
 * types it was compiled for, so the failure mode is misread memory: wrong numerics with
 * a success status. This is the silent-wrong-answer class the layout and stride gates in
 * the matcher exist to stop, one layer lower down.
 *
 * Sources for the expectations, all read off the Python and written here as literals:
 *   attention_dense_signature (rocke/library/kernels/gfx950/attention_dense.py:2078-2110)
 *     q_ptr, k_ptr, v_ptr, o_ptr as `ptr<dtype, global>`, then `scale` as f32, then
 *     `batch`, `seqlen_q`, `seqlen_kv` as i32.
 *   build_attention_dense  (attention_dense.py:369-396) declares those same parameters
 *     in that order, and declares the three shape params unconditionally.
 *   _has_shape_params      (attention_dense.py:2062-2075) gates the shape tail on
 *     `not spec.persistent`, so only a persistent body takes the 5-argument
 *     form. Every variant in gfx950_attention_dense.kdp.json declares
 *     `persistent: false`, `use_sinks: false`, `varlen: false` and `paged: false`, so the
 *     8-argument form is the only one that ships and the sink / cu_seqlens / page-table
 *     tails of attention_dense_signature are unreachable from this catalog. The
 *     5-argument form is covered here only as a shape the pack must REFUSE.
 *
 * ROUTE. The list the pack marshals lives in attentionDenseKernelSignature(), in the
 * anonymous namespace of Gfx950AttentionDenseNative.cpp, so no test translation unit can
 * name it. Reading it through the pack's own dispatch handler instead: prepare() hands it
 * to buildIngestorKernelCode as the expected side of requireSignatureMatch, which
 * compares it against the list the descriptor records and throws before the archive is
 * opened. Every case below therefore presents a descriptor recording a candidate ABI and
 * asks which failure comes back. Acceptance of the Python form plus refusal of each
 * neighbouring form is what pins the pack's list to the Python's; copying the expected
 * values out of the C++ under test would assert only that it equals itself.
 *
 * WHAT IS PINNED. requireSignatureMatch compares `kind` and `size`, and `name` only when
 * both sides carry one. The pack records no names, so these cases reach kind, size, count
 * and order. `offset` is printed and never compared, which is why every expectation below
 * carries zero for it: it is not part of the comparison and the pack's own list does not
 * populate it either. f32 and i32 are indistinguishable at this seam -- both are
 * `by_value` of 4 bytes -- so the scale-versus-shape split within the by-value tail is
 * pinned only by position and width, not by element type.
 *
 * No device, no compile and no launch: the archive named here does not exist, and both
 * the signature refusal and the archive's absence are raised on the host.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MatchContext;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.graph_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_dense.dispatch";

// ---------------------------------------------------------------------------
// The ABI, as the Python declares it
// ---------------------------------------------------------------------------

/// `.value_kind` a device pointer parameter carries in the AMDGPU metadata note.
constexpr const char* BUFFER_KIND = "global_buffer";

/// `.value_kind` a scalar parameter carries.
constexpr const char* BY_VALUE_KIND = "by_value";

/// Bytes one `ptr<dtype, global>` occupies in the kernarg segment: a 64-bit address,
/// whatever the element type is.
constexpr uint32_t POINTER_BYTES = 8;

/// Bytes one `f32` or `i32` scalar occupies.
constexpr uint32_t SCALAR_BYTES = 4;

/// Position of `scale`, the first by-value argument.
constexpr std::size_t SCALE_INDEX = 4;

/// Position of `seqlen_kv`, the last argument.
constexpr std::size_t LAST_INDEX = 7;

/// (q_ptr, k_ptr, v_ptr, o_ptr, scale, batch, seqlen_q, seqlen_kv): four buffers, then
/// an f32, then three i32.
std::vector<KernelArgument> pythonAbi()
{
    const KernelArgument buffer{BUFFER_KIND, POINTER_BYTES, 0, ""};
    const KernelArgument scalar{BY_VALUE_KIND, SCALAR_BYTES, 0, ""};
    return {buffer, buffer, buffer, buffer, scalar, scalar, scalar, scalar};
}

/// The form a persistent body would take: the same four pointers and scale, with the
/// shape tail dropped because `num_persistent` bakes the work-item space. No shipped
/// variant is persistent, so a descriptor recording this is a descriptor the pack cannot
/// launch.
std::vector<KernelArgument> persistentAbi()
{
    auto signature = pythonAbi();
    signature.resize(SCALE_INDEX + 1);
    return signature;
}

/// `pythonAbi()` with one entry replaced.
std::vector<KernelArgument> withArgument(std::size_t index, const KernelArgument& argument)
{
    auto signature = pythonAbi();
    signature.at(index) = argument;
    return signature;
}

// ---------------------------------------------------------------------------
// A graph the engine serves
// ---------------------------------------------------------------------------

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;

/// bf16, D128, B=2, Hq=Hkv=4, Sq=Skv=256, top-left causal, dense BSHD throughout. Any
/// graph the engine accepts would do; the shape is not what these cases are about, and
/// TestGfx950AttentionDenseMatchers.cpp owns the applicability rules.
constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 4;
constexpr int64_t SEQ = 256;
constexpr int64_t HEAD_SIZE = 128;
constexpr float SCALE = 0.08838834764831843F;

/// BSHD strides for (B, H, S, D) logical dims -- token-major, head varying fastest.
std::vector<int64_t> bshdStrides()
{
    return {SEQ * HEADS * HEAD_SIZE, HEAD_SIZE, HEADS * HEAD_SIZE, 1};
}

flatbuffers::FlatBufferBuilder buildAcceptedSdpaGraph()
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> dims{BATCH, HEADS, SEQ, HEAD_SIZE};
    const std::vector<int64_t> strides = bshdStrides();

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, uid, nullptr, data_objects::DataType::BFLOAT16, &strides, &dims, false));
    }

    data_objects::SdpaAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_q_tensor_uid(Q_UID);
    attributesBuilder.add_k_tensor_uid(K_UID);
    attributesBuilder.add_v_tensor_uid(V_UID);
    attributesBuilder.add_o_tensor_uid(O_UID);
    attributesBuilder.add_left_bound(-1);
    attributesBuilder.add_right_bound(0);
    attributesBuilder.add_diagonal_alignment(data_objects::DiagonalAlignment::TOP_LEFT);
    attributesBuilder.add_causal_mask(false);
    attributesBuilder.add_causal_mask_bottom_right(false);
    attributesBuilder.add_attn_scale_value(SCALE);
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

    auto name = builder.CreateString("gfx950_attention_dense_signature");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

// ---------------------------------------------------------------------------
// The seam
// ---------------------------------------------------------------------------

/// The fragment only the signature refusal carries. Kept short: the rest of that message
/// renders both argument lists and is not stable enough to match on.
constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

/// A KPACK descriptor for this engine whose recorded argument list is @p recorded.
///
/// The archive it names is not there, which is the point: the containment and link checks
/// pass, the signature comparison runs, and only if it agrees does the loader get as far
/// as reporting the absence. Nothing here depends on the archive's contents. The metadata
/// is the completed baseline tile prepare() reads before the comparison; the tile is not
/// what these cases are about, and TestGfx950AttentionDenseDispatch.cpp owns its refusal.
KernelDefinition makeKernel(const std::vector<KernelArgument>& recorded)
{
    KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051a1");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051a2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051a3");
    kernel.name = "attention_dense.bf16_d128_hq4_kv4_ca.gfx950";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = "attention_dense";
    kernel.source.signature = recorded;
    kernel.originDirectory = "/nonexistent";
    kernel.treeRoot = "/nonexistent";
    kernel.metadata = {
        {std::string("head_size"), HEAD_SIZE},
        {std::string("block_m"), int64_t{256}},
        {std::string("block_n"), int64_t{64}},
    };
    return kernel;
}

/// Runs the pack's prepare() over a descriptor recording @p recorded and returns the
/// message of whatever it throws. Empty means it returned a prepared dispatch, which no
/// case expects.
std::string prepareFailure(const std::vector<KernelArgument>& recorded)
{
    registerNativeIngestorSymbols();

    auto builder = buildAcceptedSdpaGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    DeviceProperties properties;
    properties.gcnArchName = "gfx950";
    properties.warpSize = 64;
    const MatchContext context{graph, 0, properties};

    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    const auto bound = matcher(context);
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the fixture graph, so prepare() "
                                      "never ran and no case below means anything";
    if(!bound.has_value())
    {
        return {};
    }

    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
    EXPECT_NE(handler, nullptr);
    if(handler == nullptr)
    {
        return {};
    }

    try
    {
        handler->prepare(context, *bound, makeKernel(recorded));
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

/// Asserts the pack refuses @p recorded on signature grounds rather than on any of the
/// other grounds prepare() can fail for.
void expectRefused(const std::vector<KernelArgument>& recorded, const char* what)
{
    const std::string failure = prepareFailure(recorded);
    ASSERT_FALSE(failure.empty()) << what << ": prepare() did not fail at all";
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ", but the failure was: " << failure;
}

} // namespace

// =============================================================================
// The shipped form is accepted
// =============================================================================

/// The positive control every refusal below depends on. A pack that refused every
/// argument list would pass all of them and launch nothing.
///
/// Failure is still expected -- the archive is absent -- but it must arrive from the
/// loader, past the comparison, rather than from the comparison itself.
TEST(TestGfx950AttentionDenseSignature, AcceptsTheAbiThePythonDeclares)
{
    const std::string failure = prepareFailure(pythonAbi());
    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << "the pack refused the argument list the rocKE builder declares: " << failure;
}

// =============================================================================
// Count
// =============================================================================

TEST(TestGfx950AttentionDenseSignature, RefusesADroppedTrailingArgument)
{
    // seqlen_kv gone. Seven arguments against eight: the kernel would read the kernarg
    // segment one slot past what the host wrote.
    auto signature = pythonAbi();
    signature.pop_back();
    expectRefused(signature, "a seven-argument list is not this ABI");
}

TEST(TestGfx950AttentionDenseSignature, RefusesAnExtraTrailingArgument)
{
    // One i32 too many -- the shape a fourth shape parameter, or a stride argument, would
    // add. Every argument the host does write still lands correctly, so nothing faults.
    auto signature = pythonAbi();
    signature.push_back(KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""});
    expectRefused(signature, "a nine-argument list is not this ABI");
}

TEST(TestGfx950AttentionDenseSignature, RefusesThePersistentFiveArgumentForm)
{
    // attention_dense_signature drops batch/seqlen_q/seqlen_kv when the spec is
    // persistent. No variant in this catalog is, and the pack marshals the shape
    // unconditionally, so a persistent code object reaching this pack must be refused
    // rather than launched with three arguments it never declared.
    expectRefused(persistentAbi(), "the persistent form does not ship in this catalog");
}

// =============================================================================
// Kind
// =============================================================================

TEST(TestGfx950AttentionDenseSignature, RefusesAScalarInAPointerSlot)
{
    // q_ptr as a by-value scalar. The count still agrees, so only the kind catches it.
    expectRefused(withArgument(0, KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""}),
                  "q_ptr is a global buffer");
}

TEST(TestGfx950AttentionDenseSignature, RefusesAPointerInAScalarSlot)
{
    // scale as a buffer: the kernel would take the float's bit pattern for an address.
    expectRefused(withArgument(SCALE_INDEX, KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, ""}),
                  "scale is passed by value");
}

TEST(TestGfx950AttentionDenseSignature, RefusesBuffersAndScalarsTransposed)
{
    // The by-value tail ahead of the pointers. Same eight entries, same four of each
    // kind, and the comparison is positional, so only the order distinguishes this from
    // the shipped ABI.
    const KernelArgument buffer{BUFFER_KIND, POINTER_BYTES, 0, ""};
    const KernelArgument scalar{BY_VALUE_KIND, SCALAR_BYTES, 0, ""};
    const std::vector<KernelArgument> transposed{
        scalar, scalar, scalar, scalar, buffer, buffer, buffer, buffer};
    expectRefused(transposed, "the four pointers come first");
}

// =============================================================================
// Size
// =============================================================================

TEST(TestGfx950AttentionDenseSignature, RefusesANarrowedPointerSlot)
{
    // o_ptr as a 32-bit buffer. The kind still agrees; the width does not, and every
    // argument after it would be read from the wrong offset.
    expectRefused(withArgument(3, KernelArgument{BUFFER_KIND, SCALAR_BYTES, 0, ""}),
                  "a device pointer is eight bytes");
}

TEST(TestGfx950AttentionDenseSignature, RefusesAWidenedScalarSlot)
{
    // seqlen_kv as an i64. The shape parameters are i32 and the pack marshals int32_t, so
    // a 64-bit slot would take the next four bytes of the segment as its high half.
    expectRefused(withArgument(LAST_INDEX, KernelArgument{BY_VALUE_KIND, POINTER_BYTES, 0, ""}),
                  "the shape parameters are 32-bit");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
