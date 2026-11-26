// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdGeometry.hpp"

/**
 * @file Gfx950ConvFwdAotCommon.hpp
 * @brief What the gfx950 forward-convolution AOT engines share: the convolution's graph
 *        rules, the matched problem and its token, the decoding of an AOT instance from
 *        its metadata, the launch it admits, the kernel ABI, and the dispatch handler
 *        (single launch, split-C conv pass and fold, batch split).
 *
 * hipkernel:Gfx950ConvFwdAot (Gfx950ConvFwdNative.cpp) serves a lone convolution node.
 * hipkernel:Gfx950ConvFwdBiasAot (Gfx950ConvFwdBiasNative.cpp) serves a convolution whose
 * virtual output feeds a pointwise ADD with a per-output-channel bias, and runs the same
 * instances built with a bias epilogue. What differs reaches the shared code through a
 * ConvFwdAotEngine: the engine's graph match, its token names, and whether its kernels
 * take the bias. A matched problem carries the bias in its Binding; the instances,
 * launches and kernel arguments are otherwise the same.
 */

// The provider's plugin handle (core/Handle.hpp), named here only as the handler's handle type.
struct Handle;

namespace hip_kernel_provider::compilation
{
class KpackKernelLoader;
}

namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd_aot
{

namespace conv = gfx950_conv_fwd;

/// The tensors a matched graph binds. @p bias is set by an engine whose kernels add a
/// per-output-channel bias (dims (1, K, 1, 1) or (K, 1, 1), contiguous along K) to the
/// convolution before storing y; y is then the ADD's output, not the convolution's.
struct Binding
{
    int64_t x;
    int64_t w;
    int64_t y;
    std::optional<int64_t> bias = std::nullopt;
};

struct MatchedProblem
{
    Binding binding;
    conv::Problem problem;
    /// The whole problem's geometry, or, when a tensor's byte count passes the i32
    /// launch ABI, the geometry of its largest batch chunk (wholeGeometry false).
    conv::Geometry geometry;
    hipdnn_flatbuffers_sdk::data_objects::DataType dtype;
    /// conv::nongroupedBatchChunks of the problem: empty when one launch serves it.
    std::vector<int64_t> batchChunks;
    /// Whether geometry is the whole problem's, so a single-launch kernel can run it.
    bool wholeGeometry = true;
};

/// The AOT problem block of one family's ABI, the arguments after the leading six.
using ConvKernargs = std::variant<conv::ImplicitGemmKernargs, conv::DirectKernargs>;

/// One launch of a batch-split kernel: the images it computes, after those of the
/// launches before it, as a problem of its own.
struct BatchChunkLaunch
{
    int64_t images;
    conv::LaunchGeometry geometry;
    ConvKernargs kernargs;
    std::array<int32_t, 3> tensorBytes;
};

/// Everything a kernel needs at launch beyond the bound device pointers. A split-C
/// direct kernel launches twice: its conv pass writes fp32 partial sums into a workspace
/// of workspaceBytes, and the fold (reduce, same kernargs) sums them into the output.
/// A batch-split kernel launches once per entry of batchChunks, each on a window of the
/// activation and output; geometry and kernargs are then the first chunk's.
struct ConvLaunch
{
    conv::LaunchGeometry geometry;
    ConvKernargs kernargs;
    std::optional<conv::LaunchGeometry> reduce = std::nullopt;
    int64_t workspaceBytes = 0;
    // `= {}` keeps the two-field aggregate inits clear of -Wmissing-field-initializers.
    // NOLINTNEXTLINE(readability-redundant-member-init)
    std::vector<BatchChunkLaunch> batchChunks = {};
};

/// One engine over the shared code: how it matches a graph, the tokens it binds, and
/// the name its errors carry. Held as strings, not views: the kernel matcher looks the
/// problem token up for every kernel, and an unordered_map lookup takes a std::string.
struct ConvFwdAotEngine
{
    /// The prefix of every error this engine raises ("gfx950_conv_fwd").
    std::string label;
    std::string xToken;
    std::string wToken;
    std::string yToken;
    /// The bias token; empty for an engine without a bias.
    std::string biasToken;
    /// The whole matched problem, so a kernel matcher reads it from what graph_match
    /// bound instead of walking the graph again for every kernel: problemTokenValues().
    /// Private (`_`): the graph features already describe the problem, so this is no
    /// feature.
    std::string problemToken;
    /// The engine's graph match. Its Binding holds a bias exactly when biasToken is set.
    std::optional<MatchedProblem> (*matchProblem)(const hipdnn_plugin_sdk::ingestor::MatchContext&);
};

// ---- Graph match ----

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    findTensor(const hipdnn_plugin_sdk::ingestor::MatchContext& context, int64_t uid);

/// Whether @p context is a valid gfx950, wave64 graph of @p nodeCount nodes without
/// shape overrides: what every engine of this family requires before it reads a node.
bool servesDeviceAndGraph(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                          unsigned int nodeCount);

/// Whether @p output is packed NHWK with @p matched's inferred convolution dimensions.
/// hipDNN's frontend infers a sole, non-virtual graph output's dims/strides during
/// Graph::build()'s validate() pass, strictly before engine matching, so this already
/// holds by match time -- a graph match relies on it to decline a graph whose output
/// isn't NHWK instead of matching it and failing later in prepare().
bool outputMatches(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* output,
                   const MatchedProblem& matched);

/// The match of @p problem: its whole geometry, or, for a batch rocKE runs in pieces
/// (conv::nongroupedBatchChunks) whose tensors pass the i32 launch ABI, its largest
/// chunk's, which only a batch-split launch reads.
std::optional<MatchedProblem> matchedOf(const Binding& binding,
                                        const conv::Problem& problem,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType dtype);

/// The convolution rules every engine of this family applies to its convolution node
/// (@p attributes, whose node computes in FLOAT): cross-correlation, 2-D symmetric
/// padding, packed NHWC/KYXC device tensors of one 16-bit dtype, groups encoded by the
/// filter, and a packed NHWK output. @p yUid is the tensor the kernel stores, the
/// convolution's own output or the output of an operation fused after it; @p bias is
/// carried into the Binding unchecked.
std::optional<MatchedProblem> matchConvolution(
    const hipdnn_plugin_sdk::ingestor::MatchContext& context,
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& attributes,
    int64_t yUid,
    std::optional<int64_t> bias = std::nullopt);

/// The engine's problem token value: the binding (x, w, y), the problem in
/// conv::Problem's field order, the dtype, then the bias uid when the binding has one.
/// The geometry is derived from the problem again on reading.
std::vector<int64_t> problemTokenValues(const MatchedProblem& matched);

/// The problem the engine's graph match resolved for @p context, read back from
/// @p bound: the SDK passes every kernel matcher what graph_match bound for that same
/// context. Falls back to matching the graph when the token is absent, as for a caller
/// with no graph match.
std::optional<MatchedProblem>
    matchedFromBound(const ConvFwdAotEngine& engine,
                     const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                     const hipdnn_plugin_sdk::ingestor::BoundTokens& bound);

/// The engine's graph match as the SDK calls it: its tensor tokens and problem token.
std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>
    graphMatches(const ConvFwdAotEngine& engine,
                 const hipdnn_plugin_sdk::ingestor::MatchContext& context);

// ---- Kernel match ----

/// The KernelPrepareFn both engines register under their kernel matcher's symbol: the
/// SDK runs it once per kernel when the engine loads, so matching a graph reads the
/// decoded instance instead of parsing about fifty metadata fields per kernel.
std::shared_ptr<const void> aotPrepare(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// The launch of @p kernel on @p matched, or nullopt when the kernel does not serve it:
/// its dtype, its bias (a kernel carries one exactly when the binding does), and the
/// capability check of its family.
std::optional<ConvLaunch>
    aotKernelFits(const MatchedProblem& matched,
                  const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

bool aotKernelMatches(const ConvFwdAotEngine& engine,
                      const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                      const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                      const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

double aotScore(const ConvFwdAotEngine& engine,
                const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

std::vector<bool> aotCandidateFilter(
    const ConvFwdAotEngine& engine,
    const hipdnn_plugin_sdk::ingestor::MatchContext& context,
    const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
    const std::vector<const hipdnn_plugin_sdk::ingestor::KernelDefinition*>& candidates);

// ---- Dispatch ----

/// The kernel ABI buildIngestorKernelCode verifies the loaded symbol against: rocKE's AOT
/// conv kernels open with (A*, B*, D*, A_bytes:i32, B_bytes:i32, D_bytes:i32) and follow
/// with the family's problem block as i32s (rocke/library/kernels/common/conv_abi.py). A
/// kernel built with a bias appends (Bias*, Bias_bytes:i32) after every other argument.
const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>&
    kernelSignature(const ConvKernargs& kernargs, bool bias);

/// The dispatch handler of one engine's packs. A single-launch handler (no @p
/// launchNames) runs each kernel's `kernel_source`; the split-C handler declares the
/// launches "conv" and "reduce" of its kernels' `sources` and runs both, the conv pass
/// into the workspace and the fold from it into the output, in that order on one stream.
/// Process-lifetime: the caller keeps it alive while it is registered, and @p engine and
/// @p loader for as long as the handler lives.
std::unique_ptr<hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>>
    makeDispatchHandler(const ConvFwdAotEngine& engine,
                        const compilation::KpackKernelLoader& loader,
                        std::set<std::string> launchNames = {});

/// The launches of a split-C kernel's `sources` (make_conv_aot_config.SPLIT_C_LAUNCHES).
inline constexpr const char* SPLIT_C_CONV_LAUNCH = "conv";
inline constexpr const char* SPLIT_C_REDUCE_LAUNCH = "reduce";

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd_aot

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
