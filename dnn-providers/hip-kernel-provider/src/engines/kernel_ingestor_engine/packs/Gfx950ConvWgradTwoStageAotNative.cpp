// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvWgradGeometry.hpp"

/**
 * @file Gfx950ConvWgradTwoStageAotNative.cpp
 * @brief The hipkernel:Gfx950ConvWgradTwoStageAot engine's native half: the sibling of
 *        hipkernel:Gfx950ConvWgradAot for the 100 wgrad instances whose split-K reduction
 *        is a workspace fold rather than an atomic store. Every one is a two-launch KDP
 *        (`sources: {stage1, stage2}`), kept in its own engine because the descriptor
 *        loader requires one consistent launch set per engine
 *        (WGRAD_AOT_PROGRESS.md's "two engines" decision).
 *
 * Stage 1 is rocKE's implicit-GEMM wgrad kernel built with `two_stage=True`: every K-slice
 * atomic-adds its partial sum into an f32 scratch slab instead of storing once. Stage 2
 * (conv_wgrad_workspace_reduce.py) folds the `ws_replicas` slabs per group into the final
 * dW. The scratch is hipDNN's own workspace (workspaceBytes() below), not a self-managed
 * allocation: unlike Dgrad's internal sub-GEMM table, this scratch is sized by the graph's
 * own problem and is exactly the kind of thing the caller's workspace exists for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace wgrad = gfx950_conv_wgrad;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_conv_wgrad_two_stage_aot.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx950_conv_wgrad_two_stage_aot.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_conv_wgrad_two_stage_aot.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_conv_wgrad_two_stage_aot.dispatch";
constexpr std::string_view X_TOKEN = "gfx950_conv_wgrad_two_stage.x.uid";
constexpr std::string_view DY_TOKEN = "gfx950_conv_wgrad_two_stage.dy.uid";
constexpr std::string_view DW_TOKEN = "gfx950_conv_wgrad_two_stage.dw.uid";
constexpr std::string_view PROBLEM_TOKEN = "_gfx950_conv_wgrad_two_stage.problem";
constexpr size_t PROBLEM_TOKEN_SIZE = 18;

constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view TILE_M_FIELD = "tile_m";
constexpr std::string_view TILE_N_FIELD = "tile_n";
constexpr std::string_view TILE_K_FIELD = "tile_k";
constexpr std::string_view SPLIT_K_FIELD = "split_k";
constexpr std::string_view TWO_STAGE_FLAG_FIELD = "two_stage_flag";
constexpr std::string_view WS_REPLICAS_FIELD = "ws_replicas";
constexpr std::string_view VECTOR_SIZE_A_FIELD = "vector_size_a";
constexpr std::string_view VECTOR_SIZE_B_FIELD = "vector_size_b";

constexpr std::string_view STAGE1_LAUNCH = "stage1";
constexpr std::string_view STAGE2_LAUNCH = "stage2";
/// Stage 2's own block shape (conv_wgrad_workspace_reduce.py: one thread per output
/// element). Not a KMD field -- the instance carries the implicit-GEMM tile only -- so a
/// fixed, conservative square tile is used until a per-instance Stage 2 tile is wired.
constexpr int64_t STAGE2_TILE = 16;

struct Binding
{
    int64_t x;
    int64_t dy;
    int64_t dw;
};

struct MatchedProblem
{
    Binding binding;
    wgrad::Problem problem;
    wgrad::Geometry geometry;
    data_objects::DataType dtype;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

bool isDeviceTensor(const data_objects::TensorAttributes* tensor)
{
    return tensor != nullptr && !tensor->virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
           && !tensor->ragged_offset_tensor_uid().has_value() && tensor->alignment() >= 16
           && tensor->alignment() % 16 == 0
           && (tensor->data_type() == data_objects::DataType::HALF
               || tensor->data_type() == data_objects::DataType::BFLOAT16);
}

bool hasPackedNhwcStrides(const data_objects::TensorAttributes* tensor)
{
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->strides() == nullptr
       || tensor->dims()->size() != 4 || tensor->strides()->size() != 4)
    {
        return false;
    }
    const auto* dims = tensor->dims();
    const std::array<int64_t, 4> extents{dims->Get(0), dims->Get(1), dims->Get(2), dims->Get(3)};
    if(std::any_of(extents.begin(), extents.end(), [](int64_t e) { return e <= 0; })
       || !wgrad::wideProduct({extents[0], extents[1], extents[2], extents[3], 2}))
    {
        return false;
    }
    const std::array<int64_t, 4> expected{
        extents[1] * extents[2] * extents[3], 1, extents[1] * extents[3], extents[1]};
    for(flatbuffers::uoffset_t axis = 0; axis < 4; ++axis)
    {
        if(extents[axis] != 1 && tensor->strides()->Get(axis) != expected[axis])
        {
            return false;
        }
    }
    return true;
}

bool hasSpatialRankTwo(const flatbuffers::Vector<int64_t>* values)
{
    return values != nullptr && values->size() == 2;
}

/// Identical graph contract to hipkernel:Gfx950ConvWgradAot's (Gfx950ConvWgradAotNative.cpp)
/// -- both engines match the same ConvolutionWrwAttributes problem; they differ only in
/// which packaged instances they dispatch.
std::optional<MatchedProblem> matchProblem(const MatchContext& context)
{
    if(!context.graph.isValid() || context.graph.getGraph().is_override_shape_enabled()
       || !hipdnn_plugin_sdk::archMatches(
           context.deviceProperties.gcnArchName, "gfx950", hipdnn_plugin_sdk::ArchMatchMode::PREFIX)
       || context.deviceProperties.warpSize != 64 || context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionWrwAttributes
       || node.attributes() == nullptr || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& attributes = node.attributesAs<data_objects::ConvolutionWrwAttributes>();
    if(attributes.conv_mode() != data_objects::ConvMode::CROSS_CORRELATION
       || !hasSpatialRankTwo(attributes.pre_padding())
       || !hasSpatialRankTwo(attributes.post_padding()) || !hasSpatialRankTwo(attributes.stride())
       || !hasSpatialRankTwo(attributes.dilation()))
    {
        return std::nullopt;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < 2; ++axis)
    {
        if(attributes.pre_padding()->Get(axis) != attributes.post_padding()->Get(axis))
        {
            return std::nullopt;
        }
    }

    const Binding binding{
        attributes.x_tensor_uid(), attributes.dy_tensor_uid(), attributes.dw_tensor_uid()};
    const auto* x = findTensor(context, binding.x);
    const auto* dy = findTensor(context, binding.dy);
    const auto* dw = findTensor(context, binding.dw);
    if(binding.x == binding.dy || binding.x == binding.dw || binding.dy == binding.dw
       || !isDeviceTensor(x) || !isDeviceTensor(dy) || !isDeviceTensor(dw)
       || !hasPackedNhwcStrides(x) || !hasPackedNhwcStrides(dy) || !hasPackedNhwcStrides(dw)
       || x->data_type() != dy->data_type() || x->data_type() != dw->data_type())
    {
        return std::nullopt;
    }
    const auto* xDims = x->dims();
    const auto* dwDims = dw->dims();
    const auto* dyDims = dy->dims();
    const auto channelsPerGroup = dwDims->Get(1);
    if(xDims->Get(1) % channelsPerGroup != 0)
    {
        return std::nullopt;
    }
    const auto groups = xDims->Get(1) / channelsPerGroup;
    const wgrad::Problem problem{xDims->Get(0),
                                 xDims->Get(1),
                                 dwDims->Get(0),
                                 xDims->Get(2),
                                 xDims->Get(3),
                                 dwDims->Get(2),
                                 dwDims->Get(3),
                                 attributes.stride()->Get(0),
                                 attributes.stride()->Get(1),
                                 attributes.pre_padding()->Get(0),
                                 attributes.pre_padding()->Get(1),
                                 attributes.dilation()->Get(0),
                                 attributes.dilation()->Get(1),
                                 groups};
    const auto geometry = wgrad::deriveGeometry(problem);
    if(!geometry || dyDims->Get(0) != problem.n || dyDims->Get(1) != problem.k
       || dyDims->Get(2) != geometry->ho || dyDims->Get(3) != geometry->wo)
    {
        return std::nullopt;
    }
    return MatchedProblem{binding, problem, *geometry, x->data_type()};
}

std::vector<int64_t> problemTokenValues(const MatchedProblem& matched)
{
    const auto& p = matched.problem;
    return {matched.binding.x,
            matched.binding.dy,
            matched.binding.dw,
            p.n,
            p.c,
            p.k,
            p.hi,
            p.wi,
            p.y,
            p.x,
            p.strideH,
            p.strideW,
            p.padH,
            p.padW,
            p.dilationH,
            p.dilationW,
            p.groups,
            static_cast<int64_t>(matched.dtype)};
}

std::optional<MatchedProblem> matchedFromBound(const MatchContext& context,
                                               const BoundTokens& bound)
{
    static const std::string s_problemToken(PROBLEM_TOKEN);
    const auto it = bound.find(s_problemToken);
    const auto* values
        = it == bound.end() ? nullptr : std::get_if<std::vector<int64_t>>(&it->second);
    if(values == nullptr || values->size() != PROBLEM_TOKEN_SIZE)
    {
        return matchProblem(context);
    }
    const auto& v = *values;
    const wgrad::Problem problem{
        v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16]};
    const auto geometry = wgrad::deriveGeometry(problem);
    if(!geometry)
    {
        return std::nullopt;
    }
    return MatchedProblem{
        Binding{v[0], v[1], v[2]}, problem, *geometry, static_cast<data_objects::DataType>(v[17])};
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchProblem(context);
    if(!matched)
    {
        return std::nullopt;
    }
    return BoundTokens{{std::string(X_TOKEN), matched->binding.x},
                       {std::string(DY_TOKEN), matched->binding.dy},
                       {std::string(DW_TOKEN), matched->binding.dw},
                       {std::string(PROBLEM_TOKEN), problemTokenValues(*matched)}};
}

/// See the single-launch engine's shardDtypeCriterion: only dtype distinguishes a shard
/// at the graph level, pipeline is a kernel-level choice.
template <data_objects::DataType Dtype>
bool shardDtypeCriterion(const MatchContext& context, const BoundTokens& bound)
{
    const auto matched = matchedFromBound(context, bound);
    return matched.has_value() && matched->dtype == Dtype;
}

struct KernelMetadata
{
    const KernelDefinition& kernel;

    std::optional<int64_t> intField(std::string_view field) const
    {
        const auto it = kernel.metadata.find(std::string(field));
        const auto* value
            = it == kernel.metadata.end() ? nullptr : std::get_if<int64_t>(&it->second);
        return value == nullptr ? std::nullopt : std::optional<int64_t>(*value);
    }

    const std::string* stringField(std::string_view field) const
    {
        const auto it = kernel.metadata.find(std::string(field));
        return it == kernel.metadata.end() ? nullptr : std::get_if<std::string>(&it->second);
    }
};

const char* dtypeName(data_objects::DataType dtype)
{
    return dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
}

struct TwoStagePlan
{
    wgrad::ImplicitGemmTwoStageKernargs stage1Values;
    int64_t tileM;
    int64_t tileN;
    int64_t wsReplicas;
    int64_t wsBytes;
};

/// This kernel's two-stage plan on @p matched, or nullopt when its baked knobs do not
/// admit the problem -- every instance in this engine's catalog carries two_stage_flag=1
/// (Phase 2's enumeration split on exactly that field), so a kernel this declines belongs
/// to the sibling single-launch engine instead.
std::optional<TwoStagePlan> twoStagePlan(const MatchedProblem& matched, const KernelDefinition& kernel)
{
    const KernelMetadata md{kernel};
    const auto* dtype = md.stringField(DTYPE_FIELD);
    if(dtype == nullptr || *dtype != dtypeName(matched.dtype))
    {
        return std::nullopt;
    }
    if(md.intField(TWO_STAGE_FLAG_FIELD).value_or(0) == 0)
    {
        return std::nullopt;
    }
    const auto tileM = md.intField(TILE_M_FIELD);
    const auto tileN = md.intField(TILE_N_FIELD);
    const auto tileK = md.intField(TILE_K_FIELD);
    const auto splitK = md.intField(SPLIT_K_FIELD);
    const auto wsReplicas = md.intField(WS_REPLICAS_FIELD).value_or(1);
    if(!tileM || !tileN || !tileK || !splitK || *splitK <= 1)
    {
        return std::nullopt;
    }
    // Same build-time-vector-width discipline as the single-launch engine's
    // implicitGemmPlan (Gfx950ConvWgradAotNative.cpp): the dY/X free-axis load width is
    // baked against the probe's kpg/cpg, not read from a kernarg, so a real problem's
    // per-group kpg/cpg must themselves be multiples of it. 0 means "derived by the
    // builder from the group_merge run", which needs no check here.
    const auto vectorSizeA = md.intField(VECTOR_SIZE_A_FIELD).value_or(0);
    const auto vectorSizeB = md.intField(VECTOR_SIZE_B_FIELD).value_or(0);
    if((vectorSizeA > 0 && matched.geometry.kpg % vectorSizeA != 0)
       || (vectorSizeB > 0 && matched.geometry.cpg % vectorSizeB != 0))
    {
        return std::nullopt;
    }
    const auto stage1 = wgrad::implicitGemmTwoStageKernargs(
        matched.problem, matched.geometry, *tileM, *tileN, *tileK, *splitK, wsReplicas);
    if(!stage1)
    {
        return std::nullopt;
    }
    const auto wgN = wgrad::boundedProduct({matched.problem.y, matched.problem.x, matched.geometry.cpg});
    const auto wsBytes = wgrad::boundedProduct(
        {matched.problem.groups, wsReplicas, matched.geometry.kpg, wgN.value_or(0), 4});
    if(!wsBytes)
    {
        return std::nullopt;
    }
    return TwoStagePlan{*stage1, *tileM, *tileN, wsReplicas, *wsBytes};
}

bool kernelMatches(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
{
    const auto matched = matchedFromBound(context, bound);
    return matched && twoStagePlan(*matched, kernel).has_value();
}

double scoreKernel(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
{
    // Deterministic fallback only -- see Gfx950ConvWgradAotNative.cpp's scoreKernel.
    const auto matched = matchedFromBound(context, bound);
    if(!matched)
    {
        return -1.0;
    }
    const auto plan = twoStagePlan(*matched, kernel);
    if(!plan)
    {
        return -1.0;
    }
    return static_cast<double>(plan->tileM) * static_cast<double>(plan->tileN);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

std::vector<KernelArgument> stage1Signature()
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    const auto addPtr = [&](std::string_view name) {
        offset = (offset + sizeof(void*) - 1) / sizeof(void*) * sizeof(void*);
        signature.push_back({"global_buffer", static_cast<uint32_t>(sizeof(void*)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(void*));
    };
    const auto addI32 = [&](std::string_view name) {
        offset = (offset + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
        signature.push_back({"by_value", static_cast<uint32_t>(sizeof(int32_t)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(int32_t));
    };
    addPtr("dY");
    addPtr("X");
    addPtr("dW");
    addI32("dY_bytes");
    addI32("X_bytes");
    addI32("dW_bytes");
    for(const auto name : wgrad::IMPLICIT_GEMM_TWO_STAGE_KERNARG_NAMES)
    {
        if(name == "ws_ptr")
        {
            // A pointer argument amid the scalar block: emitted positionally like the
            // leading three, but its offset follows whatever i32s preceded it.
            offset = (offset + sizeof(void*) - 1) / sizeof(void*) * sizeof(void*);
            signature.push_back(
                {"global_buffer", static_cast<uint32_t>(sizeof(void*)), offset, std::string(name)});
            offset += static_cast<uint32_t>(sizeof(void*));
            continue;
        }
        addI32(name);
    }
    return signature;
}

std::vector<KernelArgument> stage2Signature()
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    const auto addPtr = [&](std::string_view name) {
        offset = (offset + sizeof(void*) - 1) / sizeof(void*) * sizeof(void*);
        signature.push_back({"global_buffer", static_cast<uint32_t>(sizeof(void*)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(void*));
    };
    const auto addI32 = [&](std::string_view name) {
        offset = (offset + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
        signature.push_back({"by_value", static_cast<uint32_t>(sizeof(int32_t)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(int32_t));
    };
    addPtr("ws_ptr");
    addPtr("dw_ptr");
    for(const auto name : wgrad::STAGE2_SCALAR_KERNARG_NAMES)
    {
        addI32(name);
    }
    return signature;
}

class PreparedConvWgradTwoStage : public PreparedDispatch
{
public:
    PreparedConvWgradTwoStage(IngestorKernelCode stage1,
                             IngestorKernelCode stage2,
                             Binding binding,
                             std::array<int64_t, 3> bytes,
                             TwoStagePlan plan)
        : _stage1(std::move(stage1))
        , _stage2(std::move(stage2))
        , _binding(binding)
        , _bytes(bytes)
        , _plan(std::move(plan))
    {
    }

    compilation::IRunnableKernel& stage1ForStream(hipStream_t stream) const
    {
        return _stage1.kernelForStream(stream);
    }
    compilation::IRunnableKernel& stage2ForStream(hipStream_t stream) const
    {
        return _stage2.kernelForStream(stream);
    }
    const Binding& binding() const
    {
        return _binding;
    }
    const std::array<int64_t, 3>& bytes() const
    {
        return _bytes;
    }
    const TwoStagePlan& plan() const
    {
        return _plan;
    }

private:
    IngestorKernelCode _stage1;
    IngestorKernelCode _stage2;
    Binding _binding;
    std::array<int64_t, 3> _bytes;
    TwoStagePlan _plan;
};

class ConvWgradTwoStageDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit ConvWgradTwoStageDispatchHandler(const compilation::KpackKernelLoader& loader)
        : _loader(loader)
    {
    }

    /// The f32 scratch Stage 1 atomic-adds into and Stage 2 reads: hipDNN's own workspace,
    /// not a self-managed allocation (see the file doc comment).
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto matched = matchedFromBound(context, bound);
        const auto plan = matched ? twoStagePlan(*matched, kernel) : std::nullopt;
        return plan ? static_cast<size_t>(plan->wsBytes) : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto matched = matchedFromBound(context, bound);
        const auto plan = matched ? twoStagePlan(*matched, kernel) : std::nullopt;
        if(!matched || !plan)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_wgrad_two_stage graph or packaged kernel does not satisfy the "
                "convolution contract");
        }
        const auto stage1Geom = wgrad::implicitGemmLaunchGeometry(matched->problem,
                                                                   matched->geometry,
                                                                   plan->tileM,
                                                                   plan->tileN,
                                                                   plan->stage1Values[43],
                                                                   256);
        const auto stage2Geom
            = wgrad::stage2LaunchGeometry(matched->geometry, matched->problem, STAGE2_TILE, STAGE2_TILE);
        if(!stage1Geom || !stage2Geom)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_wgrad_two_stage launch grid exceeds device limits");
        }
        auto stage1Code = buildIngestorKernelCode(
            _loader, context, launchKernelDefinition(kernel, std::string(STAGE1_LAUNCH)), stage1Signature());
        stage1Code.setBlockSize(stage1Geom->blockX, 1, 1);
        stage1Code.setGridSize(stage1Geom->gridX, stage1Geom->gridY, stage1Geom->gridZ);

        auto stage2Code = buildIngestorKernelCode(
            _loader, context, launchKernelDefinition(kernel, std::string(STAGE2_LAUNCH)), stage2Signature());
        stage2Code.setBlockSize(stage2Geom->blockX, 1, 1);
        stage2Code.setGridSize(stage2Geom->gridX, stage2Geom->gridY, stage2Geom->gridZ);

        return std::make_unique<PreparedConvWgradTwoStage>(std::move(stage1Code),
                                                           std::move(stage2Code),
                                                           matched->binding,
                                                           matched->geometry.tensorBytes,
                                                           *plan);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& twoStage = dynamic_cast<const PreparedConvWgradTwoStage&>(prepared);
        const auto& binding = twoStage.binding();
        void* dy = hipdnn_plugin_sdk::findDeviceBuffer(binding.dy, deviceBuffers, numDeviceBuffers).ptr;
        void* x = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers).ptr;
        void* dw = hipdnn_plugin_sdk::findDeviceBuffer(binding.dw, deviceBuffers, numDeviceBuffers).ptr;
        if(dy == nullptr || x == nullptr || dw == nullptr || workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_wgrad_two_stage requires device buffers and a workspace");
        }
        const auto& plan = twoStage.plan();
        const auto& bytes = twoStage.bytes();
        const auto stream = handle.getStream();

        // conv_implicit_gemm_wgrad.py's two-stage contract: Stage 1 f32-atomic-adds into
        // the scratch, it never stores directly, so "the caller must zero the scratch
        // first" (same discipline as the single-launch engine's split-K atomic path, see
        // Gfx950ConvWgradAotNative.cpp::launch) -- stale residue from a prior workspace
        // lease would otherwise accumulate into this launch's result.
        const hipError_t memsetErr
            = hipMemsetAsync(workspace, 0, static_cast<size_t>(plan.wsBytes), stream);
        if(memsetErr != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string("gfx950_conv_wgrad_two_stage: failed to zero the f32 scratch "
                            "workspace before Stage 1, error: ")
                    + hipGetErrorString(memsetErr));
        }

        // Stage 1: dY, X, dW(unused store target, kept in the ABI for layout parity), the
        // byte triple, every scalar but ws_ptr, then ws_ptr (the caller workspace) at its
        // ABI position.
        std::array<int32_t, 3> byteArgs{
            static_cast<int32_t>(bytes[0]), static_cast<int32_t>(bytes[1]), static_cast<int32_t>(bytes[2])};
        std::vector<void*> stage1Params{&dy, &x, &dw, &byteArgs[0], &byteArgs[1], &byteArgs[2]};
        for(size_t i = 0; i < 40; ++i)
        {
            stage1Params.push_back(const_cast<int32_t*>(&plan.stage1Values[i]));
        }
        stage1Params.push_back(&workspace);
        for(size_t i = 41; i < plan.stage1Values.size(); ++i)
        {
            stage1Params.push_back(const_cast<int32_t*>(&plan.stage1Values[i]));
        }
        twoStage.stage1ForStream(stream).launchWithParams(stream, stage1Params.data());

        // Stage 2: ws_ptr, dw_ptr, wg_M, wg_N, ws_bytes, dw_bytes, groups.
        const int32_t wgM = static_cast<int32_t>(plan.stage1Values[18]);
        const int32_t wgN = static_cast<int32_t>(plan.stage1Values[19]);
        const int32_t wsBytes = static_cast<int32_t>(plan.wsBytes);
        const int32_t dwBytes = static_cast<int32_t>(bytes[2]);
        const int32_t groups = static_cast<int32_t>(plan.stage1Values[13]);
        std::vector<void*> stage2Params{&workspace,
                                        &dw,
                                        const_cast<int32_t*>(&wgM),
                                        const_cast<int32_t*>(&wgN),
                                        const_cast<int32_t*>(&wsBytes),
                                        const_cast<int32_t*>(&dwBytes),
                                        const_cast<int32_t*>(&groups)};
        twoStage.stage2ForStream(stream).launchWithParams(stream, stage2Params.data());
    }

private:
    const compilation::KpackKernelLoader& _loader;
};

const ConvWgradTwoStageDispatchHandler& dispatchHandler()
{
    static const compilation::KpackKernelLoader s_loader(gfx950ConvWgradTwoStageAotKpackModuleCache());
    static const ConvWgradTwoStageDispatchHandler s_handler(s_loader);
    return s_handler;
}

} // namespace

compilation::KpackModuleCache& gfx950ConvWgradTwoStageAotKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void registerGfx950ConvWgradTwoStageAotSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());

    // See the single-launch engine's registration comment: these are GraphCriterionFn
    // (not graph_match's GraphMatchFn) and only dtype distinguishes a shard.
    for(const auto* shard : {"bf16_compv3", "bf16_compv4", "bf16_mem"})
    {
        scope.add(std::string("hipkernel.gfx950_conv_wgrad_two_stage_aot.") + shard + "_match",
                  &shardDtypeCriterion<data_objects::DataType::BFLOAT16>);
    }
    for(const auto* shard : {"fp16_compv3", "fp16_compv4", "fp16_mem"})
    {
        scope.add(std::string("hipkernel.gfx950_conv_wgrad_two_stage_aot.") + shard + "_match",
                  &shardDtypeCriterion<data_objects::DataType::HALF>);
    }
}

void resetGfx950ConvWgradTwoStageAotModuleCache()
{
    gfx950ConvWgradTwoStageAotKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
