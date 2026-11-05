// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvWgradGeometry.hpp"

/// @file Gfx950ConvWgradNative.cpp
/// hipkernel:Gfx950ConvWgradAot: shape-generic AOT instances of rocKE's backward-weight
/// (wgrad) convolution kernels. Each kernel is one rocKE wgrad KernelIdentity: an
/// implicit-GEMM tile (kernel_family 0), one of rocKE's seven direct wgrad pipelines
/// (kernel_family 1), or a helper (kernel_family 2, never matched): the partial-slot fold
/// the five folded pipelines run after their pass, or the Stage 2 fold a two-stage
/// implicit-GEMM instance runs after its own; a dispatch loads it beside its instance.
/// Matching is the launch plan of Gfx950ConvWgradGeometry.hpp: a kernel serves a graph
/// exactly when it has a plan there, and the dispatch runs that plan. Mirrors
/// rocke/library/builders/common/convolution_backward_filter.py.
///
/// The graph's dW dtype picks the family: an implicit-GEMM binary stores dW in the
/// activation dtype, every direct pipeline accumulates an fp32 dW.
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace wgrad = gfx950_conv_wgrad;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_conv_wgrad_aot.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_conv_wgrad_aot.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_conv_wgrad_aot.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_conv_wgrad_aot.dispatch";
constexpr std::string_view X_TOKEN = "gfx950_conv_wgrad.x.uid";
constexpr std::string_view DY_TOKEN = "gfx950_conv_wgrad.dy.uid";
constexpr std::string_view DW_TOKEN = "gfx950_conv_wgrad.dw.uid";
/// The whole matched problem, so a kernel matcher reads it from what graph_match bound
/// instead of walking the graph again for every kernel: problemTokenValues() order.
/// Private (`_`): the graph features already describe the problem, so this is no feature.
constexpr std::string_view PROBLEM_TOKEN = "_gfx950_conv_wgrad.problem";
constexpr size_t PROBLEM_TOKEN_SIZE = 19;

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
    /// dW is fp32 (served by the direct pipelines) rather than the activation dtype
    /// (served by implicit GEMM).
    bool dwFloat;

    /// dY, X and dW storage sizes in bytes.
    std::array<int64_t, 3> tensorBytes() const
    {
        return {geometry.tensorBytes[0],
                geometry.tensorBytes[1],
                geometry.dwElements * (dwFloat ? 4 : 2)};
    }
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
           && tensor->alignment() % 16 == 0;
}

bool isHalfType(data_objects::DataType dtype)
{
    return dtype == data_objects::DataType::HALF || dtype == data_objects::DataType::BFLOAT16;
}

/// hipDNN [N, C, H, W] dims stored NHWC: X [N, C, Hi, Wi] as NHWC, dY [N, K, Ho, Wo] as
/// NHWK, dW [K, C/G, Y, X] as KYXC.
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
       || !wgrad::wideProduct({extents[0], extents[1], extents[2], extents[3], 4}))
    {
        return false;
    }
    // The byte-size guard above bounds every multiplication here.
    const std::array<int64_t, 4> expected{
        extents[1] * extents[2] * extents[3], 1, extents[1] * extents[3], extents[1]};
    for(flatbuffers::uoffset_t axis = 0; axis < 4; ++axis)
    {
        // A unit-extent axis never contributes to an address, whatever its stride.
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

/// The graph contract: one ConvolutionWrwAttributes node on gfx950 (wave64) with FLOAT
/// accumulation, cross-correlation, 2-D symmetric padding, three distinct device tensors
/// in packed NHWC -- X and dY of one 16-bit dtype, dW of that dtype or fp32 -- groups =
/// X.C / dW.dims[1] in [1, 65535] dividing K, and dY's dims equal to the forward output of
/// X (whose dims define the problem: several Hi give one Ho under a stride).
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
       || !isHalfType(x->data_type()) || dy->data_type() != x->data_type()
       || (dw->data_type() != x->data_type() && dw->data_type() != data_objects::DataType::FLOAT))
    {
        return std::nullopt;
    }
    const auto* xDims = x->dims();
    const auto* dwDims = dw->dims();
    const auto* dyDims = dy->dims();
    // hipDNN carries no groups attribute: a filter holding C / groups channels encodes it.
    // The layout checks above already made every extent positive.
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
                                 groups,
                                 x->data_type() == data_objects::DataType::HALF
                                     ? wgrad::ElementType::FP16
                                     : wgrad::ElementType::BF16};
    // groups in [1, 65535] dividing C and K, a positive dY, every byte count in i32.
    const auto geometry = wgrad::deriveGeometry(problem);
    if(!geometry || dyDims->Get(0) != problem.n || dyDims->Get(1) != problem.k
       || dyDims->Get(2) != geometry->ho || dyDims->Get(3) != geometry->wo)
    {
        return std::nullopt;
    }
    return MatchedProblem{binding,
                          problem,
                          *geometry,
                          x->data_type(),
                          dw->data_type() == data_objects::DataType::FLOAT};
}

/// PROBLEM_TOKEN's value: the binding, the problem in wgrad::Problem's field order but its
/// dtype, then the activation dtype and the dW-is-fp32 flag. The geometry is derived from
/// the problem again on reading.
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
            static_cast<int64_t>(matched.dtype),
            matched.dwFloat ? 1 : 0};
}

/// The problem graphMatches resolved for @p context, read back from @p bound: the SDK
/// passes every kernel matcher what graph_match bound for that same context. Falls back
/// to matching the graph when the token is absent, as for a caller with no graph match.
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
    const auto dtype = static_cast<data_objects::DataType>(v[17]);
    if(!isHalfType(dtype))
    {
        return std::nullopt;
    }
    const wgrad::Problem problem{v[3],
                                 v[4],
                                 v[5],
                                 v[6],
                                 v[7],
                                 v[8],
                                 v[9],
                                 v[10],
                                 v[11],
                                 v[12],
                                 v[13],
                                 v[14],
                                 v[15],
                                 v[16],
                                 dtype == data_objects::DataType::HALF ? wgrad::ElementType::FP16
                                                                       : wgrad::ElementType::BF16};
    const auto geometry = wgrad::deriveGeometry(problem);
    if(!geometry)
    {
        return std::nullopt;
    }
    return MatchedProblem{Binding{v[0], v[1], v[2]}, problem, *geometry, dtype, v[18] != 0};
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

/// A kernel's metadata as parseAotInstance reads it.
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

/// The instance @p kernel describes, when it is a kpack kernel whose metadata is a
/// complete instance of one family (wgrad::parseAotInstance).
std::optional<wgrad::AotInstance> aotInstance(const KernelDefinition& kernel)
{
    if(kernel.source.kind != KernelSourceKind::KPACK)
    {
        return std::nullopt;
    }
    return wgrad::parseAotInstance(KernelMetadata{kernel});
}

/// Where a helper is looked up: the same pack, the same archive, the instance's dtype, the
/// helper variant and what it bakes -- the fold's taps (gfx950_conv_wgrad_aot_fold_fields)
/// or the Stage 2 fold's ws_replicas (gfx950_conv_wgrad_aot_ws_reduce_fields).
struct HelperKey
{
    std::string archive;
    DescriptorId packId;
    wgrad::ElementType dtype;
    wgrad::DirectVariant variant;
    int64_t parameter;

    bool operator<(const HelperKey& other) const
    {
        return std::tie(archive, packId, dtype, variant, parameter)
               < std::tie(other.archive, other.packId, other.dtype, other.variant, other.parameter);
    }
};

/// The archive @p kernel's code object is read from, as buildIngestorKernelCode resolves
/// it; empty for a definition built outside a descriptor tree.
std::string archiveOf(const KernelDefinition& kernel)
{
    if(kernel.source.library.empty())
    {
        return {};
    }
    std::error_code ignored;
    const auto origin = std::filesystem::weakly_canonical(kernel.originDirectory, ignored);
    return std::filesystem::weakly_canonical(origin / kernel.source.library, ignored)
        .generic_string();
}

/// Every fold helper the engine has loaded, filed by HelperKey. The kernel prepare hook
/// files each one when the engine's shared state is built, which precedes any match and
/// any dispatch; the folded pipelines' matcher and dispatch look their fold up here, since
/// no native hook sees the catalog. A key holding two different kernels is ambiguous and
/// serves nothing.
class HelperRegistry
{
public:
    void add(HelperKey key, const KernelDefinition& kernel)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        auto& entries = _helpers[std::move(key)];
        for(auto& entry : entries)
        {
            if(entry.kernelId == kernel.kernelId)
            {
                entry = stripped(kernel);
                return;
            }
        }
        entries.push_back(stripped(kernel));
    }

    std::optional<KernelDefinition> find(const HelperKey& key) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        const auto it = _helpers.find(key);
        if(it == _helpers.end() || it->second.size() != 1)
        {
            return std::nullopt;
        }
        return it->second.front();
    }

private:
    static KernelDefinition stripped(const KernelDefinition& kernel)
    {
        auto copy = kernel;
        copy.prepared = nullptr;
        copy.preparedBy = nullptr;
        return copy;
    }

    mutable std::mutex _mutex;
    std::map<HelperKey, std::vector<KernelDefinition>> _helpers;
};

HelperRegistry& helperRegistry()
{
    static HelperRegistry s_registry;
    return s_registry;
}

/// The helper a plan step of @p instance launches on @p problem: a folded instance's fold,
/// whose taps follow the problem's filter and (per-channel folds) cpg, which the
/// capability match pins to the instance's own; a two-stage instance's Stage 2, keyed by
/// its ws_replicas.
std::optional<KernelDefinition> findHelper(const KernelDefinition& kernel,
                                           const wgrad::AotInstance& instance,
                                           wgrad::PlanKernel helper,
                                           const wgrad::Problem& problem,
                                           const wgrad::Geometry& geometry)
{
    if(helper == wgrad::PlanKernel::FOLD && instance.direct)
    {
        return helperRegistry().find(HelperKey{
            archiveOf(kernel),
            kernel.packId,
            instance.dtype,
            wgrad::DirectVariant::FOLD,
            wgrad::foldTaps(instance.direct->variant, problem.y, problem.x, geometry.cpg)});
    }
    if(helper == wgrad::PlanKernel::WS_REDUCE && instance.implicit)
    {
        return helperRegistry().find(HelperKey{archiveOf(kernel),
                                               kernel.packId,
                                               instance.dtype,
                                               wgrad::DirectVariant::WS_REDUCE,
                                               wgrad::effectiveWsReplicas(*instance.implicit)});
    }
    return std::nullopt;
}

/// What hipkernel:Gfx950ConvWgradAot reads of one kernel on every graph, decoded from its
/// metadata once: its instance (nullopt for metadata no reviewed launch contract
/// describes).
struct AotPrepared
{
    std::optional<wgrad::AotInstance> instance;
};

AotPrepared decodeAot(const KernelDefinition& kernel)
{
    return AotPrepared{aotInstance(kernel)};
}

/// The KernelPrepareFn registered under the kernel matcher's symbol: the SDK runs it once
/// per kernel when the engine loads, so matching a graph reads AotPrepared instead of
/// parsing some fifty metadata fields per kernel. It also files each helper for the
/// instances that launch it to find. Must not throw.
std::shared_ptr<const void> aotPrepare(const KernelDefinition& kernel)
{
    try
    {
        auto prepared = std::make_shared<const AotPrepared>(decodeAot(kernel));
        const auto& instance = prepared->instance;
        if(instance && instance->family == wgrad::KernelFamily::HELPER && instance->direct)
        {
            const auto variant = instance->direct->variant;
            helperRegistry().add(HelperKey{archiveOf(kernel),
                                           kernel.packId,
                                           instance->dtype,
                                           variant,
                                           variant == wgrad::DirectVariant::WS_REDUCE
                                               ? instance->direct->wsReplicas
                                               : instance->direct->knobs.taps},
                                 kernel);
        }
        return prepared;
    }
    catch(...)
    {
        return nullptr;
    }
}

/// @p kernel's AotPrepared: the one aotPrepare stored, or one decoded into @p scratch for
/// a definition that carries none (one built outside the engine's shared state).
const AotPrepared& preparedOf(const KernelDefinition& kernel, AotPrepared& scratch)
{
    if(kernel.preparedBy == &aotPrepare && kernel.prepared != nullptr)
    {
        return *static_cast<const AotPrepared*>(kernel.prepared.get());
    }
    scratch = decodeAot(kernel);
    return scratch;
}

/// The plan of @p kernel on @p matched when it is packaged code of a reviewed launch
/// contract that computes it: the family the graph's dW dtype admits, the instance's plan
/// (dtype pinned there), and every helper the plan launches present in its pack. The one
/// path every native hook takes, so the matcher, workspaceBytes() and prepare() can never
/// disagree on what is served.
std::optional<wgrad::Plan> aotKernelPlan(const MatchedProblem& matched,
                                         const KernelDefinition& kernel)
{
    AotPrepared scratch;
    const auto& prepared = preparedOf(kernel, scratch);
    if(!prepared.instance
       || (prepared.instance->family == wgrad::KernelFamily::DIRECT) != matched.dwFloat)
    {
        return std::nullopt;
    }
    auto plan = wgrad::instancePlan(matched.problem, *prepared.instance);
    if(!plan)
    {
        return std::nullopt;
    }
    for(const auto& step : plan->steps)
    {
        if(step.kernel != wgrad::PlanKernel::SELF
           && !findHelper(
               kernel, *prepared.instance, step.kernel, matched.problem, matched.geometry))
        {
            return std::nullopt;
        }
    }
    return plan;
}

bool aotKernelMatches(const MatchContext& context,
                      const BoundTokens& bound,
                      const KernelDefinition& kernel)
{
    const auto matched = matchedFromBound(context, bound);
    return matched && aotKernelPlan(*matched, kernel).has_value();
}

double
    aotScore(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
{
    // Deterministic fallback only; a trained ranker or BenchmarkPlan replaces it. Only
    // kernels the matcher admitted are scored, so a refusal (and the fold) ranks last.
    const auto matched = matchedFromBound(context, bound);
    if(!matched)
    {
        return -1.0;
    }
    const auto plan = aotKernelPlan(*matched, kernel);
    if(!plan)
    {
        return -1.0;
    }
    AotPrepared scratch;
    return wgrad::planScore(*preparedOf(kernel, scratch).instance, *plan);
}

void requireBindings(const BoundTokens& bound, const Binding& binding)
{
    if(tryGetBoundInt(bound, X_TOKEN) != std::optional<int64_t>(binding.x)
       || tryGetBoundInt(bound, DY_TOKEN) != std::optional<int64_t>(binding.dy)
       || tryGetBoundInt(bound, DW_TOKEN) != std::optional<int64_t>(binding.dw))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950_conv_wgrad dispatch requires this graph matcher's tensor bindings");
    }
}

/// One argument of a kernel signature: its recorded name and whether it is a pointer.
struct ArgumentSpec
{
    std::string_view name;
    bool pointer;
};

/// The ABI buildIngestorKernelCode verifies the loaded symbol against, laid out as the
/// kernarg segment is: each argument at the next offset aligned to its own size. NAMES ARE
/// LOAD-BEARING: every problem argument is a by_value i32, so kind and size alone cannot
/// tell p_Hi from p_Wi, and requireSignatureMatch compares the names hkp_pack recorded.
std::vector<KernelArgument> kernelSignature(const std::vector<ArgumentSpec>& arguments)
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    for(const auto& argument : arguments)
    {
        const uint32_t size = argument.pointer ? sizeof(void*) : sizeof(int32_t);
        offset = (offset + size - 1) / size * size;
        signature.push_back({argument.pointer ? "global_buffer" : "by_value",
                             size,
                             offset,
                             std::string(argument.name)});
        offset += size;
    }
    return signature;
}

/// The @p pointers, then their byte sizes @p sizes and the @p names as i32.
template <size_t Pointers, size_t Count>
std::vector<ArgumentSpec> arguments(const std::array<std::string_view, Pointers>& pointers,
                                    const std::array<std::string_view, Pointers>& sizes,
                                    const std::array<std::string_view, Count>& names)
{
    std::vector<ArgumentSpec> out;
    for(const auto name : pointers)
    {
        out.push_back({name, true});
    }
    for(const auto name : sizes)
    {
        out.push_back({name, false});
    }
    for(const auto name : names)
    {
        out.push_back({name, false});
    }
    return out;
}

/// The two-stage implicit-GEMM block: the one-stage one with (ws_ptr*, ws_bytes) before
/// ks / ks_count (conv_args_signature(direction="wgrad", two_stage=True)).
std::vector<ArgumentSpec> twoStageArguments()
{
    auto out = arguments<3>(
        {"dY", "X", "dW"}, {"dY_bytes", "X_bytes", "dW_bytes"}, wgrad::IMPLICIT_KERNARG_NAMES);
    const auto at = out.begin() + 6 + wgrad::TWO_STAGE_WS_POSITION;
    out.insert(at, {{"ws_ptr", true}, {"ws_bytes", false}});
    return out;
}

/// The Stage 2 fold's (ws_ptr*, dw_ptr*, wg_M, wg_N, ws_bytes, dw_bytes, groups)
/// (wgrad_reduce_signature).
std::vector<ArgumentSpec> wsReduceArguments()
{
    std::vector<ArgumentSpec> out{{"ws_ptr", true}, {"dw_ptr", true}};
    for(const auto name : wgrad::WS_REDUCE_KERNARG_NAMES)
    {
        out.push_back({name, false});
    }
    return out;
}

/// The signature of a step's ABI: the 48-argument one-stage implicit-GEMM wgrad block
/// (conv_args_signature(direction="wgrad")) and its 50-argument two-stage form, the
/// 20-argument direct wgrad block with its fp32 D, the fold's (A*, D*, A_bytes, D_bytes,
/// parts, channels) (dw_wgrad_fold_signature), and the Stage 2 fold's.
const std::vector<KernelArgument>& stepSignature(wgrad::Abi abi)
{
    static const auto s_implicit = kernelSignature(arguments<3>(
        {"dY", "X", "dW"}, {"dY_bytes", "X_bytes", "dW_bytes"}, wgrad::IMPLICIT_KERNARG_NAMES));
    static const auto s_twoStage = kernelSignature(twoStageArguments());
    static const auto s_wsReduce = kernelSignature(wsReduceArguments());
    static const auto s_direct = kernelSignature(arguments<3>(
        {"A", "B", "D"}, {"A_bytes", "B_bytes", "D_bytes"}, wgrad::DIRECT_WGRAD_KERNARG_NAMES));
    static const auto s_fold = kernelSignature(
        arguments<2>({"A", "D"}, {"A_bytes", "D_bytes"}, wgrad::FOLD_KERNARG_NAMES));
    switch(abi)
    {
    case wgrad::Abi::IMPLICIT:
        return s_implicit;
    case wgrad::Abi::DIRECT_WGRAD:
        return s_direct;
    case wgrad::Abi::IMPLICIT_TWO_STAGE:
        return s_twoStage;
    case wgrad::Abi::WS_REDUCE:
        return s_wsReduce;
    case wgrad::Abi::FOLD:
    default:
        return s_fold;
    }
}

IngestorKernelCode loadKernel(const compilation::KpackKernelLoader& loader,
                              const MatchContext& context,
                              const KernelDefinition& kernel,
                              const wgrad::Step& step)
{
    auto code = buildIngestorKernelCode(loader, context, kernel, stepSignature(step.abi));
    code.setBlockSize(static_cast<unsigned int>(step.block), 1, 1);
    code.setGridSize(static_cast<unsigned int>(step.grid[0]),
                     static_cast<unsigned int>(step.grid[1]),
                     static_cast<unsigned int>(step.grid[2]));
    // Every kernel of this pack uses statically allocated LDS only.
    code.setSharedMemBytes(0);
    return code;
}

void checkHip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string("gfx950_conv_wgrad ") + what + " failed: " + hipGetErrorString(status));
    }
}

/// One step of a prepared plan: its loaded kernel and the step it runs.
struct PreparedStep
{
    IngestorKernelCode code;
    wgrad::Step step;
};

class PreparedConvWgrad : public PreparedDispatch
{
public:
    PreparedConvWgrad(Binding binding,
                      std::array<int64_t, 3> bytes,
                      std::vector<PreparedStep> steps,
                      std::vector<wgrad::Workspace> workspaces,
                      bool zeroDw,
                      bool zeroWs)
        : _binding(binding)
        , _bytes(bytes)
        , _steps(std::move(steps))
        , _workspaces(std::move(workspaces))
        , _zeroDw(zeroDw)
        , _zeroWs(zeroWs)
    {
    }

    const Binding& binding() const
    {
        return _binding;
    }
    /// dY, X, dW storage sizes in bytes.
    const std::array<int64_t, 3>& bytes() const
    {
        return _bytes;
    }
    const std::vector<PreparedStep>& steps() const
    {
        return _steps;
    }
    const std::vector<wgrad::Workspace>& workspaces() const
    {
        return _workspaces;
    }
    bool zeroDw() const
    {
        return _zeroDw;
    }
    bool zeroWs() const
    {
        return _zeroWs;
    }

private:
    Binding _binding;
    std::array<int64_t, 3> _bytes;
    // Holds every program/module and kernel view; owns no MatchContext data.
    std::vector<PreparedStep> _steps;
    std::vector<wgrad::Workspace> _workspaces;
    bool _zeroDw;
    bool _zeroWs;
};

class ConvWgradDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit ConvWgradDispatchHandler(const compilation::KpackKernelLoader& loader)
        : _loader(loader)
    {
    }

    /// The plan's workspace: the folded pipelines' partial slots or a two-stage instance's
    /// f32 scratch; none for one-stage implicit GEMM or the atomic direct pipelines. A
    /// kernel the graph does not admit needs none: prepare() refuses it, through the same
    /// aotKernelPlan.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto matched = matchProblem(context);
        const auto plan = matched ? aotKernelPlan(*matched, kernel) : std::nullopt;
        return plan ? static_cast<size_t>(plan->workspaceBytes()) : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto matched = matchProblem(context);
        auto plan = matched ? aotKernelPlan(*matched, kernel) : std::nullopt;
        if(!plan)
        {
            AotPrepared scratch;
            const auto& instance = preparedOf(kernel, scratch).instance;
            const auto refusal = instance ? wgrad::staticRefusal(*instance) : std::nullopt;
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("gfx950_conv_wgrad graph or packaged kernel does not satisfy the "
                            "convolution contract")
                    + (refusal ? ": " + std::string(*refusal) : std::string()));
        }
        requireBindings(bound, matched->binding);
        AotPrepared scratch;
        const auto& instance = *preparedOf(kernel, scratch).instance;
        // This pack ships prebuilt code objects only; aotKernelPlan already refused every
        // other source kind. Each step loads its own kernel under its ABI's signature.
        std::vector<PreparedStep> steps;
        for(auto& step : plan->steps)
        {
            if(step.kernel == wgrad::PlanKernel::SELF)
            {
                steps.push_back({loadKernel(_loader, context, kernel, step), std::move(step)});
                continue;
            }
            const auto helper
                = findHelper(kernel, instance, step.kernel, matched->problem, matched->geometry);
            if(!helper)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_wgrad instance has no helper of its plan in its pack");
            }
            steps.push_back({loadKernel(_loader, context, *helper, step), std::move(step)});
        }
        return std::make_unique<PreparedConvWgrad>(matched->binding,
                                                   matched->tensorBytes(),
                                                   std::move(steps),
                                                   plan->workspaces,
                                                   plan->zeroDw,
                                                   plan->zeroWs);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& wgradPrepared = dynamic_cast<const PreparedConvWgrad&>(prepared);
        if(deviceBuffers == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "gfx950_conv_wgrad requires device buffers");
        }
        const auto& binding = wgradPrepared.binding();
        void* dy
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dy, deviceBuffers, numDeviceBuffers).ptr;
        void* x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers).ptr;
        void* dw
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dw, deviceBuffers, numDeviceBuffers).ptr;
        for(const auto* ptr : {dy, x, dw})
        {
            if(ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % 16 != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_wgrad requires non-null device pointers aligned to 16 bytes");
            }
        }
        const auto& bytes = wgradPrepared.bytes();
        if(!wgrad::bufferRangesDoNotOverlap({reinterpret_cast<uintptr_t>(dy),
                                             reinterpret_cast<uintptr_t>(x),
                                             reinterpret_cast<uintptr_t>(dw)},
                                            bytes))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_wgrad requires nonoverlapping gradient, input, and "
                "filter-gradient storage");
        }
        const auto& workspaces = wgradPrepared.workspaces();
        if(!workspaces.empty()
           && (workspace == nullptr || reinterpret_cast<uintptr_t>(workspace) % 16 != 0))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_wgrad requires a workspace aligned to 16 bytes");
        }
        const auto region = [&](wgrad::Buffer name) -> std::pair<char*, int64_t> {
            switch(name)
            {
            case wgrad::Buffer::DY:
                return {static_cast<char*>(dy), bytes[0]};
            case wgrad::Buffer::X:
                return {static_cast<char*>(x), bytes[1]};
            case wgrad::Buffer::DW:
                return {static_cast<char*>(dw), bytes[2]};
            default:
                break;
            }
            for(const auto& entry : workspaces)
            {
                if(entry.name == name)
                {
                    return {static_cast<char*>(workspace) + entry.offset, entry.bytes};
                }
            }
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "gfx950_conv_wgrad plan binds a buffer it does not lay out");
        };
        // Every step below goes on the launch stream, so the plan stays stream-ordered.
        const auto stream = handle.getStream();
        if(wgradPrepared.zeroDw())
        {
            // Split-K slices and the atomic direct pipelines accumulate into dW.
            checkHip(hipMemsetAsync(dw, 0, static_cast<size_t>(bytes[2]), stream), "dW zero-fill");
        }
        if(wgradPrepared.zeroWs())
        {
            // Two-stage slices accumulate into the f32 scratch.
            const auto [ws, wsBytes] = region(wgrad::Buffer::WS);
            checkHip(hipMemsetAsync(ws, 0, static_cast<size_t>(wsBytes), stream),
                     "workspace zero-fill");
        }
        for(const auto& [code, step] : wgradPrepared.steps())
        {
            std::array<void*, 3> pointers{};
            std::array<int32_t, 3> sizes{};
            for(size_t i = 0; i < step.buffers.size(); ++i)
            {
                if(step.buffers[i] == wgrad::Buffer::NONE)
                {
                    continue;
                }
                const auto [base, size] = region(step.buffers[i]);
                pointers[i] = base;
                sizes[i] = static_cast<int32_t>(size);
            }
            std::vector<void*> params;
            void* wsPointer = nullptr;
            if(step.abi == wgrad::Abi::FOLD)
            {
                params = {&pointers[0], &pointers[2], &sizes[0], &sizes[2]};
            }
            else if(step.abi == wgrad::Abi::WS_REDUCE)
            {
                params = {&pointers[0], &pointers[2]};
            }
            else
            {
                params
                    = {&pointers[0], &pointers[1], &pointers[2], &sizes[0], &sizes[1], &sizes[2]};
            }
            for(size_t i = 0; i < step.values.size(); ++i)
            {
                if(step.abi == wgrad::Abi::IMPLICIT_TWO_STAGE && i == wgrad::TWO_STAGE_WS_POSITION)
                {
                    wsPointer = region(wgrad::Buffer::WS).first;
                    params.push_back(&wsPointer);
                }
                params.push_back(const_cast<int32_t*>(&step.values[i]));
            }
            code.kernelForStream(stream).launchWithParams(stream, params.data());
        }
    }

private:
    const compilation::KpackKernelLoader& _loader;
};

const ConvWgradDispatchHandler& dispatchHandler()
{
    // The registries retain non-owning pointers; both dependencies live for the process,
    // including across provider handle destruction and recreation.
    static const compilation::KpackKernelLoader s_loader(gfx950ConvWgradAotKpackModuleCache());
    static const ConvWgradDispatchHandler s_handler(s_loader);
    return s_handler;
}

} // namespace

compilation::KpackModuleCache& gfx950ConvWgradAotKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void registerGfx950ConvWgradAotSymbols(SymbolScope<Handle>& scope)
{
    // No CandidateFilterFn: the catalog is small, so every serving instance is a
    // candidate and the ranker chooses.
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &aotKernelMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &aotPrepare);
    scope.add(std::string(SCORE_SYMBOL), &aotScore);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
}

void resetGfx950ConvWgradAotModuleCache()
{
    gfx950ConvWgradAotKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
