// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
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

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
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
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvDgradGeometry.hpp"

/// @file Gfx950ConvDgradNative.cpp
/// hipkernel:Gfx950ConvDgradAot: shape-generic AOT instances of rocKE's backward-data
/// (dgrad) convolution kernels. Each kernel is one rocKE dgrad KernelIdentity: an
/// implicit-GEMM tile (kernel_family 0), one of rocKE's direct dgrad pipelines
/// (kernel_family 1), or one of the MFMA pipelines' weight transforms (kernel_family 2,
/// never matched; an MFMA dispatch loads it beside its instance). Its metadata holds
/// tuning and capability fields only and the problem reaches the kernel as AOT
/// arguments, so matching is the launch plan of Gfx950ConvDgradGeometry.hpp, not an
/// equality on the geometry: a kernel serves a graph exactly when it has a plan there,
/// and the dispatch runs that plan. Mirrors
/// rocke/library/builders/common/convolution_backward_data.py.
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace dgrad = gfx950_conv_dgrad;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_conv_dgrad_aot.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_conv_dgrad_aot.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_conv_dgrad_aot.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_conv_dgrad_aot.dispatch";
constexpr std::string_view DY_TOKEN = "gfx950_conv_dgrad.dy.uid";
constexpr std::string_view W_TOKEN = "gfx950_conv_dgrad.w.uid";
constexpr std::string_view DX_TOKEN = "gfx950_conv_dgrad.dx.uid";
/// The whole matched problem, so a kernel matcher reads it from what graph_match bound
/// instead of walking the graph again for every kernel: problemTokenValues() order.
/// Private (`_`): the graph features already describe the problem, so this is no feature.
constexpr std::string_view PROBLEM_TOKEN = "_gfx950_conv_dgrad.problem";
constexpr size_t PROBLEM_TOKEN_SIZE = 18;

struct Binding
{
    int64_t dy;
    int64_t w;
    int64_t dx;
};

struct MatchedProblem
{
    Binding binding;
    dgrad::Problem problem;
    dgrad::Geometry geometry;
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

/// hipDNN [N, C, H, W] dims stored NHWC: dY [N, K, Ho, Wo] as NHWK, W [K, C/G, Y, X] as
/// KYXC, dX [N, C, Hi, Wi] as NHWC.
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
       || !dgrad::wideProduct({extents[0], extents[1], extents[2], extents[3], 2}))
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

/// The graph contract: one ConvolutionBwdAttributes node on gfx950 (wave64) with FLOAT
/// accumulation, cross-correlation, 2-D symmetric padding, three distinct device tensors
/// of one 16-bit dtype in packed NHWC, groups = dX.C / W.dims[1] in [1, 65535] dividing
/// K, and dY's dims equal to the forward output of dX (whose dims define the problem:
/// several Hi give one Ho under a stride).
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
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionBwdAttributes
       || node.attributes() == nullptr || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& attributes = node.attributesAs<data_objects::ConvolutionBwdAttributes>();
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
        attributes.dy_tensor_uid(), attributes.w_tensor_uid(), attributes.dx_tensor_uid()};
    const auto* dy = findTensor(context, binding.dy);
    const auto* w = findTensor(context, binding.w);
    const auto* dx = findTensor(context, binding.dx);
    if(binding.dy == binding.w || binding.dy == binding.dx || binding.w == binding.dx
       || !isDeviceTensor(dy) || !isDeviceTensor(w) || !isDeviceTensor(dx)
       || !hasPackedNhwcStrides(dy) || !hasPackedNhwcStrides(w) || !hasPackedNhwcStrides(dx)
       || dy->data_type() != w->data_type() || dy->data_type() != dx->data_type())
    {
        return std::nullopt;
    }
    const auto* dxDims = dx->dims();
    const auto* wDims = w->dims();
    const auto* dyDims = dy->dims();
    // hipDNN carries no groups attribute: a filter holding C / groups channels encodes it.
    // The layout checks above already made every extent positive.
    const auto channelsPerGroup = wDims->Get(1);
    if(dxDims->Get(1) % channelsPerGroup != 0)
    {
        return std::nullopt;
    }
    const auto groups = dxDims->Get(1) / channelsPerGroup;
    const dgrad::Problem problem{dxDims->Get(0),
                                 dxDims->Get(1),
                                 wDims->Get(0),
                                 dxDims->Get(2),
                                 dxDims->Get(3),
                                 wDims->Get(2),
                                 wDims->Get(3),
                                 attributes.stride()->Get(0),
                                 attributes.stride()->Get(1),
                                 attributes.pre_padding()->Get(0),
                                 attributes.pre_padding()->Get(1),
                                 attributes.dilation()->Get(0),
                                 attributes.dilation()->Get(1),
                                 groups};
    // groups in [1, 65535] dividing C and K, a positive dY, every byte count in i32.
    const auto geometry = dgrad::deriveGeometry(problem);
    if(!geometry || dyDims->Get(0) != problem.n || dyDims->Get(1) != problem.k
       || dyDims->Get(2) != geometry->ho || dyDims->Get(3) != geometry->wo)
    {
        return std::nullopt;
    }
    return MatchedProblem{binding, problem, *geometry, dx->data_type()};
}

/// PROBLEM_TOKEN's value: the binding, the problem in dgrad::Problem's field order, then
/// the dtype. The geometry is derived from the problem again on reading.
std::vector<int64_t> problemTokenValues(const MatchedProblem& matched)
{
    const auto& p = matched.problem;
    return {matched.binding.dy,
            matched.binding.w,
            matched.binding.dx,
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
    const dgrad::Problem problem{
        v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16]};
    const auto geometry = dgrad::deriveGeometry(problem);
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
    return BoundTokens{{std::string(DY_TOKEN), matched->binding.dy},
                       {std::string(W_TOKEN), matched->binding.w},
                       {std::string(DX_TOKEN), matched->binding.dx},
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
/// complete instance of one family (dgrad::parseAotInstance).
std::optional<dgrad::AotInstance> aotInstance(const KernelDefinition& kernel)
{
    if(kernel.source.kind != KernelSourceKind::KPACK)
    {
        return std::nullopt;
    }
    return dgrad::parseAotInstance(KernelMetadata{kernel});
}

/// Where a helper is looked up: the same pack, the same archive, the instance's dtype,
/// and the identity fields gfx950_conv_dgrad_aot_helper_fields names.
struct HelperKey
{
    std::string archive;
    DescriptorId packId;
    std::string dtype;
    dgrad::DirectVariant variant;
    int64_t filter;
    int64_t cpg;
    int64_t kpg;
    int64_t foldK32;

    bool operator<(const HelperKey& other) const
    {
        return std::tie(archive, packId, dtype, variant, filter, cpg, kpg, foldK32)
               < std::tie(other.archive,
                          other.packId,
                          other.dtype,
                          other.variant,
                          other.filter,
                          other.cpg,
                          other.kpg,
                          other.foldK32);
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

/// Every helper kernel the engine has loaded, filed by HelperKey. The kernel prepare hook
/// files each one when the engine's shared state is built, which precedes any match and
/// any dispatch; the MFMA matcher and dispatch look their helpers up here, since no
/// native hook sees the catalog. A key holding two different kernels is ambiguous and
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

/// A helper's own key, from its instance.
HelperKey helperKey(const KernelDefinition& kernel,
                    const std::string& dtype,
                    const dgrad::DirectInstance& helper)
{
    const bool reorganize = helper.variant == dgrad::DirectVariant::REORGANIZE
                            || helper.variant == dgrad::DirectVariant::REORGANIZE_RTC;
    return HelperKey{archiveOf(kernel),
                     kernel.packId,
                     dtype,
                     helper.variant,
                     helper.compiled.filterH,
                     helper.compiled.cpg,
                     helper.compiled.kpg,
                     reorganize ? helper.knobs.foldK32 : 0};
}

/// The helper a plan step of an MFMA instance launches: the transform of that kind
/// for the instance's filter and channel counts (and, for a reorganize, its fold_k32).
std::optional<KernelDefinition> findHelper(const KernelDefinition& kernel,
                                           const std::string& dtype,
                                           const dgrad::DirectInstance& instance,
                                           dgrad::PlanKernel step)
{
    dgrad::DirectVariant variant{};
    switch(step)
    {
    case dgrad::PlanKernel::TRANSPOSE:
        variant = dgrad::DirectVariant::TRANSPOSE;
        break;
    case dgrad::PlanKernel::REORGANIZE:
        variant = dgrad::DirectVariant::REORGANIZE;
        break;
    case dgrad::PlanKernel::TRANSPOSE_RTC:
        variant = dgrad::DirectVariant::TRANSPOSE_RTC;
        break;
    case dgrad::PlanKernel::REORGANIZE_RTC:
        variant = dgrad::DirectVariant::REORGANIZE_RTC;
        break;
    case dgrad::PlanKernel::SELF:
    default:
        return std::nullopt;
    }
    const bool reorganize
        = step == dgrad::PlanKernel::REORGANIZE || step == dgrad::PlanKernel::REORGANIZE_RTC;
    return helperRegistry().find(HelperKey{archiveOf(kernel),
                                           kernel.packId,
                                           dtype,
                                           variant,
                                           instance.compiled.filterH,
                                           instance.compiled.cpg,
                                           instance.compiled.kpg,
                                           reorganize ? instance.knobs.foldK32 : 0});
}

/// What hipkernel:Gfx950ConvDgradAot reads of one kernel on every graph, decoded from its
/// metadata once: its dtype and its instance (nullopt for metadata no reviewed launch
/// contract describes).
struct AotPrepared
{
    std::string dtype;
    std::optional<dgrad::AotInstance> instance;
};

AotPrepared decodeAot(const KernelDefinition& kernel)
{
    const auto* dtype = KernelMetadata{kernel}.stringField("dtype");
    return AotPrepared{dtype == nullptr ? std::string() : *dtype, aotInstance(kernel)};
}

/// The KernelPrepareFn registered under the kernel matcher's symbol: the SDK runs it once
/// per kernel when the engine loads, so matching a graph reads AotPrepared instead of
/// parsing some sixty metadata fields per kernel. It also files each helper kernel for
/// the MFMA pipelines to find. Must not throw.
std::shared_ptr<const void> aotPrepare(const KernelDefinition& kernel)
{
    try
    {
        auto prepared = std::make_shared<const AotPrepared>(decodeAot(kernel));
        const auto& instance = prepared->instance;
        if(instance && instance->direct && dgrad::isHelperVariant(instance->direct->variant)
           && (prepared->dtype == "fp16" || prepared->dtype == "bf16"))
        {
            helperRegistry().add(helperKey(kernel, prepared->dtype, *instance->direct), kernel);
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

const char* dtypeName(data_objects::DataType dtype)
{
    return dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
}

/// The plan of @p kernel on @p matched when it is packaged code of a reviewed launch
/// contract that computes it: the dtype pinned, then the instance's plan, and every
/// helper the plan launches present in its pack.
std::optional<dgrad::Plan> aotKernelPlan(const MatchedProblem& matched,
                                         const KernelDefinition& kernel)
{
    AotPrepared scratch;
    const auto& prepared = preparedOf(kernel, scratch);
    if(prepared.dtype != dtypeName(matched.dtype) || !prepared.instance)
    {
        return std::nullopt;
    }
    auto plan = dgrad::instancePlan(matched.problem, *prepared.instance);
    if(!plan)
    {
        return std::nullopt;
    }
    for(const auto& step : plan->steps)
    {
        if(step.kernel != dgrad::PlanKernel::SELF
           && !findHelper(kernel, prepared.dtype, *prepared.instance->direct, step.kernel))
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
    // kernels the matcher admitted are scored, so a refusal (and every helper) ranks last.
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
    return dgrad::planScore(*preparedOf(kernel, scratch).instance, *plan);
}

void requireBindings(const BoundTokens& bound, const Binding& binding)
{
    if(tryGetBoundInt(bound, DY_TOKEN) != std::optional<int64_t>(binding.dy)
       || tryGetBoundInt(bound, W_TOKEN) != std::optional<int64_t>(binding.w)
       || tryGetBoundInt(bound, DX_TOKEN) != std::optional<int64_t>(binding.dx))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950_conv_dgrad dispatch requires this graph matcher's tensor bindings");
    }
}

/// One argument of a kernel signature: its kind, size and recorded name.
struct ArgumentSpec
{
    std::string_view name;
    bool pointer;
};

/// The ABI buildIngestorKernelCode verifies the loaded symbol against, laid out as the
/// kernarg segment is: each argument at the next offset aligned to its own size (so the
/// implicit-GEMM sub_gemm_buf sits at 184 after a 4-byte pad). NAMES ARE LOAD-BEARING:
/// every problem argument is a by_value i32, so kind and size alone cannot tell p_Hi from
/// p_Wi, and requireSignatureMatch compares the names hkp_pack recorded.
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

/// (A*, B*, D*, A_bytes, B_bytes, D_bytes) under @p leading's names, then @p names as i32.
template <size_t Count>
std::vector<ArgumentSpec> directArguments(const std::array<std::string_view, 6>& leading,
                                          const std::array<std::string_view, Count>& names)
{
    std::vector<ArgumentSpec> arguments;
    for(size_t i = 0; i < leading.size(); ++i)
    {
        arguments.push_back({leading[i], i < 3});
    }
    for(const auto name : names)
    {
        arguments.push_back({name, false});
    }
    return arguments;
}

/// The signature of a step's ABI: the 44-argument dgrad implicit-GEMM block (sub_gemm_buf
/// a device pointer, num_sub_gemms last), the 20-argument direct dgrad block, the forward
/// direct block, and the weight transforms' (A*, D*, A_bytes, D_bytes) with, for the
/// runtime-channel ones, p_cpg and p_kpg.
const std::vector<KernelArgument>& stepSignature(dgrad::Abi abi)
{
    static const auto s_implicit = [] {
        auto arguments = directArguments({"dY", "W", "dX", "dY_bytes", "W_bytes", "dX_bytes"},
                                         dgrad::IMPLICIT_GEMM_KERNARG_NAMES);
        arguments.push_back({"sub_gemm_buf", true});
        arguments.push_back({"num_sub_gemms", false});
        return kernelSignature(arguments);
    }();
    static const auto s_directDgrad = kernelSignature(directArguments(
        {"A", "B", "D", "A_bytes", "B_bytes", "D_bytes"}, dgrad::DIRECT_DGRAD_KERNARG_NAMES));
    static const auto s_directFwd = kernelSignature(directArguments(
        {"A", "B", "D", "A_bytes", "B_bytes", "D_bytes"}, dgrad::DIRECT_KERNARG_NAMES));
    static const auto s_transform
        = kernelSignature({{"A", true}, {"D", true}, {"A_bytes", false}, {"D_bytes", false}});
    static const auto s_transformRtc = kernelSignature({{"A", true},
                                                        {"D", true},
                                                        {"A_bytes", false},
                                                        {"D_bytes", false},
                                                        {"p_cpg", false},
                                                        {"p_kpg", false}});
    switch(abi)
    {
    case dgrad::Abi::IMPLICIT:
        return s_implicit;
    case dgrad::Abi::DIRECT_DGRAD:
        return s_directDgrad;
    case dgrad::Abi::DIRECT_FWD:
        return s_directFwd;
    case dgrad::Abi::TRANSFORM:
        return s_transform;
    case dgrad::Abi::TRANSFORM_RTC:
    default:
        return s_transformRtc;
    }
}

IngestorKernelCode loadKernel(const compilation::KpackKernelLoader& loader,
                              const MatchContext& context,
                              const KernelDefinition& kernel,
                              const dgrad::Step& step)
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
            std::string("gfx950_conv_dgrad ") + what + " failed: " + hipGetErrorString(status));
    }
}

/// One step of a prepared plan: its loaded kernel and the step it runs.
struct PreparedStep
{
    IngestorKernelCode code;
    dgrad::Step step;
};

/// The plan's SUB_GEMM region, owned by the prepared plan: every implicit step's record
/// table, uploaded once when the plan is prepared, as the rocKE sweep uploads it once
/// before timing. A table depends on the problem and the instance only, so no launch
/// re-uploads it, and the caller's workspace (which it may reuse between launches)
/// never holds it.
class DeviceTables
{
public:
    DeviceTables(const dgrad::Workspace& region, const std::vector<PreparedStep>& steps)
    {
        checkHip(hipMalloc(&_data, static_cast<size_t>(region.bytes)), "sub-GEMM table alloc");
        for(const auto& [code, step] : steps)
        {
            if(step.abi != dgrad::Abi::IMPLICIT)
            {
                continue;
            }
            const auto status = hipMemcpy(static_cast<char*>(_data) + step.tableOffset,
                                          step.table.data(),
                                          step.table.size() * sizeof(int32_t),
                                          hipMemcpyHostToDevice);
            if(status != hipSuccess)
            {
                static_cast<void>(hipFree(_data));
                checkHip(status, "sub-GEMM table upload");
            }
        }
    }
    DeviceTables(const DeviceTables&) = delete;
    DeviceTables& operator=(const DeviceTables&) = delete;
    DeviceTables(DeviceTables&&) = delete;
    DeviceTables& operator=(DeviceTables&&) = delete;
    ~DeviceTables()
    {
        // A failure here (a device already reset) leaves nothing to release.
        static_cast<void>(hipFree(_data));
    }

    char* data() const
    {
        return static_cast<char*>(_data);
    }

private:
    void* _data = nullptr;
};

/// The workspace the caller provides for @p plan: every region but SUB_GEMM, which the
/// prepared plan owns (DeviceTables). An implicit-GEMM plan lays out SUB_GEMM alone, a
/// direct plan never does, so no other region moves.
int64_t callerWorkspaceBytes(const dgrad::Plan& plan)
{
    int64_t end = 0;
    for(const auto& region : plan.workspaces)
    {
        if(region.name != dgrad::Buffer::SUB_GEMM)
        {
            end = std::max(end, region.offset + region.bytes);
        }
    }
    return end;
}

class PreparedConvDgrad : public PreparedDispatch
{
public:
    PreparedConvDgrad(Binding binding,
                      std::array<int64_t, 3> bytes,
                      std::vector<PreparedStep> steps,
                      std::vector<dgrad::Workspace> workspaces,
                      std::unique_ptr<DeviceTables> tables,
                      bool zeroDx)
        : _binding(binding)
        , _bytes(bytes)
        , _steps(std::move(steps))
        , _workspaces(std::move(workspaces))
        , _tables(std::move(tables))
        , _zeroDx(zeroDx)
    {
    }

    const Binding& binding() const
    {
        return _binding;
    }
    /// dY, W, dX storage sizes in bytes.
    const std::array<int64_t, 3>& bytes() const
    {
        return _bytes;
    }
    const std::vector<PreparedStep>& steps() const
    {
        return _steps;
    }
    /// The caller-workspace regions (every one but SUB_GEMM).
    const std::vector<dgrad::Workspace>& workspaces() const
    {
        return _workspaces;
    }
    /// The SUB_GEMM region, or null when no step is implicit GEMM.
    char* tables() const
    {
        return _tables ? _tables->data() : nullptr;
    }
    bool zeroDx() const
    {
        return _zeroDx;
    }

private:
    Binding _binding;
    std::array<int64_t, 3> _bytes;
    // Holds every program/module and kernel view; owns no MatchContext data.
    std::vector<PreparedStep> _steps;
    std::vector<dgrad::Workspace> _workspaces;
    std::unique_ptr<DeviceTables> _tables;
    bool _zeroDx;
};

class ConvDgradDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit ConvDgradDispatchHandler(const compilation::KpackKernelLoader& loader)
        : _loader(loader)
    {
    }

    /// The plan's caller workspace: the MFMA pipelines' transposed (and reorganized)
    /// weights; none for implicit GEMM (the prepared plan owns its record tables) or the
    /// single-kernel direct pipelines. A kernel the graph does not admit needs none:
    /// prepare() refuses it.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto matched = matchProblem(context);
        const auto plan = matched ? aotKernelPlan(*matched, kernel) : std::nullopt;
        return plan ? static_cast<size_t>(callerWorkspaceBytes(*plan)) : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto matched = matchProblem(context);
        auto plan = matched ? aotKernelPlan(*matched, kernel) : std::nullopt;
        if(!plan)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_dgrad graph or packaged kernel does not satisfy the convolution "
                "contract");
        }
        requireBindings(bound, matched->binding);
        AotPrepared scratch;
        const auto& prepared = preparedOf(kernel, scratch);
        // This pack ships prebuilt code objects only; aotKernelPlan already refused every
        // other source kind. Each step loads its own kernel under its ABI's signature.
        std::vector<PreparedStep> steps;
        for(auto& step : plan->steps)
        {
            if(step.kernel == dgrad::PlanKernel::SELF)
            {
                steps.push_back({loadKernel(_loader, context, kernel, step), std::move(step)});
                continue;
            }
            const auto helper
                = findHelper(kernel, prepared.dtype, *prepared.instance->direct, step.kernel);
            if(!helper)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_dgrad MFMA instance has no weight-transform helper in its "
                    "pack");
            }
            steps.push_back({loadKernel(_loader, context, *helper, step), std::move(step)});
        }
        std::unique_ptr<DeviceTables> tables;
        std::vector<dgrad::Workspace> workspaces;
        for(const auto& region : plan->workspaces)
        {
            if(region.name == dgrad::Buffer::SUB_GEMM)
            {
                tables = std::make_unique<DeviceTables>(region, steps);
                continue;
            }
            workspaces.push_back(region);
        }
        return std::make_unique<PreparedConvDgrad>(matched->binding,
                                                   matched->geometry.tensorBytes,
                                                   std::move(steps),
                                                   std::move(workspaces),
                                                   std::move(tables),
                                                   plan->zeroDx);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dgradPrepared = dynamic_cast<const PreparedConvDgrad&>(prepared);
        if(deviceBuffers == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "gfx950_conv_dgrad requires device buffers");
        }
        const auto& binding = dgradPrepared.binding();
        void* dy
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dy, deviceBuffers, numDeviceBuffers).ptr;
        void* w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers).ptr;
        void* dx
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dx, deviceBuffers, numDeviceBuffers).ptr;
        for(const auto* ptr : {dy, w, dx})
        {
            if(ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % 16 != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_dgrad requires non-null device pointers aligned to 16 bytes");
            }
        }
        const auto& bytes = dgradPrepared.bytes();
        if(!dgrad::bufferRangesDoNotOverlap({reinterpret_cast<uintptr_t>(dy),
                                             reinterpret_cast<uintptr_t>(w),
                                             reinterpret_cast<uintptr_t>(dx)},
                                            bytes))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_dgrad requires nonoverlapping gradient, filter, and input-gradient "
                "storage");
        }
        const auto& workspaces = dgradPrepared.workspaces();
        if(!workspaces.empty()
           && (workspace == nullptr || reinterpret_cast<uintptr_t>(workspace) % 16 != 0))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_dgrad requires a workspace aligned to 16 bytes");
        }
        // Each buffer a step can bind, with its size: the graph tensors and the regions
        // of the caller's workspace (SUB_GEMM is the prepared plan's own).
        const auto region = [&](dgrad::Buffer name) -> std::pair<char*, int64_t> {
            switch(name)
            {
            case dgrad::Buffer::DY:
                return {static_cast<char*>(dy), bytes[0]};
            case dgrad::Buffer::W:
                return {static_cast<char*>(w), bytes[1]};
            case dgrad::Buffer::DX:
                return {static_cast<char*>(dx), bytes[2]};
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
                "gfx950_conv_dgrad plan binds a buffer it does not lay out");
        };
        // Every step below goes on the launch stream, so the plan stays stream-ordered.
        const auto stream = handle.getStream();
        if(dgradPrepared.zeroDx())
        {
            // The records leave some dX rows or columns unwritten (a strided problem whose
            // filter misses a residue): _dx_fully_written.
            checkHip(hipMemsetAsync(dx, 0, static_cast<size_t>(bytes[2]), stream), "dX zero-fill");
        }
        for(const auto& [code, step] : dgradPrepared.steps())
        {
            // The step's record table, already on the device (DeviceTables).
            void* table = step.abi == dgrad::Abi::IMPLICIT
                              ? dgradPrepared.tables() + step.tableOffset
                              : nullptr;
            // The operands, each through the step's view of it where it has one.
            std::array<void*, 3> pointers{};
            std::array<int32_t, 3> sizes{};
            for(size_t i = 0; i < step.buffers.size(); ++i)
            {
                if(step.buffers[i] == dgrad::Buffer::NONE)
                {
                    continue;
                }
                auto [base, size] = region(step.buffers[i]);
                if(const auto& view = step.views[i])
                {
                    base += view->offset;
                    size = view->bytes;
                }
                pointers[i] = base;
                sizes[i] = static_cast<int32_t>(size);
            }
            std::vector<void*> params;
            if(step.abi == dgrad::Abi::TRANSFORM || step.abi == dgrad::Abi::TRANSFORM_RTC)
            {
                params = {&pointers[0], &pointers[2], &sizes[0], &sizes[2]};
            }
            else
            {
                params
                    = {&pointers[0], &pointers[1], &pointers[2], &sizes[0], &sizes[1], &sizes[2]};
            }
            // Every value but num_sub_gemms, which follows the table pointer.
            const auto plain
                = step.abi == dgrad::Abi::IMPLICIT ? step.values.size() - 1 : step.values.size();
            for(size_t i = 0; i < plain; ++i)
            {
                params.push_back(const_cast<int32_t*>(&step.values[i]));
            }
            if(step.abi == dgrad::Abi::IMPLICIT)
            {
                params.push_back(&table);
                params.push_back(const_cast<int32_t*>(&step.values.back()));
            }
            code.kernelForStream(stream).launchWithParams(stream, params.data());
        }
    }

private:
    const compilation::KpackKernelLoader& _loader;
};

const ConvDgradDispatchHandler& dispatchHandler()
{
    // The registries retain non-owning pointers; both dependencies live for the process,
    // including across provider handle destruction and recreation.
    static const compilation::KpackKernelLoader s_loader(gfx950ConvDgradAotKpackModuleCache());
    static const ConvDgradDispatchHandler s_handler(s_loader);
    return s_handler;
}

} // namespace

compilation::KpackModuleCache& gfx950ConvDgradAotKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void registerGfx950ConvDgradAotSymbols(SymbolScope<Handle>& scope)
{
    // No CandidateFilterFn: the catalog is small, so every serving instance is a
    // candidate and the ranker chooses.
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &aotKernelMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &aotPrepare);
    scope.add(std::string(SCORE_SYMBOL), &aotScore);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
}

void resetGfx950ConvDgradAotModuleCache()
{
    gfx950ConvDgradAotKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
