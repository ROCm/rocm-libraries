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
/// implicit-GEMM tile over the tilde sub-GEMMs (kernel_family 0), a direct depthwise or
/// grouped-MFMA kernel (kernel_family 1), or one of the MFMA pipeline's weight transforms
/// (kernel_family 2, never matched; the MFMA dispatch loads it beside its instance). Its
/// metadata holds tuning and capability fields only and the problem reaches the kernel as
/// AOT arguments, so matching is a capability check (Gfx950ConvDgradGeometry.hpp), not an
/// equality on the geometry. Mirrors rocke/library/builders/common/convolution_backward_data.py.
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
    if(!dgrad::tensorByteSize(extents))
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

template <typename T>
bool metadataEquals(const KernelDefinition& kernel, const char* field, const T& expected)
{
    const auto it = kernel.metadata.find(field);
    if(it == kernel.metadata.end())
    {
        return false;
    }
    const auto* value = std::get_if<T>(&it->second);
    return value != nullptr && *value == expected;
}

std::optional<int64_t> intMetadata(const KernelDefinition& kernel, const char* field)
{
    const auto it = kernel.metadata.find(field);
    const auto* value = it == kernel.metadata.end() ? nullptr : std::get_if<int64_t>(&it->second);
    return value == nullptr ? std::nullopt : std::optional<int64_t>(*value);
}

const std::string* stringMetadata(const KernelDefinition& kernel, const char* field)
{
    const auto it = kernel.metadata.find(field);
    return it == kernel.metadata.end() ? nullptr : std::get_if<std::string>(&it->second);
}

/// The implicit-GEMM tuning fields; a direct or helper kernel holds 0 in each.
constexpr std::array<const char*, 8> GEMM_TUNING_FIELDS{
    "tile_m", "tile_n", "tile_k", "warp_m", "warp_n", "warp_tile_m", "warp_tile_n", "warp_tile_k"};
/// The implicit-GEMM fields after the tuning; a direct or helper kernel holds 0 in each.
constexpr std::array<const char*, 6> GEMM_ONLY_FIELDS{
    "vector_size_a", "vector_size_b", "vector_size_c", "lds_k_outer", "max_sub_gemms", "grouped"};
/// What a direct or helper binary compiles in, in dgrad::DirectCapabilities order.
constexpr std::array<const char*, 10> DIRECT_CAPABILITY_FIELDS{"filter_h",
                                                               "filter_w",
                                                               "stride_h",
                                                               "stride_w",
                                                               "pad_h",
                                                               "pad_w",
                                                               "dilation_h",
                                                               "dilation_w",
                                                               "cpg",
                                                               "kpg"};
/// The direct and helper knobs, each 0 where its variant does not use it
/// (_KNOB_DEFAULTS in convolution_backward_data.py).
constexpr std::array<const char*, 10> DIRECT_KNOB_FIELDS{"block_w",
                                                         "block_waves",
                                                         "block_h",
                                                         "block_q",
                                                         "block_groups",
                                                         "fold_k32",
                                                         "waves_k",
                                                         "waves_q",
                                                         "persistent_grid",
                                                         "runtime_k_loop"};
/// The 0/1 knobs.
constexpr std::array<std::string_view, 3> DIRECT_FLAG_KNOBS{
    "fold_k32", "persistent_grid", "runtime_k_loop"};

/// The knobs @p variant states (AOT_DGRAD_DIRECT_KNOBS in convolution_backward_data.py);
/// every other one must hold 0.
bool directVariantUses(dgrad::DirectVariant variant, std::string_view field)
{
    const auto among = [&](std::initializer_list<std::string_view> fields) {
        return std::find(fields.begin(), fields.end(), field) != fields.end();
    };
    switch(variant)
    {
    case dgrad::DirectVariant::DEPTHWISE:
        return among({"block_w", "block_waves"});
    case dgrad::DirectVariant::MFMA:
        return among({"block_h",
                      "block_q",
                      "block_groups",
                      "fold_k32",
                      "waves_k",
                      "waves_q",
                      "persistent_grid",
                      "runtime_k_loop"});
    case dgrad::DirectVariant::TRANSPOSE:
    case dgrad::DirectVariant::REORGANIZE:
        return among({"fold_k32"});
    case dgrad::DirectVariant::NONE:
    default:
        return false;
    }
}

/// One instance of either family: implicit GEMM, or a direct or helper kernel.
using AotKernel = std::variant<dgrad::ImplicitInstance, dgrad::DirectInstance>;

/// The implicit-GEMM instance @p kernel describes (supports_gfx950_conv_dgrad_aot): an
/// mem/compv3/compv4 pipeline, a default or cshuffle epilogue, positive tuning, widths in
/// {1, 2, 4, 8}, 0/1 lds_k_outer and grouped, a positive max_sub_gemms, and every direct
/// field and knob at its KMD default.
std::optional<dgrad::ImplicitInstance> aotImplicitGemmInstance(const KernelDefinition& kernel)
{
    const auto* pipeline = stringMetadata(kernel, "pipeline");
    const auto* epilogue = stringMetadata(kernel, "epilogue");
    if(pipeline == nullptr || epilogue == nullptr
       || (*pipeline != "mem" && *pipeline != "compv3" && *pipeline != "compv4")
       || (*epilogue != "default" && *epilogue != "cshuffle"))
    {
        return std::nullopt;
    }
    const auto isZero
        = [&](const char* field) { return metadataEquals(kernel, field, int64_t{0}); };
    if(!std::all_of(DIRECT_CAPABILITY_FIELDS.begin(), DIRECT_CAPABILITY_FIELDS.end(), isZero)
       || !std::all_of(DIRECT_KNOB_FIELDS.begin(), DIRECT_KNOB_FIELDS.end(), isZero))
    {
        return std::nullopt;
    }
    std::array<int64_t, GEMM_TUNING_FIELDS.size()> tuning{};
    for(size_t i = 0; i < GEMM_TUNING_FIELDS.size(); ++i)
    {
        const auto value = intMetadata(kernel, GEMM_TUNING_FIELDS[i]);
        if(!value || *value < 1 || *value > dgrad::I32_MAX)
        {
            return std::nullopt;
        }
        tuning[i] = *value;
    }
    std::array<int64_t, 3> vectors{};
    for(size_t i = 0; i < vectors.size(); ++i)
    {
        const auto value = intMetadata(kernel, GEMM_ONLY_FIELDS[i]);
        if(!value || (*value != 1 && *value != 2 && *value != 4 && *value != 8))
        {
            return std::nullopt;
        }
        vectors[i] = *value;
    }
    const auto ldsKOuter = intMetadata(kernel, "lds_k_outer");
    const auto grouped = intMetadata(kernel, "grouped");
    const auto maxSubGemms = intMetadata(kernel, "max_sub_gemms");
    if(!ldsKOuter || (*ldsKOuter != 0 && *ldsKOuter != 1) || !grouped
       || (*grouped != 0 && *grouped != 1) || !maxSubGemms || *maxSubGemms < 1
       || *maxSubGemms > dgrad::I32_MAX)
    {
        return std::nullopt;
    }
    return dgrad::ImplicitInstance{tuning[0],
                                   tuning[1],
                                   tuning[2],
                                   tuning[3],
                                   tuning[4],
                                   vectors[0],
                                   vectors[1],
                                   vectors[2],
                                   *maxSubGemms,
                                   *grouped == 1};
}

/// The direct or helper instance @p kernel describes (_direct_error): the direct
/// identity's placeholders (0 tuning, widths, lds_k_outer, max_sub_gemms and grouped;
/// "direct" pipeline and epilogue), a variant of its family, a square undilated filter,
/// positive channel counts, one positive stride and padding for a direct kernel and 0 for
/// a helper, cpg == kpg == 1 for depthwise, and every knob its variant does not use at 0.
/// An MFMA instance's knobs also split its own K / groups (_mfma_knobs_reason).
std::optional<dgrad::DirectInstance> aotDirectInstance(const KernelDefinition& kernel,
                                                       dgrad::KernelFamily family,
                                                       dgrad::DirectVariant variant)
{
    const bool helper = dgrad::isHelperVariant(variant);
    if(variant == dgrad::DirectVariant::NONE || (family == dgrad::KernelFamily::HELPER) != helper
       || !metadataEquals(kernel, "pipeline", std::string("direct"))
       || !metadataEquals(kernel, "epilogue", std::string("direct")))
    {
        return std::nullopt;
    }
    const auto isZero
        = [&](const char* field) { return metadataEquals(kernel, field, int64_t{0}); };
    if(!std::all_of(GEMM_TUNING_FIELDS.begin(), GEMM_TUNING_FIELDS.end(), isZero)
       || !std::all_of(GEMM_ONLY_FIELDS.begin(), GEMM_ONLY_FIELDS.end(), isZero))
    {
        return std::nullopt;
    }
    std::array<int64_t, DIRECT_CAPABILITY_FIELDS.size()> compiled{};
    for(size_t i = 0; i < DIRECT_CAPABILITY_FIELDS.size(); ++i)
    {
        const auto value = intMetadata(kernel, DIRECT_CAPABILITY_FIELDS[i]);
        if(!value || *value < 0 || *value > dgrad::I32_MAX)
        {
            return std::nullopt;
        }
        compiled[i] = *value;
    }
    const dgrad::DirectCapabilities capabilities{compiled[0],
                                                 compiled[1],
                                                 compiled[2],
                                                 compiled[3],
                                                 compiled[4],
                                                 compiled[5],
                                                 compiled[6],
                                                 compiled[7],
                                                 compiled[8],
                                                 compiled[9]};
    if(capabilities.filterH < 1 || capabilities.filterH != capabilities.filterW
       || capabilities.dilationH != 1 || capabilities.dilationW != 1 || capabilities.cpg < 1
       || capabilities.kpg < 1)
    {
        return std::nullopt;
    }
    if(helper)
    {
        // A helper identity zeroes stride and padding: its code reads neither.
        if(capabilities.strideH != 0 || capabilities.strideW != 0 || capabilities.padH != 0
           || capabilities.padW != 0)
        {
            return std::nullopt;
        }
    }
    else if(capabilities.strideH < 1 || capabilities.strideH != capabilities.strideW
            || capabilities.padH != capabilities.padW)
    {
        return std::nullopt;
    }
    if(variant == dgrad::DirectVariant::DEPTHWISE
       && (capabilities.cpg != 1 || capabilities.kpg != 1))
    {
        return std::nullopt;
    }
    std::array<int64_t, DIRECT_KNOB_FIELDS.size()> knobs{};
    for(size_t i = 0; i < DIRECT_KNOB_FIELDS.size(); ++i)
    {
        const std::string_view field = DIRECT_KNOB_FIELDS[i];
        const auto value = intMetadata(kernel, DIRECT_KNOB_FIELDS[i]);
        if(!value)
        {
            return std::nullopt;
        }
        const bool flag = std::find(DIRECT_FLAG_KNOBS.begin(), DIRECT_FLAG_KNOBS.end(), field)
                          != DIRECT_FLAG_KNOBS.end();
        if(!directVariantUses(variant, field))
        {
            if(*value != 0)
            {
                return std::nullopt;
            }
        }
        else if(flag ? (*value != 0 && *value != 1) : (*value < 1 || *value > dgrad::I32_MAX))
        {
            return std::nullopt;
        }
        knobs[i] = *value;
    }
    dgrad::DirectInstance instance{variant, capabilities};
    instance.blockW = knobs[0];
    instance.blockWaves = knobs[1];
    instance.blockH = knobs[2];
    instance.blockQ = knobs[3];
    instance.blockGroups = knobs[4];
    instance.foldK32 = knobs[5];
    instance.wavesK = knobs[6];
    instance.wavesQ = knobs[7];
    instance.persistentGrid = knobs[8];
    instance.runtimeKLoop = knobs[9];
    // A transpose writes the plain layout; only a reorganize helper is keyed on fold_k32.
    if(variant == dgrad::DirectVariant::TRANSPOSE && instance.foldK32 != 0)
    {
        return std::nullopt;
    }
    if(variant == dgrad::DirectVariant::DEPTHWISE
       && instance.blockWaves > dgrad::MAX_BLOCK_THREADS / dgrad::WAVE_SIZE)
    {
        return std::nullopt;
    }
    if(variant == dgrad::DirectVariant::MFMA
       && (!dgrad::mfmaKnobsFit(instance, capabilities.kpg)
           || instance.blockGroups * instance.wavesQ * instance.wavesK
                  > dgrad::MAX_BLOCK_THREADS / dgrad::WAVE_SIZE))
    {
        return std::nullopt;
    }
    return instance;
}

/// The instance @p kernel describes, when its metadata is a complete instance of one
/// family with a launch contract this pack reviewed (supports_gfx950_conv_dgrad_aot):
/// a kpack kernel, NHWC, wave64, then the family's own fields. kernel_family is read
/// first, so no family's checks run on another's placeholders. Both value sets are closed
/// and a missing field (the loader completes omitted ones from the KMD defaults) is
/// malformed metadata.
std::optional<AotKernel> aotInstance(const KernelDefinition& kernel)
{
    if(kernel.source.kind != KernelSourceKind::KPACK
       || !metadataEquals(kernel, "layout", std::string("NHWC"))
       || !metadataEquals(kernel, "wave_size", int64_t{64}))
    {
        return std::nullopt;
    }
    const auto familyValue = intMetadata(kernel, "kernel_family");
    const auto family = familyValue ? dgrad::parseKernelFamily(*familyValue) : std::nullopt;
    const auto* variantName = stringMetadata(kernel, "direct_variant");
    const auto variant
        = variantName == nullptr ? std::nullopt : dgrad::parseDirectVariant(*variantName);
    if(!family || !variant)
    {
        return std::nullopt;
    }
    if(*family == dgrad::KernelFamily::IMPLICIT_GEMM)
    {
        if(*variant != dgrad::DirectVariant::NONE)
        {
            return std::nullopt;
        }
        return aotImplicitGemmInstance(kernel);
    }
    return aotDirectInstance(kernel, *family, *variant);
}

/// Where a helper is looked up: the same pack, the same archive, the instance's dtype,
/// and the identity fields gfx950_conv_dgrad_aot_helper_fields names (CONTRACT §4).
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

HelperKey helperKey(const KernelDefinition& kernel,
                    const std::string& dtype,
                    const dgrad::DirectInstance& instance,
                    dgrad::DirectVariant variant)
{
    return HelperKey{archiveOf(kernel),
                     kernel.packId,
                     dtype,
                     variant,
                     instance.compiled.filterH,
                     instance.compiled.cpg,
                     instance.compiled.kpg,
                     variant == dgrad::DirectVariant::REORGANIZE ? instance.foldK32 : 0};
}

/// The helpers an MFMA instance's pipeline launches, in launch order: the transpose
/// always, the reorganize when it reads coalesced weights. std::nullopt when either is
/// missing from its pack, which refuses the instance.
struct MfmaHelpers
{
    KernelDefinition transpose;
    std::optional<KernelDefinition> reorganize;
};

std::optional<MfmaHelpers> findMfmaHelpers(const KernelDefinition& kernel,
                                           const std::string& dtype,
                                           const dgrad::DirectInstance& instance)
{
    auto transpose = helperRegistry().find(
        helperKey(kernel, dtype, instance, dgrad::DirectVariant::TRANSPOSE));
    if(!transpose)
    {
        return std::nullopt;
    }
    MfmaHelpers helpers{std::move(*transpose), std::nullopt};
    if(dgrad::usesCoalescedWeights(instance))
    {
        helpers.reorganize = helperRegistry().find(
            helperKey(kernel, dtype, instance, dgrad::DirectVariant::REORGANIZE));
        if(!helpers.reorganize)
        {
            return std::nullopt;
        }
    }
    return helpers;
}

/// What hipkernel:Gfx950ConvDgradAot reads of one kernel on every graph, decoded from its
/// metadata once: its dtype and its instance (nullopt for metadata no reviewed launch
/// contract describes).
struct AotPrepared
{
    std::string dtype;
    std::optional<AotKernel> instance;
};

AotPrepared decodeAot(const KernelDefinition& kernel)
{
    const auto* dtype = stringMetadata(kernel, "dtype");
    return AotPrepared{dtype == nullptr ? std::string() : *dtype, aotInstance(kernel)};
}

/// The KernelPrepareFn registered under the kernel matcher's symbol: the SDK runs it once
/// per kernel when the engine loads, so matching a graph reads AotPrepared instead of
/// parsing some forty metadata fields per kernel. It also files each helper kernel for
/// the MFMA pipeline to find. Must not throw.
std::shared_ptr<const void> aotPrepare(const KernelDefinition& kernel)
{
    try
    {
        auto prepared = std::make_shared<const AotPrepared>(decodeAot(kernel));
        const auto* direct = prepared->instance
                                 ? std::get_if<dgrad::DirectInstance>(&*prepared->instance)
                                 : nullptr;
        if(direct != nullptr && dgrad::isHelperVariant(direct->variant)
           && (prepared->dtype == "fp16" || prepared->dtype == "bf16"))
        {
            helperRegistry().add(helperKey(kernel, prepared->dtype, *direct, direct->variant),
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

/// A depthwise launch: one kernel, no workspace.
struct DepthwiseLaunch
{
    dgrad::LaunchGeometry geometry;
};

/// Everything a kernel needs at launch on one problem, per family.
using DgradLaunch = std::variant<dgrad::ImplicitLaunch, DepthwiseLaunch, dgrad::MfmaPlan>;

/// The launch of @p instance on @p matched, through the family's own served predicate
/// (dgrad::implicitInstanceLaunch, dgrad::depthwiseLaunch, dgrad::mfmaLaunchPlan); for
/// MFMA its helpers must be in its pack too. A helper serves nothing.
std::optional<DgradLaunch> aotInstanceLaunch(const MatchedProblem& matched,
                                             const KernelDefinition& kernel,
                                             const std::string& dtype,
                                             const AotKernel& instance)
{
    if(const auto* gemm = std::get_if<dgrad::ImplicitInstance>(&instance))
    {
        auto launch = dgrad::implicitInstanceLaunch(matched.problem, matched.geometry, *gemm);
        if(!launch
           || !dgrad::implicitGemmKernargs(
               matched.problem, matched.geometry, gemm->tileM, gemm->tileN))
        {
            return std::nullopt;
        }
        return DgradLaunch{std::move(*launch)};
    }
    const auto& direct = std::get<dgrad::DirectInstance>(instance);
    if(direct.variant == dgrad::DirectVariant::DEPTHWISE)
    {
        const auto launch = dgrad::depthwiseLaunch(matched.problem, matched.geometry, direct);
        return launch ? std::optional<DgradLaunch>(DepthwiseLaunch{*launch}) : std::nullopt;
    }
    if(direct.variant == dgrad::DirectVariant::MFMA)
    {
        const auto plan = dgrad::mfmaLaunchPlan(matched.problem, matched.geometry, direct);
        if(!plan || !findMfmaHelpers(kernel, dtype, direct))
        {
            return std::nullopt;
        }
        return DgradLaunch{*plan};
    }
    return std::nullopt;
}

const char* dtypeName(data_objects::DataType dtype)
{
    return dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
}

/// The launch when @p kernel is packaged code that computes @p matched under a launch
/// contract this pack reviewed: the dtype pinned, then the family's served predicate.
std::optional<DgradLaunch> aotKernelFits(const MatchedProblem& matched,
                                         const KernelDefinition& kernel)
{
    AotPrepared scratch;
    const auto& prepared = preparedOf(kernel, scratch);
    if(prepared.dtype != dtypeName(matched.dtype) || !prepared.instance)
    {
        return std::nullopt;
    }
    return aotInstanceLaunch(matched, kernel, prepared.dtype, *prepared.instance);
}

bool aotKernelMatches(const MatchContext& context,
                      const BoundTokens& bound,
                      const KernelDefinition& kernel)
{
    const auto matched = matchedFromBound(context, bound);
    return matched && aotKernelFits(*matched, kernel).has_value();
}

double
    aotScore(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
{
    // Deterministic fallback only; a trained ranker or BenchmarkPlan replaces it. Only
    // kernels the matcher admitted are scored, so a refusal (and every helper) ranks last.
    // Depthwise ranks first, then MFMA, then implicit GEMM by its useful tile fraction
    // (dgrad::directInstanceScore, dgrad::aotInstanceScore).
    const auto matched = matchedFromBound(context, bound);
    AotPrepared scratch;
    const auto& prepared = preparedOf(kernel, scratch);
    if(!matched || !prepared.instance || prepared.dtype != dtypeName(matched->dtype))
    {
        return -1.0;
    }
    const auto launch = aotInstanceLaunch(*matched, kernel, prepared.dtype, *prepared.instance);
    if(!launch)
    {
        return -1.0;
    }
    if(const auto* gemm = std::get_if<dgrad::ImplicitInstance>(&*prepared.instance))
    {
        return dgrad::aotInstanceScore(
            matched->problem, matched->geometry, *gemm, std::get<dgrad::ImplicitLaunch>(*launch));
    }
    const auto& direct = std::get<dgrad::DirectInstance>(*prepared.instance);
    const auto geometry = std::holds_alternative<DepthwiseLaunch>(*launch)
                              ? std::get<DepthwiseLaunch>(*launch).geometry
                              : std::get<dgrad::MfmaPlan>(*launch).compute;
    return dgrad::directInstanceScore(matched->problem, matched->geometry, direct, geometry);
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

/// The 44-argument dgrad implicit-GEMM ABI: (dY*, W*, dX*, dY_bytes, W_bytes, dX_bytes),
/// the 36 i32 problem values, then sub_gemm_buf (a device pointer) and num_sub_gemms.
const std::vector<KernelArgument>& implicitGemmSignature()
{
    static const auto s_signature = [] {
        auto arguments = directArguments({"dY", "W", "dX", "dY_bytes", "W_bytes", "dX_bytes"},
                                         dgrad::IMPLICIT_GEMM_KERNARG_NAMES);
        arguments.push_back({"sub_gemm_buf", true});
        arguments.push_back({"num_sub_gemms", false});
        return kernelSignature(arguments);
    }();
    return s_signature;
}

/// The 20-argument dgrad direct ABI of the depthwise kernels.
const std::vector<KernelArgument>& depthwiseSignature()
{
    static const auto s_signature = kernelSignature(directArguments(
        {"A", "B", "D", "A_bytes", "B_bytes", "D_bytes"}, dgrad::DEPTHWISE_KERNARG_NAMES));
    return s_signature;
}

/// The forward direct ABI the MFMA compute pass takes on the transposed problem.
const std::vector<KernelArgument>& mfmaComputeSignature()
{
    static const auto s_signature = kernelSignature(directArguments(
        {"A", "B", "D", "A_bytes", "B_bytes", "D_bytes"}, dgrad::DIRECT_KERNARG_NAMES));
    return s_signature;
}

/// The weight transforms' (A*, D*, A_bytes, D_bytes).
const std::vector<KernelArgument>& transformSignature()
{
    static const auto s_signature
        = kernelSignature({{"A", true}, {"D", true}, {"A_bytes", false}, {"D_bytes", false}});
    return s_signature;
}

IngestorKernelCode loadKernel(const compilation::KpackKernelLoader& loader,
                              const MatchContext& context,
                              const KernelDefinition& kernel,
                              const std::vector<KernelArgument>& signature,
                              const dgrad::LaunchGeometry& geometry)
{
    auto code = buildIngestorKernelCode(loader, context, kernel, signature);
    code.setBlockSize(geometry.blockX, 1, 1);
    code.setGridSize(geometry.gridX, geometry.gridY, geometry.gridZ);
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

/// The implicit-GEMM launch: the kernel, its problem block, and the packed sub-GEMM table
/// launch() uploads into the workspace on the stream before it runs.
struct ImplicitPrepared
{
    IngestorKernelCode code;
    dgrad::ImplicitGemmKernargs kernargs;
    std::vector<int32_t> table;
    int32_t numSubGemms;
    bool zeroDx;
};

struct DepthwisePrepared
{
    IngestorKernelCode code;
    dgrad::DepthwiseKernargs kernargs;
};

/// The MFMA pipeline: transpose -> (reorganize) -> compute, with the workspace layout.
struct MfmaPrepared
{
    IngestorKernelCode transpose;
    std::optional<IngestorKernelCode> reorganize;
    IngestorKernelCode compute;
    dgrad::MfmaComputeKernargs kernargs;
    int64_t wsTBytes;
    int64_t wsCoaOffset;
    int64_t wsCoaBytes;
};

class PreparedConvDgrad : public PreparedDispatch
{
public:
    using Launch = std::variant<ImplicitPrepared, DepthwisePrepared, MfmaPrepared>;

    PreparedConvDgrad(Binding binding, std::array<int32_t, 3> bytes, Launch launch)
        : _binding(binding)
        , _bytes(bytes)
        , _launch(std::move(launch))
    {
    }

    const Binding& binding() const
    {
        return _binding;
    }
    /// dY, W, dX storage sizes in bytes.
    const std::array<int32_t, 3>& bytes() const
    {
        return _bytes;
    }
    const Launch& launch() const
    {
        return _launch;
    }

private:
    Binding _binding;
    std::array<int32_t, 3> _bytes;
    // Holds every program/module and kernel view; owns no MatchContext data.
    Launch _launch;
};

class ConvDgradDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit ConvDgradDispatchHandler(const compilation::KpackKernelLoader& loader)
        : _loader(loader)
    {
    }

    /// The implicit-GEMM sub-GEMM table, (22 + 8) * num_sub_gemms * 4 bytes; the MFMA
    /// pipeline's ws_t plus, when it reorganizes, ws_coa at ws_t rounded up to 256;
    /// nothing for depthwise. A kernel the graph does not admit needs none: prepare()
    /// refuses it.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto matched = matchProblem(context);
        const auto launch = matched ? aotKernelFits(*matched, kernel) : std::nullopt;
        if(!launch)
        {
            return 0;
        }
        if(const auto* gemm = std::get_if<dgrad::ImplicitLaunch>(&*launch))
        {
            return gemm->workspaceBytes;
        }
        if(const auto* plan = std::get_if<dgrad::MfmaPlan>(&*launch))
        {
            return static_cast<size_t>(plan->workspaceBytes);
        }
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto matched = matchProblem(context);
        auto launch = matched ? aotKernelFits(*matched, kernel) : std::nullopt;
        if(!launch)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_dgrad graph or packaged kernel does not satisfy the convolution "
                "contract");
        }
        requireBindings(bound, matched->binding);
        const auto& p = matched->problem;
        const auto& g = matched->geometry;

        // This pack ships prebuilt code objects only; aotKernelFits already refused every
        // other source kind. Each family has its own ABI, so the signature follows it.
        if(auto* gemm = std::get_if<dgrad::ImplicitLaunch>(&*launch))
        {
            AotPrepared scratch;
            const auto& instance
                = std::get<dgrad::ImplicitInstance>(*preparedOf(kernel, scratch).instance);
            const auto kernargs = dgrad::implicitGemmKernargs(p, g, instance.tileM, instance.tileN);
            return std::make_unique<PreparedConvDgrad>(
                matched->binding,
                g.tensorBytes,
                ImplicitPrepared{
                    loadKernel(_loader, context, kernel, implicitGemmSignature(), gemm->geometry),
                    *kernargs,
                    std::move(gemm->table),
                    static_cast<int32_t>(gemm->numSubGemms),
                    gemm->zeroDx});
        }
        if(const auto* depthwise = std::get_if<DepthwiseLaunch>(&*launch))
        {
            return std::make_unique<PreparedConvDgrad>(
                matched->binding,
                g.tensorBytes,
                DepthwisePrepared{
                    loadKernel(_loader, context, kernel, depthwiseSignature(), depthwise->geometry),
                    dgrad::depthwiseKernargs(p, g)});
        }
        const auto& plan = std::get<dgrad::MfmaPlan>(*launch);
        AotPrepared scratch;
        const auto& prepared = preparedOf(kernel, scratch);
        const auto& instance = std::get<dgrad::DirectInstance>(*prepared.instance);
        const auto helpers = findMfmaHelpers(kernel, prepared.dtype, instance);
        if(!helpers || helpers->reorganize.has_value() != plan.reorganize.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_dgrad MFMA instance has no weight-transform helper in its pack");
        }
        std::optional<IngestorKernelCode> reorganize;
        if(plan.reorganize)
        {
            reorganize = loadKernel(
                _loader, context, *helpers->reorganize, transformSignature(), *plan.reorganize);
        }
        return std::make_unique<PreparedConvDgrad>(
            matched->binding,
            g.tensorBytes,
            MfmaPrepared{
                loadKernel(
                    _loader, context, helpers->transpose, transformSignature(), plan.transpose),
                std::move(reorganize),
                loadKernel(_loader, context, kernel, mfmaComputeSignature(), plan.compute),
                dgrad::mfmaComputeKernargs(p, g, plan),
                plan.wsTBytes,
                plan.wsCoaOffset,
                plan.wsCoaBytes});
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
        const auto stream = handle.getStream();
        const auto requireWorkspace = [&] {
            if(workspace == nullptr || reinterpret_cast<uintptr_t>(workspace) % 16 != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_dgrad requires a workspace aligned to 16 bytes");
            }
        };
        // Every step below goes on the launch stream, so the plan stays stream-ordered.
        std::visit(
            [&](const auto& launch) {
                using T = std::decay_t<decltype(launch)>;
                if constexpr(std::is_same_v<T, ImplicitPrepared>)
                {
                    requireWorkspace();
                    if(launch.zeroDx)
                    {
                        // A strided problem leaves the dX rows and columns no tap reaches
                        // unwritten: needs_zeroed_dx.
                        checkHip(hipMemsetAsync(dx, 0, static_cast<size_t>(bytes[2]), stream),
                                 "dX zero-fill");
                    }
                    // The table lives in the prepared plan, so the host source outlives the
                    // copy even when it completes asynchronously. The upload runs on every
                    // launch, since the caller may reuse the workspace between launches; its
                    // cost is part of every implicit-GEMM candidate's measured time.
                    checkHip(hipMemcpyAsync(workspace,
                                            launch.table.data(),
                                            launch.table.size() * sizeof(int32_t),
                                            hipMemcpyHostToDevice,
                                            stream),
                             "sub-GEMM table upload");
                    void* table = workspace;
                    const int32_t numSubGemms = launch.numSubGemms;
                    const auto& kernel = launch.code.kernelForStream(stream);
                    std::apply(
                        [&](const auto&... values) {
                            kernel.launch(stream,
                                          dy,
                                          w,
                                          dx,
                                          bytes[0],
                                          bytes[1],
                                          bytes[2],
                                          values...,
                                          table,
                                          numSubGemms);
                        },
                        launch.kernargs);
                }
                else if constexpr(std::is_same_v<T, DepthwisePrepared>)
                {
                    const auto& kernel = launch.code.kernelForStream(stream);
                    std::apply(
                        [&](const auto&... values) {
                            kernel.launch(
                                stream, dy, w, dx, bytes[0], bytes[1], bytes[2], values...);
                        },
                        launch.kernargs);
                }
                else
                {
                    requireWorkspace();
                    auto* base = static_cast<char*>(workspace);
                    void* wsT = base;
                    const auto wsTBytes = static_cast<int32_t>(launch.wsTBytes);
                    launch.transpose.kernelForStream(stream).launch(
                        stream, w, wsT, bytes[1], wsTBytes);
                    void* weights = wsT;
                    int32_t weightBytes = wsTBytes;
                    if(launch.reorganize)
                    {
                        void* wsCoa = base + launch.wsCoaOffset;
                        const auto wsCoaBytes = static_cast<int32_t>(launch.wsCoaBytes);
                        launch.reorganize->kernelForStream(stream).launch(
                            stream, wsT, wsCoa, wsTBytes, wsCoaBytes);
                        weights = wsCoa;
                        weightBytes = wsCoaBytes;
                    }
                    const auto& kernel = launch.compute.kernelForStream(stream);
                    std::apply(
                        [&](const auto&... values) {
                            kernel.launch(stream,
                                          dy,
                                          weights,
                                          dx,
                                          bytes[0],
                                          weightBytes,
                                          bytes[2],
                                          values...);
                        },
                        launch.kernargs);
                }
            },
            dgradPrepared.launch());
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
