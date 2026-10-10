// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
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
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdAotSelection.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdGeometry.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace conv = gfx950_conv_fwd;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_conv_fwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_conv_fwd.dispatch";
constexpr std::string_view X_TOKEN = "gfx950_conv_fwd.x.uid";
constexpr std::string_view W_TOKEN = "gfx950_conv_fwd.w.uid";
constexpr std::string_view Y_TOKEN = "gfx950_conv_fwd.y.uid";
/// The whole matched problem, so a kernel matcher reads it from what graph_match bound
/// instead of walking the graph again for every kernel: problemTokenValues() order.
constexpr std::string_view PROBLEM_TOKEN = "gfx950_conv_fwd.problem";

struct Binding
{
    int64_t x;
    int64_t w;
    int64_t y;
};

struct MatchedProblem
{
    Binding binding;
    conv::Problem problem;
    conv::Geometry geometry;
    data_objects::DataType dtype;
};

/// The AOT problem block of one family's ABI, the arguments after the leading six.
using ConvKernargs = std::variant<conv::ImplicitGemmKernargs, conv::DirectKernargs>;

/// Everything a kernel needs at launch beyond the bound device pointers.
struct ConvLaunch
{
    conv::LaunchGeometry geometry;
    ConvKernargs kernargs;
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
    if(!conv::tensorByteSize(extents))
    {
        return false;
    }
    // hipDNN [N,C,H,W] -> NHWC; filter [K,C,Y,X] -> KYXC; output [N,K,Ho,Wo] -> NHWK.
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
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes
       || node.attributes() == nullptr || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& attributes = node.attributesAs<data_objects::ConvolutionFwdAttributes>();
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
        attributes.x_tensor_uid(), attributes.w_tensor_uid(), attributes.y_tensor_uid()};
    const auto* x = findTensor(context, binding.x);
    const auto* w = findTensor(context, binding.w);
    const auto* y = findTensor(context, binding.y);
    if(binding.x == binding.w || binding.x == binding.y || binding.w == binding.y
       || !isDeviceTensor(x) || !isDeviceTensor(w) || !isDeviceTensor(y) || !hasPackedNhwcStrides(x)
       || !hasPackedNhwcStrides(w) || x->data_type() != w->data_type()
       || x->data_type() != y->data_type())
    {
        return std::nullopt;
    }
    // Output dims/strides may not yet be inferred. Validate them in prepare(), before
    // loading any code or computing device addresses; they do not define this problem.
    const auto* xDims = x->dims();
    const auto* wDims = w->dims();
    // hipDNN carries no groups attribute: a filter holding C / groups channels encodes it.
    // The layout checks above already made both channel counts positive.
    const auto channelsPerGroup = wDims->Get(1);
    if(channelsPerGroup <= 0 || xDims->Get(1) % channelsPerGroup != 0)
    {
        return std::nullopt;
    }
    const auto groups = xDims->Get(1) / channelsPerGroup;
    // gridDim.z carries the group, and each group owns K / groups output channels.
    if(groups < 1 || groups > 65535 || wDims->Get(0) % groups != 0)
    {
        return std::nullopt;
    }
    const conv::Problem problem{xDims->Get(0),
                                xDims->Get(1),
                                wDims->Get(0),
                                xDims->Get(2),
                                xDims->Get(3),
                                wDims->Get(2),
                                wDims->Get(3),
                                attributes.stride()->Get(0),
                                attributes.stride()->Get(1),
                                attributes.pre_padding()->Get(0),
                                attributes.pre_padding()->Get(1),
                                attributes.dilation()->Get(0),
                                attributes.dilation()->Get(1),
                                groups};
    // Grouped pointwise graphs are matched too. rocKE's pointwise shortcut ignores groups,
    // but no packaged pointwise binary is grouped (the adapter refuses to build one) and
    // conv::implicitGemmServes gives a grouped graph only grouped binaries, which index
    // through their descriptors.
    const auto geometry = conv::deriveGeometry(problem);
    if(!geometry)
    {
        return std::nullopt;
    }
    return MatchedProblem{binding, problem, *geometry, x->data_type()};
}

/// PROBLEM_TOKEN's value: the binding, the problem in conv::Problem's field order, then
/// the dtype. The geometry is derived from the problem again on reading.
std::vector<int64_t> problemTokenValues(const MatchedProblem& matched)
{
    const auto& p = matched.problem;
    return {matched.binding.x,
            matched.binding.w,
            matched.binding.y,
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
    if(values == nullptr || values->size() != 18)
    {
        return matchProblem(context);
    }
    const auto& v = *values;
    const conv::Problem problem{
        v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16]};
    const auto geometry = conv::deriveGeometry(problem);
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
                       {std::string(W_TOKEN), matched->binding.w},
                       {std::string(Y_TOKEN), matched->binding.y},
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

/// The implicit-GEMM tuning fields; a direct kernel holds 0 in each.
constexpr std::array<const char*, 8> GEMM_TUNING_FIELDS{
    "tile_m", "tile_n", "tile_k", "warp_m", "warp_n", "warp_tile_m", "warp_tile_n", "warp_tile_k"};

/// The problem a packaged kernel was compiled for, from its metadata. rocKE's AOT kernels
/// take the problem as kernel arguments, so this no longer has to equal the graph; it
/// records which build-time capabilities the binary has (see conv::implicitGemmCapabilities
/// and conv::directServes).
std::optional<conv::Problem> buildProblem(const KernelDefinition& kernel)
{
    std::array<int64_t, 14> values{};
    const std::array<const char*, 14> fields{
        "N", "C", "K", "Hi", "Wi", "Y", "X", "sH", "sW", "pH", "pW", "dH", "dW", "groups"};
    for(size_t i = 0; i < fields.size(); ++i)
    {
        const auto value = intMetadata(kernel, fields[i]);
        if(!value)
        {
            return std::nullopt;
        }
        values[i] = *value;
    }
    const conv::Problem problem{values[0],
                                values[1],
                                values[2],
                                values[3],
                                values[4],
                                values[5],
                                values[6],
                                values[7],
                                values[8],
                                values[9],
                                values[10],
                                values[11],
                                values[12],
                                values[13]};
    // A malformed build problem says nothing reliable about the binary's capabilities.
    if(!conv::deriveGeometry(problem))
    {
        return std::nullopt;
    }
    return problem;
}

/// Whether @p kernel's implicit-GEMM pipeline has the launch this pack computes. These
/// pipelines all launch warp_m * warp_n * 64 threads over an (N-tiles, M-tiles, groups)
/// grid with static LDS; "wavelet" appends load waves and is not accepted.
bool hasReviewedPipeline(const KernelDefinition& kernel)
{
    const auto* pipelineName = stringMetadata(kernel, "pipeline");
    return pipelineName != nullptr
           && (*pipelineName == "mem" || *pipelineName == "compv3" || *pipelineName == "compv4"
               || *pipelineName == "basic");
}

std::optional<ConvLaunch> implicitGemmLaunch(const MatchedProblem& matched,
                                             const KernelDefinition& kernel,
                                             const conv::Problem& build)
{
    // What stays reviewed here is the launch contract and the binary's capabilities.
    const auto& p = matched.problem;
    if(!hasReviewedPipeline(kernel))
    {
        return std::nullopt;
    }
    // rocKE builds the default epilogue only with scalar stores, which its default
    // vector width gives only for an odd build K / groups; any other descriptor is
    // malformed. A scalar store then serves every graph's K / groups.
    const bool cshuffle = metadataEquals(kernel, "epilogue", std::string("cshuffle"));
    if(!cshuffle
       && !(metadataEquals(kernel, "epilogue", std::string("default"))
            && (build.k / build.groups) % 2 != 0))
    {
        return std::nullopt;
    }

    std::array<int64_t, GEMM_TUNING_FIELDS.size()> tuning{};
    for(size_t i = 0; i < GEMM_TUNING_FIELDS.size(); ++i)
    {
        const auto value = intMetadata(kernel, GEMM_TUNING_FIELDS[i]);
        if(!value || *value <= 0)
        {
            return std::nullopt;
        }
        tuning[i] = *value;
    }
    const auto capabilities = conv::implicitGemmCapabilities(
        build, tuning[0], tuning[1], tuning[2], tuning[3], tuning[4], cshuffle);
    if(!capabilities || !conv::implicitGemmServes(*capabilities, p))
    {
        return std::nullopt;
    }
    const auto launch
        = conv::launchGeometry(p, matched.geometry, tuning[0], tuning[1], tuning[3], tuning[4], 64);
    // The kernel decodes its workgroup id against tile counts passed at launch, so they
    // come from the same tile the grid did.
    const auto kernargs
        = launch ? conv::implicitGemmKernargs(p, matched.geometry, *launch) : std::nullopt;
    if(!kernargs)
    {
        return std::nullopt;
    }
    return ConvLaunch{*launch, *kernargs};
}

/// Mirrors _direct_serves in rocke/library/builders/common/convolution_forward.py and the
/// padding rule of rocKE's direct validators: see conv::directInstanceLaunch. Everything
/// the guard declines stays on implicit GEMM.
std::optional<ConvLaunch> directLaunch(const MatchedProblem& matched,
                                       const KernelDefinition& kernel,
                                       const conv::Problem& build,
                                       conv::DirectVariant variant,
                                       int64_t blockW,
                                       int64_t blockWaves)
{
    // The implicit-GEMM fields a direct kernel does not read hold exact placeholders, so
    // a forced tile_k of 64 or 128 never selects one and no descriptor is ambiguous.
    for(const auto* field : GEMM_TUNING_FIELDS)
    {
        if(!metadataEquals(kernel, field, int64_t{0}))
        {
            return std::nullopt;
        }
    }
    // The exact-shape engine packages the std and spatial kernels only.
    if(!metadataEquals(kernel, "pipeline", std::string("none"))
       || !metadataEquals(kernel, "epilogue", std::string("none"))
       || (variant != conv::DirectVariant::STD && variant != conv::DirectVariant::SPATIAL))
    {
        return std::nullopt;
    }
    const auto launch = conv::directInstanceLaunch(
        matched.problem,
        matched.geometry,
        {conv::directCapabilities(build), variant, blockW, blockWaves});
    if(!launch)
    {
        return std::nullopt;
    }
    return ConvLaunch{*launch, conv::directKernargs(matched.problem, matched.geometry)};
}

/// The launch when @p kernel is packaged code that computes this graph under a launch
/// contract this pack reviewed; std::nullopt otherwise. rocKE's AOT kernels take the
/// problem as kernel arguments, so a kernel serves every graph its build-time
/// capabilities admit, not only the shape it was compiled for.
std::optional<ConvLaunch> kernelFits(const MatchedProblem& matched, const KernelDefinition& kernel)
{
    if(kernel.source.kind != KernelSourceKind::KPACK)
    {
        return std::nullopt;
    }
    const auto build = buildProblem(kernel);
    const std::string dtype = matched.dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
    if(!build || !metadataEquals(kernel, "dtype", dtype)
       || !metadataEquals(kernel, "layout", std::string("NHWC"))
       || !metadataEquals(kernel, "wave_size", int64_t{64}))
    {
        return std::nullopt;
    }

    // Both value sets are closed: an unknown family or variant is a kernel this build
    // cannot launch, never one to guess at. The descriptor loader completes omitted
    // fields with their KMD defaults, so a missing field is malformed metadata.
    const auto familyValue = intMetadata(kernel, "kernel_family");
    const auto family = familyValue ? conv::parseKernelFamily(*familyValue) : std::nullopt;
    const auto* variantName = stringMetadata(kernel, "direct_variant");
    const auto variant
        = variantName == nullptr ? std::nullopt : conv::parseDirectVariant(*variantName);
    const auto blockW = intMetadata(kernel, "block_w");
    const auto blockWaves = intMetadata(kernel, "block_waves");
    if(!family || !variant || !blockW || !blockWaves)
    {
        return std::nullopt;
    }
    if(*family == conv::KernelFamily::IMPLICIT_GEMM)
    {
        // The direct fields stay at their KMD defaults, so no descriptor carries a
        // direct arm that this branch would silently launch as implicit GEMM.
        if(*variant != conv::DirectVariant::NONE || *blockW != 0 || *blockWaves != 0)
        {
            return std::nullopt;
        }
        return implicitGemmLaunch(matched, kernel, *build);
    }
    return directLaunch(matched, kernel, *build, *variant, *blockW, *blockWaves);
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchProblem(context);
    return matched && kernelFits(*matched, kernel).has_value();
}

/// Whether @p kernel is gfx950_conv_fwd_direct_spec_for_request's choice with no
/// overrides for the group count it was built for (see conv::isDefaultDirectArm).
bool isDefaultDirectArm(const KernelDefinition& kernel)
{
    const auto groups = intMetadata(kernel, "groups");
    const auto* variantName = stringMetadata(kernel, "direct_variant");
    const auto variant
        = variantName == nullptr ? std::nullopt : conv::parseDirectVariant(*variantName);
    const auto blockW = intMetadata(kernel, "block_w");
    const auto blockWaves = intMetadata(kernel, "block_waves");
    return groups && variant && blockW && blockWaves
           && conv::isDefaultDirectArm(*groups, *variant, *blockW, *blockWaves);
}

/// How well @p kernel's build-time specialization suits @p matched, in [0, 0.85): the
/// kernel compiled for exactly this graph (rocKE's dispatcher pick for it) first, then the
/// specialized code paths the graph can use (the pointwise GEMM, the ungrouped path), then
/// wider load and store widths. Below one step of score()'s family and tile levels.
double specializationBonus(const MatchedProblem& matched, const KernelDefinition& kernel)
{
    const auto build = buildProblem(kernel);
    if(!build)
    {
        return 0.0;
    }
    const auto& p = matched.problem;
    const bool exact = build->n == p.n && build->c == p.c && build->k == p.k && build->hi == p.hi
                       && build->wi == p.wi && build->y == p.y && build->x == p.x
                       && build->strideH == p.strideH && build->strideW == p.strideW
                       && build->padH == p.padH && build->padW == p.padW
                       && build->dilationH == p.dilationH && build->dilationW == p.dilationW
                       && build->groups == p.groups;
    double bonus = exact ? 0.5 : 0.0;
    if(metadataEquals(
           kernel, "kernel_family", static_cast<int64_t>(conv::KernelFamily::IMPLICIT_GEMM)))
    {
        const auto tile = [&](const char* field) { return intMetadata(kernel, field).value_or(0); };
        const auto capabilities = conv::implicitGemmCapabilities(
            *build,
            tile("tile_m"),
            tile("tile_n"),
            tile("tile_k"),
            tile("warp_m"),
            tile("warp_n"),
            metadataEquals(kernel, "epilogue", std::string("cshuffle")));
        if(capabilities)
        {
            bonus += capabilities->pointwise ? 0.1 : 0.0;
            bonus += !capabilities->grouped && p.groups == 1 ? 0.1 : 0.0;
            bonus += 0.02 * std::log2(static_cast<double>(capabilities->loadWidth))
                     + 0.02 * std::log2(static_cast<double>(capabilities->storeWidth));
        }
    }
    return bonus;
}

double
    score(const MatchContext& context, const BoundTokens& /*bound*/, const KernelDefinition& kernel)
{
    // Deterministic fallback only. BenchmarkPlan replaces this order with measurements
    // across both families when benchmarking is enabled. A direct depthwise kernel ranks
    // first where one fits, its default arm highest. Next is the gfx950 dispatcher's own
    // implicit-GEMM tile, so an unbenchmarked run without a direct kernel serves what
    // rocKE would; its tile_k=128 sibling follows. Every kernel the matcher admits computes
    // the graph, and specializationBonus orders those that share a level.
    const auto matched = matchProblem(context);
    const auto bonus = matched ? specializationBonus(*matched, kernel) : 0.0;
    if(metadataEquals(
           kernel, "kernel_family", static_cast<int64_t>(conv::KernelFamily::DIRECT_DEPTHWISE)))
    {
        return (isDefaultDirectArm(kernel) ? 4.0 : 3.0) + bonus;
    }
    const bool dispatcherTile = metadataEquals(kernel, "tile_m", int64_t{64})
                                && metadataEquals(kernel, "tile_n", int64_t{64})
                                && metadataEquals(kernel, "warp_m", int64_t{2})
                                && metadataEquals(kernel, "warp_n", int64_t{2})
                                && metadataEquals(kernel, "warp_tile_m", int64_t{32})
                                && metadataEquals(kernel, "warp_tile_n", int64_t{32})
                                && metadataEquals(kernel, "warp_tile_k", int64_t{16})
                                && metadataEquals(kernel, "pipeline", std::string("mem"));
    if(!dispatcherTile)
    {
        return bonus;
    }
    return (metadataEquals(kernel, "tile_k", int64_t{64}) ? 2.0 : 1.0) + bonus;
}

// ---- hipkernel:Gfx950ConvFwdAot: shape-generic AOT instances ----
//
// Each kernel is one rocKE forward KernelIdentity, implicit GEMM or direct depthwise: its
// metadata holds tuning and capability fields only, and the problem reaches the kernel as
// AOT arguments computed from the graph. Matching is therefore a capability check
// (conv::aotInstanceLaunch, conv::directInstanceLaunch), not an equality on the geometry.

constexpr std::string_view AOT_GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd_aot.graph_match";
constexpr std::string_view AOT_KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd_aot.kernel_match";
constexpr std::string_view AOT_SCORE_SYMBOL = "hipkernel.gfx950_conv_fwd_aot.score";
constexpr std::string_view AOT_DISPATCH_SYMBOL = "hipkernel.gfx950_conv_fwd_aot.dispatch";

/// Integer instance fields; supports_gfx950_conv_fwd_aot validated every one at build.
constexpr std::array<const char*, 13> AOT_INT_FIELDS{"tile_m",
                                                     "tile_n",
                                                     "tile_k",
                                                     "warp_m",
                                                     "warp_n",
                                                     "warp_tile_m",
                                                     "warp_tile_n",
                                                     "warp_tile_k",
                                                     "vector_size_a",
                                                     "vector_size_b",
                                                     "vector_size_c",
                                                     "async_chunk_a",
                                                     "async_chunk_b"};
constexpr std::array<const char*, 3> AOT_FLAG_FIELDS{"unroll_k", "async_dma", "grouped"};
/// What a direct instance compiles in, in conv::DirectCapabilities order.
constexpr std::array<const char*, 8> AOT_DIRECT_FIELDS{
    "filter_h", "filter_w", "stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w"};

/// The block knobs after block_w and block_waves, with the KMD default each holds where
/// its variant does not use it (_AOT_DIRECT_KNOB_DEFAULTS in convolution_forward.py, less
/// the string "atom"). iglp -1 and waves_per_eu 0 stand for rocKE's None.
constexpr std::array<std::pair<const char*, int64_t>, 16> AOT_DIRECT_KNOB_DEFAULTS{{
    {"block_h", 0},
    {"unroll_rows", 0},
    {"block_q", 0},
    {"block_groups", 0},
    {"double_buffer", 0},
    {"cpg", 0},
    {"kpg", 0},
    {"tile_h", 0},
    {"tile_w", 0},
    {"tile_k_out", 0},
    {"ck", 0},
    {"waves_m", 0},
    {"waves_n", 0},
    {"swizzle_wgm", 0},
    {"iglp", -1},
    {"waves_per_eu", 0},
}};

/// The fields of block_w, block_waves, AOT_DIRECT_KNOB_DEFAULTS and "atom" that
/// @p variant states (AOT_DIRECT_KNOBS and _AOT_DIRECT_EXTRA_FIELDS in
/// convolution_forward.py); every other one must hold its default.
bool aotDirectVariantUses(conv::DirectVariant variant, std::string_view field)
{
    const auto among = [&](std::initializer_list<std::string_view> fields) {
        return std::find(fields.begin(), fields.end(), field) != fields.end();
    };
    switch(variant)
    {
    case conv::DirectVariant::STD:
        return among({"block_w", "block_waves"});
    case conv::DirectVariant::SPATIAL:
        return among({"block_waves"});
    case conv::DirectVariant::TILED:
        return among({"block_w", "block_h", "block_waves", "unroll_rows"});
    case conv::DirectVariant::COL:
        return among({"block_w", "block_h", "block_waves"});
    case conv::DirectVariant::GROUPED:
        return among({"block_h", "block_q", "block_groups", "double_buffer", "cpg", "kpg"});
    case conv::DirectVariant::NONGROUPED:
        return among({"tile_h",
                      "tile_w",
                      "tile_k_out",
                      "ck",
                      "waves_m",
                      "waves_n",
                      "atom",
                      "swizzle_wgm",
                      "iglp",
                      "waves_per_eu"});
    case conv::DirectVariant::NONE:
    default:
        return false;
    }
}

/// One instance of either family: an implicit-GEMM identity or a direct one.
using AotKernel = std::variant<conv::AotInstance, conv::DirectInstance>;

/// The implicit-GEMM instance @p kernel describes: a reviewed pipeline (unroll_k and
/// async_dma change only the K loop), a default or cshuffle epilogue, and positive
/// tuning. Unlike the exact-shape engine, the store width is the identity's
/// vector_size_c, not derived.
std::optional<conv::AotInstance> aotImplicitGemmInstance(const KernelDefinition& kernel)
{
    if(!hasReviewedPipeline(kernel)
       || (!metadataEquals(kernel, "epilogue", std::string("default"))
           && !metadataEquals(kernel, "epilogue", std::string("cshuffle"))))
    {
        return std::nullopt;
    }
    std::array<int64_t, AOT_INT_FIELDS.size()> values{};
    for(size_t i = 0; i < AOT_INT_FIELDS.size(); ++i)
    {
        const auto value = intMetadata(kernel, AOT_INT_FIELDS[i]);
        // Every field but the async chunk widths is positive; those are 0 without async.
        if(!value || *value < (i >= 11 ? 0 : 1))
        {
            return std::nullopt;
        }
        values[i] = *value;
    }
    std::array<bool, AOT_FLAG_FIELDS.size()> flags{};
    for(size_t i = 0; i < AOT_FLAG_FIELDS.size(); ++i)
    {
        const auto value = intMetadata(kernel, AOT_FLAG_FIELDS[i]);
        if(!value || (*value != 0 && *value != 1))
        {
            return std::nullopt;
        }
        flags[i] = *value == 1;
    }
    return conv::AotInstance{values[0],
                             values[1],
                             values[3],
                             values[4],
                             values[8],
                             values[9],
                             values[10],
                             flags[1],
                             values[11],
                             values[12],
                             flags[2]};
}

/// The direct instance @p kernel describes. Its implicit-GEMM fields hold the values
/// rocKE's direct sweep writes into a direct identity (0, and "direct" for the pipeline
/// and epilogue), so a forced tile_k of 64 or 128 never selects one. Each variant states
/// its own block knobs (aotDirectVariantUses) and leaves every other one at its KMD
/// default; the flags hold 0 or 1. The knobs' ranges are checked where they are used, by
/// conv::directInstanceLaunch.
std::optional<conv::DirectInstance> aotDirectInstance(const KernelDefinition& kernel,
                                                      conv::DirectVariant variant,
                                                      int64_t blockW,
                                                      int64_t blockWaves)
{
    if(variant == conv::DirectVariant::NONE
       || !metadataEquals(kernel, "pipeline", std::string("direct"))
       || !metadataEquals(kernel, "epilogue", std::string("direct")))
    {
        return std::nullopt;
    }
    const auto isZero
        = [&](const char* field) { return metadataEquals(kernel, field, int64_t{0}); };
    if(!std::all_of(AOT_INT_FIELDS.begin(), AOT_INT_FIELDS.end(), isZero)
       || !std::all_of(AOT_FLAG_FIELDS.begin(), AOT_FLAG_FIELDS.end(), isZero))
    {
        return std::nullopt;
    }
    std::array<int64_t, AOT_DIRECT_FIELDS.size()> compiled{};
    for(size_t i = 0; i < AOT_DIRECT_FIELDS.size(); ++i)
    {
        const auto value = intMetadata(kernel, AOT_DIRECT_FIELDS[i]);
        if(!value)
        {
            return std::nullopt;
        }
        compiled[i] = *value;
    }
    if((!aotDirectVariantUses(variant, "block_w") && blockW != 0)
       || (!aotDirectVariantUses(variant, "block_waves") && blockWaves != 0))
    {
        return std::nullopt;
    }
    std::array<int64_t, AOT_DIRECT_KNOB_DEFAULTS.size()> knobs{};
    for(size_t i = 0; i < AOT_DIRECT_KNOB_DEFAULTS.size(); ++i)
    {
        const auto& [field, unset] = AOT_DIRECT_KNOB_DEFAULTS[i];
        const auto value = intMetadata(kernel, field);
        if(!value || (!aotDirectVariantUses(variant, field) && *value != unset))
        {
            return std::nullopt;
        }
        knobs[i] = *value;
    }
    const auto knob = [&](std::string_view field) {
        for(size_t i = 0; i < AOT_DIRECT_KNOB_DEFAULTS.size(); ++i)
        {
            if(AOT_DIRECT_KNOB_DEFAULTS[i].first == field)
            {
                return knobs[i];
            }
        }
        return int64_t{-1};
    };
    // Compile-only knobs: they shape the code, not the launch, but must be well formed.
    for(const auto* flag : {"unroll_rows", "double_buffer"})
    {
        if(knob(flag) != 0 && knob(flag) != 1)
        {
            return std::nullopt;
        }
    }
    if(knob("iglp") < -1 || knob("waves_per_eu") < 0 || knob("swizzle_wgm") < 0)
    {
        return std::nullopt;
    }
    const auto* atom = stringMetadata(kernel, "atom");
    if(atom == nullptr)
    {
        return std::nullopt;
    }
    int64_t atomTile = 0;
    if(aotDirectVariantUses(variant, "atom"))
    {
        const auto tile = conv::nongroupedAtomTile(*atom);
        if(!tile || knob("swizzle_wgm") < 1)
        {
            return std::nullopt;
        }
        atomTile = *tile;
    }
    else if(*atom != "none")
    {
        return std::nullopt;
    }
    conv::DirectInstance instance{{compiled[0],
                                   compiled[1],
                                   compiled[2],
                                   compiled[3],
                                   compiled[4],
                                   compiled[5],
                                   compiled[6],
                                   compiled[7]},
                                  variant,
                                  blockW,
                                  blockWaves};
    instance.blockH = knob("block_h");
    instance.blockQ = knob("block_q");
    instance.blockGroups = knob("block_groups");
    instance.cpg = knob("cpg");
    instance.kpg = knob("kpg");
    instance.tileH = knob("tile_h");
    instance.tileW = knob("tile_w");
    instance.tileKOut = knob("tile_k_out");
    instance.ck = knob("ck");
    instance.wavesM = knob("waves_m");
    instance.wavesN = knob("waves_n");
    instance.atomTile = atomTile;
    return instance;
}

/// The instance @p kernel describes, when its metadata is a complete instance of either
/// family with a launch contract this pack reviewed. kernel_family is read first, so
/// neither family's checks run on the other's placeholders. As in kernelFits, both value
/// sets are closed and a missing field (the loader completes them from the KMD defaults)
/// is malformed metadata.
std::optional<AotKernel> aotInstance(const KernelDefinition& kernel)
{
    if(kernel.source.kind != KernelSourceKind::KPACK
       || !metadataEquals(kernel, "layout", std::string("NHWC"))
       || !metadataEquals(kernel, "wave_size", int64_t{64}))
    {
        return std::nullopt;
    }
    const auto familyValue = intMetadata(kernel, "kernel_family");
    const auto family = familyValue ? conv::parseKernelFamily(*familyValue) : std::nullopt;
    const auto* variantName = stringMetadata(kernel, "direct_variant");
    const auto variant
        = variantName == nullptr ? std::nullopt : conv::parseDirectVariant(*variantName);
    const auto blockW = intMetadata(kernel, "block_w");
    const auto blockWaves = intMetadata(kernel, "block_waves");
    if(!family || !variant || !blockW || !blockWaves)
    {
        return std::nullopt;
    }
    if(*family == conv::KernelFamily::DIRECT_DEPTHWISE)
    {
        return aotDirectInstance(kernel, *variant, *blockW, *blockWaves);
    }
    // An implicit-GEMM instance leaves every direct field at its KMD default, so no
    // descriptor carries a direct arm that this branch would launch as implicit GEMM.
    if(*variant != conv::DirectVariant::NONE || *blockW != 0 || *blockWaves != 0)
    {
        return std::nullopt;
    }
    if(!std::all_of(AOT_DIRECT_FIELDS.begin(),
                    AOT_DIRECT_FIELDS.end(),
                    [&](const char* field) { return metadataEquals(kernel, field, int64_t{0}); })
       || !std::all_of(
           AOT_DIRECT_KNOB_DEFAULTS.begin(),
           AOT_DIRECT_KNOB_DEFAULTS.end(),
           [&](const auto& knob) { return metadataEquals(kernel, knob.first, knob.second); })
       || !metadataEquals(kernel, "atom", std::string("none")))
    {
        return std::nullopt;
    }
    return aotImplicitGemmInstance(kernel);
}

/// What hipkernel:Gfx950ConvFwdAot reads of one kernel on every graph, decoded from its
/// metadata once: its dtype and epilogue, and its instance (nullopt for metadata no
/// reviewed launch contract describes).
struct AotPrepared
{
    std::string dtype;
    std::optional<std::string> epilogue;
    std::optional<AotKernel> instance;
};

AotPrepared decodeAot(const KernelDefinition& kernel)
{
    const auto* dtype = stringMetadata(kernel, "dtype");
    const auto* epilogue = stringMetadata(kernel, "epilogue");
    return AotPrepared{dtype == nullptr ? std::string() : *dtype,
                       epilogue == nullptr ? std::nullopt : std::optional<std::string>(*epilogue),
                       aotInstance(kernel)};
}

/// The KernelPrepareFn registered under the AOT kernel matcher's symbol: the SDK runs it
/// once per kernel when the engine loads, so matching a graph reads AotPrepared instead
/// of parsing about fifty metadata fields per kernel.
std::shared_ptr<const void> aotPrepare(const KernelDefinition& kernel)
{
    return std::make_shared<const AotPrepared>(decodeAot(kernel));
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

/// The launch of @p instance on @p matched, through the family's own capability check
/// and ABI (conv::aotInstanceLaunch or conv::directInstanceLaunch).
std::optional<ConvLaunch> aotInstanceLaunch(const MatchedProblem& matched,
                                            const AotKernel& instance)
{
    if(const auto* direct = std::get_if<conv::DirectInstance>(&instance))
    {
        const auto launch = conv::directInstanceLaunch(matched.problem, matched.geometry, *direct);
        if(!launch)
        {
            return std::nullopt;
        }
        return ConvLaunch{*launch, conv::directKernargs(matched.problem, matched.geometry)};
    }
    const auto launch = conv::aotInstanceLaunch(
        matched.problem, matched.geometry, std::get<conv::AotInstance>(instance));
    const auto kernargs
        = launch ? conv::implicitGemmKernargs(matched.problem, matched.geometry, *launch)
                 : std::nullopt;
    if(!kernargs)
    {
        return std::nullopt;
    }
    return ConvLaunch{*launch, *kernargs};
}

std::optional<ConvLaunch> aotKernelFits(const MatchedProblem& matched,
                                        const KernelDefinition& kernel)
{
    const char* dtype = matched.dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
    AotPrepared scratch;
    const auto& prepared = preparedOf(kernel, scratch);
    if(prepared.dtype != dtype || !prepared.instance)
    {
        return std::nullopt;
    }
    return aotInstanceLaunch(matched, *prepared.instance);
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
    // Deterministic fallback only. Only kernels the kernel matcher admitted are scored,
    // so a refusal here is a contract violation, ranked last (-1). As in score(), a
    // depthwise direct instance (std, spatial, tiled, col) ranks above every
    // implicit-GEMM one (conv::aotInstanceScore stays in [0, 2)), the default direct arm
    // first: implicit GEMM leaves most lanes idle on one channel per group. An instance
    // carries no build group count, so the default arm is the one for the graph's.
    //
    // The grouped and non-grouped direct kernels compete with implicit GEMM on ordinary
    // graphs, and nothing yet measures which wins where. Until rocKE's dispatcher or a
    // tuned heuristic ranks them, they take a neutral placeholder below every
    // implicit-GEMM instance (and above a refusal): an unbenchmarked plan keeps serving
    // implicit GEMM wherever it can, and the direct kernels are chosen by benchmarking,
    // a recorded winner, the kernel_family knob, or where nothing else serves.
    const auto matched = matchedFromBound(context, bound);
    AotPrepared scratch;
    const auto& instance = preparedOf(kernel, scratch).instance;
    const auto launch = matched && instance ? aotInstanceLaunch(*matched, *instance) : std::nullopt;
    if(!launch)
    {
        return -1.0;
    }
    if(const auto* direct = std::get_if<conv::DirectInstance>(&*instance))
    {
        if(!conv::isDepthwiseDirectVariant(direct->variant))
        {
            return -0.5;
        }
        return conv::isDefaultDirectArm(
                   matched->problem.groups, direct->variant, direct->blockW, direct->blockWaves)
                   ? 4.0
                   : 3.0;
    }
    return conv::aotInstanceScore(matched->problem,
                                  matched->geometry,
                                  std::get<conv::AotInstance>(*instance),
                                  launch->geometry);
}

/// Which of the instances serving a graph run: the CandidateFilterFn registered under the
/// kernel matcher's symbol, so it sees every instance aotKernelMatches admitted at once.
/// Among implicit-GEMM instances it is rocKE's own runtime selection
/// (conv::widestVectorSelection); "serves" is the capability, this is the policy. rocKE's
/// direct benchmark runs every direct kernel that serves, so direct instances all stay.
std::vector<bool> aotCandidateFilter(const MatchContext& /*context*/,
                                     const BoundTokens& /*bound*/,
                                     const std::vector<const KernelDefinition*>& candidates)
{
    std::vector<bool> keep(candidates.size(), true);
    std::vector<size_t> implicitGemm;
    std::vector<conv::AotSelectionKey> keys;
    AotPrepared scratch;
    for(size_t i = 0; i < candidates.size(); ++i)
    {
        const auto& prepared = preparedOf(*candidates[i], scratch);
        const auto* gemm
            = prepared.instance ? std::get_if<conv::AotInstance>(&*prepared.instance) : nullptr;
        if(gemm != nullptr && prepared.epilogue)
        {
            implicitGemm.push_back(i);
            keys.push_back({*prepared.epilogue, gemm->vectorA, gemm->vectorB, gemm->vectorC});
        }
    }
    const auto selected = conv::widestVectorSelection(keys);
    for(size_t j = 0; j < implicitGemm.size(); ++j)
    {
        keep[implicitGemm[j]] = selected[j];
    }
    return keep;
}

/// How one engine of this pack admits a kernel for a matched graph and launches it:
/// kernelFits for hipkernel:Gfx950ConvFwd, aotKernelFits for hipkernel:Gfx950ConvFwdAot.
using KernelFits = std::optional<ConvLaunch> (*)(const MatchedProblem&, const KernelDefinition&);

void requireBindings(const BoundTokens& bound, const Binding& binding)
{
    if(tryGetBoundInt(bound, X_TOKEN) != std::optional<int64_t>(binding.x)
       || tryGetBoundInt(bound, W_TOKEN) != std::optional<int64_t>(binding.w)
       || tryGetBoundInt(bound, Y_TOKEN) != std::optional<int64_t>(binding.y))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950_conv_fwd dispatch requires this graph matcher's tensor bindings");
    }
}

void requireOutput(const MatchContext& context, const MatchedProblem& matched)
{
    const auto* output = findTensor(context, matched.binding.y);
    if(!hasPackedNhwcStrides(output) || output->dims()->Get(0) != matched.problem.n
       || output->dims()->Get(1) != matched.problem.k
       || output->dims()->Get(2) != matched.geometry.ho
       || output->dims()->Get(3) != matched.geometry.wo)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950_conv_fwd requires packed NHWK output with the inferred convolution dimensions");
    }
}

/// The kernel ABI buildIngestorKernelCode verifies the loaded symbol against: rocKE's AOT
/// conv kernels open with (A*, B*, D*, A_bytes:i32, B_bytes:i32, D_bytes:i32) and follow
/// with @p names as i32s (rocke/library/kernels/common/conv_abi.py).
///
/// NAMES ARE LOAD-BEARING. Every problem argument is a by_value i32, so kind and size alone
/// cannot tell p_Hi from p_Wi; a reordered list would launch with every argument past the
/// first divergence shifted. requireSignatureMatch compares names whenever both sides
/// carry one, and hkp_pack records the builder's parameter names.
template <size_t Count>
std::vector<KernelArgument> convFwdKernelSignature(const std::array<std::string_view, Count>& names)
{
    constexpr auto POINTER_BYTES = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32_BYTES = static_cast<uint32_t>(sizeof(int32_t));
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    for(const auto* name : {"A", "B", "D"})
    {
        signature.push_back({"global_buffer", POINTER_BYTES, offset, name});
        offset += POINTER_BYTES;
    }
    for(const auto* name : {"A_bytes", "B_bytes", "D_bytes"})
    {
        signature.push_back({"by_value", I32_BYTES, offset, name});
        offset += I32_BYTES;
    }
    for(const auto name : names)
    {
        signature.push_back({"by_value", I32_BYTES, offset, std::string(name)});
        offset += I32_BYTES;
    }
    return signature;
}

const std::vector<KernelArgument>& kernelSignature(const ConvKernargs& kernargs)
{
    static const auto s_implicitGemm = convFwdKernelSignature(conv::IMPLICIT_GEMM_KERNARG_NAMES);
    static const auto s_direct = convFwdKernelSignature(conv::DIRECT_KERNARG_NAMES);
    return std::holds_alternative<conv::DirectKernargs>(kernargs) ? s_direct : s_implicitGemm;
}

class PreparedConvFwd : public PreparedDispatch
{
public:
    PreparedConvFwd(IngestorKernelCode code,
                    Binding binding,
                    std::array<int32_t, 3> bytes,
                    ConvKernargs kernargs)
        : _code(std::move(code))
        , _binding(binding)
        , _bytes(bytes)
        , _kernargs(kernargs)
    {
    }

    /// A hipModule_t belongs to the device it loaded on, so the kernel is resolved per
    /// launch stream rather than held as one pointer.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }
    const Binding& binding() const
    {
        return _binding;
    }
    const std::array<int32_t, 3>& bytes() const
    {
        return _bytes;
    }
    const ConvKernargs& kernargs() const
    {
        return _kernargs;
    }

private:
    // Holds both the program/module and its kernel view; owns no MatchContext data.
    IngestorKernelCode _code;
    Binding _binding;
    std::array<int32_t, 3> _bytes;
    ConvKernargs _kernargs;
};

class ConvFwdDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvFwdDispatchHandler(const compilation::KpackKernelLoader& loader, KernelFits fits)
        : _loader(loader)
        , _fits(fits)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        // Implicit GEMM uses registers and statically allocated LDS only; the direct
        // depthwise kernels use no LDS at all.
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto matched = matchProblem(context);
        const auto launch = matched ? _fits(*matched, kernel) : std::nullopt;
        if(!launch)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_fwd graph or packaged kernel does not satisfy the convolution "
                "contract");
        }
        requireBindings(bound, matched->binding);
        requireOutput(context, *matched);

        // This pack ships prebuilt code objects only; _fits already refused every
        // other source kind. Each family has its own ABI, so the signature follows it.
        auto code
            = buildIngestorKernelCode(_loader, context, kernel, kernelSignature(launch->kernargs));
        // _fits computed this kernel's launch for this graph's geometry before
        // loading the archive. No kernel of this pack uses dynamic LDS.
        code.setBlockSize(launch->geometry.blockX, 1, 1);
        code.setGridSize(launch->geometry.gridX, launch->geometry.gridY, launch->geometry.gridZ);
        code.setSharedMemBytes(0);
        return std::make_unique<PreparedConvFwd>(
            std::move(code), matched->binding, matched->geometry.tensorBytes, launch->kernargs);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& convPrepared = dynamic_cast<const PreparedConvFwd&>(prepared);
        if(deviceBuffers == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "gfx950_conv_fwd requires device buffers");
        }
        const auto& binding = convPrepared.binding();
        const auto a
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto b
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto d
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);
        for(const auto* ptr : {a.ptr, b.ptr, d.ptr})
        {
            if(ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % 16 != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_fwd requires non-null device pointers aligned to 16 bytes");
            }
        }
        const auto& bytes = convPrepared.bytes();
        if(!conv::bufferRangesDoNotOverlap({reinterpret_cast<uintptr_t>(a.ptr),
                                            reinterpret_cast<uintptr_t>(b.ptr),
                                            reinterpret_cast<uintptr_t>(d.ptr)},
                                           bytes))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_fwd requires nonoverlapping input, filter, and output storage");
        }
        // A*, B*, D*, A_bytes:i32, B_bytes:i32, D_bytes:i32, then the family's AOT problem
        // block in kernelSignature's order. The byte counts are storage sizes, not element
        // counts.
        const auto& kernel = convPrepared.kernelForStream(handle.getStream());
        std::visit(
            [&](const auto& kernargs) {
                std::apply(
                    [&](const auto&... values) {
                        kernel.launch(handle.getStream(),
                                      a.ptr,
                                      b.ptr,
                                      d.ptr,
                                      bytes[0],
                                      bytes[1],
                                      bytes[2],
                                      values...);
                    },
                    kernargs);
            },
            convPrepared.kernargs());
    }

private:
    const compilation::KpackKernelLoader& _loader;
    KernelFits _fits;
};

const ConvFwdDispatchHandler& dispatchHandler()
{
    // The registries retain non-owning pointers; both dependencies live for the
    // process, including across provider handle destruction and recreation.
    static const compilation::KpackKernelLoader s_loader(gfx950ConvFwdKpackModuleCache());
    static const ConvFwdDispatchHandler s_handler(s_loader, &kernelFits);
    return s_handler;
}

const ConvFwdDispatchHandler& aotDispatchHandler()
{
    static const compilation::KpackKernelLoader s_loader(gfx950ConvFwdAotKpackModuleCache());
    static const ConvFwdDispatchHandler s_handler(s_loader, &aotKernelFits);
    return s_handler;
}

} // namespace

compilation::KpackModuleCache& gfx950ConvFwdKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void registerGfx950ConvFwdSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
}

void resetGfx950ConvFwdModuleCache()
{
    gfx950ConvFwdKpackModuleCache().clear();
}

compilation::KpackModuleCache& gfx950ConvFwdAotKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void registerGfx950ConvFwdAotSymbols(SymbolScope<Handle>& scope)
{
    // Both engines read the graph the same way; only the kernels they admit differ.
    scope.add(std::string(AOT_GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(AOT_KERNEL_MATCHER_SYMBOL), &aotKernelMatches);
    scope.add(std::string(AOT_KERNEL_MATCHER_SYMBOL), &aotCandidateFilter);
    scope.add(std::string(AOT_KERNEL_MATCHER_SYMBOL), &aotPrepare);
    scope.add(std::string(AOT_SCORE_SYMBOL), &aotScore);
    scope.add(std::string(AOT_DISPATCH_SYMBOL), &aotDispatchHandler());
}

void resetGfx950ConvFwdAotModuleCache()
{
    gfx950ConvFwdAotKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
