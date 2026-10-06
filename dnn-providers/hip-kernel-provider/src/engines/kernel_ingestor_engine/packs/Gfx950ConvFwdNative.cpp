// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
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
    // Known rocKE bug: the is_pointwise shortcut (1x1 filter, unit stride, no padding)
    // indexes A and D as a plain GEMM and ignores groups, so grouped pointwise kernels
    // compute wrong results. Same predicate as ConvProblem.is_pointwise.
    const bool pointwise = problem.y == 1 && problem.x == 1 && problem.strideH == 1
                           && problem.strideW == 1 && problem.padH == 0 && problem.padW == 0;
    if(groups > 1 && pointwise)
    {
        return std::nullopt;
    }
    const auto geometry = conv::deriveGeometry(problem);
    if(!geometry)
    {
        return std::nullopt;
    }
    return MatchedProblem{binding, problem, *geometry, x->data_type()};
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
                       {std::string(Y_TOKEN), matched->binding.y}};
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

std::optional<ConvLaunch> implicitGemmLaunch(const MatchedProblem& matched,
                                             const KernelDefinition& kernel)
{
    // The tuning values are whatever the build validated for this exact geometry: rocKE's
    // is_valid_spec_for_problem ran on every variant before it was compiled, and the
    // geometry fields above pin each kernel to one problem. What stays reviewed here is the
    // launch contract. These pipelines all launch warp_m * warp_n * 64 threads over an
    // (N-tiles, M-tiles, groups) grid with static LDS; "wavelet" appends load waves and is
    // not accepted. rocKE derives vec_c from the per-group K (default_vector_sizes(cpg,
    // kpg)) and picks vec_c > 1 when it is even; the default epilogue then needs scalar
    // stores, so an even K / groups takes cshuffle only.
    const auto& p = matched.problem;
    const auto* pipelineName = stringMetadata(kernel, "pipeline");
    if(pipelineName == nullptr
       || (*pipelineName != "mem" && *pipelineName != "compv3" && *pipelineName != "compv4"
           && *pipelineName != "basic"))
    {
        return std::nullopt;
    }
    // The default epilogue stores scalars, so it is valid only when K/groups is odd.
    const bool defaultEpilogueAllowed
        = (p.k / p.groups) % 2 != 0 && metadataEquals(kernel, "epilogue", std::string("default"));
    if(!metadataEquals(kernel, "epilogue", std::string("cshuffle")) && !defaultEpilogueAllowed)
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

/// Mirrors _direct_error in rocke/library/builders/common/convolution_forward.py and the
/// padding rule of rocKE's direct validators: see conv::directProblemSupported. Everything
/// the guard declines stays on implicit GEMM.
std::optional<ConvLaunch> directLaunch(const MatchedProblem& matched,
                                       const KernelDefinition& kernel,
                                       conv::DirectVariant variant,
                                       int64_t blockW,
                                       int64_t blockWaves)
{
    const auto& p = matched.problem;
    if(!conv::directProblemSupported(p))
    {
        return std::nullopt;
    }
    // The implicit-GEMM fields a direct kernel does not read hold exact placeholders, so
    // a forced tile_k of 64 or 128 never selects one and no descriptor is ambiguous.
    for(const auto* field : GEMM_TUNING_FIELDS)
    {
        if(!metadataEquals(kernel, field, int64_t{0}))
        {
            return std::nullopt;
        }
    }
    if(!metadataEquals(kernel, "pipeline", std::string("none"))
       || !metadataEquals(kernel, "epilogue", std::string("none")))
    {
        return std::nullopt;
    }
    const auto launch = conv::directLaunchGeometry(
        p, matched.geometry, variant, blockW, blockWaves, conv::DIRECT_WAVE_SIZE);
    if(!launch)
    {
        return std::nullopt;
    }
    return ConvLaunch{*launch, conv::directKernargs(p, matched.geometry)};
}

/// The launch when @p kernel is packaged code compiled for exactly this problem under a
/// launch contract this pack reviewed; std::nullopt otherwise.
std::optional<ConvLaunch> kernelFits(const MatchedProblem& matched, const KernelDefinition& kernel)
{
    if(kernel.source.kind != KernelSourceKind::KPACK)
    {
        return std::nullopt;
    }
    const auto& p = matched.problem;
    const std::array<std::pair<const char*, int64_t>, 13> fields{{{"N", p.n},
                                                                  {"C", p.c},
                                                                  {"K", p.k},
                                                                  {"Hi", p.hi},
                                                                  {"Wi", p.wi},
                                                                  {"Y", p.y},
                                                                  {"X", p.x},
                                                                  {"sH", p.strideH},
                                                                  {"sW", p.strideW},
                                                                  {"pH", p.padH},
                                                                  {"pW", p.padW},
                                                                  {"dH", p.dilationH},
                                                                  {"dW", p.dilationW}}};
    for(const auto& field : fields)
    {
        if(!metadataEquals(kernel, field.first, field.second))
        {
            return std::nullopt;
        }
    }
    const std::string dtype = matched.dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
    if(!metadataEquals(kernel, "dtype", dtype)
       || !metadataEquals(kernel, "layout", std::string("NHWC"))
       || !metadataEquals(kernel, "groups", p.groups)
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
        return implicitGemmLaunch(matched, kernel);
    }
    return directLaunch(matched, kernel, *variant, *blockW, *blockWaves);
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchProblem(context);
    return matched && kernelFits(*matched, kernel).has_value();
}

/// Whether @p kernel is gfx950_conv_fwd_direct_spec_for_request's choice with no
/// overrides: spatial with one wave below 64 groups, otherwise std block_w 4, one wave.
bool isDefaultDirectArm(const KernelDefinition& kernel)
{
    const auto groups = intMetadata(kernel, "groups");
    if(!groups)
    {
        return false;
    }
    if(*groups < conv::DIRECT_WAVE_SIZE)
    {
        return metadataEquals(kernel, "direct_variant", std::string("spatial"))
               && metadataEquals(kernel, "block_w", int64_t{0})
               && metadataEquals(kernel, "block_waves", conv::DEFAULT_DIRECT_BLOCK_WAVES);
    }
    return metadataEquals(kernel, "direct_variant", std::string("std"))
           && metadataEquals(kernel, "block_w", conv::DEFAULT_DIRECT_BLOCK_W)
           && metadataEquals(kernel, "block_waves", conv::DEFAULT_DIRECT_BLOCK_WAVES);
}

double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& kernel)
{
    // Deterministic fallback only. BenchmarkPlan replaces this order with measurements
    // across both families when benchmarking is enabled. A direct depthwise kernel ranks
    // first where one fits, its default arm highest. Next is the gfx950 dispatcher's own
    // implicit-GEMM pick, so an unbenchmarked run without a direct kernel serves what
    // rocKE would; its tile_k=128 sibling follows.
    if(metadataEquals(
           kernel, "kernel_family", static_cast<int64_t>(conv::KernelFamily::DIRECT_DEPTHWISE)))
    {
        return isDefaultDirectArm(kernel) ? 4.0 : 3.0;
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
        return 0.0;
    }
    return metadataEquals(kernel, "tile_k", int64_t{64}) ? 2.0 : 1.0;
}

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
    explicit ConvFwdDispatchHandler(const compilation::KpackKernelLoader& loader)
        : _loader(loader)
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
        const auto launch = matched ? kernelFits(*matched, kernel) : std::nullopt;
        if(!launch)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_fwd graph or packaged kernel does not satisfy the convolution "
                "contract");
        }
        requireBindings(bound, matched->binding);
        requireOutput(context, *matched);

        // This pack ships prebuilt code objects only; kernelFits already refused every
        // other source kind. Each family has its own ABI, so the signature follows it.
        auto code
            = buildIngestorKernelCode(_loader, context, kernel, kernelSignature(launch->kernargs));
        // kernelFits computed this family's launch for this exact geometry before loading
        // the archive. Neither family uses dynamic LDS.
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
};

const ConvFwdDispatchHandler& dispatchHandler()
{
    // The registries retain non-owning pointers; both dependencies live for the
    // process, including across provider handle destruction and recreation.
    static const compilation::KpackKernelLoader s_loader(gfx950ConvFwdKpackModuleCache());
    static const ConvFwdDispatchHandler s_handler(s_loader);
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

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
