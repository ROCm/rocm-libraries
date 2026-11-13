// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/HipConvDgradDwRdnaLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file HipConvDgradDwRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_dgrad_dw_rdna`
 *        (descriptors/hip/conv_dgrad_dw_rdna/): matching, scoring, dispatch, and the function
 *        that registers them.
 *
 * The engine serves a graph of exactly one depthwise conv_dgrad node (groups == C >= 2, K a
 * multiple of C) on the gfx11 and gfx12 RDNA targets, packed channels-first or channels-last,
 * 1-D to 3-D, FLOAT / HALF / BFLOAT16, any stride, dilation and padding. That is the backward
 * data of a depthwise convolution and the forward of a depthwise transposed convolution. The
 * problem is read in the forward view: x is dx, y is dy.
 *
 * kernels/ConvDgradDepthwise.cpp holds two kernels. The specialized one takes 1-D NCL with
 * K == C and dilation 1 and compiles the filter width and stride in; the bundle carries it for
 * the shapes in SHAPES. Everything else runs on the generic kernel. Each graph has one
 * candidate.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
using conv_rdna::Layout;
using Geometry = HipConvDgradDwRdnaGeometry;
using DwBinding = HipConvDgradDwRdnaBinding;
using LaunchInputs = HipConvDgradDwRdnaLaunchInputs;

// The contract with hip_conv_dgrad_dw_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.hip_conv_dgrad_dw_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.hip_conv_dgrad_dw_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_dgrad_dw_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_dgrad_dw_rdna.dispatch";

constexpr std::string_view FAMILY_FIELD = "kernel_family";
constexpr std::string_view VARIANT_FIELD = "variant";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view LAYOUT_FIELD = "layout";
constexpr std::string_view FILTER_W_FIELD = "filter_w";
constexpr std::string_view STRIDE_FIELD = "stride";
constexpr std::string_view DGRAD_DW_FAMILY = "dgrad_dw_direct";
constexpr std::string_view SPECIALIZED_VARIANT = "specialized";
constexpr std::string_view GENERIC_VARIANT = "generic";

// The names of the saved launch values (HipConvDgradDwRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_DGRAD_DW_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view DY_UID_VALUE = "dy.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view DX_UID_VALUE = "dx.uid";
constexpr std::string_view N_VALUE = "n";
constexpr std::string_view C_VALUE = "c";
constexpr std::string_view K_VALUE = "k";
// Per spatial axis (depth, height, width).
constexpr std::array<std::string_view, 3> IN_VALUES{"in_d", "in_h", "in_w"};
constexpr std::array<std::string_view, 3> FILTER_VALUES{"filter_d", "filter_h", "filter_w"};
constexpr std::array<std::string_view, 3> OUT_VALUES{"out_d", "out_h", "out_w"};
constexpr std::array<std::string_view, 3> PAD_VALUES{"pad_d", "pad_h", "pad_w"};
constexpr std::array<std::string_view, 3> STRIDE_VALUES{"stride_d", "stride_h", "stride_w"};
constexpr std::array<std::string_view, 3> DILATION_VALUES{"dilation_d", "dilation_h", "dilation_w"};
constexpr std::string_view GENERIC_VALUE = "generic";
constexpr std::string_view TILE_Q_VALUE = "tile_q";

// bindConvFeatures binds the forward view: x is dx, y is dy.
constexpr std::string_view TOKEN_PREFIX = "conv_dgrad_dw";
constexpr std::string_view DX_TOKEN = "conv_dgrad_dw.x.uid";
constexpr std::string_view W_TOKEN = "conv_dgrad_dw.w.uid";
constexpr std::string_view DY_TOKEN = "conv_dgrad_dw.y.uid";

/// A specialized shape: filter width and stride along the width axis.
struct Shape
{
    int filterW;
    int stride;
};

// Must match configs/hip_conv_dgrad_dw_rdna.yaml: k = 12 at stride 2 (BigVGAN-style 2x
// upsampling), k = 29 / 43 at stride 2 / 3 (Hann resamplers), k = 7 at stride 1.
constexpr std::array<Shape, 4> SHAPES{{{12, 2}, {29, 2}, {43, 3}, {7, 1}}};

// Must match kernels/ConvDgradDepthwise.cpp.
constexpr int64_t BLOCK_SIZE = 256;
// The specialized grid puts channels on y and the batch on z.
constexpr int64_t MAX_GRID_YZ = 0xffff;
constexpr int64_t MAX_GRID_X_ITEMS = 0x7fffffff;
// The generic kernel walks dx in a grid-stride loop.
constexpr int64_t MAX_GENERIC_BLOCKS = int64_t{1} << 20;

// ---------------------------------------------------------------------------
// Variants and launch geometry
// ---------------------------------------------------------------------------

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

/// Output phases per lane: one was fastest in fp32 on gfx1151; in fp16 / bf16 two, written
/// with one vector store (MIOpen's ConvHipDirectDepthwiseBwd1d).
int tileQOf(data_objects::DataType dtype)
{
    return dtype == data_objects::DataType::FLOAT ? 1 : 2;
}

/// Blocks along x: the specialized kernel covers the phases q of dx (length in[2], padding
/// pad[2]), tileQ per lane; the generic one covers every dx element in a grid-stride loop.
int64_t gridXBlocks(const ConvProblem& problem, const Geometry& geometry)
{
    if(geometry.generic)
    {
        const int64_t total
            = int64_t{problem.n} * problem.c * problem.in[0] * problem.in[1] * problem.in[2];
        return std::min(ceilDiv(total, BLOCK_SIZE), MAX_GENERIC_BLOCKS);
    }
    const int64_t phases = ceilDiv(int64_t{problem.in[2]} + problem.pad[2], problem.stride[2]);
    return ceilDiv(ceilDiv(phases, geometry.tileQ), BLOCK_SIZE);
}

bool launchFits(const ConvProblem& problem, const Geometry& geometry)
{
    if(geometry.generic)
    {
        return true;
    }
    // The specialized kernel indexes dx positions as int, up to stride x (dx length + padding)
    // rounded up to a whole block of lanes.
    const int64_t lastPosition
        = int64_t{problem.stride[2]} * gridXBlocks(problem, geometry) * BLOCK_SIZE * geometry.tileQ;
    return problem.c <= MAX_GRID_YZ && problem.n <= MAX_GRID_YZ
           && gridXBlocks(problem, geometry) * BLOCK_SIZE <= MAX_GRID_X_ITEMS
           && lastPosition <= std::numeric_limits<int>::max();
}

/// A 1-D problem the specialized kernel takes: NCL (unit outer spatial axes), K == C,
/// dilation 1, a built filter width and stride, and a grid within the limits.
bool isSpecialized(const ConvProblem& problem)
{
    if(problem.layout != Layout::CHANNELS_FIRST || problem.k != problem.c
       || problem.dilation[2] != 1)
    {
        return false;
    }
    for(size_t axis = 0; axis < 2; ++axis)
    {
        if(problem.in[axis] != 1 || problem.out[axis] != 1 || problem.filter[axis] != 1
           || problem.pad[axis] != 0)
        {
            return false;
        }
    }
    const bool built = std::any_of(SHAPES.begin(), SHAPES.end(), [&problem](const Shape& shape) {
        return problem.filter[2] == shape.filterW && problem.stride[2] == shape.stride;
    });
    return built && launchFits(problem, Geometry{false, tileQOf(problem.dtype)});
}

Geometry geometryOf(const ConvProblem& problem)
{
    if(isSpecialized(problem))
    {
        return Geometry{false, tileQOf(problem.dtype)};
    }
    return Geometry{true, 1};
}

/// A depthwise problem: one dx channel per group, K a multiple of C.
bool serves(const ConvProblem& problem)
{
    return problem.groups >= 2 && problem.groups == problem.c && problem.k % problem.c == 0;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The single conv_dgrad node and the problem it poses, or nullopt for any other graph.
struct MatchedConv
{
    const data_objects::ConvolutionBwdAttributes* attributes = nullptr;
    const data_objects::TensorAttributes* dy = nullptr;
    const data_objects::TensorAttributes* w = nullptr;
    const data_objects::TensorAttributes* dx = nullptr;
    ConvProblem problem;
};

std::optional<MatchedConv> matchConv(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionBwdAttributes)
    {
        return std::nullopt;
    }

    MatchedConv matched;
    matched.attributes = &node.attributesAs<data_objects::ConvolutionBwdAttributes>();
    matched.dy = conv_rdna::findTensor(context, matched.attributes->dy_tensor_uid());
    matched.w = conv_rdna::findTensor(context, matched.attributes->w_tensor_uid());
    matched.dx = conv_rdna::findTensor(context, matched.attributes->dx_tensor_uid());
    if(matched.dy == nullptr || matched.w == nullptr || matched.dx == nullptr)
    {
        return std::nullopt;
    }

    const conv_rdna::ConvAttributes attributes{matched.attributes->conv_mode(),
                                               matched.attributes->stride(),
                                               matched.attributes->dilation(),
                                               matched.attributes->pre_padding(),
                                               matched.attributes->post_padding()};
    // The forward view: dx is x, dy is y.
    const auto problem = conv_rdna::convProblem(*matched.dx, *matched.w, *matched.dy, attributes);
    if(!problem || !serves(*problem))
    {
        return std::nullopt;
    }
    matched.problem = *problem;
    return matched;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchConv(context);
    if(!matched)
    {
        return std::nullopt;
    }
    BoundTokens bound;
    conv_rdna::bindConvFeatures(bound,
                                TOKEN_PREFIX,
                                matched->problem,
                                matched->attributes->dx_tensor_uid(),
                                matched->attributes->w_tensor_uid(),
                                matched->attributes->dy_tensor_uid(),
                                *matched->dx,
                                *matched->w,
                                *matched->dy);
    return bound;
}

/// The variant geometryOf picks: the specialization of the graph's filter width and stride,
/// or the generic variant; dtype and layout always match. One candidate per graph.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchConv(context);
    if(!matched)
    {
        return false;
    }
    const auto& problem = matched->problem;
    if(kernel.getStringMetadata(std::string(FAMILY_FIELD)) != DGRAD_DW_FAMILY
       || kernel.getStringMetadata(std::string(DTYPE_FIELD))
              != data_objects::EnumNameDataType(problem.dtype)
       || kernel.getStringMetadata(std::string(LAYOUT_FIELD))
              != conv_rdna::layoutName(problem.layout))
    {
        return false;
    }
    if(geometryOf(problem).generic)
    {
        return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == GENERIC_VARIANT;
    }
    return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == SPECIALIZED_VARIANT
           && kernel.getIntMetadata(std::string(FILTER_W_FIELD)) == problem.filter[2]
           && kernel.getIntMetadata(std::string(STRIDE_FIELD)) == problem.stride[2];
}

/// The kernel matcher leaves a single candidate to rank.
double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& /*kernel*/)
{
    return 1.0;
}

DwBinding dwBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_dgrad_dw_rdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(DY_TOKEN), read(W_TOKEN), read(DX_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvDgradDwRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvDgradDwRdna(IngestorKernelCode code, LaunchInputs launch)
        : _code(std::move(code))
        , _launch(launch)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const LaunchInputs& launch() const
    {
        return _launch;
    }

private:
    IngestorKernelCode _code;
    LaunchInputs _launch;
};

/// Mirrors the argument lists of the two kernels and launch() below one for one.
const std::vector<KernelArgument>& kernelSignature(bool generic)
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_scalar{"by_value", static_cast<uint32_t>(sizeof(int)), 0, ""};
    const auto make = [](size_t scalars) {
        std::vector<KernelArgument> signature{s_buffer, s_buffer, s_buffer};
        signature.insert(signature.end(), scalars, s_scalar);
        return signature;
    };
    // Specialized: C, Li (dy length), Lo (dx length), P. Generic: N, C, Di, Hi, Wi, K, Do, Ho,
    // Wo, Z, Y, X, sd, sh, sw, pd, ph, pw, dd, dh, dw.
    static const std::vector<KernelArgument> s_specialized = make(4);
    static const std::vector<KernelArgument> s_generic = make(21);
    return generic ? s_generic : s_specialized;
}

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePrepared(IngestorKernelCode code, const LaunchInputs& launch)
{
    // The grid was bounded by geometryOf or by readLaunchInputs.
    const auto blocks = static_cast<unsigned int>(gridXBlocks(launch.problem, launch.geometry));
    code.setBlockSize(static_cast<unsigned int>(BLOCK_SIZE), 1, 1);
    if(launch.geometry.generic)
    {
        code.setGridSize(blocks, 1, 1);
    }
    else
    {
        code.setGridSize(blocks,
                         static_cast<unsigned int>(launch.problem.c),
                         static_cast<unsigned int>(launch.problem.n));
    }
    return std::make_unique<PreparedHipConvDgradDwRdna>(std::move(code), launch);
}

/// @see hipConvDgradDwRdnaLaunchValues
MetadataValues
    launchValues(const DwBinding& binding, const ConvProblem& problem, const Geometry& geometry)
{
    MetadataValues values{{std::string(DY_UID_VALUE), binding.dy},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(DX_UID_VALUE), binding.dx},
                          {std::string(N_VALUE), int64_t{problem.n}},
                          {std::string(C_VALUE), int64_t{problem.c}},
                          {std::string(K_VALUE), int64_t{problem.k}},
                          {std::string(GENERIC_VALUE), int64_t{geometry.generic ? 1 : 0}},
                          {std::string(TILE_Q_VALUE), int64_t{geometry.tileQ}}};
    for(size_t axis = 0; axis < 3; ++axis)
    {
        values[std::string(IN_VALUES[axis])] = int64_t{problem.in[axis]};
        values[std::string(FILTER_VALUES[axis])] = int64_t{problem.filter[axis]};
        values[std::string(OUT_VALUES[axis])] = int64_t{problem.out[axis]};
        values[std::string(PAD_VALUES[axis])] = int64_t{problem.pad[axis]};
        values[std::string(STRIDE_VALUES[axis])] = int64_t{problem.stride[axis]};
        values[std::string(DILATION_VALUES[axis])] = int64_t{problem.dilation[axis]};
    }
    return values;
}

/// Whether @p geometry is one the bundle builds.
bool isBuiltGeometry(const Geometry& geometry)
{
    return geometry.generic ? geometry.tileQ == 1 : (geometry.tileQ == 1 || geometry.tileQ == 2);
}

/// @see readHipConvDgradDwRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{DY_UID_VALUE,
                                        W_UID_VALUE,
                                        DX_UID_VALUE,
                                        N_VALUE,
                                        C_VALUE,
                                        K_VALUE,
                                        GENERIC_VALUE,
                                        TILE_Q_VALUE};
    for(const auto* axisNames :
        {&IN_VALUES, &FILTER_VALUES, &OUT_VALUES, &PAD_VALUES, &STRIDE_VALUES, &DILATION_VALUES})
    {
        names.insert(names.end(), axisNames->begin(), axisNames->end());
    }
    requireOnlyLaunchValues(inputs, names);

    constexpr int64_t INT_MAX_VALUE = std::numeric_limits<int>::max();
    const auto positive = [&inputs](std::string_view name) {
        return static_cast<int>(requireLaunchInt(inputs, name, 1, INT_MAX_VALUE));
    };

    LaunchInputs launch;
    launch.binding.dy = requireLaunchInt(inputs, DY_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, W_UID_VALUE);
    launch.binding.dx = requireLaunchInt(inputs, DX_UID_VALUE);
    auto& problem = launch.problem;
    problem.spatialRank = 3;
    problem.n = positive(N_VALUE);
    problem.c = positive(C_VALUE);
    problem.k = positive(K_VALUE);
    problem.groups = problem.c;
    for(size_t axis = 0; axis < 3; ++axis)
    {
        problem.in[axis] = positive(IN_VALUES[axis]);
        problem.filter[axis] = positive(FILTER_VALUES[axis]);
        problem.out[axis] = positive(OUT_VALUES[axis]);
        problem.pad[axis]
            = static_cast<int>(requireLaunchInt(inputs, PAD_VALUES[axis], 0, INT_MAX_VALUE));
        problem.stride[axis] = positive(STRIDE_VALUES[axis]);
        problem.dilation[axis] = positive(DILATION_VALUES[axis]);
    }
    auto& geometry = launch.geometry;
    geometry.generic = requireLaunchInt(inputs, GENERIC_VALUE, 0, 1) == 1;
    geometry.tileQ = static_cast<int>(requireLaunchInt(inputs, TILE_Q_VALUE, 1, 2));

    if(problem.k % problem.c != 0 || !isBuiltGeometry(geometry) || !launchFits(problem, geometry))
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values of dispatch '" + inputs.dispatchSymbol
                + "' give a K that is not a multiple of C, a geometry the engine does not "
                  "build, or a grid beyond the launch limits");
    }
    return launch;
}

/// The DW_DTYPE of both kernels for a dtype.
int dtypeCode(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return 0;
    }
    return dtype == data_objects::DataType::HALF ? 1 : 2;
}

class HipConvDgradDwRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvDgradDwRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                      const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The kernels read every operand in place.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        LaunchInputs launch;
        launch.binding = dwBinding(bound);
        const auto matched = matchConv(context);
        if(!matched)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_dgrad_dw_rdna prepare() got a graph its matcher does not admit");
        }
        launch.problem = matched->problem;
        launch.geometry = geometryOf(launch.problem);
        const auto& problem = launch.problem;
        const auto& geometry = launch.geometry;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->dx,
                                                  context.deviceProperties.gcnArchName);
        options.add("DW_DTYPE", dtypeCode(problem.dtype));
        if(geometry.generic)
        {
            options.add("DW_GENERIC", 1);
            options.add("DW_CHANNELS_LAST", problem.layout == Layout::CHANNELS_LAST ? 1 : 0);
        }
        else
        {
            options.add("DW_K", problem.filter[2]);
            options.add("DW_S", problem.stride[2]);
            options.add("DW_TW", geometry.tileQ);
        }

        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            kernelSignature(geometry.generic));
        return makePrepared(std::move(code), launch);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        // The signature depends on the saved geometry, so read it before the restore checks
        // the saved code against it.
        const auto launch = readLaunchInputs(inputs);
        auto restored
            = restoreIngestorLaunch<LaunchInputs>(inputs,
                                                  std::move(code),
                                                  kernelSignature(launch.geometry.generic),
                                                  &readLaunchInputs,
                                                  deviceOrdinal);
        return makePrepared(std::move(restored.code), restored.launch);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& dw = dynamic_cast<const PreparedHipConvDgradDwRdna&>(prepared);
        const auto& launch = dw.launch();
        return SavedLaunchInputs{std::string(DISPATCH_SYMBOL_V1),
                                 launchValues(launch.binding, launch.problem, launch.geometry)};
    }

    bool supportsSaving() const override
    {
        return true;
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& dw = dynamic_cast<const PreparedHipConvDgradDwRdna&>(prepared);
        const auto& binding = dw.launch().binding;
        const auto& p = dw.launch().problem;

        const auto dy
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dy, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto dx
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dx, deviceBuffers, numDeviceBuffers);

        auto& kernel = dw.kernelForStream(handle.getStream());
        if(dw.launch().geometry.generic)
        {
            kernel.launch(handle.getStream(),
                          dy.ptr,
                          w.ptr,
                          dx.ptr,
                          p.n,
                          p.c,
                          p.in[0],
                          p.in[1],
                          p.in[2],
                          p.k,
                          p.out[0],
                          p.out[1],
                          p.out[2],
                          p.filter[0],
                          p.filter[1],
                          p.filter[2],
                          p.stride[0],
                          p.stride[1],
                          p.stride[2],
                          p.pad[0],
                          p.pad[1],
                          p.pad[2],
                          p.dilation[0],
                          p.dilation[1],
                          p.dilation[2]);
            return;
        }
        // Filter width and stride are compiled into the specialized variant; in the forward
        // view, out is the dy length and in the dx length.
        kernel.launch(handle.getStream(), dy.ptr, w.ptr, dx.ptr, p.c, p.out[2], p.in[2], p.pad[2]);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvDgradDwRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(hipConvDgradDwRdnaKpackModuleCache());
    static const HipConvDgradDwRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                     s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

HipConvDgradDwRdnaGeometry hipConvDgradDwRdnaGeometry(const conv_rdna::ConvProblem& problem)
{
    return geometryOf(problem);
}

MetadataValues hipConvDgradDwRdnaLaunchValues(const HipConvDgradDwRdnaBinding& binding,
                                              const conv_rdna::ConvProblem& problem,
                                              const HipConvDgradDwRdnaGeometry& geometry)
{
    return launchValues(binding, problem, geometry);
}

HipConvDgradDwRdnaLaunchInputs readHipConvDgradDwRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvDgradDwRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvDgradDwRdnaModuleCache()
{
    hipConvDgradDwRdnaKpackModuleCache().clear();
}

void registerHipConvDgradDwRdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(DISPATCH_SYMBOL_V1), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
