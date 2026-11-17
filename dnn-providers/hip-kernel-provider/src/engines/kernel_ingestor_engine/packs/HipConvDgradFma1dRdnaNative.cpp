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
#include "engines/kernel_ingestor_engine/packs/HipConvDgradFma1dRdnaLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file HipConvDgradFma1dRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_dgrad_fma_1d_rdna`
 *        (descriptors/hip/conv_dgrad_fma_1d_rdna/): matching, scoring, dispatch, and the
 *        function that registers them.
 *
 * The engine serves a graph of exactly one 1-D conv_dgrad node that is not depthwise
 * (C / groups >= 2) on the gfx11 and gfx12 RDNA targets: rank 3, or rank 4 with a unit H axis,
 * packed channels-first or channels-last, FLOAT / HALF / BFLOAT16, any stride, dilation and
 * padding. That is the backward data of a 1-D convolution and the forward of a ConvTranspose1d.
 * The problem is read in the forward view: x is dx, y is dy. No variant needs a workspace.
 *
 * kernels/ConvDgradFma1d.cpp holds two kernels. The specialized one takes NCL with groups == 1
 * and dilation 1 and compiles the filter width, stride and tile in; the bundle carries it for
 * the shapes in SHAPES at every channel tile. Everything else runs on the generic kernel. Each
 * graph has one candidate.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
using conv_rdna::Layout;
using Geometry = HipConvDgradFma1dRdnaGeometry;
using Binding = HipConvDgradFma1dRdnaBinding;
using LaunchInputs = HipConvDgradFma1dRdnaLaunchInputs;

// The contract with hip_conv_dgrad_fma_1d_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.hip_conv_dgrad_fma_1d_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.hip_conv_dgrad_fma_1d_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_dgrad_fma_1d_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_dgrad_fma_1d_rdna.dispatch";

constexpr std::string_view FAMILY_FIELD = "kernel_family";
constexpr std::string_view VARIANT_FIELD = "variant";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view LAYOUT_FIELD = "layout";
constexpr std::string_view FILTER_W_FIELD = "filter_w";
constexpr std::string_view STRIDE_FIELD = "stride";
constexpr std::string_view CHANNELS_PER_WAVE_FIELD = "channels_per_wave";
constexpr std::string_view DGRAD_FMA_1D_FAMILY = "dgrad_fma_1d";
constexpr std::string_view SPECIALIZED_VARIANT = "specialized";
constexpr std::string_view GENERIC_VARIANT = "generic";

// The names of the saved launch values (HipConvDgradFma1dRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_DGRAD_FMA_1D_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view DY_UID_VALUE = "dy.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view DX_UID_VALUE = "dx.uid";
constexpr std::string_view N_VALUE = "n";
constexpr std::string_view C_VALUE = "c";
constexpr std::string_view K_VALUE = "k";
constexpr std::string_view GROUPS_VALUE = "groups";
constexpr std::string_view IN_W_VALUE = "in_w";
constexpr std::string_view FILTER_W_VALUE = "filter_w";
constexpr std::string_view OUT_W_VALUE = "out_w";
constexpr std::string_view PAD_W_VALUE = "pad_w";
constexpr std::string_view STRIDE_W_VALUE = "stride_w";
constexpr std::string_view DILATION_W_VALUE = "dilation_w";
constexpr std::string_view GENERIC_VALUE = "generic";
constexpr std::string_view Q_PER_LANE_VALUE = "q_per_lane";
constexpr std::string_view CHANNELS_PER_WAVE_VALUE = "channels_per_wave";

// bindConvFeatures binds the forward view: x is dx, y is dy.
constexpr std::string_view TOKEN_PREFIX = "conv_dgrad_fma_1d";
constexpr std::string_view DX_TOKEN = "conv_dgrad_fma_1d.x.uid";
constexpr std::string_view W_TOKEN = "conv_dgrad_fma_1d.w.uid";
constexpr std::string_view DY_TOKEN = "conv_dgrad_fma_1d.y.uid";

/// The width axis of ConvProblem's right-aligned spatial slots.
constexpr size_t W = 2;

/// A specialized shape: filter width and stride.
struct Shape
{
    int filterW;
    int stride;
};

// Must match configs/hip_conv_dgrad_fma_1d_rdna.yaml: the ConvTranspose1d shapes of audio
// decoders and vocoders (MIOpen's ConvHipImplicitGemmFmaBwdData1d measurements).
constexpr std::array<Shape, 4> SHAPES{{{16, 8}, {8, 4}, {4, 2}, {9, 5}}};

// Must match kernels/ConvDgradFma1d.cpp.
constexpr int64_t BLOCK_SIZE = 256;
constexpr int64_t WAVES = 8;
constexpr int64_t LANES = 32;
// The specialized grid puts dx channel tiles on y and the batch on z.
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

/// The tile of MIOpen's GetTiles() for the workspace-free modes: two q per lane, 8 channels per
/// wave up to stride 4 and 4 above; with 16-bit weights read in place, 4 q x 2 channels for an
/// odd filter width and at most 128 weight words per wave; fewer channels for a layer with few
/// dx channels.
Geometry specializedTile(const ConvProblem& problem)
{
    Geometry tile{false, 2, problem.stride[W] <= 4 ? 8 : 4};
    if(problem.dtype != data_objects::DataType::FLOAT)
    {
        const int k = problem.filter[W];
        if(k % 2 != 0)
        {
            tile.qPerLane = 4;
            tile.channelsPerWave = 2;
        }
        while(tile.channelsPerWave > 1 && tile.channelsPerWave * ((k + 1) / 2) > 128)
        {
            tile.channelsPerWave /= 2;
        }
    }
    while(tile.channelsPerWave > 1 && WAVES * (tile.channelsPerWave / 2) >= problem.c)
    {
        tile.channelsPerWave /= 2;
    }
    return tile;
}

/// Blocks along x: the specialized kernel covers the phases q of dx (length in[W], padding
/// pad[W]), 32 x qPerLane per block; the generic one covers every dx element in a grid-stride
/// loop.
int64_t gridXBlocks(const ConvProblem& problem, const Geometry& geometry)
{
    if(geometry.generic)
    {
        const int64_t total = int64_t{problem.n} * problem.c * problem.in[W];
        return std::min(ceilDiv(total, BLOCK_SIZE), MAX_GENERIC_BLOCKS);
    }
    const int64_t phases = ceilDiv(int64_t{problem.in[W]} + problem.pad[W], problem.stride[W]);
    return ceilDiv(phases, LANES * geometry.qPerLane);
}

int64_t gridYBlocks(const ConvProblem& problem, const Geometry& geometry)
{
    return geometry.generic ? 1 : ceilDiv(problem.c, WAVES * geometry.channelsPerWave);
}

bool launchFits(const ConvProblem& problem, const Geometry& geometry)
{
    if(geometry.generic)
    {
        return true;
    }
    // The specialized kernel indexes dx positions as int, up to stride x the q its grid covers.
    const int64_t blocks = gridXBlocks(problem, geometry);
    const int64_t lastPosition = int64_t{problem.stride[W]} * blocks * LANES * geometry.qPerLane;
    return gridYBlocks(problem, geometry) <= MAX_GRID_YZ && problem.n <= MAX_GRID_YZ
           && blocks * BLOCK_SIZE <= MAX_GRID_X_ITEMS
           && lastPosition <= std::numeric_limits<int>::max();
}

/// A 1-D problem: rank 3, or unit outer spatial axes with no padding along them.
bool isOneD(const ConvProblem& problem)
{
    if(problem.spatialRank > 2)
    {
        return false;
    }
    for(size_t axis = 0; axis < W; ++axis)
    {
        if(problem.in[axis] != 1 || problem.out[axis] != 1 || problem.filter[axis] != 1
           || problem.pad[axis] != 0)
        {
            return false;
        }
    }
    return true;
}

/// NCL, groups == 1, dilation 1, a built filter width and stride, and a grid within the limits.
bool isSpecialized(const ConvProblem& problem)
{
    if(problem.layout != Layout::CHANNELS_FIRST || problem.groups != 1 || problem.dilation[W] != 1)
    {
        return false;
    }
    const bool built = std::any_of(SHAPES.begin(), SHAPES.end(), [&problem](const Shape& shape) {
        return problem.filter[W] == shape.filterW && problem.stride[W] == shape.stride;
    });
    return built && launchFits(problem, specializedTile(problem));
}

Geometry geometryOf(const ConvProblem& problem)
{
    if(isSpecialized(problem))
    {
        return specializedTile(problem);
    }
    return Geometry{true, 1, 1};
}

/// A 1-D problem that is not depthwise: at least two dx channels per group.
bool serves(const ConvProblem& problem)
{
    return isOneD(problem) && problem.channelsPerGroup() >= 2 && problem.k % problem.groups == 0;
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

/// The variant geometryOf picks: the specialization of the graph's filter width, stride and
/// channel tile, or the generic variant; dtype and layout always match. One candidate per graph.
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
    if(kernel.getStringMetadata(std::string(FAMILY_FIELD)) != DGRAD_FMA_1D_FAMILY
       || kernel.getStringMetadata(std::string(DTYPE_FIELD))
              != data_objects::EnumNameDataType(problem.dtype)
       || kernel.getStringMetadata(std::string(LAYOUT_FIELD))
              != conv_rdna::layoutName(problem.layout))
    {
        return false;
    }
    const auto geometry = geometryOf(problem);
    if(geometry.generic)
    {
        return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == GENERIC_VARIANT;
    }
    return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == SPECIALIZED_VARIANT
           && kernel.getIntMetadata(std::string(FILTER_W_FIELD)) == problem.filter[W]
           && kernel.getIntMetadata(std::string(STRIDE_FIELD)) == problem.stride[W]
           && kernel.getIntMetadata(std::string(CHANNELS_PER_WAVE_FIELD))
                  == geometry.channelsPerWave;
}

/// The kernel matcher leaves a single candidate to rank.
double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& /*kernel*/)
{
    return 1.0;
}

Binding bindingOf(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_dgrad_fma_1d_rdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(DY_TOKEN), read(W_TOKEN), read(DX_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvDgradFma1dRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvDgradFma1dRdna(IngestorKernelCode code, LaunchInputs launch)
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
    // Specialized: dy channels, dx channels, dy length, dx length, padding. Generic: N, C, K,
    // groups, dx length, dy length, filter width, stride, padding, dilation.
    static const std::vector<KernelArgument> s_specialized = make(5);
    static const std::vector<KernelArgument> s_generic = make(10);
    return generic ? s_generic : s_specialized;
}

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePrepared(IngestorKernelCode code, const LaunchInputs& launch)
{
    // The grid was bounded by geometryOf or by readLaunchInputs.
    const auto& problem = launch.problem;
    const auto& geometry = launch.geometry;
    code.setBlockSize(static_cast<unsigned int>(BLOCK_SIZE), 1, 1);
    code.setGridSize(static_cast<unsigned int>(gridXBlocks(problem, geometry)),
                     static_cast<unsigned int>(gridYBlocks(problem, geometry)),
                     geometry.generic ? 1U : static_cast<unsigned int>(problem.n));
    return std::make_unique<PreparedHipConvDgradFma1dRdna>(std::move(code), launch);
}

/// @see hipConvDgradFma1dRdnaLaunchValues
MetadataValues
    launchValues(const Binding& binding, const ConvProblem& problem, const Geometry& geometry)
{
    return MetadataValues{
        {std::string(DY_UID_VALUE), binding.dy},
        {std::string(W_UID_VALUE), binding.w},
        {std::string(DX_UID_VALUE), binding.dx},
        {std::string(N_VALUE), int64_t{problem.n}},
        {std::string(C_VALUE), int64_t{problem.c}},
        {std::string(K_VALUE), int64_t{problem.k}},
        {std::string(GROUPS_VALUE), int64_t{problem.groups}},
        {std::string(IN_W_VALUE), int64_t{problem.in[W]}},
        {std::string(FILTER_W_VALUE), int64_t{problem.filter[W]}},
        {std::string(OUT_W_VALUE), int64_t{problem.out[W]}},
        {std::string(PAD_W_VALUE), int64_t{problem.pad[W]}},
        {std::string(STRIDE_W_VALUE), int64_t{problem.stride[W]}},
        {std::string(DILATION_W_VALUE), int64_t{problem.dilation[W]}},
        {std::string(GENERIC_VALUE), int64_t{geometry.generic ? 1 : 0}},
        {std::string(Q_PER_LANE_VALUE), int64_t{geometry.qPerLane}},
        {std::string(CHANNELS_PER_WAVE_VALUE), int64_t{geometry.channelsPerWave}}};
}

/// Whether @p geometry is one the bundle builds.
bool isBuiltGeometry(const Geometry& geometry)
{
    if(geometry.generic)
    {
        return geometry.qPerLane == 1 && geometry.channelsPerWave == 1;
    }
    const int c = geometry.channelsPerWave;
    return (geometry.qPerLane == 2 || geometry.qPerLane == 4)
           && (c == 1 || c == 2 || c == 4 || c == 8);
}

/// @see readHipConvDgradFma1dRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    const std::vector<std::string_view> names{DY_UID_VALUE,
                                              W_UID_VALUE,
                                              DX_UID_VALUE,
                                              N_VALUE,
                                              C_VALUE,
                                              K_VALUE,
                                              GROUPS_VALUE,
                                              IN_W_VALUE,
                                              FILTER_W_VALUE,
                                              OUT_W_VALUE,
                                              PAD_W_VALUE,
                                              STRIDE_W_VALUE,
                                              DILATION_W_VALUE,
                                              GENERIC_VALUE,
                                              Q_PER_LANE_VALUE,
                                              CHANNELS_PER_WAVE_VALUE};
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
    problem.spatialRank = 1;
    problem.n = positive(N_VALUE);
    problem.c = positive(C_VALUE);
    problem.k = positive(K_VALUE);
    problem.groups = positive(GROUPS_VALUE);
    problem.in[W] = positive(IN_W_VALUE);
    problem.filter[W] = positive(FILTER_W_VALUE);
    problem.out[W] = positive(OUT_W_VALUE);
    problem.pad[W] = static_cast<int>(requireLaunchInt(inputs, PAD_W_VALUE, 0, INT_MAX_VALUE));
    problem.stride[W] = positive(STRIDE_W_VALUE);
    problem.dilation[W] = positive(DILATION_W_VALUE);
    auto& geometry = launch.geometry;
    geometry.generic = requireLaunchInt(inputs, GENERIC_VALUE, 0, 1) == 1;
    geometry.qPerLane = static_cast<int>(requireLaunchInt(inputs, Q_PER_LANE_VALUE, 1, 4));
    geometry.channelsPerWave
        = static_cast<int>(requireLaunchInt(inputs, CHANNELS_PER_WAVE_VALUE, 1, 8));

    if(problem.c % problem.groups != 0 || problem.k % problem.groups != 0
       || !isBuiltGeometry(geometry) || !launchFits(problem, geometry))
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values of dispatch '" + inputs.dispatchSymbol
                + "' give channels that are not multiples of the group count, a geometry the "
                  "engine does not build, or a grid beyond the launch limits");
    }
    return launch;
}

/// The CT_DTYPE of both kernels for a dtype.
int dtypeCode(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return 0;
    }
    return dtype == data_objects::DataType::HALF ? 1 : 2;
}

class HipConvDgradFma1dRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvDgradFma1dRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                         const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The kernels read every operand in place, 16-bit weights included.
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
        launch.binding = bindingOf(bound);
        const auto matched = matchConv(context);
        if(!matched)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_dgrad_fma_1d_rdna prepare() got a graph its matcher does not admit");
        }
        launch.problem = matched->problem;
        launch.geometry = geometryOf(launch.problem);
        const auto& problem = launch.problem;
        const auto& geometry = launch.geometry;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->dx,
                                                  context.deviceProperties.gcnArchName);
        options.add("CT_DTYPE", dtypeCode(problem.dtype));
        if(geometry.generic)
        {
            options.add("CT_GENERIC", 1);
            options.add("CT_CHANNELS_LAST", problem.layout == Layout::CHANNELS_LAST ? 1 : 0);
        }
        else
        {
            options.add("CT_K", problem.filter[W]);
            options.add("CT_S", problem.stride[W]);
            options.add("CT_QR", geometry.qPerLane);
            options.add("CT_CO_W", geometry.channelsPerWave);
            options.add("CT_WS", problem.dtype == data_objects::DataType::FLOAT ? 1 : 0);
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
        const auto& dgrad = dynamic_cast<const PreparedHipConvDgradFma1dRdna&>(prepared);
        const auto& launch = dgrad.launch();
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
        const auto& dgrad = dynamic_cast<const PreparedHipConvDgradFma1dRdna&>(prepared);
        const auto& binding = dgrad.launch().binding;
        const auto& p = dgrad.launch().problem;

        const auto dy
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dy, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto dx
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dx, deviceBuffers, numDeviceBuffers);

        auto& kernel = dgrad.kernelForStream(handle.getStream());
        if(dgrad.launch().geometry.generic)
        {
            kernel.launch(handle.getStream(),
                          dy.ptr,
                          w.ptr,
                          dx.ptr,
                          p.n,
                          p.c,
                          p.k,
                          p.groups,
                          p.in[W],
                          p.out[W],
                          p.filter[W],
                          p.stride[W],
                          p.pad[W],
                          p.dilation[W]);
            return;
        }
        // Filter width, stride and tile are compiled into the specialized variant; in the
        // forward view, k is the dy channel count and c the dx channel count.
        kernel.launch(
            handle.getStream(), dy.ptr, w.ptr, dx.ptr, p.k, p.c, p.out[W], p.in[W], p.pad[W]);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvDgradFma1dRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        hipConvDgradFma1dRdnaKpackModuleCache());
    static const HipConvDgradFma1dRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                        s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

HipConvDgradFma1dRdnaGeometry hipConvDgradFma1dRdnaGeometry(const conv_rdna::ConvProblem& problem)
{
    return geometryOf(problem);
}

MetadataValues hipConvDgradFma1dRdnaLaunchValues(const HipConvDgradFma1dRdnaBinding& binding,
                                                 const conv_rdna::ConvProblem& problem,
                                                 const HipConvDgradFma1dRdnaGeometry& geometry)
{
    return launchValues(binding, problem, geometry);
}

HipConvDgradFma1dRdnaLaunchInputs
    readHipConvDgradFma1dRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvDgradFma1dRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvDgradFma1dRdnaModuleCache()
{
    hipConvDgradFma1dRdnaKpackModuleCache().clear();
}

void registerHipConvDgradFma1dRdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(DISPATCH_SYMBOL_V1), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
