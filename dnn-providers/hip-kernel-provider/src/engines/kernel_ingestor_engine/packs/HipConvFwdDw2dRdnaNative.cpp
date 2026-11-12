// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
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
#include "engines/kernel_ingestor_engine/packs/HipConvFwdDw2dRdnaLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file HipConvFwdDw2dRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_dw2d_rdna` (descriptors/hip/conv_dw2d_rdna/):
 *        matching, scoring, dispatch, and the function that registers them.
 *
 * The engine serves a graph of exactly one depthwise 2-D (or 1-D) conv-forward node
 * (groups == C >= 2, K a multiple of C) on the gfx11 and gfx12 RDNA targets, packed NC[H]W or
 * N[H]WC, FLOAT / HALF / BFLOAT16, any stride, dilation and padding. Its specialized kernel
 * (kernels/ConvFwdDepthwise2d.cpp) takes NCHW with K == C and compiles the filter, padding,
 * stride and dilation in; the bundle carries it for the shapes in UNDILATED_SHAPES and
 * DILATED_SHAPES with "same" pre-padding. Everything else runs on the generic depthwise kernel
 * of the dw3d bundle (../conv_dw3d_rdna/kernels/ConvFwdDepthwise3d.cpp) with depth 1. Each graph
 * has one candidate.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
using conv_rdna::Layout;
using Geometry = HipConvFwdDw2dRdnaGeometry;
using DwBinding = HipConvFwdDw2dRdnaBinding;
using LaunchInputs = HipConvFwdDw2dRdnaLaunchInputs;

// The contract with hip_conv_fwd_dw2d_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.hip_conv_fwd_dw2d_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.hip_conv_fwd_dw2d_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_dw2d_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_fwd_dw2d_rdna.dispatch";

constexpr std::string_view FAMILY_FIELD = "kernel_family";
constexpr std::string_view VARIANT_FIELD = "variant";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view LAYOUT_FIELD = "layout";
constexpr std::string_view FILTER_H_FIELD = "filter_h";
constexpr std::string_view FILTER_W_FIELD = "filter_w";
constexpr std::string_view STRIDE_FIELD = "stride";
constexpr std::string_view DILATION_FIELD = "dilation";
constexpr std::string_view DW2D_FAMILY = "dw2d_direct";
constexpr std::string_view SPECIALIZED_VARIANT = "specialized";
constexpr std::string_view GENERIC_VARIANT = "generic";

// The names of the saved launch values (HipConvFwdDw2dRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_FWD_DW2D_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view X_UID_VALUE = "x.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view Y_UID_VALUE = "y.uid";
constexpr std::string_view N_VALUE = "n";
constexpr std::string_view C_VALUE = "c";
constexpr std::string_view K_VALUE = "k";
// Per spatial axis (height, width).
constexpr std::array<std::string_view, 2> IN_VALUES{"in_h", "in_w"};
constexpr std::array<std::string_view, 2> FILTER_VALUES{"filter_h", "filter_w"};
constexpr std::array<std::string_view, 2> OUT_VALUES{"out_h", "out_w"};
constexpr std::array<std::string_view, 2> PAD_VALUES{"pad_h", "pad_w"};
constexpr std::array<std::string_view, 2> STRIDE_VALUES{"stride_h", "stride_w"};
constexpr std::array<std::string_view, 2> DILATION_VALUES{"dilation_h", "dilation_w"};
constexpr std::string_view GENERIC_VALUE = "generic";
constexpr std::string_view LANE_ON_DILATION_GRID_VALUE = "lane_on_dilation_grid";
constexpr std::string_view OUT_PER_LANE_VALUE = "out_per_lane";

constexpr std::string_view TOKEN_PREFIX = "conv_fwd_dw2d";
constexpr std::string_view X_TOKEN = "conv_fwd_dw2d.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd_dw2d.w.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd_dw2d.y.uid";

/// A specialized shape: filter (height, width), stride and dilation (equal along H and W), with
/// pre-padding dilation x (filter - 1) / 2 along each axis.
struct Shape
{
    int filterH;
    int filterW;
    int stride;
    int dilation;
};

// Must match configs/hip_conv_fwd_dw2d_rdna.yaml.
constexpr std::array<Shape, 7> UNDILATED_SHAPES{{{3, 3, 1, 1},
                                                 {3, 3, 2, 1},
                                                 {5, 5, 1, 1},
                                                 {5, 5, 2, 1},
                                                 {7, 7, 1, 1},
                                                 {7, 7, 2, 1},
                                                 {1, 31, 1, 1}}};
constexpr std::array<Shape, 6> DILATED_SHAPES{
    {{3, 3, 1, 2}, {3, 3, 1, 4}, {3, 3, 1, 6}, {3, 3, 1, 12}, {3, 3, 1, 18}, {5, 5, 1, 2}}};

// Must match kernels/ConvFwdDepthwise2d.cpp and the generic kernel of
// ../conv_dw3d_rdna/kernels/ConvFwdDepthwise3d.cpp.
constexpr int64_t ROW_CHUNK = 8;
constexpr int UNDILATED_OUT_PER_LANE = 2;
constexpr int64_t GENERIC_TILE_W = 4;
constexpr unsigned int BLOCK_SIZE = 256;
// The specialized kernel walks its tiles in a grid-stride loop; more workgroups than this did
// not help in MIOpen's measurements.
constexpr int64_t MAX_SPECIALIZED_BLOCKS = 2048;
constexpr int64_t MAX_GRID_X_ITEMS = 0x7fffffff;

// ---------------------------------------------------------------------------
// Variants and launch geometry
// ---------------------------------------------------------------------------

bool isShape(const ConvProblem& problem, const Shape& shape)
{
    return problem.filter[1] == shape.filterH && problem.filter[2] == shape.filterW
           && problem.stride[1] == shape.stride && problem.stride[2] == shape.stride
           && problem.dilation[1] == shape.dilation && problem.dilation[2] == shape.dilation
           && problem.pad[1] == shape.dilation * (shape.filterH - 1) / 2
           && problem.pad[2] == shape.dilation * (shape.filterW - 1) / 2;
}

Geometry geometryOf(const ConvProblem& problem)
{
    if(problem.layout == Layout::CHANNELS_FIRST && problem.k == problem.c)
    {
        for(const auto& shape : UNDILATED_SHAPES)
        {
            if(isShape(problem, shape))
            {
                return Geometry{false, false, UNDILATED_OUT_PER_LANE};
            }
        }
        for(const auto& shape : DILATED_SHAPES)
        {
            if(isShape(problem, shape))
            {
                // Four outputs amortize the 16-bit conversion of the window; in fp32 two were
                // faster on gfx1151 (MIOpen's ConvDirectDepthwiseFwd2DDilated).
                return Geometry{
                    false, true, problem.dtype == data_objects::DataType::FLOAT ? 2 : 4};
            }
        }
    }
    return Geometry{true, false, static_cast<int>(GENERIC_TILE_W)};
}

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

int64_t gridXBlocks(const ConvProblem& problem, const Geometry& geometry)
{
    const int64_t nc = int64_t{problem.n} * problem.k;
    if(geometry.generic)
    {
        return ceilDiv(nc * problem.out[1] * ceilDiv(problem.out[2], GENERIC_TILE_W), BLOCK_SIZE);
    }
    // The tile count of the kernel: lanes over row chunks and column groups. On the dilation
    // grid a lane's outputs are a lane step apart, so the tiles come in step-sized groups.
    int64_t rowStep = 1;
    int64_t colStep = 1;
    if(geometry.laneOnDilationGrid)
    {
        rowStep = problem.dilation[1] / std::gcd(problem.stride[1], problem.dilation[1]);
        colStep = problem.dilation[2] / std::gcd(problem.stride[2], problem.dilation[2]);
    }
    const int64_t tilesW = ceilDiv(problem.out[2], geometry.outPerLane * colStep) * colStep;
    const int64_t rowTiles = ceilDiv(problem.out[1], ROW_CHUNK * rowStep) * rowStep;
    return std::min(ceilDiv(tilesW * rowTiles * nc, BLOCK_SIZE), MAX_SPECIALIZED_BLOCKS);
}

bool launchFits(const ConvProblem& problem, const Geometry& geometry)
{
    return gridXBlocks(problem, geometry) * BLOCK_SIZE <= MAX_GRID_X_ITEMS;
}

/// A 2-D (or 1-D) depthwise problem: one input channel per group, K a multiple of C.
bool serves(const ConvProblem& problem)
{
    return problem.spatialRank <= 2 && problem.groups >= 2 && problem.groups == problem.c
           && problem.k % problem.c == 0 && launchFits(problem, geometryOf(problem));
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The single conv-forward node and the problem it poses, or nullopt for any other graph.
struct MatchedConv
{
    const data_objects::ConvolutionFwdAttributes* attributes = nullptr;
    const data_objects::TensorAttributes* x = nullptr;
    const data_objects::TensorAttributes* w = nullptr;
    const data_objects::TensorAttributes* y = nullptr;
    ConvProblem problem;
};

std::optional<MatchedConv> matchConv(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes)
    {
        return std::nullopt;
    }

    MatchedConv matched;
    matched.attributes = &node.attributesAs<data_objects::ConvolutionFwdAttributes>();
    matched.x = conv_rdna::findTensor(context, matched.attributes->x_tensor_uid());
    matched.w = conv_rdna::findTensor(context, matched.attributes->w_tensor_uid());
    matched.y = conv_rdna::findTensor(context, matched.attributes->y_tensor_uid());
    if(matched.x == nullptr || matched.w == nullptr || matched.y == nullptr)
    {
        return std::nullopt;
    }

    const conv_rdna::ConvAttributes attributes{matched.attributes->conv_mode(),
                                               matched.attributes->stride(),
                                               matched.attributes->dilation(),
                                               matched.attributes->pre_padding(),
                                               matched.attributes->post_padding()};
    const auto problem = conv_rdna::convProblem(*matched.x, *matched.w, *matched.y, attributes);
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
                                matched->attributes->x_tensor_uid(),
                                matched->attributes->w_tensor_uid(),
                                matched->attributes->y_tensor_uid(),
                                *matched->x,
                                *matched->w,
                                *matched->y);
    return bound;
}

/// The variant geometryOf picks: the specialization of the graph's filter, stride and
/// dilation, or the generic variant; dtype and layout always match. One candidate per graph.
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
    if(kernel.getStringMetadata(std::string(FAMILY_FIELD)) != DW2D_FAMILY
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
    const auto intField
        = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };
    return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == SPECIALIZED_VARIANT
           && intField(FILTER_H_FIELD) == problem.filter[1]
           && intField(FILTER_W_FIELD) == problem.filter[2]
           && intField(STRIDE_FIELD) == problem.stride[1]
           && intField(DILATION_FIELD) == problem.dilation[1];
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
                "hip_conv_fwd_dw2d_rdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(X_TOKEN), read(W_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvFwdDw2dRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvFwdDw2dRdna(IngestorKernelCode code, LaunchInputs launch)
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
    // Specialized: N, C, iH, iW, oH, oW. Generic (the dw3d kernel): N, C, Di, Hi, Wi, K, Do, Ho,
    // Wo, Z, Y, X, sd, sh, sw, pd, ph, pw, dd, dh, dw.
    static const std::vector<KernelArgument> s_specialized = make(6);
    static const std::vector<KernelArgument> s_generic = make(21);
    return generic ? s_generic : s_specialized;
}

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePrepared(IngestorKernelCode code, const LaunchInputs& launch)
{
    // The grid was bounded by serves or by readLaunchInputs.
    code.setBlockSize(BLOCK_SIZE, 1, 1);
    code.setGridSize(static_cast<unsigned int>(gridXBlocks(launch.problem, launch.geometry)), 1, 1);
    return std::make_unique<PreparedHipConvFwdDw2dRdna>(std::move(code), launch);
}

/// @see hipConvFwdDw2dRdnaLaunchValues
MetadataValues
    launchValues(const DwBinding& binding, const ConvProblem& problem, const Geometry& geometry)
{
    MetadataValues values{
        {std::string(X_UID_VALUE), binding.x},
        {std::string(W_UID_VALUE), binding.w},
        {std::string(Y_UID_VALUE), binding.y},
        {std::string(N_VALUE), int64_t{problem.n}},
        {std::string(C_VALUE), int64_t{problem.c}},
        {std::string(K_VALUE), int64_t{problem.k}},
        {std::string(GENERIC_VALUE), int64_t{geometry.generic ? 1 : 0}},
        {std::string(LANE_ON_DILATION_GRID_VALUE), int64_t{geometry.laneOnDilationGrid ? 1 : 0}},
        {std::string(OUT_PER_LANE_VALUE), int64_t{geometry.outPerLane}}};
    for(size_t axis = 0; axis < 2; ++axis)
    {
        values[std::string(IN_VALUES[axis])] = int64_t{problem.in[axis + 1]};
        values[std::string(FILTER_VALUES[axis])] = int64_t{problem.filter[axis + 1]};
        values[std::string(OUT_VALUES[axis])] = int64_t{problem.out[axis + 1]};
        values[std::string(PAD_VALUES[axis])] = int64_t{problem.pad[axis + 1]};
        values[std::string(STRIDE_VALUES[axis])] = int64_t{problem.stride[axis + 1]};
        values[std::string(DILATION_VALUES[axis])] = int64_t{problem.dilation[axis + 1]};
    }
    return values;
}

/// Whether @p geometry is one the bundle builds.
bool isBuiltGeometry(const Geometry& geometry)
{
    if(geometry.generic)
    {
        return !geometry.laneOnDilationGrid && geometry.outPerLane == GENERIC_TILE_W;
    }
    return geometry.laneOnDilationGrid ? (geometry.outPerLane == 2 || geometry.outPerLane == 4)
                                       : geometry.outPerLane == UNDILATED_OUT_PER_LANE;
}

/// @see readHipConvFwdDw2dRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{X_UID_VALUE,
                                        W_UID_VALUE,
                                        Y_UID_VALUE,
                                        N_VALUE,
                                        C_VALUE,
                                        K_VALUE,
                                        GENERIC_VALUE,
                                        LANE_ON_DILATION_GRID_VALUE,
                                        OUT_PER_LANE_VALUE};
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
    launch.binding.x = requireLaunchInt(inputs, X_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, W_UID_VALUE);
    launch.binding.y = requireLaunchInt(inputs, Y_UID_VALUE);
    auto& problem = launch.problem;
    problem.spatialRank = 2;
    problem.n = positive(N_VALUE);
    problem.c = positive(C_VALUE);
    problem.k = positive(K_VALUE);
    problem.groups = problem.c;
    for(size_t axis = 0; axis < 2; ++axis)
    {
        problem.in[axis + 1] = positive(IN_VALUES[axis]);
        problem.filter[axis + 1] = positive(FILTER_VALUES[axis]);
        problem.out[axis + 1] = positive(OUT_VALUES[axis]);
        problem.pad[axis + 1]
            = static_cast<int>(requireLaunchInt(inputs, PAD_VALUES[axis], 0, INT_MAX_VALUE));
        problem.stride[axis + 1] = positive(STRIDE_VALUES[axis]);
        problem.dilation[axis + 1] = positive(DILATION_VALUES[axis]);
    }
    auto& geometry = launch.geometry;
    geometry.generic = requireLaunchInt(inputs, GENERIC_VALUE, 0, 1) == 1;
    geometry.laneOnDilationGrid = requireLaunchInt(inputs, LANE_ON_DILATION_GRID_VALUE, 0, 1) == 1;
    geometry.outPerLane = static_cast<int>(requireLaunchInt(inputs, OUT_PER_LANE_VALUE, 1, 4));

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

/// The IO_DTYPE of the specialized kernel for a dtype.
std::string ioDtype(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return "float";
    }
    return dtype == data_objects::DataType::HALF ? "__half" : "__hip_bfloat16";
}

/// The DW_DTYPE of the generic kernel for a dtype.
int genericDtypeCode(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return 0;
    }
    return dtype == data_objects::DataType::HALF ? 1 : 2;
}

class HipConvFwdDw2dRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvFwdDw2dRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
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
                "hip_conv_fwd_dw2d_rdna prepare() got a graph its matcher does not admit");
        }
        launch.problem = matched->problem;
        launch.geometry = geometryOf(launch.problem);
        const auto& problem = launch.problem;
        const auto& geometry = launch.geometry;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        if(geometry.generic)
        {
            options.add("DW_GENERIC", 1);
            options.add("DW_DTYPE", genericDtypeCode(problem.dtype));
            options.add("DW_NDHWC", problem.layout == Layout::CHANNELS_LAST ? 1 : 0);
        }
        else
        {
            options.add("IO_DTYPE", ioDtype(problem.dtype));
            options.add("FILTER_H", problem.filter[1]);
            options.add("FILTER_W", problem.filter[2]);
            options.add("PAD_H", problem.pad[1]);
            options.add("PAD_W", problem.pad[2]);
            options.add("STRIDE_H", problem.stride[1]);
            options.add("STRIDE_W", problem.stride[2]);
            options.add("DILATION_H", problem.dilation[1]);
            options.add("DILATION_W", problem.dilation[2]);
            options.add("OUT_PER_LANE", geometry.outPerLane);
            options.add("LANE_ON_DILATION_GRID", geometry.laneOnDilationGrid ? 1 : 0);
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
        const auto& dw = dynamic_cast<const PreparedHipConvFwdDw2dRdna&>(prepared);
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
        const auto& dw = dynamic_cast<const PreparedHipConvFwdDw2dRdna&>(prepared);
        const auto& binding = dw.launch().binding;
        const auto& p = dw.launch().problem;

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        auto& kernel = dw.kernelForStream(handle.getStream());
        if(dw.launch().geometry.generic)
        {
            // The dw3d generic kernel with depth 1: unit depth extent, filter, stride and
            // dilation, no depth padding.
            kernel.launch(handle.getStream(),
                          x.ptr,
                          w.ptr,
                          y.ptr,
                          p.n,
                          p.c,
                          1,
                          p.in[1],
                          p.in[2],
                          p.k,
                          1,
                          p.out[1],
                          p.out[2],
                          1,
                          p.filter[1],
                          p.filter[2],
                          1,
                          p.stride[1],
                          p.stride[2],
                          0,
                          p.pad[1],
                          p.pad[2],
                          1,
                          p.dilation[1],
                          p.dilation[2]);
            return;
        }
        // Filter, padding, stride and dilation are compiled into the specialized variant.
        kernel.launch(handle.getStream(),
                      x.ptr,
                      w.ptr,
                      y.ptr,
                      p.n,
                      p.c,
                      p.in[1],
                      p.in[2],
                      p.out[1],
                      p.out[2]);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvFwdDw2dRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(hipConvFwdDw2dRdnaKpackModuleCache());
    static const HipConvFwdDw2dRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                     s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

HipConvFwdDw2dRdnaGeometry hipConvFwdDw2dRdnaGeometry(const conv_rdna::ConvProblem& problem)
{
    return geometryOf(problem);
}

MetadataValues hipConvFwdDw2dRdnaLaunchValues(const HipConvFwdDw2dRdnaBinding& binding,
                                              const conv_rdna::ConvProblem& problem,
                                              const HipConvFwdDw2dRdnaGeometry& geometry)
{
    return launchValues(binding, problem, geometry);
}

HipConvFwdDw2dRdnaLaunchInputs readHipConvFwdDw2dRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvFwdDw2dRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvFwdDw2dRdnaModuleCache()
{
    hipConvFwdDw2dRdnaKpackModuleCache().clear();
}

void registerHipConvFwdDw2dRdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(DISPATCH_SYMBOL_V1), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
