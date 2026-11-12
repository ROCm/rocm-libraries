// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
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
#include "engines/kernel_ingestor_engine/packs/HipConvFwdDw3dRdnaLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file HipConvFwdDw3dRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_dw3d_rdna` (descriptors/hip/conv_dw3d_rdna/):
 *        matching, scoring, dispatch, and the function that registers them.
 *
 * The engine serves a graph of exactly one depthwise 3-D conv-forward node (groups == C >= 2,
 * K a multiple of C) on the gfx11 and gfx12 RDNA targets, packed NCDHW or NDHWC, FLOAT / HALF /
 * BFLOAT16, any padding. Its kernel (kernels/ConvFwdDepthwise3d.cpp) is specialized at compile
 * time on the filter, the H / W stride and dilation and the output tile; the bundle carries
 * those variants for the common filters and strides (SPECIALIZED_FILTERS, H / W stride 1 or 2,
 * H / W dilation 1) and one generic variant per dtype and layout, with filter, stride and
 * dilation as launch arguments, for everything else. Each graph has one candidate.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
using conv_rdna::Layout;
using Geometry = HipConvFwdDw3dRdnaGeometry;
using DwBinding = HipConvFwdDw3dRdnaBinding;
using LaunchInputs = HipConvFwdDw3dRdnaLaunchInputs;

// The contract with hip_conv_fwd_dw3d_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.hip_conv_fwd_dw3d_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.hip_conv_fwd_dw3d_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_dw3d_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_fwd_dw3d_rdna.dispatch";

constexpr std::string_view FAMILY_FIELD = "kernel_family";
constexpr std::string_view VARIANT_FIELD = "variant";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view LAYOUT_FIELD = "layout";
constexpr std::string_view FILTER_D_FIELD = "filter_d";
constexpr std::string_view FILTER_H_FIELD = "filter_h";
constexpr std::string_view FILTER_W_FIELD = "filter_w";
constexpr std::string_view STRIDE_H_FIELD = "stride_h";
constexpr std::string_view STRIDE_W_FIELD = "stride_w";
constexpr std::string_view TILE_D_FIELD = "tile_d";
constexpr std::string_view TILE_W_FIELD = "tile_w";
constexpr std::string_view UNIFORM_K_FIELD = "uniform_k";
constexpr std::string_view DW3D_FAMILY = "dw3d_direct";
constexpr std::string_view SPECIALIZED_VARIANT = "specialized";
constexpr std::string_view GENERIC_VARIANT = "generic";

// The names of the saved launch values (HipConvFwdDw3dRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_FWD_DW3D_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view X_UID_VALUE = "x.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view Y_UID_VALUE = "y.uid";
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
constexpr std::string_view TILE_D_VALUE = "tile_d";
constexpr std::string_view TILE_W_VALUE = "tile_w";
constexpr std::string_view UNIFORM_K_VALUE = "uniform_k";

constexpr std::string_view TOKEN_PREFIX = "conv_fwd_dw3d";
constexpr std::string_view X_TOKEN = "conv_fwd_dw3d.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd_dw3d.w.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd_dw3d.y.uid";

// The filters (depth, height, width) the bundle has specialized variants for, each with H / W
// stride 1 and 2 and H / W dilation 1. Must match configs/hip_conv_fwd_dw3d_rdna.yaml.
constexpr std::array<std::array<int, 3>, 4> SPECIALIZED_FILTERS{
    {{3, 3, 3}, {5, 5, 5}, {3, 1, 1}, {1, 3, 3}}};

// Must match kernels/ConvFwdDepthwise3d.cpp.
constexpr int64_t TILE_H = 4;
constexpr int64_t GENERIC_TILE_W = 4;
constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int UNIFORM_K_BLOCK_SIZE = 64;
// HSA grid limits: work-items along x, workgroups along y.
constexpr int64_t MAX_GRID_X_ITEMS = 0x7fffffff;
constexpr int64_t MAX_GRID_Y = 0xffff;

// ---------------------------------------------------------------------------
// Variants and launch geometry
// ---------------------------------------------------------------------------

bool isSpecializedShape(const ConvProblem& problem)
{
    bool filterBuilt = false;
    for(const auto& filter : SPECIALIZED_FILTERS)
    {
        filterBuilt = filterBuilt
                      || (problem.filter[0] == filter[0] && problem.filter[1] == filter[1]
                          && problem.filter[2] == filter[2]);
    }
    const bool strideBuilt = problem.stride[1] == problem.stride[2]
                             && (problem.stride[1] == 1 || problem.stride[1] == 2);
    return filterBuilt && strideBuilt && problem.dilation[1] == 1 && problem.dilation[2] == 1;
}

Geometry geometryOf(const ConvProblem& problem)
{
    const bool channelsLast = problem.layout == Layout::CHANNELS_LAST;
    // The NCDHW specializations use the uniform-k grid only, whose y extent is N x K.
    if(!isSpecializedShape(problem)
       || (!channelsLast && int64_t{problem.n} * problem.k > MAX_GRID_Y))
    {
        return Geometry{true, 1, static_cast<int>(GENERIC_TILE_W), false};
    }
    Geometry geometry;
    geometry.uniformK = !channelsLast;
    // Two output depths share their input planes; it needs unit depth stride and dilation.
    geometry.tileD = !channelsLast && problem.stride[0] == 1 && problem.dilation[0] == 1 ? 2 : 1;
    // A wider tile reuses more of each input row for 5-wide filters, except with two fp32
    // output depths (#15100's measurements on gfx1151).
    if(geometry.tileD == 2 && problem.dtype == data_objects::DataType::FLOAT)
    {
        geometry.tileW = 4;
    }
    else
    {
        geometry.tileW = problem.filter[2] >= 5 ? 8 : 4;
    }
    return geometry;
}

unsigned int blockSize(const Geometry& geometry)
{
    return geometry.uniformK ? UNIFORM_K_BLOCK_SIZE : BLOCK_SIZE;
}

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

/// Threads the kernel indexes along x (all of them unless uniform-k, one volume otherwise).
int64_t gridXItems(const ConvProblem& problem, const Geometry& geometry)
{
    if(geometry.generic)
    {
        return int64_t{problem.n} * problem.k * problem.out[0] * problem.out[1]
               * ceilDiv(problem.out[2], GENERIC_TILE_W);
    }
    const int64_t tiles = ceilDiv(problem.out[0], geometry.tileD) * ceilDiv(problem.out[1], TILE_H)
                          * ceilDiv(problem.out[2], geometry.tileW);
    return geometry.uniformK ? tiles : int64_t{problem.n} * problem.k * tiles;
}

int64_t gridXBlocks(const ConvProblem& problem, const Geometry& geometry)
{
    return ceilDiv(gridXItems(problem, geometry), blockSize(geometry));
}

int64_t gridY(const ConvProblem& problem, const Geometry& geometry)
{
    return geometry.uniformK ? int64_t{problem.n} * problem.k : 1;
}

bool launchFits(const ConvProblem& problem, const Geometry& geometry)
{
    return gridXBlocks(problem, geometry) * blockSize(geometry) <= MAX_GRID_X_ITEMS
           && gridY(problem, geometry) <= MAX_GRID_Y;
}

/// A 3-D depthwise problem: one input channel per group, K a multiple of C.
bool serves(const ConvProblem& problem)
{
    return problem.spatialRank == 3 && problem.groups >= 2 && problem.groups == problem.c
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

/// The variant geometryOf picks: the specialization of the graph's filter, H / W stride and
/// tile, or the generic variant; dtype and layout always match. One candidate per graph.
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
    if(kernel.getStringMetadata(std::string(FAMILY_FIELD)) != DW3D_FAMILY
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
    const auto intField
        = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };
    return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == SPECIALIZED_VARIANT
           && intField(FILTER_D_FIELD) == problem.filter[0]
           && intField(FILTER_H_FIELD) == problem.filter[1]
           && intField(FILTER_W_FIELD) == problem.filter[2]
           && intField(STRIDE_H_FIELD) == problem.stride[1]
           && intField(STRIDE_W_FIELD) == problem.stride[2]
           && intField(TILE_D_FIELD) == geometry.tileD && intField(TILE_W_FIELD) == geometry.tileW
           && intField(UNIFORM_K_FIELD) == (geometry.uniformK ? 1 : 0);
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
                "hip_conv_fwd_dw3d_rdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(X_TOKEN), read(W_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvFwdDw3dRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvFwdDw3dRdna(IngestorKernelCode code, LaunchInputs launch)
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

/// The kernel's DW_DTYPE for a dtype.
int dtypeCode(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return 0;
    }
    return dtype == data_objects::DataType::HALF ? 1 : 2;
}

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
    // Specialized: N, C, Di, Hi, Wi, K, Do, Ho, Wo, sd, pd, ph, pw, dd. Generic: N, C, Di, Hi,
    // Wi, K, Do, Ho, Wo, Z, Y, X, sd, sh, sw, pd, ph, pw, dd, dh, dw.
    static const std::vector<KernelArgument> s_specialized = make(14);
    static const std::vector<KernelArgument> s_generic = make(21);
    return generic ? s_generic : s_specialized;
}

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePrepared(IngestorKernelCode code, const LaunchInputs& launch)
{
    // Both grid dimensions were bounded by serves or by readLaunchInputs.
    code.setBlockSize(blockSize(launch.geometry), 1, 1);
    code.setGridSize(static_cast<unsigned int>(gridXBlocks(launch.problem, launch.geometry)),
                     static_cast<unsigned int>(gridY(launch.problem, launch.geometry)),
                     1);
    return std::make_unique<PreparedHipConvFwdDw3dRdna>(std::move(code), launch);
}

/// @see hipConvFwdDw3dRdnaLaunchValues
MetadataValues
    launchValues(const DwBinding& binding, const ConvProblem& problem, const Geometry& geometry)
{
    MetadataValues values{{std::string(X_UID_VALUE), binding.x},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(Y_UID_VALUE), binding.y},
                          {std::string(N_VALUE), int64_t{problem.n}},
                          {std::string(C_VALUE), int64_t{problem.c}},
                          {std::string(K_VALUE), int64_t{problem.k}},
                          {std::string(GENERIC_VALUE), int64_t{geometry.generic ? 1 : 0}},
                          {std::string(TILE_D_VALUE), int64_t{geometry.tileD}},
                          {std::string(TILE_W_VALUE), int64_t{geometry.tileW}},
                          {std::string(UNIFORM_K_VALUE), int64_t{geometry.uniformK ? 1 : 0}}};
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

/// Whether @p geometry is one the bundle builds: the generic tile, or a specialized tile.
bool isBuiltGeometry(const Geometry& geometry)
{
    if(geometry.generic)
    {
        return geometry.tileD == 1 && geometry.tileW == GENERIC_TILE_W && !geometry.uniformK;
    }
    return (geometry.tileD == 1 || (geometry.tileD == 2 && geometry.uniformK))
           && (geometry.tileW == 4 || geometry.tileW == 8);
}

/// @see readHipConvFwdDw3dRdnaLaunchInputs
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
                                        TILE_D_VALUE,
                                        TILE_W_VALUE,
                                        UNIFORM_K_VALUE};
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
    geometry.tileD = static_cast<int>(requireLaunchInt(inputs, TILE_D_VALUE, 1, 2));
    geometry.tileW = static_cast<int>(requireLaunchInt(inputs, TILE_W_VALUE, 1, 8));
    geometry.uniformK = requireLaunchInt(inputs, UNIFORM_K_VALUE, 0, 1) == 1;

    if(problem.k % problem.c != 0 || !isBuiltGeometry(geometry) || !launchFits(problem, geometry))
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values of dispatch '" + inputs.dispatchSymbol
                + "' give a K that is not a multiple of C, a tile the engine does not build, or "
                  "a grid beyond the launch limits");
    }
    return launch;
}

class HipConvFwdDw3dRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvFwdDw3dRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
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
                "hip_conv_fwd_dw3d_rdna prepare() got a graph its matcher does not admit");
        }
        launch.problem = matched->problem;
        launch.geometry = geometryOf(launch.problem);
        const auto& problem = launch.problem;
        const auto& geometry = launch.geometry;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        options.add("DW_DTYPE", dtypeCode(problem.dtype));
        options.add("DW_NDHWC", problem.layout == Layout::CHANNELS_LAST ? 1 : 0);
        if(geometry.generic)
        {
            options.add("DW_GENERIC", 1);
        }
        else
        {
            options.add("DW_KZ", problem.filter[0]);
            options.add("DW_KY", problem.filter[1]);
            options.add("DW_KX", problem.filter[2]);
            options.add("DW_SH", problem.stride[1]);
            options.add("DW_SW", problem.stride[2]);
            options.add("DW_TD", geometry.tileD);
            options.add("DW_TW", geometry.tileW);
            options.add("DW_UK", geometry.uniformK ? 1 : 0);
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
        const auto& dw = dynamic_cast<const PreparedHipConvFwdDw3dRdna&>(prepared);
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
        const auto& dw = dynamic_cast<const PreparedHipConvFwdDw3dRdna&>(prepared);
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
            kernel.launch(handle.getStream(),
                          x.ptr,
                          w.ptr,
                          y.ptr,
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
        // The filter, H / W stride and dilation are compiled into the specialized variant.
        kernel.launch(handle.getStream(),
                      x.ptr,
                      w.ptr,
                      y.ptr,
                      p.n,
                      p.c,
                      p.in[0],
                      p.in[1],
                      p.in[2],
                      p.k,
                      p.out[0],
                      p.out[1],
                      p.out[2],
                      p.stride[0],
                      p.pad[0],
                      p.pad[1],
                      p.pad[2],
                      p.dilation[0]);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvFwdDw3dRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(hipConvFwdDw3dRdnaKpackModuleCache());
    static const HipConvFwdDw3dRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                     s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

HipConvFwdDw3dRdnaGeometry hipConvFwdDw3dRdnaGeometry(const conv_rdna::ConvProblem& problem)
{
    return geometryOf(problem);
}

MetadataValues hipConvFwdDw3dRdnaLaunchValues(const HipConvFwdDw3dRdnaBinding& binding,
                                              const conv_rdna::ConvProblem& problem,
                                              const HipConvFwdDw3dRdnaGeometry& geometry)
{
    return launchValues(binding, problem, geometry);
}

HipConvFwdDw3dRdnaLaunchInputs readHipConvFwdDw3dRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvFwdDw3dRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvFwdDw3dRdnaModuleCache()
{
    hipConvFwdDw3dRdnaKpackModuleCache().clear();
}

void registerHipConvFwdDw3dRdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(DISPATCH_SYMBOL_V1), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
