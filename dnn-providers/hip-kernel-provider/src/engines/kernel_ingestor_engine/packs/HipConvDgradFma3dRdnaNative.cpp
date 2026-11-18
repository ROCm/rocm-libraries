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
#include "engines/kernel_ingestor_engine/packs/HipConvDgradFma3dRdnaLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file HipConvDgradFma3dRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_dgrad_fma_3d_rdna`
 *        (descriptors/hip/conv_dgrad_fma_3d_rdna/): matching, scoring, dispatch, and the
 *        function that registers them.
 *
 * The engine serves a graph of exactly one 3-D conv_dgrad node on the gfx11 and gfx12 RDNA
 * targets: rank 5, packed channels-first or channels-last, FLOAT / HALF / BFLOAT16, any stride,
 * dilation and padding, groups == 1 or at least two dx channels per group. That is the backward
 * data of a 3-D convolution and the forward of a ConvTranspose3d. The problem is read in the
 * forward view: x is dx, y is dy.
 *
 * ../conv_dgrad_fma_2d_rdna/kernels/ConvDgradFmaIgemm.cpp (shared with
 * hip_conv_dgrad_fma_2d_rdna) is an FMA implicit GEMM that splits dx into the stride phases,
 * so each workgroup walks only the taps that reach its positions. dy and the weights are read
 * in place, so no variant needs a workspace. One variant per dtype, layout and dx-channel tile;
 * each graph has one candidate.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
using conv_rdna::Layout;
using Binding = HipConvDgradFma3dRdnaBinding;
using LaunchInputs = HipConvDgradFma3dRdnaLaunchInputs;

// The contract with hip_conv_dgrad_fma_3d_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.hip_conv_dgrad_fma_3d_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.hip_conv_dgrad_fma_3d_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_dgrad_fma_3d_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_dgrad_fma_3d_rdna.dispatch";

constexpr std::string_view FAMILY_FIELD = "kernel_family";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view LAYOUT_FIELD = "layout";
constexpr std::string_view TILE_N_FIELD = "tile_n";
constexpr std::string_view DGRAD_FMA_IGEMM_FAMILY = "dgrad_fma_igemm";

// The names of the saved launch values (HipConvDgradFma3dRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_DGRAD_FMA_3D_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view DY_UID_VALUE = "dy.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view DX_UID_VALUE = "dx.uid";
constexpr std::string_view N_VALUE = "n";
constexpr std::string_view C_VALUE = "c";
constexpr std::string_view K_VALUE = "k";
constexpr std::string_view GROUPS_VALUE = "groups";
// Per spatial axis (depth, height, width).
constexpr std::array<std::string_view, 3> IN_VALUES{"in_d", "in_h", "in_w"};
constexpr std::array<std::string_view, 3> FILTER_VALUES{"filter_d", "filter_h", "filter_w"};
constexpr std::array<std::string_view, 3> OUT_VALUES{"out_d", "out_h", "out_w"};
constexpr std::array<std::string_view, 3> PAD_VALUES{"pad_d", "pad_h", "pad_w"};
constexpr std::array<std::string_view, 3> STRIDE_VALUES{"stride_d", "stride_h", "stride_w"};
constexpr std::array<std::string_view, 3> DILATION_VALUES{"dilation_d", "dilation_h", "dilation_w"};

// bindConvFeatures binds the forward view: x is dx, y is dy.
constexpr std::string_view TOKEN_PREFIX = "conv_dgrad_fma_3d";
constexpr std::string_view DX_TOKEN = "conv_dgrad_fma_3d.x.uid";
constexpr std::string_view W_TOKEN = "conv_dgrad_fma_3d.w.uid";
constexpr std::string_view DY_TOKEN = "conv_dgrad_fma_3d.y.uid";

// Must match ../conv_dgrad_fma_2d_rdna/kernels/ConvDgradFmaIgemm.cpp.
constexpr int64_t TILE_M = 128;
constexpr int64_t TILE_C = 16;
constexpr unsigned int BLOCK_SIZE = 256;
// One dx channel per group is the depthwise shape, which belongs to a depthwise kernel family.
constexpr int64_t MIN_DX_CHANNELS_PER_GROUP = 2;
constexpr int64_t MAX_GRID_X_BLOCKS = 0x7fffffff / BLOCK_SIZE;
constexpr int64_t MAX_GRID_Y = 0xffff;

// ---------------------------------------------------------------------------
// Launch geometry
// ---------------------------------------------------------------------------

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

int64_t tileN(const ConvProblem& problem)
{
    const int64_t cPerGroup = problem.channelsPerGroup();
    if(cPerGroup <= 32)
    {
        return 32;
    }
    return cPerGroup <= 64 ? 64 : 128;
}

/// One run of blocks per stride phase (sd x sh x sw of them), each as long as phase (0, 0, 0)
/// needs: ceil(N x ceil(Di / sd) x ceil(Hi / sh) x ceil(Wi / sw) / TILE_M). Saturates at
/// MAX_GRID_X_BLOCKS + 1, so extents and strides near INT_MAX cannot wrap the product into the
/// launch limits.
int64_t gridXBlocks(const ConvProblem& problem)
{
    constexpr int64_t TOO_MANY = MAX_GRID_X_BLOCKS + 1;
    int64_t phases = 1;
    int64_t phasePositions = problem.n;
    for(size_t axis = 0; axis < 3; ++axis)
    {
        phases *= problem.stride[axis];
        phasePositions *= ceilDiv(problem.in[axis], problem.stride[axis]);
        if(phases > MAX_GRID_X_BLOCKS || phasePositions > MAX_GRID_X_BLOCKS * TILE_M)
        {
            return TOO_MANY;
        }
    }
    const int64_t blocks = ceilDiv(phasePositions, TILE_M);
    return blocks > MAX_GRID_X_BLOCKS / phases ? TOO_MANY : phases * blocks;
}

int64_t gridY(const ConvProblem& problem)
{
    return problem.groups * ceilDiv(problem.channelsPerGroup(), tileN(problem));
}

/// The kernel counts the taps and its K iterations (taps x ceil(dy channels per group /
/// TILE_C)) in int.
bool kLoopFits(const ConvProblem& problem)
{
    int64_t iterations = ceilDiv(problem.outChannelsPerGroup(), TILE_C);
    for(const int extent : problem.filter)
    {
        iterations *= extent;
        if(iterations > std::numeric_limits<int>::max())
        {
            return false;
        }
    }
    return true;
}

bool launchFits(const ConvProblem& problem)
{
    return gridXBlocks(problem) <= MAX_GRID_X_BLOCKS && gridY(problem) <= MAX_GRID_Y
           && kLoopFits(problem);
}

/// A 3-D problem with groups == 1 or at least two dx channels per group, within the launch
/// limits.
bool serves(const ConvProblem& problem)
{
    if(problem.spatialRank != 3)
    {
        return false;
    }
    if(problem.groups > 1 && problem.channelsPerGroup() < MIN_DX_CHANNELS_PER_GROUP)
    {
        return false;
    }
    return launchFits(problem);
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

/// The kernel of the graph's dtype and layout, with the channel tile tileN picks. Any other
/// would return wrong numbers: the grid is sized for that tile, so a kernel built for another
/// one leaves dx channels unwritten or writes into the next group's.
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
    return kernel.getStringMetadata(std::string(FAMILY_FIELD)) == DGRAD_FMA_IGEMM_FAMILY
           && kernel.getStringMetadata(std::string(DTYPE_FIELD))
                  == data_objects::EnumNameDataType(problem.dtype)
           && kernel.getStringMetadata(std::string(LAYOUT_FIELD))
                  == conv_rdna::layoutName(problem.layout)
           && kernel.getIntMetadata(std::string(TILE_N_FIELD)) == tileN(problem);
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
                "hip_conv_dgrad_fma_3d_rdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(DY_TOKEN), read(W_TOKEN), read(DX_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvDgradFma3dRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvDgradFma3dRdna(IngestorKernelCode code, LaunchInputs launch)
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

/// The kernel's CONV_DTYPE for a dtype.
int dtypeCode(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return 0;
    }
    return dtype == data_objects::DataType::HALF ? 1 : 2;
}

/// Mirrors the argument list of ConvDgradFmaIgemm and the launch() below one for one.
const std::vector<KernelArgument>& kernelSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_scalar{"by_value", static_cast<uint32_t>(sizeof(int)), 0, ""};
    static const std::vector<KernelArgument> s_signature = [] {
        std::vector<KernelArgument> signature{s_buffer, s_buffer, s_buffer};
        // N, C, Di, Hi, Wi, K, Z, Y, X, Do, Ho, Wo, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups.
        signature.insert(signature.end(), 22, s_scalar);
        return signature;
    }();
    return s_signature;
}

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePrepared(IngestorKernelCode code, const LaunchInputs& launch)
{
    // Both grid dimensions were bounded by serves or by readLaunchInputs.
    code.setBlockSize(BLOCK_SIZE, 1, 1);
    code.setGridSize(static_cast<unsigned int>(gridXBlocks(launch.problem)),
                     static_cast<unsigned int>(gridY(launch.problem)),
                     1);
    return std::make_unique<PreparedHipConvDgradFma3dRdna>(std::move(code), launch);
}

/// @see hipConvDgradFma3dRdnaLaunchValues
MetadataValues launchValues(const Binding& binding, const ConvProblem& problem)
{
    MetadataValues values{{std::string(DY_UID_VALUE), binding.dy},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(DX_UID_VALUE), binding.dx},
                          {std::string(N_VALUE), int64_t{problem.n}},
                          {std::string(C_VALUE), int64_t{problem.c}},
                          {std::string(K_VALUE), int64_t{problem.k}},
                          {std::string(GROUPS_VALUE), int64_t{problem.groups}}};
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

/// @see readHipConvDgradFma3dRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{
        DY_UID_VALUE, W_UID_VALUE, DX_UID_VALUE, N_VALUE, C_VALUE, K_VALUE, GROUPS_VALUE};
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
    problem.groups = positive(GROUPS_VALUE);
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

    if(problem.c % problem.groups != 0 || problem.k % problem.groups != 0 || !launchFits(problem))
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values of dispatch '" + inputs.dispatchSymbol
                + "' give groups that do not divide 'c' and 'k', or a grid beyond the launch "
                  "limits");
    }
    return launch;
}

class HipConvDgradFma3dRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvDgradFma3dRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                         const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// dy and the weights are read in place in either layout, and the kernel writes every dx
    /// element, so nothing is staged.
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
                "hip_conv_dgrad_fma_3d_rdna prepare() got a graph its matcher does not admit");
        }
        launch.problem = matched->problem;
        const auto& problem = launch.problem;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->dx,
                                                  context.deviceProperties.gcnArchName);
        options.add("CONV_DTYPE", dtypeCode(problem.dtype));
        options.add("CONV_CHANNELS_LAST", problem.layout == Layout::CHANNELS_LAST ? 1 : 0);
        options.add("TILE_N", static_cast<int>(tileN(problem)));

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, kernelSignature());
        return makePrepared(std::move(code), launch);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        auto restored = restoreIngestorLaunch<LaunchInputs>(
            inputs, std::move(code), kernelSignature(), &readLaunchInputs, deviceOrdinal);
        return makePrepared(std::move(restored.code), restored.launch);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& dgrad = dynamic_cast<const PreparedHipConvDgradFma3dRdna&>(prepared);
        const auto& launch = dgrad.launch();
        return SavedLaunchInputs{std::string(DISPATCH_SYMBOL_V1),
                                 launchValues(launch.binding, launch.problem)};
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
        const auto& dgrad = dynamic_cast<const PreparedHipConvDgradFma3dRdna&>(prepared);
        const auto& binding = dgrad.launch().binding;
        const auto& p = dgrad.launch().problem;

        const auto dy
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dy, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto dx
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dx, deviceBuffers, numDeviceBuffers);

        // In the forward view, c is the dx channel count, k the dy one, `in` dx and `out` dy.
        dgrad.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    dy.ptr,
                    w.ptr,
                    dx.ptr,
                    p.n,
                    p.c,
                    p.in[0],
                    p.in[1],
                    p.in[2],
                    p.k,
                    p.filter[0],
                    p.filter[1],
                    p.filter[2],
                    p.out[0],
                    p.out[1],
                    p.out[2],
                    p.stride[0],
                    p.stride[1],
                    p.stride[2],
                    p.pad[0],
                    p.pad[1],
                    p.pad[2],
                    p.dilation[0],
                    p.dilation[1],
                    p.dilation[2],
                    p.groups);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvDgradFma3dRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        hipConvDgradFma3dRdnaKpackModuleCache());
    static const HipConvDgradFma3dRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                        s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

int64_t hipConvDgradFma3dRdnaTileN(const conv_rdna::ConvProblem& problem)
{
    return tileN(problem);
}

MetadataValues hipConvDgradFma3dRdnaLaunchValues(const HipConvDgradFma3dRdnaBinding& binding,
                                                 const conv_rdna::ConvProblem& problem)
{
    return launchValues(binding, problem);
}

HipConvDgradFma3dRdnaLaunchInputs
    readHipConvDgradFma3dRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvDgradFma3dRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvDgradFma3dRdnaModuleCache()
{
    hipConvDgradFma3dRdnaKpackModuleCache().clear();
}

void registerHipConvDgradFma3dRdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(DISPATCH_SYMBOL_V1), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
