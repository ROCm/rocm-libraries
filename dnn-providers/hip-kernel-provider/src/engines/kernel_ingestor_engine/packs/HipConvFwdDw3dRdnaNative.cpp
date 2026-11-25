// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
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
#include "engines/kernel_ingestor_engine/packs/ConvRdnaDw3d.hpp"
#include "engines/kernel_ingestor_engine/packs/HipConvFwdDw3dRdnaLaunchValues.hpp"

/**
 * @file HipConvFwdDw3dRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_dw3d_rdna` (descriptors/hip/conv_dw3d_rdna/):
 *        matching, scoring, dispatch, and the function that registers them.
 *
 * The engine serves a graph of exactly one depthwise 3-D conv-forward node (groups == C >= 2,
 * K a multiple of C) on the gfx11 and gfx12 RDNA targets, packed NCDHW or NDHWC, FLOAT / HALF /
 * BFLOAT16, any padding. Its kernel (kernels/ConvFwdDepthwise3d.cpp) is specialized at compile
 * time on the filter, the H / W stride and dilation and the output tile; the bundle carries
 * those variants for the common filters and strides (H / W stride 1 or 2, H / W dilation 1) and
 * one generic variant per dtype and layout, with filter, stride and dilation as launch
 * arguments, for everything else (ConvRdnaDw3d.hpp). Each graph has one candidate.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
namespace dw3d = conv_rdna::dw3d;
using Geometry = HipConvFwdDw3dRdnaGeometry;
using DwBinding = HipConvFwdDw3dRdnaBinding;
using LaunchInputs = HipConvFwdDw3dRdnaLaunchInputs;

// The contract with hip_conv_fwd_dw3d_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.hip_conv_fwd_dw3d_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.hip_conv_fwd_dw3d_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_dw3d_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_fwd_dw3d_rdna.dispatch";

// The names of the saved launch values (HipConvFwdDw3dRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_FWD_DW3D_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view X_UID_VALUE = "x.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view Y_UID_VALUE = "y.uid";

constexpr std::string_view TOKEN_PREFIX = "conv_fwd_dw3d";
constexpr std::string_view X_TOKEN = "conv_fwd_dw3d.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd_dw3d.w.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd_dw3d.y.uid";

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
    // The kernel accumulates in fp32, so a graph that asks for another compute type (half,
    // int32, double) is declined rather than computed at a different precision.
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes
       || node.computeDataType() != data_objects::DataType::FLOAT)
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
    if(!problem || !dw3d::serves(*problem))
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

/// The variant dw3d::isVariantOf picks for the graph's problem; one candidate per graph.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchConv(context);
    if(!matched)
    {
        return false;
    }
    return dw3d::isVariantOf(kernel, matched->problem);
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

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePrepared(IngestorKernelCode code, const LaunchInputs& launch)
{
    // Both grid dimensions were bounded by dw3d::serves or by readLaunchInputs.
    code.setBlockSize(dw3d::blockSize(launch.geometry), 1, 1);
    code.setGridSize(static_cast<unsigned int>(dw3d::gridXBlocks(launch.problem, launch.geometry)),
                     static_cast<unsigned int>(dw3d::gridY(launch.problem, launch.geometry)),
                     1);
    return std::make_unique<PreparedHipConvFwdDw3dRdna>(std::move(code), launch);
}

/// @see hipConvFwdDw3dRdnaLaunchValues
MetadataValues
    launchValues(const DwBinding& binding, const ConvProblem& problem, const Geometry& geometry)
{
    MetadataValues values{{std::string(X_UID_VALUE), binding.x},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(Y_UID_VALUE), binding.y}};
    dw3d::addProblemLaunchValues(values, problem, geometry);
    return values;
}

/// @see readHipConvFwdDw3dRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{X_UID_VALUE, W_UID_VALUE, Y_UID_VALUE};
    const auto problemNames = dw3d::problemLaunchValueNames();
    names.insert(names.end(), problemNames.begin(), problemNames.end());
    requireOnlyLaunchValues(inputs, names);

    LaunchInputs launch;
    launch.binding.x = requireLaunchInt(inputs, X_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, W_UID_VALUE);
    launch.binding.y = requireLaunchInt(inputs, Y_UID_VALUE);
    dw3d::readProblemLaunchValues(inputs, launch.problem, launch.geometry);
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
        launch.geometry = dw3d::geometryOf(launch.problem);

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        dw3d::addCompileOptions(options, launch.problem, launch.geometry);

        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            dw3d::kernelSignature(launch.geometry.generic));
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
                                                  dw3d::kernelSignature(launch.geometry.generic),
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

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        dw3d::launch(dw.kernelForStream(handle.getStream()),
                     handle.getStream(),
                     x.ptr,
                     w.ptr,
                     y.ptr,
                     dw.launch().problem,
                     dw.launch().geometry);
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
    return dw3d::geometryOf(problem);
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
