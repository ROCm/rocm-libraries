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
#include "engines/kernel_ingestor_engine/packs/ConvRdnaFusedEpilogue.hpp"
#include "engines/kernel_ingestor_engine/packs/HipConvFwdBiasActDw3dRdnaLaunchValues.hpp"

/**
 * @file HipConvFwdBiasActDw3dRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_bias_act_dw3d_rdna`
 *        (descriptors/hip/conv_bias_act_dw3d_rdna/): matching, scoring, dispatch, and the
 *        function that registers them.
 *
 * The engine serves, on the gfx11 and gfx12 RDNA targets, a depthwise 3-D conv-forward node
 * whose virtual output feeds a pointwise ADD with a per-output-channel bias, whose virtual output
 * feeds a forward activation; or, without the bias, a depthwise 3-D conv-forward node whose
 * virtual output feeds the activation directly. It runs the depthwise 3D kernels of
 * `hip_conv_fwd_dw3d_rdna` (ConvRdnaDw3d.hpp) built with their bias and activation epilogue, so
 * neither intermediate is stored: the bias is added and the activation applied in fp32, and the
 * result rounded once. The convolution itself is served under the same rules, and with the same
 * variant, as the unfused engine.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using conv_rdna::Activation;
using conv_rdna::ConvProblem;
namespace dw3d = conv_rdna::dw3d;

// The contract with hip_conv_fwd_bias_act_dw3d_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.dispatch";

// The names of the saved launch values (HipConvFwdBiasActDw3dRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view X_UID_VALUE = "x.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view Y_UID_VALUE = "y.uid";

constexpr std::string_view TOKEN_PREFIX = "conv_fwd_bias_act_dw3d";
constexpr std::string_view X_TOKEN = "conv_fwd_bias_act_dw3d.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd_bias_act_dw3d.w.uid";
constexpr std::string_view B_TOKEN = "conv_fwd_bias_act_dw3d.b.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd_bias_act_dw3d.y.uid";

using BiasActBinding = HipConvFwdBiasActDw3dRdnaBinding;
using LaunchInputs = HipConvFwdBiasActDw3dRdnaLaunchInputs;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// conv -> [ADD(bias) ->] activation (ConvRdnaFusedEpilogue.hpp) whose convolution the depthwise
/// 3D kernels serve (dw3d::serves), or nullopt for any other graph.
std::optional<conv_rdna::ConvEpilogueMatch> matchDw3dBiasAct(const MatchContext& context)
{
    auto matched = conv_rdna::matchConvBiasActivation(context);
    if(!matched || !dw3d::serves(matched->problem))
    {
        return std::nullopt;
    }
    return matched;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchDw3dBiasAct(context);
    if(!matched)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    conv_rdna::bindConvFeatures(bound,
                                TOKEN_PREFIX,
                                matched->problem,
                                matched->conv->x_tensor_uid(),
                                matched->conv->w_tensor_uid(),
                                matched->yUid,
                                *matched->x,
                                *matched->w,
                                *matched->y);
    if(matched->bUid)
    {
        bound[std::string(B_TOKEN)] = *matched->bUid;
    }
    return bound;
}

/// The variant dw3d::isVariantOf picks for the graph's problem; every kernel of this engine
/// carries the epilogue. One candidate per graph.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchDw3dBiasAct(context);
    return matched && dw3d::isVariantOf(kernel, matched->problem);
}

/// The kernel matcher leaves a single candidate to rank.
double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& /*kernel*/)
{
    return 1.0;
}

BiasActBinding biasActBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_bias_act_dw3d_rdna dispatch is missing bound token '"
                    + std::string(token) + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    BiasActBinding binding;
    binding.x = read(X_TOKEN);
    binding.w = read(W_TOKEN);
    // No bias token for a graph without a bias.
    binding.b = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, B_TOKEN);
    binding.y = read(Y_TOKEN);
    return binding;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvFwdBiasActDw3dRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvFwdBiasActDw3dRdna(IngestorKernelCode code, LaunchInputs launch)
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
    return std::make_unique<PreparedHipConvFwdBiasActDw3dRdna>(std::move(code), launch);
}

/// @see hipConvFwdBiasActDw3dRdnaLaunchValues
MetadataValues launchValues(const BiasActBinding& binding,
                            const ConvProblem& problem,
                            const dw3d::Geometry& geometry,
                            const Activation& activation)
{
    MetadataValues values{{std::string(X_UID_VALUE), binding.x},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(Y_UID_VALUE), binding.y}};
    conv_rdna::addEpilogueLaunchValues(values, binding.b, activation);
    dw3d::addProblemLaunchValues(values, problem, geometry);
    return values;
}

/// @see readHipConvFwdBiasActDw3dRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{X_UID_VALUE, W_UID_VALUE, Y_UID_VALUE};
    const auto epilogueNames = conv_rdna::epilogueLaunchValueNames();
    names.insert(names.end(), epilogueNames.begin(), epilogueNames.end());
    const auto problemNames = dw3d::problemLaunchValueNames();
    names.insert(names.end(), problemNames.begin(), problemNames.end());
    requireOnlyLaunchValues(inputs, names);

    LaunchInputs launch;
    launch.binding.x = requireLaunchInt(inputs, X_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, W_UID_VALUE);
    launch.binding.y = requireLaunchInt(inputs, Y_UID_VALUE);
    dw3d::readProblemLaunchValues(inputs, launch.problem, launch.geometry);
    const auto epilogue = conv_rdna::readEpilogueLaunchValues(inputs);
    launch.binding.b = epilogue.bUid;
    launch.activation = epilogue.activation;
    return launch;
}

class HipConvFwdBiasActDw3dRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvFwdBiasActDw3dRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                             const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The kernels read every operand in place and store no intermediate.
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
        launch.binding = biasActBinding(bound);
        const auto matched = matchDw3dBiasAct(context);
        if(!matched)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_bias_act_dw3d_rdna prepare() got a graph its matcher does not "
                "admit");
        }
        launch.problem = matched->problem;
        launch.geometry = dw3d::geometryOf(launch.problem);
        launch.activation = matched->activation;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        dw3d::addCompileOptions(
            options, launch.problem, launch.geometry, /*hasBias=*/true, /*hasActivation=*/true);

        auto code = buildIngestorKernelCode(
            _kernelCompiler,
            _kpackLoader,
            context,
            kernel,
            options,
            dw3d::kernelSignature(launch.geometry.generic, /*hasBias=*/true, /*hasActivation=*/true));
        return makePrepared(std::move(code), launch);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        // The signature depends on the saved geometry, so read it before the restore checks
        // the saved code against it.
        const auto launch = readLaunchInputs(inputs);
        auto restored = restoreIngestorLaunch<LaunchInputs>(
            inputs,
            std::move(code),
            dw3d::kernelSignature(launch.geometry.generic, /*hasBias=*/true, /*hasActivation=*/true),
            &readLaunchInputs,
            deviceOrdinal);
        return makePrepared(std::move(restored.code), restored.launch);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& dw = dynamic_cast<const PreparedHipConvFwdBiasActDw3dRdna&>(prepared);
        const auto& launch = dw.launch();
        return SavedLaunchInputs{
            std::string(DISPATCH_SYMBOL_V1),
            launchValues(launch.binding, launch.problem, launch.geometry, launch.activation)};
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
        const auto& dw = dynamic_cast<const PreparedHipConvFwdBiasActDw3dRdna&>(prepared);
        const auto& binding = dw.launch().binding;

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const void* b = nullptr;
        if(binding.b)
        {
            b = hipdnn_plugin_sdk::findDeviceBuffer(*binding.b, deviceBuffers, numDeviceBuffers)
                    .ptr;
        }
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        dw3d::launch(dw.kernelForStream(handle.getStream()),
                     handle.getStream(),
                     x.ptr,
                     w.ptr,
                     y.ptr,
                     dw.launch().problem,
                     dw.launch().geometry,
                     b,
                     binding.b.has_value(),
                     &dw.launch().activation);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvFwdBiasActDw3dRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        hipConvFwdBiasActDw3dRdnaKpackModuleCache());
    static const HipConvFwdBiasActDw3dRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                            s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

MetadataValues
    hipConvFwdBiasActDw3dRdnaLaunchValues(const HipConvFwdBiasActDw3dRdnaBinding& binding,
                                          const conv_rdna::ConvProblem& problem,
                                          const conv_rdna::dw3d::Geometry& geometry,
                                          const conv_rdna::Activation& activation)
{
    return launchValues(binding, problem, geometry, activation);
}

HipConvFwdBiasActDw3dRdnaLaunchInputs
    readHipConvFwdBiasActDw3dRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvFwdBiasActDw3dRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvFwdBiasActDw3dRdnaModuleCache()
{
    hipConvFwdBiasActDw3dRdnaKpackModuleCache().clear();
}

void registerHipConvFwdBiasActDw3dRdnaSymbols(
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(DISPATCH_SYMBOL_V1), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
