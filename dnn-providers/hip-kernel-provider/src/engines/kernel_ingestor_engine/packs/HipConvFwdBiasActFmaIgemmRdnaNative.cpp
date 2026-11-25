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
#include "engines/kernel_ingestor_engine/packs/ConvRdnaFmaIgemm.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaFusedEpilogue.hpp"
#include "engines/kernel_ingestor_engine/packs/HipConvFwdBiasActFmaIgemmRdnaLaunchValues.hpp"

/**
 * @file HipConvFwdBiasActFmaIgemmRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_bias_act_fma_igemm_rdna`
 *        (descriptors/hip/conv_bias_act_rdna/): matching, scoring, dispatch, and the function
 *        that registers them.
 *
 * The engine serves, on the gfx11 and gfx12 RDNA targets, a conv-forward node whose virtual
 * output feeds a pointwise ADD with a per-output-channel bias, whose virtual output feeds a
 * forward activation; or, without the bias, a conv-forward node whose virtual output feeds the
 * activation directly. It runs the FMA implicit-GEMM kernel of
 * `hip_conv_fwd_fma_igemm_rdna` (ConvRdnaFmaIgemm.hpp) built with its bias and activation
 * epilogues, so neither intermediate is stored: the bias is added and the activation applied
 * in fp32, and the result rounded once. The convolution itself is served under the same rules
 * as the unfused engine.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using conv_rdna::ConvProblem;
namespace fma_igemm = conv_rdna::fma_igemm;
using conv_rdna::Activation;

// The contract with hip_conv_fwd_bias_act_fma_igemm_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.dispatch";

// The names of the saved launch values (HipConvFwdBiasActFmaIgemmRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1
    = HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view X_UID_VALUE = "x.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view Y_UID_VALUE = "y.uid";

constexpr std::string_view TOKEN_PREFIX = "conv_fwd_bias_act";
constexpr std::string_view X_TOKEN = "conv_fwd_bias_act.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd_bias_act.w.uid";
constexpr std::string_view B_TOKEN = "conv_fwd_bias_act.b.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd_bias_act.y.uid";

using BiasActBinding = HipConvFwdBiasActFmaIgemmRdnaBinding;
using LaunchInputs = HipConvFwdBiasActFmaIgemmRdnaLaunchInputs;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// conv -> [ADD(bias) ->] activation (ConvRdnaFusedEpilogue.hpp); graphMatches and kernelMatches
/// add the rules under which the FMA implicit-GEMM kernel serves the convolution.
std::optional<conv_rdna::ConvEpilogueMatch> matchConvBiasAct(const MatchContext& context)
{
    return conv_rdna::matchConvBiasActivation(context);
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchConvBiasAct(context);
    if(!matched || !fma_igemm::serves(matched->problem))
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

/// The variant of the graph's problem (fma_igemm::isVariantOf); every kernel of this engine
/// carries the bias and activation epilogues.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchConvBiasAct(context);
    return matched && fma_igemm::isVariantOf(kernel, matched->problem);
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
                "hip_conv_fwd_bias_act_fma_igemm_rdna dispatch is missing bound token '"
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

class PreparedHipConvFwdBiasActFmaIgemmRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvFwdBiasActFmaIgemmRdna(IngestorKernelCode code,
                                          BiasActBinding binding,
                                          ConvProblem problem,
                                          Activation activation)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _activation(activation)
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

    const BiasActBinding& binding() const
    {
        return _binding;
    }

    const ConvProblem& problem() const
    {
        return _problem;
    }

    const Activation& activation() const
    {
        return _activation;
    }

private:
    IngestorKernelCode _code;
    BiasActBinding _binding;
    ConvProblem _problem;
    Activation _activation;
};

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePrepared(IngestorKernelCode code,
                                               const BiasActBinding& binding,
                                               const ConvProblem& problem,
                                               const Activation& activation)
{
    // Both grid dimensions were bounded by fma_igemm::serves or by readLaunchInputs.
    code.setBlockSize(fma_igemm::BLOCK_SIZE, 1, 1);
    code.setGridSize(static_cast<unsigned int>(fma_igemm::gridXBlocks(problem)),
                     static_cast<unsigned int>(fma_igemm::gridY(problem)),
                     1);
    return std::make_unique<PreparedHipConvFwdBiasActFmaIgemmRdna>(
        std::move(code), binding, problem, activation);
}

/// @see hipConvFwdBiasActFmaIgemmRdnaLaunchValues
MetadataValues launchValues(const BiasActBinding& binding,
                            const ConvProblem& problem,
                            const Activation& activation)
{
    MetadataValues values{{std::string(X_UID_VALUE), binding.x},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(Y_UID_VALUE), binding.y}};
    conv_rdna::addEpilogueLaunchValues(values, binding.b, activation);
    fma_igemm::addProblemLaunchValues(values, problem);
    return values;
}

/// @see readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{X_UID_VALUE, W_UID_VALUE, Y_UID_VALUE};
    const auto epilogueNames = conv_rdna::epilogueLaunchValueNames();
    names.insert(names.end(), epilogueNames.begin(), epilogueNames.end());
    const auto problemNames = fma_igemm::problemLaunchValueNames();
    names.insert(names.end(), problemNames.begin(), problemNames.end());
    requireOnlyLaunchValues(inputs, names);

    LaunchInputs launch;
    launch.binding.x = requireLaunchInt(inputs, X_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, W_UID_VALUE);
    launch.binding.y = requireLaunchInt(inputs, Y_UID_VALUE);
    launch.problem = fma_igemm::readProblemLaunchValues(inputs);
    const auto epilogue = conv_rdna::readEpilogueLaunchValues(inputs);
    launch.binding.b = epilogue.bUid;
    launch.activation = epilogue.activation;
    return launch;
}

class HipConvFwdBiasActFmaIgemmRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvFwdBiasActFmaIgemmRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                                 const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The weights are read in place and no intermediate is stored.
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
        const auto binding = biasActBinding(bound);
        const auto matched = matchConvBiasAct(context);
        if(!matched || !fma_igemm::serves(matched->problem))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_bias_act_fma_igemm_rdna prepare() got a graph its matcher does not "
                "admit");
        }
        const auto& problem = matched->problem;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        fma_igemm::addCompileOptions(options, problem, /*hasBias=*/true, /*hasActivation=*/true);

        auto code = buildIngestorKernelCode(
            _kernelCompiler,
            _kpackLoader,
            context,
            kernel,
            options,
            fma_igemm::kernelSignature(/*hasBias=*/true, /*hasActivation=*/true));
        return makePrepared(std::move(code), binding, problem, matched->activation);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        auto restored = restoreIngestorLaunch<LaunchInputs>(
            inputs,
            std::move(code),
            fma_igemm::kernelSignature(/*hasBias=*/true, /*hasActivation=*/true),
            &readLaunchInputs,
            deviceOrdinal);
        return makePrepared(std::move(restored.code),
                            restored.launch.binding,
                            restored.launch.problem,
                            restored.launch.activation);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& conv = dynamic_cast<const PreparedHipConvFwdBiasActFmaIgemmRdna&>(prepared);
        return SavedLaunchInputs{std::string(DISPATCH_SYMBOL_V1),
                                 launchValues(conv.binding(), conv.problem(), conv.activation())};
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
        const auto& conv = dynamic_cast<const PreparedHipConvFwdBiasActFmaIgemmRdna&>(prepared);
        const auto& binding = conv.binding();

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

        fma_igemm::launch(conv.kernelForStream(handle.getStream()),
                          handle.getStream(),
                          x.ptr,
                          w.ptr,
                          y.ptr,
                          b,
                          binding.b.has_value(),
                          conv.problem(),
                          &conv.activation());
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvFwdBiasActFmaIgemmRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        hipConvFwdBiasActFmaIgemmRdnaKpackModuleCache());
    static const HipConvFwdBiasActFmaIgemmRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                                s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

MetadataValues
    hipConvFwdBiasActFmaIgemmRdnaLaunchValues(const HipConvFwdBiasActFmaIgemmRdnaBinding& binding,
                                              const conv_rdna::ConvProblem& problem,
                                              const conv_rdna::Activation& activation)
{
    return launchValues(binding, problem, activation);
}

HipConvFwdBiasActFmaIgemmRdnaLaunchInputs
    readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvFwdBiasActFmaIgemmRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvFwdBiasActFmaIgemmRdnaModuleCache()
{
    hipConvFwdBiasActFmaIgemmRdnaKpackModuleCache().clear();
}

void registerHipConvFwdBiasActFmaIgemmRdnaSymbols(
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
