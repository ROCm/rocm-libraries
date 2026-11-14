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
#include "engines/kernel_ingestor_engine/packs/ConvRdnaFmaIgemm.hpp"
#include "engines/kernel_ingestor_engine/packs/HipConvFwdFmaIgemmRdnaLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file HipConvFwdFmaIgemmRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_fma_igemm_rdna` (descriptors/hip/conv_rdna/):
 *        matching, scoring, dispatch, and the function that registers them.
 *
 * The engine serves a graph of exactly one conv-forward node on the gfx11 and gfx12 RDNA
 * targets with the FMA implicit-GEMM kernel (ConvRdnaFmaIgemm.hpp, kernels/ConvFwdFmaIgemm.cpp):
 * 1-D, 2-D and 3-D problems, packed NC[D][H]W or N[D][H]WC, FLOAT / HALF / BFLOAT16, any stride,
 * dilation and padding, and groups of at least 16 output channels. The matcher refuses
 * everything else, so another engine gets the plan.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
namespace fma_igemm = conv_rdna::fma_igemm;

// The contract with hip_conv_fwd_fma_igemm_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_fma_igemm_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_fma_igemm_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_fma_igemm_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_fwd_fma_igemm_rdna.dispatch";

// The names of the saved launch values (HipConvFwdFmaIgemmRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_FWD_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view X_UID_VALUE = "x.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view Y_UID_VALUE = "y.uid";

constexpr std::string_view TOKEN_PREFIX = "conv_fwd";
constexpr std::string_view X_TOKEN = "conv_fwd.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd.w.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd.y.uid";

using ConvBinding = HipConvFwdFmaIgemmRdnaBinding;
using LaunchInputs = HipConvFwdFmaIgemmRdnaLaunchInputs;

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
    if(!problem)
    {
        return std::nullopt;
    }
    matched.problem = *problem;
    return matched;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchConv(context);
    // Every family the engine ships must decline for the graph to be declined; fma_igemm is
    // the only one so far.
    if(!matched || !fma_igemm::serves(matched->problem))
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

/// The variant of the graph's problem (fma_igemm::isVariantOf).
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchConv(context);
    if(!matched)
    {
        return false;
    }
    return fma_igemm::isVariantOf(kernel, matched->problem);
}

/// The kernel matcher leaves a single candidate to rank.
double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& /*kernel*/)
{
    return 1.0;
}

ConvBinding convBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_fma_igemm_rdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(X_TOKEN), read(W_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvFwdFmaIgemmRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvFwdFmaIgemmRdna(IngestorKernelCode code,
                                   ConvBinding binding,
                                   ConvProblem problem)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
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

    const ConvBinding& binding() const
    {
        return _binding;
    }

    const ConvProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _code;
    ConvBinding _binding;
    ConvProblem _problem;
};

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch>
    makePrepared(IngestorKernelCode code, const ConvBinding& binding, const ConvProblem& problem)
{
    // Both grid dimensions were bounded by fma_igemm::serves or by readLaunchInputs.
    code.setBlockSize(fma_igemm::BLOCK_SIZE, 1, 1);
    code.setGridSize(static_cast<unsigned int>(fma_igemm::gridXBlocks(problem)),
                     static_cast<unsigned int>(fma_igemm::gridY(problem)),
                     1);
    return std::make_unique<PreparedHipConvFwdFmaIgemmRdna>(std::move(code), binding, problem);
}

/// @see hipConvFwdFmaIgemmRdnaLaunchValues
MetadataValues launchValues(const ConvBinding& binding, const ConvProblem& problem)
{
    MetadataValues values{{std::string(X_UID_VALUE), binding.x},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(Y_UID_VALUE), binding.y}};
    fma_igemm::addProblemLaunchValues(values, problem);
    return values;
}

/// @see readHipConvFwdFmaIgemmRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{X_UID_VALUE, W_UID_VALUE, Y_UID_VALUE};
    const auto problemNames = fma_igemm::problemLaunchValueNames();
    names.insert(names.end(), problemNames.begin(), problemNames.end());
    requireOnlyLaunchValues(inputs, names);

    LaunchInputs launch;
    launch.binding.x = requireLaunchInt(inputs, X_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, W_UID_VALUE);
    launch.binding.y = requireLaunchInt(inputs, Y_UID_VALUE);
    launch.problem = fma_igemm::readProblemLaunchValues(inputs);
    return launch;
}

class HipConvFwdFmaIgemmRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvFwdFmaIgemmRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                          const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Channels-first weights are read in place (CONV_WEI_KCZYX) and channels-last ones
    /// are K x taps x C already, so nothing is transposed.
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
        const auto binding = convBinding(bound);
        const auto matched = matchConv(context);
        if(!matched || !fma_igemm::serves(matched->problem))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_fma_igemm_rdna prepare() got a graph its matcher does not admit");
        }
        const auto& problem = matched->problem;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        fma_igemm::addCompileOptions(options, problem, /*hasBias=*/false);

        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            fma_igemm::kernelSignature(/*hasBias=*/false));
        return makePrepared(std::move(code), binding, problem);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        auto restored
            = restoreIngestorLaunch<LaunchInputs>(inputs,
                                                  std::move(code),
                                                  fma_igemm::kernelSignature(/*hasBias=*/false),
                                                  &readLaunchInputs,
                                                  deviceOrdinal);
        return makePrepared(
            std::move(restored.code), restored.launch.binding, restored.launch.problem);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& conv = dynamic_cast<const PreparedHipConvFwdFmaIgemmRdna&>(prepared);
        return SavedLaunchInputs{std::string(DISPATCH_SYMBOL_V1),
                                 launchValues(conv.binding(), conv.problem())};
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
        const auto& conv = dynamic_cast<const PreparedHipConvFwdFmaIgemmRdna&>(prepared);
        const auto& binding = conv.binding();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        fma_igemm::launch(conv.kernelForStream(handle.getStream()),
                          handle.getStream(),
                          x.ptr,
                          w.ptr,
                          y.ptr,
                          nullptr,
                          /*hasBias=*/false,
                          conv.problem());
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvFwdFmaIgemmRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        hipConvFwdFmaIgemmRdnaKpackModuleCache());
    static const HipConvFwdFmaIgemmRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                         s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

MetadataValues hipConvFwdFmaIgemmRdnaLaunchValues(const HipConvFwdFmaIgemmRdnaBinding& binding,
                                                  const conv_rdna::ConvProblem& problem)
{
    return launchValues(binding, problem);
}

HipConvFwdFmaIgemmRdnaLaunchInputs
    readHipConvFwdFmaIgemmRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvFwdFmaIgemmRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvFwdFmaIgemmRdnaModuleCache()
{
    hipConvFwdFmaIgemmRdnaKpackModuleCache().clear();
}

void registerHipConvFwdFmaIgemmRdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(DISPATCH_SYMBOL_V1), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
