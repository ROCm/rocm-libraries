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
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
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
#include "engines/kernel_ingestor_engine/packs/HipConvFwdBiasFmaIgemmRdnaLaunchValues.hpp"

/**
 * @file HipConvFwdBiasFmaIgemmRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_bias_fma_igemm_rdna`
 *        (descriptors/hip/conv_bias_rdna/): matching, scoring, dispatch, and the function that
 *        registers them.
 *
 * The engine serves a two-node graph on the gfx11 and gfx12 RDNA targets: a conv-forward node
 * whose virtual output feeds a pointwise ADD with a per-output-channel bias, the form a biased
 * convolution takes in hipDNN. It runs the FMA implicit-GEMM kernel of
 * `hip_conv_fwd_fma_igemm_rdna` (ConvRdnaFmaIgemm.hpp) built with its bias epilogue, so the
 * convolution result is never stored: the bias is added in fp32 and the sum rounded once. The
 * convolution itself is served under the same rules as the unfused engine.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
namespace fma_igemm = conv_rdna::fma_igemm;

// The contract with hip_conv_fwd_bias_fma_igemm_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.dispatch";

// The names of the saved launch values (HipConvFwdBiasFmaIgemmRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1 = HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view X_UID_VALUE = "x.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view B_UID_VALUE = "b.uid";
constexpr std::string_view Y_UID_VALUE = "y.uid";

constexpr std::string_view TOKEN_PREFIX = "conv_fwd_bias";
constexpr std::string_view X_TOKEN = "conv_fwd_bias.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd_bias.w.uid";
constexpr std::string_view B_TOKEN = "conv_fwd_bias.b.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd_bias.y.uid";

using BiasBinding = HipConvFwdBiasFmaIgemmRdnaBinding;
using LaunchInputs = HipConvFwdBiasFmaIgemmRdnaLaunchInputs;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The conv node, the ADD node, the tensors the kernel reads and writes, and the problem.
struct MatchedConvBias
{
    const data_objects::ConvolutionFwdAttributes* conv = nullptr;
    const data_objects::TensorAttributes* x = nullptr;
    const data_objects::TensorAttributes* w = nullptr;
    const data_objects::TensorAttributes* b = nullptr;
    /// The output of the ADD: what the kernel stores.
    const data_objects::TensorAttributes* y = nullptr;
    int64_t bUid = 0;
    int64_t yUid = 0;
    ConvProblem problem;
};

bool sameDims(const data_objects::TensorAttributes& a, const data_objects::TensorAttributes& b)
{
    const auto* aDims = a.dims();
    const auto* bDims = b.dims();
    if(aDims == nullptr || bDims == nullptr || aDims->size() != bDims->size())
    {
        return false;
    }
    for(flatbuffers::uoffset_t i = 0; i < aDims->size(); ++i)
    {
        if(aDims->Get(i) != bDims->Get(i))
        {
            return false;
        }
    }
    return true;
}

/// A device tensor of @p problem's dtype holding one value per output channel: dims
/// (1, K, 1...) of the convolution's rank, the channels contiguous.
bool isChannelBias(const data_objects::TensorAttributes& bias,
                   const ConvProblem& problem,
                   flatbuffers::uoffset_t rank)
{
    if(bias.virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&bias)
       || bias.data_type() != problem.dtype)
    {
        return false;
    }
    const auto* dims = bias.dims();
    const auto* strides = bias.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != rank || strides->size() != rank)
    {
        return false;
    }
    for(flatbuffers::uoffset_t i = 0; i < rank; ++i)
    {
        if(dims->Get(i) != (i == 1 ? int64_t{problem.k} : 1))
        {
            return false;
        }
    }
    return strides->Get(1) == 1;
}

std::optional<MatchedConvBias> matchConvBias(const MatchContext& context)
{
    if(context.graph.nodeCount() != 2)
    {
        return std::nullopt;
    }
    const data_objects::ConvolutionFwdAttributes* conv = nullptr;
    const data_objects::PointwiseAttributes* add = nullptr;
    for(uint32_t i = 0; i < 2; ++i)
    {
        const auto& node = context.graph.getNodeWrapper(i);
        if(node.attributesType() == data_objects::NodeAttributes::ConvolutionFwdAttributes)
        {
            conv = &node.attributesAs<data_objects::ConvolutionFwdAttributes>();
        }
        else if(node.attributesType() == data_objects::NodeAttributes::PointwiseAttributes)
        {
            add = &node.attributesAs<data_objects::PointwiseAttributes>();
        }
    }
    if(conv == nullptr || add == nullptr || add->operation() != data_objects::PointwiseMode::ADD
       || !add->in_1_tensor_uid().has_value() || add->in_2_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    // One ADD operand is the convolution output, the other the bias, in either order.
    const int64_t convOutUid = conv->y_tensor_uid();
    const int64_t in0 = add->in_0_tensor_uid();
    const int64_t in1 = *add->in_1_tensor_uid();
    if((in0 == convOutUid) == (in1 == convOutUid))
    {
        return std::nullopt;
    }

    MatchedConvBias matched;
    matched.conv = conv;
    matched.bUid = in0 == convOutUid ? in1 : in0;
    matched.yUid = add->out_0_tensor_uid();
    matched.x = conv_rdna::findTensor(context, conv->x_tensor_uid());
    matched.w = conv_rdna::findTensor(context, conv->w_tensor_uid());
    matched.b = conv_rdna::findTensor(context, matched.bUid);
    matched.y = conv_rdna::findTensor(context, matched.yUid);
    const auto* convOut = conv_rdna::findTensor(context, convOutUid);
    if(matched.x == nullptr || matched.w == nullptr || matched.b == nullptr || matched.y == nullptr
       || convOut == nullptr)
    {
        return std::nullopt;
    }

    // The convolution output only feeds the ADD: virtual, the shape of the ADD output, and
    // either fp32 or the data dtype (the kernel keeps it in fp32 either way).
    if(!convOut->virtual_() || !sameDims(*convOut, *matched.y)
       || (convOut->data_type() != data_objects::DataType::FLOAT
           && convOut->data_type() != matched.y->data_type()))
    {
        return std::nullopt;
    }

    // The convolution as the kernel runs it, with the ADD output as its stored output.
    const conv_rdna::ConvAttributes attributes{conv->conv_mode(),
                                               conv->stride(),
                                               conv->dilation(),
                                               conv->pre_padding(),
                                               conv->post_padding()};
    const auto problem = conv_rdna::convProblem(*matched.x, *matched.w, *matched.y, attributes);
    if(!problem || !isChannelBias(*matched.b, *problem, matched.x->dims()->size()))
    {
        return std::nullopt;
    }
    matched.problem = *problem;
    return matched;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchConvBias(context);
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
    bound[std::string(B_TOKEN)] = matched->bUid;
    return bound;
}

/// The variant of the graph's problem (fma_igemm::isVariantOf); every kernel of this engine
/// carries the bias epilogue.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchConvBias(context);
    return matched && fma_igemm::isVariantOf(kernel, matched->problem);
}

/// The kernel matcher leaves a single candidate to rank.
double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& /*kernel*/)
{
    return 1.0;
}

BiasBinding biasBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_bias_fma_igemm_rdna dispatch is missing bound token '"
                    + std::string(token) + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(X_TOKEN), read(W_TOKEN), read(B_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvFwdBiasFmaIgemmRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvFwdBiasFmaIgemmRdna(IngestorKernelCode code,
                                       BiasBinding binding,
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

    const BiasBinding& binding() const
    {
        return _binding;
    }

    const ConvProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _code;
    BiasBinding _binding;
    ConvProblem _problem;
};

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch>
    makePrepared(IngestorKernelCode code, const BiasBinding& binding, const ConvProblem& problem)
{
    // Both grid dimensions were bounded by fma_igemm::serves or by readLaunchInputs.
    code.setBlockSize(fma_igemm::BLOCK_SIZE, 1, 1);
    code.setGridSize(static_cast<unsigned int>(fma_igemm::gridXBlocks(problem)),
                     static_cast<unsigned int>(fma_igemm::gridY(problem)),
                     1);
    return std::make_unique<PreparedHipConvFwdBiasFmaIgemmRdna>(std::move(code), binding, problem);
}

/// @see hipConvFwdBiasFmaIgemmRdnaLaunchValues
MetadataValues launchValues(const BiasBinding& binding, const ConvProblem& problem)
{
    MetadataValues values{{std::string(X_UID_VALUE), binding.x},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(B_UID_VALUE), binding.b},
                          {std::string(Y_UID_VALUE), binding.y}};
    fma_igemm::addProblemLaunchValues(values, problem);
    return values;
}

/// @see readHipConvFwdBiasFmaIgemmRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{X_UID_VALUE, W_UID_VALUE, B_UID_VALUE, Y_UID_VALUE};
    const auto problemNames = fma_igemm::problemLaunchValueNames();
    names.insert(names.end(), problemNames.begin(), problemNames.end());
    requireOnlyLaunchValues(inputs, names);

    LaunchInputs launch;
    launch.binding.x = requireLaunchInt(inputs, X_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, W_UID_VALUE);
    launch.binding.b = requireLaunchInt(inputs, B_UID_VALUE);
    launch.binding.y = requireLaunchInt(inputs, Y_UID_VALUE);
    launch.problem = fma_igemm::readProblemLaunchValues(inputs);
    return launch;
}

class HipConvFwdBiasFmaIgemmRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvFwdBiasFmaIgemmRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                              const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The weights are read in place and the convolution result is never stored.
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
        const auto binding = biasBinding(bound);
        const auto matched = matchConvBias(context);
        if(!matched || !fma_igemm::serves(matched->problem))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_bias_fma_igemm_rdna prepare() got a graph its matcher does not "
                "admit");
        }
        const auto& problem = matched->problem;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        fma_igemm::addCompileOptions(options, problem, /*hasBias=*/true);

        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            fma_igemm::kernelSignature(/*hasBias=*/true));
        return makePrepared(std::move(code), binding, problem);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        auto restored
            = restoreIngestorLaunch<LaunchInputs>(inputs,
                                                  std::move(code),
                                                  fma_igemm::kernelSignature(/*hasBias=*/true),
                                                  &readLaunchInputs,
                                                  deviceOrdinal);
        return makePrepared(
            std::move(restored.code), restored.launch.binding, restored.launch.problem);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& conv = dynamic_cast<const PreparedHipConvFwdBiasFmaIgemmRdna&>(prepared);
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
        const auto& conv = dynamic_cast<const PreparedHipConvFwdBiasFmaIgemmRdna&>(prepared);
        const auto& binding = conv.binding();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto b
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.b, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        fma_igemm::launch(conv.kernelForStream(handle.getStream()),
                          handle.getStream(),
                          x.ptr,
                          w.ptr,
                          y.ptr,
                          b.ptr,
                          /*hasBias=*/true,
                          conv.problem());
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvFwdBiasFmaIgemmRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        hipConvFwdBiasFmaIgemmRdnaKpackModuleCache());
    static const HipConvFwdBiasFmaIgemmRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                             s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

MetadataValues
    hipConvFwdBiasFmaIgemmRdnaLaunchValues(const HipConvFwdBiasFmaIgemmRdnaBinding& binding,
                                           const conv_rdna::ConvProblem& problem)
{
    return launchValues(binding, problem);
}

HipConvFwdBiasFmaIgemmRdnaLaunchInputs
    readHipConvFwdBiasFmaIgemmRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvFwdBiasFmaIgemmRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvFwdBiasFmaIgemmRdnaModuleCache()
{
    hipConvFwdBiasFmaIgemmRdnaKpackModuleCache().clear();
}

void registerHipConvFwdBiasFmaIgemmRdnaSymbols(
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
