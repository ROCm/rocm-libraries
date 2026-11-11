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
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"

/**
 * @file HipConvFwdRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_rdna` (descriptors/hip/conv_rdna/):
 *        matching, scoring, dispatch, and the function that registers them.
 *
 * The engine serves a graph of exactly one conv-forward node on RDNA (gfx11-generic,
 * gfx12-generic). Its kernels use no matrix instructions. Kernel family `fma_igemm` is an
 * FMA implicit GEMM (kernels/MIOpenConv3dFwdFmaHip.cpp) for 2-D and 3-D problems, packed
 * NC(D)HW or N(D)HWC, FLOAT / HALF / BFLOAT16, any stride, dilation and padding, and groups
 * with at least MIN_OUT_CHANNELS_PER_GROUP output channels each. The matcher refuses
 * everything else, so another engine gets the plan.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
using conv_rdna::Layout;

// The contract with conv_fwd_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.hip_conv_fwd_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.hip_conv_fwd_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.hip_conv_fwd_rdna.dispatch";

constexpr std::string_view FAMILY_FIELD = "kernel_family";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view LAYOUT_FIELD = "layout";
constexpr std::string_view TILE_N_FIELD = "tile_n";
constexpr std::string_view FMA_IGEMM_FAMILY = "fma_igemm";

constexpr std::string_view TOKEN_PREFIX = "conv_fwd";
constexpr std::string_view X_TOKEN = "conv_fwd.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd.w.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd.y.uid";

// Must match MIOpenConv3dFwdFmaHip.cpp.
constexpr int64_t FMA_TILE_M = 128;
constexpr unsigned int FMA_BLOCK_SIZE = 256;
// Fewer output channels per group leave most of the (at least 32-wide) channel tile idle;
// depthwise convolutions belong to a depthwise kernel family.
constexpr int64_t MIN_OUT_CHANNELS_PER_GROUP = 16;
constexpr int64_t MAX_GRID_X_BLOCKS = 0x7fffffff / FMA_BLOCK_SIZE;
constexpr int64_t MAX_GRID_Y = 0xffff;

/// The operand uids a match binds, in kernel argument order.
struct ConvBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
};

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// Output channels per workgroup: the narrowest tile that holds one group's channels.
int64_t fmaTileN(const ConvProblem& problem)
{
    const int64_t kPerGroup = problem.outChannelsPerGroup();
    return kPerGroup <= 32 ? 32 : (kPerGroup <= 64 ? 64 : 128);
}

int64_t fmaGridXBlocks(const ConvProblem& problem)
{
    const int64_t m = int64_t{problem.n} * problem.out[0] * problem.out[1] * problem.out[2];
    return (m + FMA_TILE_M - 1) / FMA_TILE_M;
}

int64_t fmaGridY(const ConvProblem& problem)
{
    const int64_t kPerGroup = problem.outChannelsPerGroup();
    const int64_t tile = fmaTileN(problem);
    return problem.groups * ((kPerGroup + tile - 1) / tile);
}

bool fmaServes(const ConvProblem& problem)
{
    if(problem.groups > 1 && problem.outChannelsPerGroup() < MIN_OUT_CHANNELS_PER_GROUP)
    {
        return false;
    }
    return fmaGridXBlocks(problem) <= MAX_GRID_X_BLOCKS && fmaGridY(problem) <= MAX_GRID_Y;
}

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
    if(!matched || !fmaServes(matched->problem))
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

/// The kernel of the graph's family, dtype and layout, with the channel tile fmaTileN picks.
/// A wrong dtype or layout would return wrong numbers; a wrong tile is only slower, but one
/// tile per graph leaves exactly one candidate.
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
    return kernel.getStringMetadata(std::string(FAMILY_FIELD)) == FMA_IGEMM_FAMILY
           && fmaServes(problem)
           && kernel.getStringMetadata(std::string(DTYPE_FIELD))
                  == data_objects::EnumNameDataType(problem.dtype)
           && kernel.getStringMetadata(std::string(LAYOUT_FIELD))
                  == conv_rdna::layoutName(problem.layout)
           && kernel.getIntMetadata(std::string(TILE_N_FIELD)) == fmaTileN(problem);
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
                "hip_conv_fwd_rdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(X_TOKEN), read(W_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvFwdRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvFwdRdna(IngestorKernelCode code, ConvBinding binding, ConvProblem problem)
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

/// The kernel's CONV3D_DTYPE for a dtype.
int fmaDtypeCode(data_objects::DataType dtype)
{
    return dtype == data_objects::DataType::FLOAT ? 0
                                                  : (dtype == data_objects::DataType::HALF ? 1 : 2);
}

/// Mirrors the argument list of MIOpenConv3dFwdFmaHip and the launch() below one for one.
const std::vector<KernelArgument>& fmaKernelSignature()
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

class HipConvFwdRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvFwdRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                  const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Channels-first weights are read in place (CONV3D_WEI_KCZYX) and channels-last ones
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
        if(!matched || !fmaServes(matched->problem))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_rdna prepare() got a graph its matcher does not admit");
        }
        const auto& problem = matched->problem;
        const int channelsLast = problem.layout == Layout::CHANNELS_LAST ? 1 : 0;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        options.add("CONV3D_DTYPE", fmaDtypeCode(problem.dtype));
        options.add("CONV3D_NDHWC", channelsLast);
        options.add("CONV3D_WEI_KCZYX", 1 - channelsLast);
        options.add("TILE_N", static_cast<int>(fmaTileN(problem)));

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, fmaKernelSignature());

        // fmaServes bounded both against the grid limits.
        code.setBlockSize(FMA_BLOCK_SIZE, 1, 1);
        code.setGridSize(static_cast<unsigned int>(fmaGridXBlocks(problem)),
                         static_cast<unsigned int>(fmaGridY(problem)),
                         1);

        return std::make_unique<PreparedHipConvFwdRdna>(std::move(code), binding, problem);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& conv = dynamic_cast<const PreparedHipConvFwdRdna&>(prepared);
        const auto& binding = conv.binding();
        const auto& p = conv.problem();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        // A 2-D problem carries depth 1, depth stride and dilation 1 and depth padding 0.
        conv.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    w.ptr,
                    y.ptr,
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

const HipConvFwdRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(hipConvFwdRdnaKpackModuleCache());
    static const HipConvFwdRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

compilation::KpackModuleCache& hipConvFwdRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvFwdRdnaModuleCache()
{
    hipConvFwdRdnaKpackModuleCache().clear();
}

void registerHipConvFwdRdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
