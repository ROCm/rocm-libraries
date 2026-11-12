// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
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
#include "engines/kernel_ingestor_engine/descriptors/conv/fft/kernels/ConvFftArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvFftProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvFftNative.cpp
 * @brief The native half of the FFT convolution engines (CONV_FFT_FWD, CONV_FFT_BWD,
 *        CONV_FFT_WRW): graph matching, kernel matching, workspace sizing and dispatch of the
 *        transform, product and inverse kernels (ConvFft.cpp).
 *
 * Forward and backward data: the filter spectra once, then per chunk of tiles the data
 * spectra, the per-bin product over channels and the inverse to the output tiles. Backward
 * weights: per chunk the x and dy spectra and the per-bin product summed over the chunk's tiles
 * into the weight-gradient spectra (chunk 0 writes, later chunks add, in order), then the
 * inverse to dw. Every launch is in stream order and no sum is shared between threads, so the
 * result is the same on every run.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace
{

constexpr std::string_view FWD_GRAPH_SYMBOL = "hipkernel.conv_fft_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_SYMBOL = "hipkernel.conv_fft_bwd.graph_match";
constexpr std::string_view WRW_GRAPH_SYMBOL = "hipkernel.conv_fft_wrw.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_fft.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_fft.dispatch";

constexpr std::string_view X_TOKEN = "conv_fft.x.uid";
constexpr std::string_view W_TOKEN = "conv_fft.w.uid";
constexpr std::string_view Y_TOKEN = "conv_fft.y.uid";
constexpr std::string_view FLOPS_TOKEN = "conv_fft.flops";

std::optional<ConvDirection> directionOf(const std::string& name)
{
    if(name == "FWD")
    {
        return ConvDirection::FORWARD;
    }
    if(name == "BWD")
    {
        return ConvDirection::BACKWARD_DATA;
    }
    if(name == "WRW")
    {
        return ConvDirection::BACKWARD_WEIGHTS;
    }
    return std::nullopt;
}

std::string elementTypeFor(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return "float";
    case data_objects::DataType::HALF:
        return "_Float16";
    case data_objects::DataType::BFLOAT16:
        return "unsigned short";
    default:
        break;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "conv_fft: unsupported element type");
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = conv_fft::parseFftProblem(context, direction);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = problem->base.operands.x;
    bound[std::string(W_TOKEN)] = problem->base.operands.w;
    bound[std::string(Y_TOKEN)] = problem->base.operands.y;
    const auto flops = pack_features::checkedMultiply(2, problem->base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

/// The direction, problem and plan a kernel runs, or nullopt when it does not serve the graph.
struct Resolved
{
    ConvDirection direction;
    conv_fft::FftProblem problem;
    conv_fft::Plan plan;
};

std::optional<Resolved> resolve(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto direction = directionOf(kernel.getStringMetadata("direction"));
    if(!direction.has_value())
    {
        return std::nullopt;
    }
    auto problem = conv_fft::parseFftProblem(context, *direction);
    if(!problem.has_value()
       || kernel.getStringMetadata("dtype")
              != data_objects::EnumNameDataType(problem->base.dataType))
    {
        return std::nullopt;
    }
    const auto config = conv_fft::configOf(kernel);
    if(!config.has_value())
    {
        return std::nullopt;
    }
    auto plan = conv_fft::planFor(
        *problem, *direction, *config, device_units::dispatchUnits(context.deviceProperties));
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    return Resolved{*direction, std::move(*problem), std::move(*plan)};
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return resolve(context, kernel).has_value();
}

const std::vector<KernelArgument>& kernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvFftArgs)), 0, ""}};
    return s_signature;
}

/// The four launches of a plan: forward / backward data (filter, data, product, output), or
/// backward weights (x data, dy data, product, inverse).
class PreparedConvFft : public IngestorPreparedDispatch
{
public:
    PreparedConvFft(std::vector<IngestorKernelCode> kernels, Resolved resolved)
        : _kernels(std::move(kernels))
        , _resolved(std::move(resolved))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _kernels.front();
    }

    compilation::IRunnableKernel& kernel(size_t index, hipStream_t stream) const
    {
        return _kernels.at(index).kernelForStream(stream);
    }

    const Resolved& resolved() const
    {
        return _resolved;
    }

private:
    std::vector<IngestorKernelCode> _kernels;
    Resolved _resolved;
};

class ConvFftDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvFftDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                           const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto resolved = resolve(context, kernel);
        return resolved.has_value() ? resolved->plan.workspaceBytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        auto resolved = resolve(context, kernel);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!resolved.has_value() || !x.has_value() || !w.has_value() || !y.has_value()
           || *x != resolved->problem.base.operands.x || *w != resolved->problem.base.operands.w
           || *y != resolved->problem.base.operands.y)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_fft dispatch was given a graph or config its matchers did not admit");
        }
        const auto& plan = resolved->plan;
        const auto& config = plan.config;
        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_FFT_T", elementTypeFor(resolved->problem.base.dataType));
        options.add("CONV_FFT_TH", std::to_string(config.th));
        options.add("CONV_FFT_TW", std::to_string(config.tw));
        options.add("CONV_FFT_PB", std::to_string(config.pb));
        options.add("CONV_FFT_TB", std::to_string(config.tb));
        options.add("CONV_FFT_GN", std::to_string(config.gn));
        options.add("CONV_FFT_TM", std::to_string(config.tm));

        const bool wrw = resolved->direction == ConvDirection::BACKWARD_WEIGHTS;
        const auto& a = plan.args;
        struct Launch
        {
            const char* entry;
            long long grid;
            int threads;
        };
        std::vector<Launch> launches;
        if(plan.depthwise && wrw)
        {
            const long long foldGrid = (static_cast<long long>(a.groups) * plan.bins + 255) / 256;
            launches = {{"ConvFftDepthwiseWgrad", plan.productGridPerChunk, config.tb},
                        {"ConvFftDepthwiseFold", foldGrid, 256},
                        {"ConvFftWgradOut", plan.filterGrid, config.tb}};
        }
        else if(plan.depthwise)
        {
            launches = {{"ConvFftFilter", plan.filterGrid, config.tb},
                        {"ConvFftDepthwise", plan.productGridPerChunk, config.tb}};
        }
        else if(wrw)
        {
            launches = {{"ConvFftData", plan.dataGrid(plan.chunk, a.ca), config.tb},
                        {"ConvFftData", plan.dataGrid(plan.chunk, a.cb), config.tb},
                        {"ConvFftWgrad", plan.productGridPerChunk, 256},
                        {"ConvFftWgradOut", plan.filterGrid, config.tb}};
        }
        else
        {
            launches = {{"ConvFftFilter", plan.filterGrid, config.tb},
                        {"ConvFftData", plan.dataGrid(plan.chunk, a.ca), config.tb},
                        {"ConvFftProduct", plan.productGridPerChunk, 256},
                        {"ConvFftOutput", plan.dataGrid(plan.chunk, a.cb), config.tb}};
        }
        std::vector<IngestorKernelCode> kernels;
        for(const auto& launch : launches)
        {
            KernelDefinition entry = kernel;
            entry.source.entryPoint = launch.entry;
            auto code = buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, entry, options, kernelSignature());
            code.setBlockSize(static_cast<unsigned int>(launch.threads), 1, 1);
            code.setGridSize(static_cast<unsigned int>(launch.grid), 1, 1);
            kernels.push_back(std::move(code));
        }
        return std::make_unique<PreparedConvFft>(std::move(kernels), std::move(*resolved));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_fft needs its frequency-domain workspace");
        }
        const auto& dispatch = dynamic_cast<const PreparedConvFft&>(prepared);
        const auto& resolved = dispatch.resolved();
        const auto& problem = resolved.problem;
        const auto& plan = resolved.plan;
        const auto& uids = problem.base.operands;
        void* xPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.x, deviceBuffers, numDeviceBuffers).ptr;
        void* wPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.w, deviceBuffers, numDeviceBuffers).ptr;
        void* yPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.y, deviceBuffers, numDeviceBuffers).ptr;
        auto* base = static_cast<char*>(workspace);
        auto* fixed = reinterpret_cast<ConvFftComplex*>(base + plan.fixedOffset);
        auto* specA = reinterpret_cast<ConvFftComplex*>(base + plan.aOffset);
        auto* specB = reinterpret_cast<ConvFftComplex*>(base + plan.bOffset);
        const auto stream = handle.getStream();
        const auto copy4 = [](long long(&to)[4], const long long(&from)[4]) {
            for(int i = 0; i < 4; ++i)
            {
                to[i] = from[i];
            }
        };

        if(plan.depthwise && resolved.direction == ConvDirection::BACKWARD_WEIGHTS)
        {
            ConvFftArgs partial = plan.args;
            partial.src = xPtr; // x windows: th x tw, at the graph's padding
            copy4(partial.srcStr, problem.xStr);
            partial.srcH = problem.h;
            partial.srcW = problem.w;
            partial.padH = problem.padH;
            partial.padW = problem.padW;
            partial.validH = plan.config.th;
            partial.validW = plan.config.tw;
            partial.src2 = yPtr; // dy windows: the oh x ow tile, zero-padded
            copy4(partial.src2Str, problem.yStr);
            partial.src2H = problem.ho;
            partial.src2W = problem.wo;
            partial.valid2H = plan.args.oh;
            partial.valid2W = plan.args.ow;
            partial.wsOut = specA;
            dispatch.kernel(0, stream).launch(stream, partial);
            ConvFftArgs fold = plan.args;
            fold.wsA = specA;
            fold.wsY = fixed;
            dispatch.kernel(1, stream).launch(stream, fold);
            ConvFftArgs inverse = plan.args;
            inverse.wsY = fixed;
            inverse.dst = wPtr;
            copy4(inverse.dstStr, problem.wStr);
            dispatch.kernel(2, stream).launch(stream, inverse);
            return;
        }
        if(plan.depthwise)
        {
            const bool bwd = resolved.direction == ConvDirection::BACKWARD_DATA;
            ConvFftArgs filter = plan.args;
            filter.filt = wPtr;
            filter.wsOut = fixed;
            dispatch.kernel(0, stream).launch(stream, filter);
            ConvFftArgs fused = plan.args;
            fused.src = bwd ? yPtr : xPtr;
            copy4(fused.srcStr, bwd ? problem.yStr : problem.xStr);
            fused.srcH = bwd ? problem.ho : problem.h;
            fused.srcW = bwd ? problem.wo : problem.w;
            fused.padH = bwd ? problem.spanH() - 1 - problem.padH : problem.padH;
            fused.padW = bwd ? problem.spanW() - 1 - problem.padW : problem.padW;
            fused.validH = plan.config.th;
            fused.validW = plan.config.tw;
            fused.wsB = fixed;
            fused.dst = bwd ? xPtr : yPtr;
            copy4(fused.dstStr, bwd ? problem.xStr : problem.yStr);
            dispatch.kernel(1, stream).launch(stream, fused);
            return;
        }
        if(resolved.direction == ConvDirection::BACKWARD_WEIGHTS)
        {
            for(int chunk = 0; chunk < plan.chunks; ++chunk)
            {
                ConvFftArgs args = plan.args;
                args.p0 = chunk * plan.chunk;
                args.pc = std::min(plan.chunk, plan.tiles - args.p0);
                ConvFftArgs dx = args; // x windows: th x tw, at the graph's padding
                dx.src = xPtr;
                copy4(dx.srcStr, problem.xStr);
                dx.srcH = problem.h;
                dx.srcW = problem.w;
                dx.padH = problem.padH;
                dx.padW = problem.padW;
                dx.validH = plan.config.th;
                dx.validW = plan.config.tw;
                dx.chans = args.ca;
                dx.wsOut = specA;
                dispatch.kernel(0, stream).launch(stream, dx);
                ConvFftArgs dd = args; // dy windows: the oh x ow tile, zero-padded
                dd.src = yPtr;
                copy4(dd.srcStr, problem.yStr);
                dd.srcH = problem.ho;
                dd.srcW = problem.wo;
                dd.padH = 0;
                dd.padW = 0;
                dd.validH = args.oh;
                dd.validW = args.ow;
                dd.chans = args.cb;
                dd.wsOut = specB;
                dispatch.kernel(1, stream).launch(stream, dd);
                ConvFftArgs product = args;
                product.wsA = specA;
                product.wsB = specB;
                product.wsY = fixed;
                product.accumulate = chunk > 0 ? 1 : 0;
                dispatch.kernel(2, stream).launch(stream, product);
            }
            ConvFftArgs inverse = plan.args;
            inverse.wsY = fixed;
            inverse.dst = wPtr;
            copy4(inverse.dstStr, problem.wStr);
            dispatch.kernel(3, stream).launch(stream, inverse);
            return;
        }

        const bool bwd = resolved.direction == ConvDirection::BACKWARD_DATA;
        ConvFftArgs filter = plan.args;
        filter.filt = wPtr;
        filter.wsOut = fixed;
        dispatch.kernel(0, stream).launch(stream, filter);
        for(int chunk = 0; chunk < plan.chunks; ++chunk)
        {
            ConvFftArgs args = plan.args;
            args.p0 = chunk * plan.chunk;
            args.pc = std::min(plan.chunk, plan.tiles - args.p0);
            ConvFftArgs data = args;
            data.src = bwd ? yPtr : xPtr;
            copy4(data.srcStr, bwd ? problem.yStr : problem.xStr);
            data.srcH = bwd ? problem.ho : problem.h;
            data.srcW = bwd ? problem.wo : problem.w;
            // Backward data reads dy at (filter span - 1 - pre-padding) before each dx tile.
            data.padH = bwd ? problem.spanH() - 1 - problem.padH : problem.padH;
            data.padW = bwd ? problem.spanW() - 1 - problem.padW : problem.padW;
            data.validH = plan.config.th;
            data.validW = plan.config.tw;
            data.chans = args.ca;
            data.wsOut = specA;
            dispatch.kernel(1, stream).launch(stream, data);
            ConvFftArgs product = args;
            product.wsA = specA;
            product.wsB = fixed;
            product.wsY = specB;
            dispatch.kernel(2, stream).launch(stream, product);
            ConvFftArgs output = args;
            output.wsY = specB;
            output.dst = bwd ? xPtr : yPtr;
            copy4(output.dstStr, bwd ? problem.xStr : problem.yStr);
            dispatch.kernel(3, stream).launch(stream, output);
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convFftKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvFftDispatchHandler& convFftDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convFftKpackModuleCache());
    static const ConvFftDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

std::optional<BoundTokens> graphMatchesFwd(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::FORWARD);
}
std::optional<BoundTokens> graphMatchesBwd(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::BACKWARD_DATA);
}
std::optional<BoundTokens> graphMatchesWrw(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::BACKWARD_WEIGHTS);
}

} // namespace

void registerConvFftSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_SYMBOL), &graphMatchesFwd);
    scope.add(std::string(BWD_GRAPH_SYMBOL), &graphMatchesBwd);
    scope.add(std::string(WRW_GRAPH_SYMBOL), &graphMatchesWrw);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convFftDispatchHandler());
}

void resetConvFftModuleCache()
{
    convFftKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
