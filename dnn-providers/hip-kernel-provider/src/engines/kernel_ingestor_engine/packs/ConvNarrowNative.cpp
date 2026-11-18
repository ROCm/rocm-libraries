// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
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
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNarrowNative.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvNarrowNative.cpp
 * @brief The native half of CONV_NARROW_FWD / CONV_NARROW_BWD: graph matching (the problem in
 *        the kernels' roles), kernel matching, workspace sizing and dispatch of the prepare and
 *        main kernels (ConvNarrow.cpp). The workspace holds the prepared filter; both launches
 *        are in stream order and every output element is written by one thread: deterministic.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace cn = conv_narrow;
using conv_naive::ConvDirection;

namespace conv_narrow
{

namespace
{

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();

bool fitsInt(int64_t value)
{
    return value >= 0 && value <= INT_LIMIT;
}

/// The byte alignment the graph declares for tensor @p uid's pointer, at most 16; 0 if missing.
int pointerAlignment(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto found = tensors.find(uid);
    if(found == tensors.end() || found->second == nullptr)
    {
        return 0;
    }
    const int64_t declared = found->second->alignment();
    int align = 16;
    while(align > 1 && declared % align != 0)
    {
        align /= 2;
    }
    return align;
}

std::optional<int> intField(const KernelDefinition& kernel, const char* name)
{
    const auto value = kernel.tryGetMetadata(name);
    const auto* number = value.has_value() ? std::get_if<int64_t>(&*value) : nullptr;
    if(number == nullptr || !fitsInt(*number))
    {
        return std::nullopt;
    }
    return static_cast<int>(*number);
}

std::optional<std::string> stringField(const KernelDefinition& kernel, const char* name)
{
    const auto value = kernel.tryGetMetadata(name);
    const auto* text = value.has_value() ? std::get_if<std::string>(&*value) : nullptr;
    if(text == nullptr)
    {
        return std::nullopt;
    }
    return *text;
}

} // namespace

std::optional<NarrowProblem> parseNarrowProblem(const MatchContext& context,
                                                ConvDirection direction)
{
    if(direction != ConvDirection::FORWARD && direction != ConvDirection::BACKWARD_DATA)
    {
        return std::nullopt;
    }
    auto base = conv_naive::parseConvNaiveProblem(context, direction);
    if(!base.has_value() || base->spatialRank < 1 || base->spatialRank > 3)
    {
        return std::nullopt;
    }
    // fp32 sums: a compute type of fp32 (or none) only.
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& a = base->args;
    if(a.groups != 1)
    {
        return std::nullopt;
    }
    const bool bwd = direction == ConvDirection::BACKWARD_DATA;
    // The narrow side is the output of the kernels: K forward, C backward data, at most MAX_KO,
    // and narrower than the side it is reduced from.
    const int64_t narrow = bwd ? a.c : a.k;
    const int64_t wide = bwd ? a.k : a.c;
    if(narrow < 1 || narrow > MAX_KO || wide <= narrow || !fitsInt(a.n) || !fitsInt(wide))
    {
        return std::nullopt;
    }
    NarrowShape shape;
    shape.backward = bwd;
    shape.n = static_cast<int>(a.n);
    shape.ci = static_cast<int>(wide);
    shape.ko = static_cast<int>(narrow);
    for(int axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        for(const long long value : {a.inSpatial[axis],
                                     a.outSpatial[axis],
                                     a.filter[axis],
                                     a.prePadding[axis],
                                     a.stride[axis],
                                     a.dilation[axis]})
        {
            if(!fitsInt(value))
            {
                return std::nullopt;
            }
        }
        const auto i = static_cast<size_t>(axis);
        const int in = static_cast<int>(a.inSpatial[axis]);
        const int out = static_cast<int>(a.outSpatial[axis]);
        shape.in.at(i) = bwd ? out : in;
        shape.out.at(i) = bwd ? in : out;
        shape.filter.at(i) = static_cast<int>(a.filter[axis]);
        shape.pad.at(i) = static_cast<int>(a.prePadding[axis]);
        shape.stride.at(i) = static_cast<int>(a.stride[axis]);
        shape.dilation.at(i) = static_cast<int>(a.dilation[axis]);
        if(shape.in.at(i) < 1 || shape.out.at(i) < 1 || shape.filter.at(i) < 1
           || shape.stride.at(i) < 1 || shape.dilation.at(i) < 1)
        {
            return std::nullopt;
        }
    }
    shape.flip = a.flipFilter != 0 ? 1 : 0;
    shape.elementBytes = base->dataType == data_objects::DataType::FLOAT ? 4 : 2;
    for(size_t i = 0; i < 5; ++i)
    {
        shape.is.at(i) = bwd ? a.yStrides[i] : a.xStrides[i];
        shape.os.at(i) = bwd ? a.xStrides[i] : a.yStrides[i];
        shape.ws.at(i) = a.wStrides[i];
    }
    shape.wGraphK = static_cast<int>(a.k);
    shape.wGraphC = static_cast<int>(a.c);
    shape.inAlign = pointerAlignment(context, bwd ? base->operands.y : base->operands.x);
    const int outAlign = pointerAlignment(context, bwd ? base->operands.x : base->operands.y);
    if(shape.inAlign < shape.elementBytes || outAlign < shape.elementBytes)
    {
        return std::nullopt;
    }
    return NarrowProblem{*base, shape};
}

std::optional<Config> configOf(const KernelDefinition& kernel)
{
    Config config;
    const auto direction = stringField(kernel, "direction");
    const auto layout = stringField(kernel, "layout");
    const auto atom = stringField(kernel, "atom");
    if(!direction || !layout || !atom)
    {
        return std::nullopt;
    }
    config.direction = *direction;
    config.layout = *layout;
    config.atom = *atom;
    const std::array<std::pair<const char*, int*>, 16> ints{{{"kmin", &config.kmin},
                                                             {"pd", &config.pd},
                                                             {"cmin", &config.cmin},
                                                             {"sf", &config.sf},
                                                             {"ko", &config.ko},
                                                             {"fd", &config.fd},
                                                             {"fr", &config.fr},
                                                             {"fs", &config.fs},
                                                             {"sw", &config.sw},
                                                             {"ds", &config.ds},
                                                             {"cm", &config.cm},
                                                             {"wr", &config.wr},
                                                             {"rb", &config.rb},
                                                             {"nw", &config.nw},
                                                             {"op", &config.op},
                                                             {"atom_id", nullptr}}};
    for(const auto& [name, target] : ints)
    {
        const auto value = intField(kernel, name);
        if(!value.has_value())
        {
            return std::nullopt;
        }
        if(target != nullptr)
        {
            *target = *value;
        }
    }
    return checkConfig(config);
}

bool shipsTo(std::string_view archs, std::string_view gcnArchName)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    if(arch.empty())
    {
        return false;
    }
    size_t start = 0;
    while(start <= archs.size())
    {
        const size_t end = std::min(archs.find(',', start), archs.size());
        const auto group = archs.substr(start, end - start);
        if(group == "any" || group == arch
           || ((group == "gfx11" || group == "gfx12") && arch.rfind(group, 0) == 0))
        {
            return true;
        }
        start = end + 1;
    }
    return false;
}

std::optional<Admitted> admitted(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto config = configOf(kernel);
    const auto dtype = stringField(kernel, "dtype");
    const auto archs = stringField(kernel, "archs");
    if(!config.has_value() || !dtype.has_value() || !archs.has_value())
    {
        return std::nullopt;
    }
    auto problem = parseNarrowProblem(context,
                                      config->direction == "BWD" ? ConvDirection::BACKWARD_DATA
                                                                 : ConvDirection::FORWARD);
    const auto& arch = context.deviceProperties.gcnArchName;
    if(!problem.has_value() || *dtype != data_objects::EnumNameDataType(problem->base.dataType)
       || !shipsTo(*archs, arch))
    {
        return std::nullopt;
    }
    const bool valu = config->atom == "valu";
    if(!valu
       && (!conv_igemm::atomRunsOn(config->atom, arch)
           || conv_igemm::atomElement(config->atom)
                  != conv_igemm::dataElement(problem->base.dataType)))
    {
        return std::nullopt;
    }
    const int reported = context.deviceProperties.warpSize;
    const auto base = hipdnn_plugin_sdk::stripArchFeatures(arch);
    // The reported wave size, else the architecture's (RDNA 32, CDNA 64).
    int wave = base.size() >= 5 && base.substr(0, 4) == "gfx1" ? 32 : 64;
    if(reported == 32 || reported == 64)
    {
        wave = reported;
    }
    auto plan = planFor(problem->shape,
                        *config,
                        wave,
                        device_units::dispatchUnits(context.deviceProperties),
                        ldsLimitOf(base));
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    return Admitted{std::move(*problem), std::move(*plan)};
}

} // namespace conv_narrow

namespace
{

constexpr std::string_view FWD_GRAPH_SYMBOL = "hipkernel.conv_narrow_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_SYMBOL = "hipkernel.conv_narrow_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_narrow.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_narrow.dispatch";

constexpr std::string_view X_TOKEN = "conv_narrow.x.uid";
constexpr std::string_view W_TOKEN = "conv_narrow.w.uid";
constexpr std::string_view Y_TOKEN = "conv_narrow.y.uid";
constexpr std::string_view FLOPS_TOKEN = "conv_narrow.flops";

std::string elementTypeOf(data_objects::DataType dataType)
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
                                                   "conv_narrow: unsupported element type");
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = cn::parseNarrowProblem(context, direction);
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

std::optional<BoundTokens> graphMatchesFwd(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::FORWARD);
}

std::optional<BoundTokens> graphMatchesBwd(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::BACKWARD_DATA);
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return cn::admitted(context, kernel).has_value();
}

const std::vector<KernelArgument>& kernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvNarrowArgs)), 0, ""}};
    return s_signature;
}

class PreparedConvNarrow : public IngestorPreparedDispatch
{
public:
    PreparedConvNarrow(std::vector<IngestorKernelCode> kernels, cn::Admitted admitted)
        : _kernels(std::move(kernels))
        , _admitted(std::move(admitted))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _kernels.back();
    }

    compilation::IRunnableKernel& kernel(size_t index, hipStream_t stream) const
    {
        return _kernels.at(index).kernelForStream(stream);
    }

    const cn::Admitted& admitted() const
    {
        return _admitted;
    }

private:
    std::vector<IngestorKernelCode> _kernels;
    cn::Admitted _admitted;
};

class ConvNarrowDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvNarrowDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                              const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto admitted = cn::admitted(context, kernel);
        return admitted.has_value() ? admitted->plan.workspaceBytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        auto admitted = cn::admitted(context, kernel);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!admitted.has_value() || !x.has_value() || !w.has_value() || !y.has_value()
           || *x != admitted->problem.base.operands.x || *w != admitted->problem.base.operands.w
           || *y != admitted->problem.base.operands.y)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_narrow dispatch was given a graph or config its matchers did not admit");
        }
        const auto& plan = admitted->plan;
        std::string kernelType;
        if(plan.config.atom != "valu")
        {
            const auto info = conv_igemm::atomNamed(plan.config.atom);
            if(!info.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                               "conv_narrow: unknown atom");
            }
            kernelType = std::string(info->kernelType);
        }
        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        for(const auto& [name, value] :
            cn::compileDefinesFor(plan, elementTypeOf(admitted->problem.base.dataType), kernelType))
        {
            options.add(name, value);
        }
        struct Launch
        {
            const char* entry;
            unsigned grid;
            unsigned threads;
        };
        const std::array<Launch, 2> launches{
            {{"ConvNarrowPrepare", static_cast<unsigned>(plan.prepareGrid), 256},
             {"ConvNarrow",
              static_cast<unsigned>(plan.args.blocks),
              static_cast<unsigned>(plan.threads)}}};
        std::vector<IngestorKernelCode> kernels;
        for(const auto& launch : launches)
        {
            KernelDefinition entry = kernel;
            entry.source.entryPoint = launch.entry;
            auto code = buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, entry, options, kernelSignature());
            code.setBlockSize(launch.threads, 1, 1);
            code.setGridSize(launch.grid, 1, 1);
            kernels.push_back(std::move(code));
        }
        return std::make_unique<PreparedConvNarrow>(std::move(kernels), std::move(*admitted));
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
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_narrow needs its prepared-filter workspace");
        }
        const auto& dispatch = dynamic_cast<const PreparedConvNarrow&>(prepared);
        const auto& admitted = dispatch.admitted();
        const auto& uids = admitted.problem.base.operands;
        const bool bwd = admitted.problem.shape.backward;
        void* xPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.x, deviceBuffers, numDeviceBuffers).ptr;
        void* wPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.w, deviceBuffers, numDeviceBuffers).ptr;
        void* yPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.y, deviceBuffers, numDeviceBuffers).ptr;
        ConvNarrowArgs args = admitted.plan.args;
        args.in = bwd ? yPtr : xPtr;
        args.out = bwd ? xPtr : yPtr;
        args.w = wPtr;
        args.wp = workspace;
        const auto stream = handle.getStream();
        dispatch.kernel(0, stream).launch(stream, args);
        dispatch.kernel(1, stream).launch(stream, args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convNarrowKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvNarrowDispatchHandler& convNarrowDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convNarrowKpackModuleCache());
    static const ConvNarrowDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvNarrowSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_SYMBOL), &graphMatchesFwd);
    scope.add(std::string(BWD_GRAPH_SYMBOL), &graphMatchesBwd);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convNarrowDispatchHandler());
}

void resetConvNarrowModuleCache()
{
    convNarrowKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
