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
 * @file ConvNarrowWrwNative.cpp
 * @brief The native half of CONV_NARROW_WRW: graph matching (the problem in the kernel's roles),
 *        kernel matching, workspace sizing and dispatch of the main and reduce kernels
 *        (ConvNarrowWrw.cpp). The workspace holds the fp32 partial sums; both launches are in
 *        stream order, the reduce adds the splits in a fixed order: deterministic.
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

std::optional<NarrowWrwProblem> parseNarrowWrwProblem(const MatchContext& context)
{
    auto base = conv_naive::parseConvNaiveProblem(context, ConvDirection::BACKWARD_WEIGHTS);
    if(!base.has_value() || base->spatialRank < 1 || base->spatialRank > 3)
    {
        return std::nullopt;
    }
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& a = base->args;
    if(a.groups != 1 || !fitsInt(a.n) || !fitsInt(a.c) || !fitsInt(a.k))
    {
        return std::nullopt;
    }
    // The narrow side: K (an output layer) or C (a stem), at most MAX_KO and narrower than the
    // other.
    const bool wideIsC = a.k <= MAX_KO && a.k < a.c;
    const bool wideIsK = a.c <= MAX_KO && a.c < a.k;
    if(!wideIsC && !wideIsK)
    {
        return std::nullopt;
    }
    WrwShape shape;
    shape.wideIsC = wideIsC;
    shape.n = static_cast<int>(a.n);
    shape.ci = static_cast<int>(wideIsC ? a.c : a.k);
    shape.ko = static_cast<int>(wideIsC ? a.k : a.c);
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
        const int x = static_cast<int>(a.inSpatial[axis]);
        const int y = static_cast<int>(a.outSpatial[axis]);
        shape.in.at(i) = wideIsC ? x : y;
        shape.nar.at(i) = wideIsC ? y : x;
        shape.filter.at(i) = static_cast<int>(a.filter[axis]);
        shape.pad.at(i) = static_cast<int>(a.prePadding[axis]);
        shape.stride.at(i) = static_cast<int>(a.stride[axis]);
        shape.dilation.at(i) = static_cast<int>(a.dilation[axis]);
        if(x < 1 || y < 1 || shape.filter.at(i) < 1 || shape.stride.at(i) < 1
           || shape.dilation.at(i) < 1)
        {
            return std::nullopt;
        }
    }
    shape.flip = a.flipFilter != 0 ? 1 : 0;
    shape.elementBytes = base->dataType == data_objects::DataType::FLOAT ? 4 : 2;
    for(size_t i = 0; i < 5; ++i)
    {
        shape.is.at(i) = wideIsC ? a.xStrides[i] : a.yStrides[i];
        shape.ns.at(i) = wideIsC ? a.yStrides[i] : a.xStrides[i];
        shape.ws.at(i) = a.wStrides[i];
    }
    shape.inAlign = pointerAlignment(context, wideIsC ? base->operands.x : base->operands.y);
    const int narAlign = pointerAlignment(context, wideIsC ? base->operands.y : base->operands.x);
    const int dwAlign = pointerAlignment(context, base->operands.w);
    if(shape.inAlign < shape.elementBytes || narAlign < shape.elementBytes
       || dwAlign < shape.elementBytes)
    {
        return std::nullopt;
    }
    return NarrowWrwProblem{*base, shape};
}

std::optional<WrwConfig> wrwConfigOf(const KernelDefinition& kernel)
{
    WrwConfig config;
    const auto side = stringField(kernel, "side");
    const auto layout = stringField(kernel, "layout");
    const auto atom = stringField(kernel, "atom");
    if(!side || !layout || !atom)
    {
        return std::nullopt;
    }
    config.side = *side;
    config.layout = *layout;
    config.atom = *atom;
    const std::array<std::pair<const char*, int*>, 11> ints{{{"kmin", &config.kmin},
                                                             {"ko", &config.ko},
                                                             {"fd", &config.fd},
                                                             {"fr", &config.fr},
                                                             {"fs", &config.fs},
                                                             {"wr", &config.wr},
                                                             {"rb", &config.rb},
                                                             {"nw", &config.nw},
                                                             {"mpw", &config.mpw},
                                                             {"atom_id", nullptr},
                                                             {"side_id", nullptr}}};
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
    return checkWrwConfig(config);
}

std::optional<WrwAdmitted> admittedWrw(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto config = wrwConfigOf(kernel);
    const auto dtype = stringField(kernel, "dtype");
    const auto archs = stringField(kernel, "archs");
    if(!config.has_value() || !dtype.has_value() || !archs.has_value())
    {
        return std::nullopt;
    }
    auto problem = parseNarrowWrwProblem(context);
    const auto& arch = context.deviceProperties.gcnArchName;
    if(!problem.has_value() || *dtype != data_objects::EnumNameDataType(problem->base.dataType)
       || !shipsTo(*archs, arch))
    {
        return std::nullopt;
    }
    if(config->atom != "valu"
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
    auto plan = planWrw(problem->shape,
                        *config,
                        wave,
                        device_units::dispatchUnits(context.deviceProperties),
                        ldsLimitOf(base));
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    return WrwAdmitted{std::move(*problem), std::move(*plan)};
}

} // namespace conv_narrow

namespace
{

constexpr std::string_view GRAPH_SYMBOL = "hipkernel.conv_narrow_wrw.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_narrow_wrw.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_narrow_wrw.dispatch";

constexpr std::string_view X_TOKEN = "conv_narrow_wrw.x.uid";
constexpr std::string_view W_TOKEN = "conv_narrow_wrw.w.uid";
constexpr std::string_view Y_TOKEN = "conv_narrow_wrw.y.uid";
constexpr std::string_view FLOPS_TOKEN = "conv_narrow_wrw.flops";

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
                                                   "conv_narrow_wrw: unsupported element type");
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto problem = cn::parseNarrowWrwProblem(context);
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

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return cn::admittedWrw(context, kernel).has_value();
}

const std::vector<KernelArgument>& kernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvNarrowWrwArgs)), 0, ""}};
    return s_signature;
}

class PreparedConvNarrowWrw : public IngestorPreparedDispatch
{
public:
    PreparedConvNarrowWrw(std::vector<IngestorKernelCode> kernels, cn::WrwAdmitted admitted)
        : _kernels(std::move(kernels))
        , _admitted(std::move(admitted))
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

    const cn::WrwAdmitted& admitted() const
    {
        return _admitted;
    }

private:
    std::vector<IngestorKernelCode> _kernels;
    cn::WrwAdmitted _admitted;
};

class ConvNarrowWrwDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvNarrowWrwDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                 const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto admitted = cn::admittedWrw(context, kernel);
        return admitted.has_value() ? admitted->plan.workspaceBytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        auto admitted = cn::admittedWrw(context, kernel);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!admitted.has_value() || !x.has_value() || !w.has_value() || !y.has_value()
           || *x != admitted->problem.base.operands.x || *w != admitted->problem.base.operands.w
           || *y != admitted->problem.base.operands.y)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_narrow_wrw dispatch was given a graph or config its matchers did not admit");
        }
        const auto& plan = admitted->plan;
        std::string kernelType;
        if(plan.config.atom != "valu")
        {
            const auto info = conv_igemm::atomNamed(plan.config.atom);
            if(!info.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                               "conv_narrow_wrw: unknown atom");
            }
            kernelType = std::string(info->kernelType);
        }
        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        for(const auto& [name, value] :
            cn::wrwDefinesFor(plan, elementTypeOf(admitted->problem.base.dataType), kernelType))
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
            {{"ConvNarrowWrw",
              static_cast<unsigned>(plan.args.splits) * static_cast<unsigned>(plan.args.cblocks),
              static_cast<unsigned>(plan.threads)},
             {"ConvNarrowWrwReduce", static_cast<unsigned>(plan.reduceGrid), 256}}};
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
        return std::make_unique<PreparedConvNarrowWrw>(std::move(kernels), std::move(*admitted));
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
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_narrow_wrw needs its partial-sum workspace");
        }
        const auto& dispatch = dynamic_cast<const PreparedConvNarrowWrw&>(prepared);
        const auto& admitted = dispatch.admitted();
        const auto& uids = admitted.problem.base.operands;
        const bool wideIsC = admitted.problem.shape.wideIsC;
        void* xPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.x, deviceBuffers, numDeviceBuffers).ptr;
        void* wPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.w, deviceBuffers, numDeviceBuffers).ptr;
        void* yPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.y, deviceBuffers, numDeviceBuffers).ptr;
        ConvNarrowWrwArgs args = admitted.plan.args;
        args.in = wideIsC ? xPtr : yPtr;
        args.nar = wideIsC ? yPtr : xPtr;
        args.dw = wPtr;
        args.part = static_cast<float*>(workspace);
        const auto stream = handle.getStream();
        dispatch.kernel(0, stream).launch(stream, args);
        dispatch.kernel(1, stream).launch(stream, args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convNarrowWrwKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvNarrowWrwDispatchHandler& convNarrowWrwDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convNarrowWrwKpackModuleCache());
    static const ConvNarrowWrwDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvNarrowWrwSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convNarrowWrwDispatchHandler());
}

void resetConvNarrowWrwModuleCache()
{
    convNarrowWrwKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
