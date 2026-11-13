// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
#include "engines/kernel_ingestor_engine/packs/ConvSmallcProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvSmallcNative.cpp
 * @brief The small-input-channel forward convolution engine (CONV_SMALLC_FWD): stems and other
 *        layers with C <= 8, where implicit GEMM fills few k per MMA and gathers one scalar per
 *        reduction element. A block stages its input window and its weights in LDS once and runs
 *        the whole flattened (c, t, r, s) reduction as a handful of 16x16 MMA steps
 *        (descriptors/conv/smallc/kernels/ConvSmallcCore.h). Any layout, any compile-time filter,
 *        stride and dilation; every config is admitted on correctness alone and ranked by
 *        autotune. CONV_SMALLC_BWD runs stride-1 backward data with few output-gradient channels
 *        (network output layers: dx of many channels from dy of 1-4) on the same kernels, as the
 *        forward convolution over dy with the filter transposed and reflected.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace conv_smallc
{

namespace
{

constexpr long long INT_LIMIT = std::numeric_limits<int>::max();
constexpr long long MAX_GRID_Y = 65535;

/// The 16x16 atoms the kernel takes, with their reduction step (k per MMA).
struct AtomStep
{
    std::string_view name;
    int ka;
};
constexpr std::array<AtomStep, 9> ATOMS{{
    {"mfma16_f16", 16},
    {"mfma16_bf16", 16},
    {"mfma16_f32", 4},
    {"mfma16x32_f16", 32},
    {"mfma16x32_bf16", 32},
    {"wmma11_f16", 16},
    {"wmma11_bf16", 16},
    {"wmma12_f16", 16},
    {"wmma12_bf16", 16},
}};

constexpr int FMA_THREADS = 256;

int atomStep(std::string_view atom)
{
    for(const auto& entry : ATOMS)
    {
        if(entry.name == atom)
        {
            return entry.ka;
        }
    }
    return 0;
}

std::optional<int> intField(const KernelDefinition& kernel, const char* name)
{
    const auto value = kernel.tryGetMetadata(name);
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* number = std::get_if<int64_t>(&*value);
    if(number == nullptr || *number < std::numeric_limits<int>::min() || *number > INT_LIMIT)
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

bool isPow2(int v)
{
    return v > 0 && (v & (v - 1)) == 0;
}

} // namespace

int Config::sp() const
{
    int p = 1;
    while(p < s)
    {
        p *= 2;
    }
    return p;
}

int Config::kpl() const
{
    const long long taps = static_cast<long long>(cmax) * kd * r * sp();
    return static_cast<int>((taps + ka - 1) / ka * ka);
}

std::optional<SmallcProblem> parseSmallcProblem(const MatchContext& context,
                                                ConvDirection direction)
{
    if(direction != ConvDirection::FORWARD && direction != ConvDirection::BACKWARD_DATA)
    {
        return std::nullopt;
    }
    const bool backward = direction == ConvDirection::BACKWARD_DATA;
    auto base = conv_naive::parseConvNaiveProblem(context, direction);
    if(!base.has_value())
    {
        return std::nullopt;
    }
    const auto& a = base->args;
    int elementBytes = 0;
    switch(base->dataType)
    {
    case data_objects::DataType::FLOAT:
        elementBytes = 4;
        break;
    case data_objects::DataType::HALF:
    case data_objects::DataType::BFLOAT16:
        elementBytes = 2;
        break;
    default:
        return std::nullopt;
    }
    if(a.groups != 1 || base->spatialRank < 1 || base->spatialRank > 3)
    {
        return std::nullopt;
    }
    for(const long long value : {a.n, a.c, a.k})
    {
        if(value < 1 || value > INT_LIMIT)
        {
            return std::nullopt;
        }
    }
    for(int axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        for(const long long value : {a.inSpatial[axis], a.outSpatial[axis], a.prePadding[axis]})
        {
            if(value < 0 || value > INT_LIMIT)
            {
                return std::nullopt;
            }
        }
        if(a.inSpatial[axis] < 1 || a.outSpatial[axis] < 1)
        {
            return std::nullopt;
        }
        // Backward data runs as a forward convolution over dy only at unit stride.
        if(backward && a.stride[axis] != 1)
        {
            return std::nullopt;
        }
    }
    // The kernel's input and output: x and y forward; dy and dx backward.
    const long long inChannels = backward ? a.k : a.c;
    const long long outChannels = backward ? a.c : a.k;
    const long long* inExtents = backward ? a.outSpatial : a.inSpatial;
    const long long* outExtents = backward ? a.inSpatial : a.outSpatial;
    const long long* inStrides = backward ? a.yStrides : a.xStrides;
    const long long* outStrides = backward ? a.xStrides : a.yStrides;
    // One input image within the kernel's 32-bit buffer offsets (its window loads); the span
    // covers every element an image's strides reach.
    long long span = 1;
    const long long extents[5] = {a.n, inChannels, inExtents[0], inExtents[1], inExtents[2]};
    for(int i = 1; i < 5; ++i)
    {
        if(inStrides[i] < 0)
        {
            return std::nullopt;
        }
        span += (extents[i] - 1) * inStrides[i];
    }
    if(span * elementBytes >= INT_LIMIT)
    {
        return std::nullopt;
    }

    SmallcProblem problem;
    problem.backward = backward;
    problem.elementBytes = elementBytes;
    auto& args = problem.args;
    args.n = static_cast<int>(a.n);
    args.c = static_cast<int>(inChannels);
    args.k = static_cast<int>(outChannels);
    args.di = static_cast<int>(inExtents[0]);
    args.hi = static_cast<int>(inExtents[1]);
    args.wi = static_cast<int>(inExtents[2]);
    args.dout = static_cast<int>(outExtents[0]);
    args.ho = static_cast<int>(outExtents[1]);
    args.wo = static_cast<int>(outExtents[2]);
    int pads[3];
    for(int axis = 0; axis < 3; ++axis)
    {
        // Backward: dx row i reads dy rows i + pre - f dil, i.e. the forward window from
        // i - ((filter - 1) dil - pre) over the reflected filter.
        const long long pad = backward
                                  ? (a.filter[axis] - 1) * a.dilation[axis] - a.prePadding[axis]
                                  : a.prePadding[axis];
        if(pad < -INT_LIMIT || pad > INT_LIMIT)
        {
            return std::nullopt;
        }
        pads[axis] = static_cast<int>(pad);
    }
    args.pd = pads[0];
    args.ph = pads[1];
    args.pw = pads[2];
    for(int i = 0; i < 5; ++i)
    {
        args.xs[i] = inStrides[i];
        args.ys[i] = outStrides[i];
    }
    problem.base = std::move(*base);
    return problem;
}

std::optional<Config> configOf(const KernelDefinition& kernel)
{
    Config config;
    const auto atom = stringField(kernel, "atom");
    const auto archs = stringField(kernel, "archs");
    if(!atom.has_value() || !archs.has_value())
    {
        return std::nullopt;
    }
    config.atom = *atom;
    config.archs = *archs;
    // "fma": the FMA kernel (256 threads, step 1); otherwise a 16x16 MMA atom.
    config.fma = config.atom == "fma";
    if(config.fma)
    {
        config.ka = 1;
        config.wave = 0;
    }
    else
    {
        const auto info = conv_igemm::atomNamed(config.atom);
        config.ka = atomStep(config.atom);
        if(!info.has_value() || info->wave == 0 || config.ka == 0)
        {
            return std::nullopt;
        }
        config.kernelType = info->kernelType;
        config.wave = info->wave;
    }
    int atomId = 0;
    const std::array<std::pair<const char*, int*>, 19> ints{{{"atom_id", &atomId},
                                                             {"cmax", &config.cmax},
                                                             {"kd", &config.kd},
                                                             {"r", &config.r},
                                                             {"s", &config.s},
                                                             {"sd", &config.sd},
                                                             {"sh", &config.sh},
                                                             {"sw", &config.sw},
                                                             {"dd", &config.dd},
                                                             {"dh", &config.dh},
                                                             {"dw", &config.dw},
                                                             {"fm", &config.fm},
                                                             {"fn", &config.fn},
                                                             {"wm", &config.wm},
                                                             {"wq", &config.wq},
                                                             {"wn", &config.wn},
                                                             {"wave", &config.wave},
                                                             {"ka", &config.ka},
                                                             {"ep", &config.ep}}};
    const int atomWave = config.wave, atomKa = config.ka;
    for(const auto& [name, target] : ints)
    {
        const auto value = intField(kernel, name);
        if(!value.has_value())
        {
            return std::nullopt;
        }
        *target = *value;
    }
    // The declared wave and step restate the atom's.
    if(config.wave != atomWave || config.ka != atomKa)
    {
        return std::nullopt;
    }
    for(const int value : {config.cmax,
                           config.kd,
                           config.r,
                           config.s,
                           config.sd,
                           config.sh,
                           config.sw,
                           config.dd,
                           config.dh,
                           config.dw,
                           config.fm,
                           config.fn,
                           config.wm,
                           config.wq,
                           config.wn})
    {
        if(value < 1)
        {
            return std::nullopt;
        }
    }
    // Epilogue: 0 (LDS slots) or 1 (registers, MMA only).
    if(config.ep < 0 || config.ep > 1 || (config.fma && config.ep != 0))
    {
        return std::nullopt;
    }
    if(config.fma)
    {
        // 256 threads cover the tile exactly: NG = BN / FN channel groups x FM-column runs.
        const int bn = config.blockColumns(), tw = config.wq * 16;
        if(bn % config.fn != 0 || FMA_THREADS % (bn / config.fn) != 0 || tw % config.fm != 0
           || FMA_THREADS / (bn / config.fn) * config.fm != config.wm * tw)
        {
            return std::nullopt;
        }
        return config;
    }
    // A lane's k run and the filter rows align: one divides the other.
    const int kpl = config.wave == 64 ? config.ka / 4 : config.ka;
    const int lanes = config.atom.rfind("wmma12", 0) == 0 ? config.ka / 2 : kpl;
    const int sp = config.sp();
    if(!isPow2(lanes) || (lanes >= sp ? lanes % sp != 0 : sp % lanes != 0))
    {
        return std::nullopt;
    }
    return config;
}

std::optional<Plan> planFor(const SmallcProblem& problem, const Config& config)
{
    const auto& a = problem.base.args;
    // The config's channel bound (the kernel's input channels: dy's backward), filter, stride
    // and dilation are the graph's.
    if(problem.args.c > config.cmax || a.filter[0] != config.kd || a.filter[1] != config.r
       || a.filter[2] != config.s || a.stride[0] != config.sd || a.stride[1] != config.sh
       || a.stride[2] != config.sw || a.dilation[0] != config.dd || a.dilation[1] != config.dh
       || a.dilation[2] != config.dw)
    {
        return std::nullopt;
    }
    // The register epilogue stores a lane's adjacent channels: only for channels-fastest output
    // (other layouts keep the LDS-slot epilogue's runs along the width).
    if(config.ep == 1 && problem.args.ys[1] != 1)
    {
        return std::nullopt;
    }
    Plan plan;
    plan.args = problem.args;
    auto& args = plan.args;
    const long long columns = config.blockColumns();
    const long long columnBlocks = (args.k + columns - 1) / columns;
    // Tile rows: wm fragments of fm rows (MMA), wm rows (FMA).
    const long long th = config.fma ? config.wm : static_cast<long long>(config.wm) * config.fm;
    const long long tw = static_cast<long long>(config.wq) * 16;
    const long long tilesH = (args.ho + th - 1) / th;
    const long long tilesW = (args.wo + tw - 1) / tw;
    const long long gridX = tilesH * tilesW * args.dout * a.n;
    plan.weightElems = columnBlocks * columns * config.kpl();
    // One dispatch addresses at most 2^32 - 1 work items along x.
    if(columnBlocks > MAX_GRID_Y || gridX > INT_LIMIT || plan.weightElems > INT_LIMIT
       || gridX * config.blockThreads()
              > static_cast<long long>(std::numeric_limits<uint32_t>::max()))
    {
        return std::nullopt;
    }
    args.kpad = static_cast<int>(columnBlocks * columns);
    args.tilesH = static_cast<int>(tilesH);
    args.tilesW = static_cast<int>(tilesW);
    plan.grid[0] = static_cast<unsigned int>(gridX);
    plan.grid[1] = static_cast<unsigned int>(columnBlocks);
    plan.weightBytes
        = static_cast<size_t>(plan.weightElems) * static_cast<size_t>(problem.elementBytes);
    return plan;
}

} // namespace conv_smallc

namespace
{

using conv_smallc::Config;
using conv_smallc::Plan;
using conv_smallc::SmallcProblem;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.conv_smallc_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_smallc_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_smallc.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_smallc.dispatch";
constexpr std::string_view FORMAT_ENTRY_POINT = "ConvSmallcFormat";
constexpr unsigned int AUX_BLOCK = 256;
constexpr long long AUX_MAX_BLOCKS = 1LL << 16;

constexpr std::string_view X_TOKEN = "conv_smallc.x.uid";
constexpr std::string_view W_TOKEN = "conv_smallc.w.uid";
constexpr std::string_view Y_TOKEN = "conv_smallc.y.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_smallc.dtype";
constexpr std::string_view FLOPS_TOKEN = "conv_smallc.flops";

/// The graph's tokens in @p direction: x, w and y in forward roles (x is dx backward).
std::optional<BoundTokens> graphMatchesIn(const MatchContext& context, ConvDirection direction)
{
    const auto problem = conv_smallc::parseSmallcProblem(context, direction);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = problem->base.operands.x;
    bound[std::string(W_TOKEN)] = problem->base.operands.w;
    bound[std::string(Y_TOKEN)] = problem->base.operands.y;
    bound[std::string(DTYPE_TOKEN)]
        = std::string(data_objects::EnumNameDataType(problem->base.dataType));
    const auto flops = pack_features::checkedMultiply(2, problem->base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    return graphMatchesIn(context, ConvDirection::FORWARD);
}

std::optional<BoundTokens> graphMatchesBwd(const MatchContext& context)
{
    return graphMatchesIn(context, ConvDirection::BACKWARD_DATA);
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

struct Admitted
{
    SmallcProblem problem;
    Config config;
    Plan plan;
};

std::optional<Admitted> admitted(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto config = conv_smallc::configOf(kernel);
    const auto dtype = conv_smallc::stringField(kernel, "dtype");
    const auto direction = conv_smallc::stringField(kernel, "direction");
    if(!config.has_value() || !dtype.has_value() || (direction != "FWD" && direction != "BWD"))
    {
        return std::nullopt;
    }
    // The config's direction is the graph's: the parse refuses a graph of the other one.
    auto problem = conv_smallc::parseSmallcProblem(
        context, direction == "BWD" ? ConvDirection::BACKWARD_DATA : ConvDirection::FORWARD);
    if(!problem.has_value() || *dtype != data_objects::EnumNameDataType(problem->base.dataType)
       || !conv_igemm::atomRunsOn(config->atom, context.deviceProperties.gcnArchName)
       || !shipsTo(config->archs, context.deviceProperties.gcnArchName)
       || (!config->fma
           && conv_igemm::atomElement(config->atom)
                  != conv_igemm::dataElement(problem->base.dataType)))
    {
        return std::nullopt;
    }
    const auto plan = conv_smallc::planFor(*problem, *config);
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    return Admitted{std::move(*problem), *config, *plan};
}

/// Kernel-scoped applicability: the config's direction, dtype, channel bound, filter, stride and
/// dilation are the graph's; its epilogue suits the output layout; its atom runs on this device
/// and its ship list names it; the launch fits.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return admitted(context, kernel).has_value();
}

template <class Args>
const std::vector<KernelArgument>& signatureOf()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(Args)), 0, ""}};
    return s_signature;
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
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "conv_smallc: unsupported element type");
    }
}

class PreparedConvSmallc : public IngestorPreparedDispatch
{
public:
    PreparedConvSmallc(IngestorKernelCode conv, IngestorKernelCode format, Admitted admitted)
        : _conv(std::move(conv))
        , _format(std::move(format))
        , _admitted(std::move(admitted))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _conv;
    }
    compilation::IRunnableKernel& conv(hipStream_t stream) const
    {
        return _conv.kernelForStream(stream);
    }
    compilation::IRunnableKernel& format(hipStream_t stream) const
    {
        return _format.kernelForStream(stream);
    }
    const Admitted& admittedRun() const
    {
        return _admitted;
    }

private:
    IngestorKernelCode _conv;
    IngestorKernelCode _format;
    Admitted _admitted;
};

class ConvSmallcDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvSmallcDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                              const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The formatted weights.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto run = admitted(context, kernel);
        return run.has_value() ? run->plan.weightBytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        auto run = admitted(context, kernel);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!run.has_value() || x != run->problem.base.operands.x
           || w != run->problem.base.operands.w || y != run->problem.base.operands.y)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_smallc dispatch was given a graph or config its matchers did not admit");
        }
        const auto& config = run->config;
        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_SMALLC_T", elementTypeFor(run->problem.base.dataType));
        options.add("CONV_SMALLC_FMA", std::string(config.fma ? "1" : "0"));
        if(!config.fma)
        {
            options.add("CONV_SMALLC_ATOM", std::string(config.kernelType));
        }
        const std::array<std::pair<const char*, int>, 15> fields{{{"CMAX", config.cmax},
                                                                  {"KD", config.kd},
                                                                  {"R", config.r},
                                                                  {"S", config.s},
                                                                  {"SD", config.sd},
                                                                  {"SH", config.sh},
                                                                  {"SW", config.sw},
                                                                  {"DD", config.dd},
                                                                  {"DH", config.dh},
                                                                  {"DW", config.dw},
                                                                  {"FM", config.fm},
                                                                  {"FN", config.fn},
                                                                  {"WM", config.wm},
                                                                  {"WQ", config.wq},
                                                                  {"WN", config.wn}}};
        for(const auto& [name, value] : fields)
        {
            options.add(std::string("CONV_SMALLC_") + name, std::to_string(value));
        }
        // Set only when not the default, so ep 0 configs compile exactly as before.
        if(config.ep != 0)
        {
            options.add("CONV_SMALLC_EP", std::to_string(config.ep));
        }
        auto conv = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, signatureOf<SmallcArgs>());
        conv.setBlockSize(static_cast<unsigned int>(config.blockThreads()), 1, 1);
        conv.setGridSize(run->plan.grid[0], run->plan.grid[1], 1);

        KernelDefinition aux = kernel;
        aux.source.entryPoint = std::string(FORMAT_ENTRY_POINT);
        auto format = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, aux, options, signatureOf<SmallcFormatArgs>());
        format.setBlockSize(AUX_BLOCK, 1, 1);
        format.setGridSize(
            static_cast<unsigned int>(std::clamp<long long>(
                (run->plan.weightElems + AUX_BLOCK - 1) / AUX_BLOCK, 1, AUX_MAX_BLOCKS)),
            1,
            1);
        return std::make_unique<PreparedConvSmallc>(
            std::move(conv), std::move(format), std::move(*run));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvSmallc&>(prepared);
        const auto& run = dispatch.admittedRun();
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "conv_smallc needs its workspace");
        }
        const auto& ops = run.problem.base.operands;
        auto* x = hipdnn_plugin_sdk::findDeviceBuffer(ops.x, deviceBuffers, numDeviceBuffers).ptr;
        auto* w = hipdnn_plugin_sdk::findDeviceBuffer(ops.w, deviceBuffers, numDeviceBuffers).ptr;
        auto* y = hipdnn_plugin_sdk::findDeviceBuffer(ops.y, deviceBuffers, numDeviceBuffers).ptr;
        const auto stream = handle.getStream();
        const auto& a = run.problem.base.args;
        const auto& config = run.config;
        const bool backward = run.problem.backward;

        // Backward: the kernel's output channel n is the graph's c and its reduction channel
        // the graph's k (filter strides swapped), over the reflected filter.
        SmallcFormatArgs fmt{};
        fmt.w = w;
        fmt.wf = workspace;
        for(int i = 0; i < 5; ++i)
        {
            fmt.wStrides[i] = a.wStrides[i];
        }
        if(backward)
        {
            std::swap(fmt.wStrides[0], fmt.wStrides[1]);
        }
        fmt.k = run.problem.args.k;
        fmt.c = run.problem.args.c;
        fmt.kd = config.kd;
        fmt.r = config.r;
        fmt.s = config.s;
        fmt.sp = config.sp();
        fmt.kpl = config.kpl();
        fmt.kpad = run.plan.args.kpad;
        fmt.flip = (a.flipFilter != 0) != backward ? 1 : 0;
        fmt.bn = config.fma ? config.blockColumns() : 0;
        dispatch.format(stream).launch(stream, fmt);

        SmallcArgs args = run.plan.args;
        args.x = backward ? y : x;
        args.w = workspace;
        args.y = backward ? x : y;
        dispatch.conv(stream).launch(stream, args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convSmallcKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvSmallcDispatchHandler& convSmallcDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convSmallcKpackModuleCache());
    static const ConvSmallcDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvSmallcSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &graphMatchesBwd);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convSmallcDispatchHandler());
}

void resetConvSmallcModuleCache()
{
    convSmallcKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
