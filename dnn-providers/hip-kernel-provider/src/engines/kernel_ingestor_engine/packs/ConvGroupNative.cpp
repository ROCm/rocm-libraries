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
#include "engines/kernel_ingestor_engine/packs/ConvGroupProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvGroupNative.cpp
 * @brief The small-group convolution engines (CONV_GROUP_FWD, CONV_GROUP_BWD): graph matching,
 *        kernel matching, workspace sizing and dispatch.
 *
 * Grouped convolutions of 2 to 16 channels a group (ResNeXt, RegNet), where an implicit GEMM
 * per group leaves its tiles mostly empty and reads 4- to 32-byte channel slices. The kernels
 * (descriptors/conv/group/kernels) run each 16-channel block on a 16x16 matrix-core atom with
 * the block-diagonal expansion of its groups' filters, from an input halo staged in LDS: 16 /
 * C/g groups per multiply, every input read once a workgroup. Forward gathers x into y; backward
 * data (unit stride) gathers dy into dx with the filter reflected and its channel roles swapped,
 * which the expansion does. The activations are staged channels-last by the implicit-GEMM
 * engine's own parse and staging kernels when the graph's are not. Every config is admitted on
 * correctness alone; autotune ranks them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace conv_group
{

namespace
{

constexpr long long MAX_GRID = std::numeric_limits<int>::max();
constexpr size_t WORKSPACE_ALIGNMENT = 256;

size_t alignUp(size_t bytes)
{
    return (bytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

long long divUp(long long a, long long b)
{
    return (a + b - 1) / b;
}

} // namespace

long long groupLdsBytes(int elementBytes, int r, int s, int stride, int th, int wp, int cb)
{
    const long long vec = 16 / elementBytes;
    const long long rows = static_cast<long long>(th - 1) * stride + r;
    const long long cols = static_cast<long long>(16 * wp - 1) * stride + s;
    return rows * cols * (16LL * cb + vec) * elementBytes;
}

std::optional<unsigned int> phaseBlocks(const K4Args& a, int th, int wp, int cb)
{
    if(th < 1 || wp < 1 || cb < 1 || a.c % (16 * cb) != 0)
    {
        return std::nullopt;
    }
    const long long blocks = static_cast<long long>(a.n) * divUp(a.ho, th) * divUp(a.wo, 16LL * wp)
                             * (a.c / (16 * cb));
    if(blocks < 1 || blocks > MAX_GRID)
    {
        return std::nullopt;
    }
    return static_cast<unsigned int>(blocks);
}

std::optional<GroupProblem> parseGroupProblem(const MatchContext& context, ConvDirection direction)
{
    auto igemm = conv_igemm::parseIgemmProblem(context, direction);
    if(!igemm.has_value())
    {
        return std::nullopt;
    }
    const auto& k = igemm->args;
    const auto& base = igemm->base.args;
    const bool forward = direction == ConvDirection::FORWARD;
    // Small groups, the same each way, whole 16-channel blocks.
    if(k.groups < 2 || k.c != k.k || k.c % 16 != 0)
    {
        return std::nullopt;
    }
    const int cg = k.c / k.groups;
    if(cg < 2 || cg > 16 || 16 % cg != 0)
    {
        return std::nullopt;
    }
    // 2-D (a unit depth, its filter one tap deep), unit dilation, one stride both ways
    // (backward data: 1 or 2).
    if(k.di != 1 || k.dout != 1 || k.ft != 1 || k.pd != 0 || k.sd != 1 || k.dd != 1 || k.dh != 1
       || k.dw != 1 || k.sh != k.sw || (!forward && k.sh > 2))
    {
        return std::nullopt;
    }
    GroupProblem problem;
    problem.filterR = k.fr;
    problem.filterS = k.fs;
    problem.stride = k.sh;
    const int st = k.sh;
    // The filter, any strides; true convolution reflected by the expansion.
    K4ExpandArgs expand{};
    expand.sK = base.wStrides[0];
    expand.sC = base.wStrides[1];
    expand.sR = base.wStrides[3];
    expand.sS = base.wStrides[4];
    expand.c = k.c;
    expand.cg = cg;
    expand.r = k.fr;
    expand.s = k.fs;
    expand.reflect = base.flipFilter != 0 ? 1 : 0;
    expand.elementBytes = igemm->elementBytes;
    K4Args args{};
    args.n = k.n;
    args.c = k.c;
    args.cg = cg;
    if(forward)
    {
        problem.kernelR = k.fr;
        problem.kernelS = k.fs;
        problem.kernelStride = st;
        GroupPhase phase;
        args.hi = k.hi;
        args.wi = k.wi;
        args.ho = k.ho;
        args.wo = k.wo;
        args.ph = k.ph;
        args.pw = k.pw;
        args.outH = k.ho;
        args.outW = k.wo;
        args.osh = args.osw = 1;
        phase.args = args;
        expand.rc = k.fr;
        expand.sc = k.fs;
        expand.rStep = expand.sStep = 1;
        phase.expand = expand;
        problem.phases.push_back(phase);
    }
    else
    {
        // dx[i] gathers dy[(i + pad - r) / st] over the taps r that divide: by phases p = i mod
        // st, dx[st j + p] = sum over q < ceil(R / st) of dy[j + dmin + q] w[p + pad - st (dmin
        // + q)], dmin = ceil((p + pad - R + 1) / st), taps past the filter zero.
        problem.kernelR = static_cast<int>(divUp(k.fr, st));
        problem.kernelS = static_cast<int>(divUp(k.fs, st));
        problem.kernelStride = 1;
        const auto ceilDiv
            = [](long long a, long long b) { return a >= 0 ? (a + b - 1) / b : -((-a) / b); };
        for(int pr = 0; pr < st; ++pr)
        {
            for(int pc = 0; pc < st; ++pc)
            {
                GroupPhase phase;
                phase.args = args;
                auto& p = phase.args;
                p.hi = k.ho;
                p.wi = k.wo;
                p.ho = static_cast<int>(divUp(k.hi - pr, st));
                p.wo = static_cast<int>(divUp(k.wi - pc, st));
                if(p.ho < 1 || p.wo < 1)
                {
                    continue;
                }
                const long long dminH = ceilDiv(pr + k.ph - k.fr + 1, st);
                const long long dminW = ceilDiv(pc + k.pw - k.fs + 1, st);
                p.ph = static_cast<int>(-dminH);
                p.pw = static_cast<int>(-dminW);
                p.outH = k.hi;
                p.outW = k.wi;
                p.osh = p.osw = st;
                p.ooh = pr;
                p.oow = pc;
                phase.expand = expand;
                auto& e = phase.expand;
                e.rc = problem.kernelR;
                e.sc = problem.kernelS;
                e.rBase = static_cast<int>(pr + k.ph - st * dminH);
                e.rStep = -st;
                e.sBase = static_cast<int>(pc + k.pw - st * dminW);
                e.sStep = -st;
                e.swap = 1;
                problem.phases.push_back(phase);
            }
        }
    }
    if(problem.phases.empty())
    {
        return std::nullopt;
    }
    // Workspace: the staged activations (the implicit-GEMM layout of them), then each phase's
    // expanded filter; the implicit-GEMM filter staging is not used.
    igemm->w = conv_igemm::Staging{};
    size_t offset = 0;
    for(auto* staging : {&igemm->x, &igemm->y})
    {
        if(staging->needed)
        {
            staging->workspaceOffset = offset;
            offset = alignUp(offset + staging->bytes);
        }
    }
    const size_t expandBytes = static_cast<size_t>(k.c / 16)
                               * static_cast<size_t>(problem.kernelR * problem.kernelS) * 256
                               * static_cast<size_t>(igemm->elementBytes);
    for(auto& phase : problem.phases)
    {
        phase.expandOffset = offset;
        offset = alignUp(offset + expandBytes);
        // The smallest tile's grid within the launch limits.
        if(!phaseBlocks(phase.args, 1, 1, 1).has_value())
        {
            return std::nullopt;
        }
    }
    problem.workspaceBytes = offset;
    problem.igemm = std::move(*igemm);
    return problem;
}

} // namespace conv_group

namespace
{

using conv_group::GroupProblem;

constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_group_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_group_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_group.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_group.dispatch";

constexpr std::string_view EXPAND_ENTRY_POINT = "ConvGroupExpand";
constexpr std::string_view COPY_ENTRY_POINT = "ConvGroupCopy";
constexpr std::string_view TRANSPOSE_ENTRY_POINT = "ConvGroupTranspose";
constexpr unsigned int COPY_BLOCK = 256;
constexpr long long COPY_MAX_BLOCKS = 1LL << 20;
/// LDS a workgroup may take (every target arch has at least 64 KB).
constexpr long long LDS_LIMIT = 64 * 1024;

constexpr std::string_view X_TOKEN = "conv_group.x.uid";
constexpr std::string_view W_TOKEN = "conv_group.w.uid";
constexpr std::string_view Y_TOKEN = "conv_group.y.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_group.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_group.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_group.flops";

std::optional<ConvDirection> directionFor(const KernelDefinition& kernel)
{
    const auto& direction = kernel.getStringMetadata("direction");
    if(direction == "FWD")
    {
        return ConvDirection::FORWARD;
    }
    if(direction == "BWD")
    {
        return ConvDirection::BACKWARD_DATA;
    }
    return std::nullopt;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = conv_group::parseGroupProblem(context, direction);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    const auto& igemm = problem->igemm;
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = igemm.xUid;
    bound[std::string(W_TOKEN)] = igemm.wUid;
    bound[std::string(Y_TOKEN)] = igemm.yUid;
    bound[std::string(DTYPE_TOKEN)]
        = std::string(data_objects::EnumNameDataType(igemm.base.dataType));
    bound[std::string(DIRECTION_TOKEN)]
        = std::string(direction == ConvDirection::FORWARD ? "FWD" : "BWD");
    const auto flops = pack_features::checkedMultiply(2, igemm.base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

std::optional<BoundTokens> fwdGraphMatches(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::FORWARD);
}

std::optional<BoundTokens> bwdGraphMatches(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::BACKWARD_DATA);
}

/// A config's compile-time choices, read from its metadata.
struct Tile
{
    std::string atom;
    conv_igemm::AtomInfo info;
    int64_t r = 0, s = 0, stride = 0, th = 0, wp = 0, cb = 0;
};

std::optional<Tile> tileOf(const KernelDefinition& kernel)
{
    Tile tile;
    tile.atom = kernel.getStringMetadata("atom");
    const auto info = conv_igemm::atomNamed(tile.atom);
    if(!info.has_value() || info->wave == 0)
    {
        return std::nullopt;
    }
    tile.info = *info;
    tile.r = kernel.getIntMetadata("r");
    tile.s = kernel.getIntMetadata("s");
    tile.stride = kernel.getIntMetadata("stride");
    tile.th = kernel.getIntMetadata("th");
    tile.wp = kernel.getIntMetadata("wp");
    tile.cb = kernel.getIntMetadata("cb");
    if(tile.r < 1 || tile.s < 1 || tile.stride < 1 || tile.th < 1 || tile.wp < 1 || tile.cb < 1)
    {
        return std::nullopt;
    }
    return tile;
}

/// Kernel-scoped applicability: the config's atom runs on this device and multiplies the graph's
/// element type (the 16x16 atoms whose k-step divides 16), its compile-time filter and stride
/// are the graph's, its channel tile divides the channels, and its halo fits the LDS.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto direction = directionFor(kernel);
    const auto tile = tileOf(kernel);
    if(!direction.has_value() || !tile.has_value())
    {
        return false;
    }
    const auto problem = conv_group::parseGroupProblem(context, *direction);
    if(!problem.has_value()
       || kernel.getStringMetadata("dtype")
              != data_objects::EnumNameDataType(problem->igemm.base.dataType)
       || !conv_igemm::atomRunsOn(tile->atom, context.deviceProperties.gcnArchName)
       || conv_igemm::atomElement(tile->atom)
              != conv_igemm::dataElement(problem->igemm.base.dataType)
       || tile->info.tf32)
    {
        return false;
    }
    if(tile->r != problem->filterR || tile->s != problem->filterS
       || tile->stride != problem->stride)
    {
        return false;
    }
    // The halo the kernel stages: its compile-time filter and gather stride (backward data's
    // stride phases gather at unit stride over the taps a phase reaches).
    if(conv_group::groupLdsBytes(problem->igemm.elementBytes,
                                 problem->kernelR,
                                 problem->kernelS,
                                 problem->kernelStride,
                                 static_cast<int>(tile->th),
                                 static_cast<int>(tile->wp),
                                 static_cast<int>(tile->cb))
       > LDS_LIMIT)
    {
        return false;
    }
    return std::all_of(problem->phases.begin(), problem->phases.end(), [&](const auto& phase) {
        return conv_group::phaseBlocks(phase.args,
                                       static_cast<int>(tile->th),
                                       static_cast<int>(tile->wp),
                                       static_cast<int>(tile->cb))
            .has_value();
    });
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
                                                       "conv_group: unsupported element type");
    }
}

template <class Args>
const std::vector<KernelArgument>& signatureOf()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(Args)), 0, ""}};
    return s_signature;
}

class PreparedConvGroup : public IngestorPreparedDispatch
{
public:
    PreparedConvGroup(IngestorKernelCode conv,
                      IngestorKernelCode expand,
                      std::array<std::optional<IngestorKernelCode>, 2> copies,
                      GroupProblem problem,
                      std::vector<unsigned int> blocks)
        : _conv(std::move(conv))
        , _expand(std::move(expand))
        , _copies(std::move(copies))
        , _problem(std::move(problem))
        , _blocks(std::move(blocks))
    {
    }

    /// Each phase's workgroups.
    const std::vector<unsigned int>& blocks() const
    {
        return _blocks;
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _conv;
    }

    compilation::IRunnableKernel& convForStream(hipStream_t stream) const
    {
        return _conv.kernelForStream(stream);
    }

    compilation::IRunnableKernel& expandForStream(hipStream_t stream) const
    {
        return _expand.kernelForStream(stream);
    }

    /// The staging of x (0) or y (1).
    compilation::IRunnableKernel& copyForStream(size_t operand, hipStream_t stream) const
    {
        return _copies[operand]->kernelForStream(stream);
    }

    const GroupProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _conv;
    IngestorKernelCode _expand;
    std::array<std::optional<IngestorKernelCode>, 2> _copies;
    GroupProblem _problem;
    std::vector<unsigned int> _blocks;
};

class ConvGroupDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvGroupDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                             const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The staged activations and the expanded filter.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto direction = directionFor(kernel);
        if(!direction.has_value())
        {
            return 0;
        }
        const auto problem = conv_group::parseGroupProblem(context, *direction);
        return problem.has_value() ? problem->workspaceBytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto direction = directionFor(kernel);
        const auto tile = tileOf(kernel);
        std::optional<GroupProblem> problem;
        if(direction.has_value())
        {
            problem = conv_group::parseGroupProblem(context, *direction);
        }
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!problem.has_value() || !tile.has_value() || !x.has_value() || !w.has_value()
           || !y.has_value() || *x != problem->igemm.xUid || *w != problem->igemm.wUid
           || *y != problem->igemm.yUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_group dispatch was given a graph its matcher did not admit");
        }
        std::vector<unsigned int> blocks;
        for(const auto& phase : problem->phases)
        {
            const auto phaseGrid = conv_group::phaseBlocks(phase.args,
                                                           static_cast<int>(tile->th),
                                                           static_cast<int>(tile->wp),
                                                           static_cast<int>(tile->cb));
            if(!phaseGrid.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "conv_group dispatch was given a config its kernel match declines");
            }
            blocks.push_back(*phaseGrid);
        }

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_GROUP_T", elementTypeFor(problem->igemm.base.dataType));
        options.add("CONV_GROUP_ATOM", std::string(tile->info.kernelType));
        options.add("CONV_GROUP_R", std::to_string(problem->kernelR));
        options.add("CONV_GROUP_S", std::to_string(problem->kernelS));
        options.add("CONV_GROUP_ST", std::to_string(problem->kernelStride));
        options.add("CONV_GROUP_TH", std::to_string(tile->th));
        options.add("CONV_GROUP_WP", std::to_string(tile->wp));
        options.add("CONV_GROUP_CB", std::to_string(tile->cb));

        auto conv = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, signatureOf<K4Args>());
        conv.setBlockSize(static_cast<unsigned int>(4 * tile->info.wave), 1, 1);
        conv.setGridSize(blocks.front(), 1, 1);

        KernelDefinition expandKernel = kernel;
        expandKernel.source.entryPoint = std::string(EXPAND_ENTRY_POINT);
        auto expand = buildIngestorKernelCode(_kernelCompiler,
                                              _kpackLoader,
                                              context,
                                              expandKernel,
                                              options,
                                              signatureOf<K4ExpandArgs>());
        const long long expandElems = static_cast<long long>(problem->igemm.args.c / 16)
                                      * problem->kernelR * problem->kernelS * 256;
        expand.setBlockSize(COPY_BLOCK, 1, 1);
        expand.setGridSize(static_cast<unsigned int>(std::clamp<long long>(
                               (expandElems + COPY_BLOCK - 1) / COPY_BLOCK, 1, COPY_MAX_BLOCKS)),
                           1,
                           1);

        std::array<std::optional<IngestorKernelCode>, 2> copies;
        const std::array<const conv_igemm::Staging*, 2> stagings{&problem->igemm.x,
                                                                 &problem->igemm.y};
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(!stagings[i]->needed)
            {
                continue;
            }
            KernelDefinition copyKernel = kernel;
            const bool transpose = stagings[i]->transpose;
            copyKernel.source.entryPoint
                = std::string(transpose ? TRANSPOSE_ENTRY_POINT : COPY_ENTRY_POINT);
            auto copy = buildIngestorKernelCode(_kernelCompiler,
                                                _kpackLoader,
                                                context,
                                                copyKernel,
                                                options,
                                                transpose ? signatureOf<ConvIgemmTransposeArgs>()
                                                          : signatureOf<ConvIgemmCopyArgs>());
            long long grid = 0;
            if(transpose)
            {
                const auto& t = stagings[i]->transposeArgs;
                constexpr long long TILE = CONV_IGEMM_TRANSPOSE_TILE;
                grid = (t.cols + TILE - 1) / TILE * ((t.rows + TILE - 1) / TILE) * t.batches;
                if(grid > MAX_GRID_BLOCKS)
                {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                        "conv_group staging transpose exceeds the launch grid");
                }
            }
            else
            {
                grid = std::clamp<long long>(
                    (stagings[i]->copy.total + COPY_BLOCK - 1) / COPY_BLOCK, 1, COPY_MAX_BLOCKS);
            }
            copy.setBlockSize(COPY_BLOCK, 1, 1);
            copy.setGridSize(static_cast<unsigned int>(grid), 1, 1);
            copies[i].emplace(std::move(copy));
        }
        return std::make_unique<PreparedConvGroup>(
            std::move(conv), std::move(expand), std::move(copies), *problem, std::move(blocks));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvGroup&>(prepared);
        const auto& problem = dispatch.problem();
        const auto& igemm = problem.igemm;
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "conv_group needs its workspace");
        }
        auto* x
            = hipdnn_plugin_sdk::findDeviceBuffer(igemm.xUid, deviceBuffers, numDeviceBuffers).ptr;
        auto* w
            = hipdnn_plugin_sdk::findDeviceBuffer(igemm.wUid, deviceBuffers, numDeviceBuffers).ptr;
        auto* y
            = hipdnn_plugin_sdk::findDeviceBuffer(igemm.yUid, deviceBuffers, numDeviceBuffers).ptr;
        const auto stream = handle.getStream();
        const bool forward = igemm.direction == ConvDirection::FORWARD;
        auto* base = static_cast<char*>(workspace);
        const std::array<void*, 2> graph{x, y};
        const std::array<const conv_igemm::Staging*, 2> stagings{&igemm.x, &igemm.y};
        // x is read forward and written backward; y the other way round.
        const std::array<bool, 2> read{forward, !forward};
        std::array<void*, 2> operand = graph;
        auto copy = [&](size_t i, bool in) {
            char* staged = base + stagings[i]->workspaceOffset;
            void* src = in ? graph[i] : staged;
            void* dst = in ? staged : graph[i];
            if(stagings[i]->transpose)
            {
                ConvIgemmTransposeArgs args = stagings[i]->transposeArgs;
                args.src = src;
                args.dst = dst;
                dispatch.copyForStream(i, stream).launch(stream, args);
            }
            else
            {
                ConvIgemmCopyArgs args = stagings[i]->copy;
                args.src = src;
                args.dst = dst;
                dispatch.copyForStream(i, stream).launch(stream, args);
            }
        };
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(stagings[i]->needed)
            {
                operand[i] = base + stagings[i]->workspaceOffset;
                if(read[i])
                {
                    copy(i, true);
                }
            }
        }
        // Each phase: its filter expansion, then its gather.
        for(size_t i = 0; i < problem.phases.size(); ++i)
        {
            const auto& phase = problem.phases[i];
            void* expanded = base + phase.expandOffset;
            K4ExpandArgs expand = phase.expand;
            expand.src = w;
            expand.dst = expanded;
            dispatch.expandForStream(stream).launch(stream, expand);
            K4Args args = phase.args;
            args.in = forward ? operand[0] : operand[1];
            args.w = expanded;
            args.out = forward ? operand[1] : operand[0];
            auto& conv = dispatch.convForStream(stream);
            conv.setGridSize(dispatch.blocks()[i], 1, 1);
            conv.launch(stream, args);
        }
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(stagings[i]->needed && !read[i])
            {
                copy(i, false);
            }
        }
    }

private:
    static constexpr long long MAX_GRID_BLOCKS = std::numeric_limits<int>::max();
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convGroupKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvGroupDispatchHandler& convGroupDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convGroupKpackModuleCache());
    static const ConvGroupDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvGroupSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &fwdGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convGroupDispatchHandler());
}

void resetConvGroupModuleCache()
{
    convGroupKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
