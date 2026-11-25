// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/descriptors/conv/winograd/kernels/ConvWinogradArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvWinogradNdProblem.hpp"
#include <algorithm>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>
#include <memory>
#include <string>
#include <vector>
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
using conv_naive::ConvDirection;
namespace nd = conv_winograd_nd;
std::optional<BoundTokens> ndGraph(const MatchContext& context, ConvDirection dir)
{
    const auto p = nd::parseProblem(context, dir);
    if(!p)
    {
        return std::nullopt;
    }
    BoundTokens b;
    b["conv_winograd_nd.x.uid"] = p->base.operands.x;
    b["conv_winograd_nd.w.uid"] = p->base.operands.w;
    b["conv_winograd_nd.y.uid"] = p->base.operands.y;
    b["conv_winograd_nd.dtype"]
        = std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(p->base.dataType));
    return b;
}
std::optional<BoundTokens> ndFwd(const MatchContext& c)
{
    return ndGraph(c, ConvDirection::FORWARD);
}
struct NdResolved
{
    nd::Problem problem;
    nd::Config config;
    nd::Plan plan;
};
std::optional<NdResolved> ndResolve(const MatchContext& context, const KernelDefinition& kernel)
{
    try
    {
        const auto direction = kernel.getStringMetadata("direction");
        if(direction != "FWD")
        {
            return std::nullopt;
        }
        auto p = nd::parseProblem(context, ConvDirection::FORWARD);
        auto c = nd::configOf(kernel);
        if(!p || !c || !nd::configRunsOn(*c, context.deviceProperties.gcnArchName)
           || kernel.getStringMetadata("dtype")
                  != hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(p->base.dataType))
        {
            return std::nullopt;
        }
        auto plan = nd::planFor(*p, *c);
        if(!plan)
        {
            return std::nullopt;
        }
        return NdResolved{*p, *c, *plan};
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }
}
bool ndMatches(const MatchContext& context, const BoundTokens& /*bound*/, const KernelDefinition& k)
{
    return ndResolve(context, k).has_value();
}
template <class A>
const std::vector<KernelArgument>& ndSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(A)), 0, ""}};
    return s_signature;
}
class NdPrepared : public IngestorPreparedDispatch
{
public:
    NdPrepared(std::vector<IngestorKernelCode> k, NdResolved r)
        : kernels(std::move(k))
        , resolved(std::move(r))
    {
    }
    const IngestorKernelCode& kernelCode() const override
    {
        return kernels[1];
    }
    std::vector<IngestorKernelCode> kernels;
    NdResolved resolved;
};
class NdHandler : public IKernelDispatchHandler<Handle>
{
public:
    NdHandler(const compilation::IKernelCompiler& c, const compilation::KpackKernelLoader& l)
        : _compiler(c)
        , _loader(l)
    {
    }
    size_t workspaceBytes(const MatchContext& ctx,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& k) const override
    {
        const auto r = ndResolve(ctx, k);
        return r ? r->plan.workspaceBytes : 0;
    }
    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& ctx,
                                              const BoundTokens& b,
                                              const KernelDefinition& k) const override
    {
        const auto r = ndResolve(ctx, k);
        const auto x = tryGetBoundInt(b, "conv_winograd_nd.x.uid");
        const auto w = tryGetBoundInt(b, "conv_winograd_nd.w.uid");
        const auto y = tryGetBoundInt(b, "conv_winograd_nd.y.uid");
        if(!r || !x || !w || !y || *x != r->problem.base.operands.x
           || *w != r->problem.base.operands.w || *y != r->problem.base.operands.y)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "winograd_nd: unmatched graph/config");
        }
        compilation::KernelCompileOptions opts(ctx.deviceProperties.gcnArchName);
        const auto& c = r->config;
        const auto& p = r->plan;
        opts.add("CONV_ND_BM", std::to_string(c.bm));
        opts.add("CONV_ND_BN", std::to_string(c.bn));
        opts.add("CONV_ND_BK", std::to_string(c.bk));
        opts.add("CONV_WINO_GFX1250", 1);
        std::vector<IngestorKernelCode> codes;
        const auto build = [&](const char* entry,
                               const std::vector<KernelArgument>& signature,
                               unsigned int gx,
                               unsigned int gy,
                               unsigned int gz) {
            KernelDefinition def = k;
            def.source.entryPoint = entry;
            auto code = buildIngestorKernelCode(_compiler, _loader, ctx, def, opts, signature);
            code.setBlockSize(256, 1, 1);
            code.setGridSize(gx, gy, gz);
            codes.push_back(std::move(code));
        };
        build("ConvWinogradNdFilter", ndSignature<KNdArgs>(), p.filterBlocks, 1, 1);
        build("ConvWinogradNdGemm",
              ndSignature<K5GemmArgs>(),
              p.gemmBlocks[0],
              p.gemmBlocks[1],
              p.gemmBlocks[2]);
        build("ConvWinogradNdInput", ndSignature<KNdArgs>(), p.inputBlocks, 1, 1);
        build("ConvWinogradNdOutput", ndSignature<KNdArgs>(), p.outputBlocks, 1, 1);
        return std::make_unique<NdPrepared>(std::move(codes), *r);
    }
    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* buffers,
                uint32_t count,
                void* workspace) const override
    {
        const auto& d = dynamic_cast<const NdPrepared&>(prepared);
        const auto& p = d.resolved.problem;
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "winograd_nd: workspace required");
        }
        auto* x = hipdnn_plugin_sdk::findDeviceBuffer(p.base.operands.x, buffers, count).ptr;
        auto* w = hipdnn_plugin_sdk::findDeviceBuffer(p.base.operands.w, buffers, count).ptr;
        auto* y = hipdnn_plugin_sdk::findDeviceBuffer(p.base.operands.y, buffers, count).ptr;
        const bool fwd = p.base.direction == ConvDirection::FORWARD;
        auto a = d.resolved.plan.args;
        auto* ws = static_cast<char*>(workspace);
        a.in = fwd ? x : y;
        a.out = fwd ? y : x;
        a.w = w;
        a.u = reinterpret_cast<float*>(ws);
        a.v = reinterpret_cast<float*>(ws + d.resolved.plan.vOffset);
        a.product = reinterpret_cast<float*>(ws + d.resolved.plan.mOffset);
        const auto stream = handle.getStream();
        // Re-transform every execute: bindings and filter contents may change even at one address.
        d.kernels[0].kernelForStream(stream).launch(stream, a);
        const int capacity = a.tiles;
        for(long long first = 0; first < d.resolved.plan.totalTiles;)
        {
            a.tileBegin = first;
            a.tiles = static_cast<int>(
                std::min<long long>(capacity, d.resolved.plan.totalTiles - first));
            K5GemmArgs g{};
            g.a = a.u;
            g.b = a.v;
            g.c = a.product;
            g.m = static_cast<int>(a.co / a.groups);
            g.n = a.tiles;
            g.k = static_cast<int>(a.ci / a.groups);
            g.lda = g.k;
            g.ldb = g.n;
            g.ldc = g.n;
            g.sa = static_cast<long long>(g.m) * g.k;
            g.sb = static_cast<long long>(g.k) * g.n;
            g.sc = static_cast<long long>(g.m) * g.n;
            d.kernels[2].kernelForStream(stream).launch(stream, a);
            d.kernels[1].kernelForStream(stream).launch(stream, g);
            d.kernels[3].kernelForStream(stream).launch(stream, a);
            first += a.tiles;
        }
    }

private:
    const compilation::IKernelCompiler& _compiler;
    const compilation::KpackKernelLoader& _loader;
};
compilation::KpackModuleCache& ndCache()
{
    static compilation::KpackModuleCache s_cache;
    return s_cache;
}
const NdHandler& ndHandler()
{
    static const HipMlopsKernelCompiler s_compiler;
    static const compilation::KpackKernelLoader s_loader(ndCache());
    static const NdHandler s_handler(s_compiler, s_loader);
    return s_handler;
}
}
void registerConvWinogradNdSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add("hipkernel.conv_winograd_nd_fwd.graph_match", &ndFwd);
    scope.add("hipkernel.conv_winograd_nd.kernel_match", &ndMatches);
    scope.add("hipkernel.conv_winograd_nd.dispatch", &ndHandler());
}
void resetConvWinogradNdModuleCache()
{
    ndCache().clear();
}
}
#endif
