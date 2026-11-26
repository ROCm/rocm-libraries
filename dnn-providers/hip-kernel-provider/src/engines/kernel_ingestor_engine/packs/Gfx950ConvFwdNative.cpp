// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdAotCommon.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

namespace aot = gfx950_conv_fwd_aot;

// ---- hipkernel:Gfx950ConvFwdAot: shape-generic AOT instances ----
//
// A lone convolution node; the instances, their admission, launch and ABI are those of
// Gfx950ConvFwdAotCommon.hpp, which hipkernel:Gfx950ConvFwdBiasAot shares.

constexpr std::string_view AOT_GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd_aot.graph_match";
constexpr std::string_view AOT_KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd_aot.kernel_match";
constexpr std::string_view AOT_SCORE_SYMBOL = "hipkernel.gfx950_conv_fwd_aot.score";
constexpr std::string_view AOT_DISPATCH_SYMBOL = "hipkernel.gfx950_conv_fwd_aot.dispatch";
/// The packs of split-C kernels (make_conv_aot_config.SPLIT_C_DISPATCH): two launches.
constexpr std::string_view AOT_SPLIT_C_DISPATCH_SYMBOL
    = "hipkernel.gfx950_conv_fwd_aot.split_c_dispatch";

std::optional<aot::MatchedProblem> matchProblem(const MatchContext& context)
{
    if(!aot::servesDeviceAndGraph(context, 1))
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes
       || node.attributes() == nullptr || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& attributes = node.attributesAs<data_objects::ConvolutionFwdAttributes>();
    return aot::matchConvolution(context, attributes, attributes.y_tensor_uid());
}

const aot::ConvFwdAotEngine& engine()
{
    static const aot::ConvFwdAotEngine s_engine{"gfx950_conv_fwd",
                                                "gfx950_conv_fwd.x.uid",
                                                "gfx950_conv_fwd.w.uid",
                                                "gfx950_conv_fwd.y.uid",
                                                "",
                                                "_gfx950_conv_fwd.problem",
                                                &matchProblem};
    return s_engine;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    return aot::graphMatches(engine(), context);
}

bool aotKernelMatches(const MatchContext& context,
                      const BoundTokens& bound,
                      const KernelDefinition& kernel)
{
    return aot::aotKernelMatches(engine(), context, bound, kernel);
}

double
    aotScore(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
{
    return aot::aotScore(engine(), context, bound, kernel);
}

std::vector<bool> aotCandidateFilter(const MatchContext& context,
                                     const BoundTokens& bound,
                                     const std::vector<const KernelDefinition*>& candidates)
{
    return aot::aotCandidateFilter(engine(), context, bound, candidates);
}

const compilation::KpackKernelLoader& aotKpackLoader()
{
    static const compilation::KpackKernelLoader s_loader(gfx950ConvFwdAotKpackModuleCache());
    return s_loader;
}

const IKernelDispatchHandler<Handle>& aotDispatchHandler()
{
    static const auto s_handler = aot::makeDispatchHandler(engine(), aotKpackLoader());
    return *s_handler;
}

/// The handler of the split-C packs: the same admission, two launches per kernel.
const IKernelDispatchHandler<Handle>& aotSplitCDispatchHandler()
{
    static const auto s_handler = aot::makeDispatchHandler(
        engine(), aotKpackLoader(), {aot::SPLIT_C_CONV_LAUNCH, aot::SPLIT_C_REDUCE_LAUNCH});
    return *s_handler;
}

} // namespace

compilation::KpackModuleCache& gfx950ConvFwdAotKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void registerGfx950ConvFwdAotSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(AOT_GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(AOT_KERNEL_MATCHER_SYMBOL), &aotKernelMatches);
    scope.add(std::string(AOT_KERNEL_MATCHER_SYMBOL), &aotCandidateFilter);
    scope.add(std::string(AOT_KERNEL_MATCHER_SYMBOL), &aot::aotPrepare);
    scope.add(std::string(AOT_SCORE_SYMBOL), &aotScore);
    scope.add(std::string(AOT_DISPATCH_SYMBOL), &aotDispatchHandler());
    scope.add(std::string(AOT_SPLIT_C_DISPATCH_SYMBOL), &aotSplitCDispatchHandler());
}

void resetGfx950ConvFwdAotModuleCache()
{
    gfx950ConvFwdAotKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
