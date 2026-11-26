// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvAotGraphPredicates.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdAotCommon.hpp"

/**
 * @file Gfx950ConvFwdBiasNative.cpp
 * @brief The native half of `hipkernel:Gfx950ConvFwdBiasAot`
 *        (descriptors/rocKE/gfx950_conv_fwd_bias_aot/): its graph match, symbols, module
 *        cache and registration.
 *
 * The engine serves a two-node graph on gfx950: a forward convolution whose virtual
 * output feeds a pointwise ADD with a per-output-channel bias, the form a biased
 * convolution takes in hipDNN. It runs hipkernel:Gfx950ConvFwdAot's shape-generic AOT
 * instances built with rocKE's bias epilogue (`bias` = 1 in their metadata): the kernel
 * adds bias[k] to the fp32 accumulator and rounds the sum once, so the convolution
 * result is never stored. The convolution is served under the same rules as the unfused
 * engine (Gfx950ConvFwdAotCommon.hpp), with the ADD's output as the stored output.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

namespace aot = gfx950_conv_fwd_aot;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd_bias_aot.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx950_conv_fwd_bias_aot.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_conv_fwd_bias_aot.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_conv_fwd_bias_aot.dispatch";
/// The packs of split-C kernels (make_conv_aot_config.SPLIT_C_DISPATCH): two launches.
constexpr std::string_view SPLIT_C_DISPATCH_SYMBOL
    = "hipkernel.gfx950_conv_fwd_bias_aot.split_c_dispatch";

/// Whether @p bias holds one value per output channel of a @p k-channel convolution
/// output, as the kernel reads it: a device tensor of the io dtype @p dtype, contiguous
/// along the channels, dims (1, K, 1, 1) or (K, 1, 1). hipDNN broadcasts from the
/// right, so (K, 1, 1) also lands K on the channel axis; a rank-1 (K) would land on W
/// and is declined, as is every other rank.
bool isChannelBias(const data_objects::TensorAttributes* bias,
                   int64_t k,
                   data_objects::DataType dtype)
{
    if(!conv_aot::isDeviceTensor(bias) || bias->data_type() != dtype)
    {
        return false;
    }
    const auto* dims = bias->dims();
    const auto* strides = bias->strides();
    if(dims == nullptr || strides == nullptr || dims->size() != strides->size()
       || (dims->size() != 4 && dims->size() != 3))
    {
        return false;
    }
    // The channel axis: 1 of (1, K, 1, 1), 0 of (K, 1, 1).
    const flatbuffers::uoffset_t channel = dims->size() == 4 ? 1 : 0;
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        if(dims->Get(axis) != (axis == channel ? k : 1))
        {
            return false;
        }
    }
    return k == 1 || strides->Get(channel) == 1;
}

/// Whether two dims vectors are equal; a virtual tensor whose dims the frontend did not
/// infer (none, or empty) agrees with any.
bool dimsAgree(const data_objects::TensorAttributes& inner, const data_objects::TensorAttributes& y)
{
    const auto* innerDims = inner.dims();
    if(innerDims == nullptr || innerDims->size() == 0)
    {
        return true;
    }
    const auto* yDims = y.dims();
    if(yDims == nullptr || innerDims->size() != yDims->size())
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < innerDims->size(); ++axis)
    {
        if(innerDims->Get(axis) != yDims->Get(axis))
        {
            return false;
        }
    }
    return true;
}

/// conv(x, w) -> yc (virtual) ; ADD(yc, b) -> y, in either node and operand order.
std::optional<aot::MatchedProblem> matchProblem(const MatchContext& context)
{
    if(!aot::servesDeviceAndGraph(context, 2))
    {
        return std::nullopt;
    }
    const data_objects::ConvolutionFwdAttributes* conv = nullptr;
    const data_objects::PointwiseAttributes* add = nullptr;
    for(uint32_t i = 0; i < 2; ++i)
    {
        const auto& node = context.graph.getNodeWrapper(i);
        // Both nodes compute in fp32: the kernel accumulates the convolution and adds the
        // bias in fp32, so another compute type would be served at another precision.
        if(node.attributes() == nullptr || node.computeDataType() != data_objects::DataType::FLOAT)
        {
            return std::nullopt;
        }
        if(node.attributesType() == data_objects::NodeAttributes::ConvolutionFwdAttributes
           && conv == nullptr)
        {
            conv = &node.attributesAs<data_objects::ConvolutionFwdAttributes>();
        }
        else if(node.attributesType() == data_objects::NodeAttributes::PointwiseAttributes
                && add == nullptr)
        {
            add = &node.attributesAs<data_objects::PointwiseAttributes>();
        }
        else
        {
            return std::nullopt;
        }
    }
    if(conv == nullptr || add == nullptr || add->operation() != data_objects::PointwiseMode::ADD
       || !add->in_1_tensor_uid().has_value() || add->in_2_tensor_uid().has_value()
       || add->axis_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    // One ADD operand is the convolution's output, the other the bias.
    const int64_t convOutUid = conv->y_tensor_uid();
    const int64_t in0 = add->in_0_tensor_uid();
    const int64_t in1 = *add->in_1_tensor_uid();
    if((in0 == convOutUid) == (in1 == convOutUid))
    {
        return std::nullopt;
    }
    const int64_t biasUid = in0 == convOutUid ? in1 : in0;
    const int64_t yUid = add->out_0_tensor_uid();
    if(biasUid == conv->x_tensor_uid() || biasUid == conv->w_tensor_uid() || biasUid == yUid
       || yUid == convOutUid)
    {
        return std::nullopt;
    }

    // The convolution with the ADD's output as its stored output: x, w and y as the
    // unfused engine requires them.
    auto matched = aot::matchConvolution(context, *conv, yUid, biasUid);
    if(!matched)
    {
        return std::nullopt;
    }
    // The convolution's output only feeds the ADD and is never stored: virtual, the
    // shape of the ADD's output, fp32 or the io dtype (the kernel keeps it in fp32).
    const auto* convOut = aot::findTensor(context, convOutUid);
    const auto* y = aot::findTensor(context, yUid);
    if(convOut == nullptr || !convOut->virtual_() || !dimsAgree(*convOut, *y)
       || (convOut->data_type() != data_objects::DataType::FLOAT
           && convOut->data_type() != matched->dtype)
       || !isChannelBias(aot::findTensor(context, biasUid), matched->problem.k, matched->dtype))
    {
        return std::nullopt;
    }
    return matched;
}

const aot::ConvFwdAotEngine& engine()
{
    static const aot::ConvFwdAotEngine s_engine{"gfx950_conv_fwd_bias",
                                                "gfx950_conv_fwd_bias.x.uid",
                                                "gfx950_conv_fwd_bias.w.uid",
                                                "gfx950_conv_fwd_bias.y.uid",
                                                "gfx950_conv_fwd_bias.b.uid",
                                                "_gfx950_conv_fwd_bias.problem",
                                                &matchProblem};
    return s_engine;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    return aot::graphMatches(engine(), context);
}

/// Every kernel of this engine carries the bias epilogue (`bias` = 1): the shared
/// admission refuses any other, as the unfused engine refuses these.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    return aot::aotKernelMatches(engine(), context, bound, kernel);
}

/// The unfused engine's deterministic fallback order: the bias add does not change which
/// instance runs the convolution best.
double score(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
{
    return aot::aotScore(engine(), context, bound, kernel);
}

std::vector<bool> candidateFilter(const MatchContext& context,
                                  const BoundTokens& bound,
                                  const std::vector<const KernelDefinition*>& candidates)
{
    return aot::aotCandidateFilter(engine(), context, bound, candidates);
}

const compilation::KpackKernelLoader& kpackLoader()
{
    static const compilation::KpackKernelLoader s_loader(gfx950ConvFwdBiasAotKpackModuleCache());
    return s_loader;
}

const IKernelDispatchHandler<Handle>& dispatchHandler()
{
    static const auto s_handler = aot::makeDispatchHandler(engine(), kpackLoader());
    return *s_handler;
}

/// The handler of the split-C packs: the same admission, two launches per kernel, the
/// fold adding the bias.
const IKernelDispatchHandler<Handle>& splitCDispatchHandler()
{
    static const auto s_handler = aot::makeDispatchHandler(
        engine(), kpackLoader(), {aot::SPLIT_C_CONV_LAUNCH, aot::SPLIT_C_REDUCE_LAUNCH});
    return *s_handler;
}

} // namespace

compilation::KpackModuleCache& gfx950ConvFwdBiasAotKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void registerGfx950ConvFwdBiasAotSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &candidateFilter);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &aot::aotPrepare);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(SPLIT_C_DISPATCH_SYMBOL), &splitCDispatchHandler());
}

void resetGfx950ConvFwdBiasAotModuleCache()
{
    gfx950ConvFwdBiasAotKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
