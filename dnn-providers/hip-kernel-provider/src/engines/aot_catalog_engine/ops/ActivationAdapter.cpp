// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ops/ActivationAdapter.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "launch/PluginError.hpp"

namespace aot_catalog_engine::ops
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper;

namespace
{

// hipDNN element dtype -> the provider dtype token kernel authors use in
// family.json constraints. Unsupported dtypes yield nullopt (decode declines).
std::optional<std::string> providerDtype(data_objects::DataType dtype)
{
    switch(dtype)
    {
    case data_objects::DataType::HALF:
        return std::string("f16");
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    default:
        return std::nullopt;
    }
}

// Map a PointwiseMode to the rocKE elementwise op token baked into the .co
// symbol / constrained in family.json. Returns nullopt for any mode this adapter
// does not serve (binary ops, exact erf GELU, bwd modes, etc.), which makes
// decode decline. `beta` is the SWISH_FWD swish_beta (absent -> unit).
std::optional<std::string> activationToken(data_objects::PointwiseMode op,
                                           ::flatbuffers::Optional<float> beta)
{
    switch(op)
    {
    case data_objects::PointwiseMode::SWISH_FWD:
        // SiLU is Swish with beta == 1. A non-unit beta is a different function
        // our kernel does not implement, so decline it.
        if(beta.has_value() && *beta != 1.0f)
        {
            return std::nullopt;
        }
        return std::string("silu");
    case data_objects::PointwiseMode::GELU_APPROX_TANH_FWD:
        return std::string("gelu_tanh");
    default:
        // Exact erf GELU_FWD has no rocKE builder op yet; every other mode is
        // out of scope for the activation family.
        return std::nullopt;
    }
}

int64_t problemInt(const catalog::ProblemShape& problem, const std::string& key)
{
    auto it = problem.find(key);
    if(it == problem.end() || !std::holds_alternative<int64_t>(it->second))
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog(pointwise): problem missing integer key '" + key + "'");
    }
    return std::get<int64_t>(it->second);
}

// Total element count of a tensor, or nullopt if any dim is non-positive.
std::optional<int64_t> numElements(const TensorAttributesWrapper& t)
{
    const auto dims = t.dims();
    if(dims.empty())
    {
        return std::nullopt;
    }
    int64_t numel = 1;
    for(const int64_t d : dims)
    {
        if(d <= 0)
        {
            return std::nullopt;
        }
        numel *= d;
    }
    return numel;
}

// The elementwise kernel walks the buffer as one flat contiguous run, so both
// operands must be packed C-contiguous (last dim stride 1, each earlier stride =
// product of the trailing dims).
bool isPackedContiguous(const TensorAttributesWrapper& t)
{
    const auto dims = t.dims();
    const auto strides = t.strides();
    if(dims.empty() || strides.size() != dims.size())
    {
        return false;
    }
    int64_t expected = 1;
    for(size_t i = dims.size(); i-- > 0;)
    {
        if(strides[i] != expected)
        {
            return false;
        }
        expected *= dims[i];
    }
    return true;
}

} // namespace

std::optional<catalog::ProblemShape> ActivationAdapter::decode(const IGraph& graph) const
{
    // Single PointwiseAttributes node only (Tier-D allowlist for the POC).
    if(!graph.isValid() || graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return std::nullopt;
    }
    const auto& attrs = node.attributesAs<data_objects::PointwiseAttributes>();

    // Unary activations only: a second/third input means a binary/ternary
    // pointwise (add/mul/select/...), which is a different kernel.
    if(attrs.in_1_tensor_uid().has_value() || attrs.in_2_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    const auto activation = activationToken(attrs.operation(), attrs.swish_beta());
    if(!activation.has_value())
    {
        return std::nullopt;
    }

    const auto& tensorMap = graph.getTensorMap();
    auto findTensor = [&](int64_t uid) -> const data_objects::TensorAttributes* {
        auto it = tensorMap.find(uid);
        return it == tensorMap.end() ? nullptr : it->second;
    };

    const auto* inTensor = findTensor(attrs.in_0_tensor_uid());
    const auto* outTensor = findTensor(attrs.out_0_tensor_uid());
    if(inTensor == nullptr || outTensor == nullptr)
    {
        return std::nullopt;
    }

    try
    {
        const TensorAttributesWrapper in(inTensor);
        const TensorAttributesWrapper out(outTensor);

        // A single supported element dtype (f16 or bf16), identical in and out.
        const auto dtype = providerDtype(in.dataType());
        if(!dtype.has_value() || out.dataType() != in.dataType())
        {
            return std::nullopt;
        }

        // Flat contiguous run of numel elements, in and out identically shaped.
        const auto inNumel = numElements(in);
        const auto outNumel = numElements(out);
        if(!inNumel.has_value() || !outNumel.has_value() || *inNumel != *outNumel)
        {
            return std::nullopt;
        }
        if(!isPackedContiguous(in) || !isPackedContiguous(out))
        {
            return std::nullopt;
        }
        // numel is passed to the kernel as an i32; decline anything that would
        // overflow rather than launch with a truncated element count.
        if(*inNumel > std::numeric_limits<int32_t>::max())
        {
            return std::nullopt;
        }

        catalog::ProblemShape shape;
        shape.emplace("dtype", catalog::ShapeValue{*dtype});
        shape.emplace("activation", catalog::ShapeValue{*activation});
        shape.emplace("numel", catalog::ShapeValue{*inNumel});
        return shape;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "aot-catalog(pointwise): declining graph, decode failed: " << e.what());
        return std::nullopt;
    }
}

catalog::LaunchBindings ActivationAdapter::buildBindings(const IGraph& graph,
                                                         const catalog::ProblemShape& problem,
                                                         const catalog::KernelEntry& kernel) const
{
    (void)kernel; // activation binds a fixed (A,C,N) ABI regardless of kernel

    const auto& node = graph.getNodeWrapper(0);
    const auto& attrs = node.attributesAs<data_objects::PointwiseAttributes>();

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", attrs.in_0_tensor_uid());
    bindings.pointerUids.emplace("C", attrs.out_0_tensor_uid());
    bindings.scalars.emplace("N", catalog::ScalarValue{problemInt(problem, "numel")});
    return bindings;
}

launch::SymbolTable ActivationAdapter::gridSymbols(const catalog::ProblemShape& problem,
                                                   const catalog::KernelEntry& kernel) const
{
    (void)kernel;

    launch::SymbolTable symbols;
    symbols.emplace("numel", problemInt(problem, "numel"));
    return symbols;
}

} // namespace aot_catalog_engine::ops
