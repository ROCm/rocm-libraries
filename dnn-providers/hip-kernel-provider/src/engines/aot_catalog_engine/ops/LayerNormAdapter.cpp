// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ops/LayerNormAdapter.hpp"

#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/norm_common_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

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

int64_t problemInt(const catalog::ProblemShape& problem, const std::string& key)
{
    auto it = problem.find(key);
    if(it == problem.end() || !std::holds_alternative<int64_t>(it->second))
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog(layernorm): problem missing integer key '" + key + "'");
    }
    return std::get<int64_t>(it->second);
}

// Resolve epsilon into a baked float, or nullopt if it cannot be baked at
// plan-build time (pure runtime user-supplied) or its dtype is unsupported.
// The rocKE kernel takes eps as an f32 by-value arg; hipDNN passes it as a
// scalar tensor operand, which for our real targets (ComfyUI/PyTorch) is a
// compile-time constant baked into the graph.
std::optional<float> bakedEpsilon(const IGraph& graph, int64_t epsilonUid)
{
    try
    {
        const auto& tensorMap = graph.getTensorMap();
        const hipdnn_plugin_sdk::ScalarOperand op
            = hipdnn_plugin_sdk::makeScalarOperand(tensorMap, epsilonUid, "epsilon");
        if(op.isRuntimeUserSupplied)
        {
            // Deferring the read to execute would need LaunchAbi support we do
            // not have; fail closed rather than launch with a wrong epsilon.
            return std::nullopt;
        }
        return static_cast<float>(hipdnn_plugin_sdk::toDouble(op.bakedDefault));
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO("aot-catalog(layernorm): epsilon not bakeable: " << e.what());
        return std::nullopt;
    }
}

} // namespace

std::optional<catalog::ProblemShape> LayerNormAdapter::decode(const IGraph& graph) const
{
    // Single LayernormAttributes node only (Tier-D allowlist for the POC).
    if(!graph.isValid() || graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::LayernormAttributes)
    {
        return std::nullopt;
    }
    const auto& attrs = node.attributesAs<data_objects::LayernormAttributes>();

    // Our forward kernel saves neither mean nor inv_variance; a graph asking for
    // either is a training/backward-stat request, so decline.
    if(attrs.mean_tensor_uid().has_value() || attrs.inv_variance_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Forward inference only. NOT_SET is treated as inference (matches how the
    // frontend leaves the phase unset for pure forward graphs).
    if(attrs.forward_phase() == data_objects::NormFwdPhase::TRAINING)
    {
        return std::nullopt;
    }

    const auto& tensorMap = graph.getTensorMap();
    auto findTensor = [&](int64_t uid) -> const data_objects::TensorAttributes* {
        auto it = tensorMap.find(uid);
        return it == tensorMap.end() ? nullptr : it->second;
    };

    const auto* xTensor = findTensor(attrs.x_tensor_uid());
    const auto* scaleTensor = findTensor(attrs.scale_tensor_uid());
    const auto* biasTensor = findTensor(attrs.bias_tensor_uid());
    const auto* yTensor = findTensor(attrs.y_tensor_uid());
    if(xTensor == nullptr || scaleTensor == nullptr || biasTensor == nullptr || yTensor == nullptr)
    {
        return std::nullopt;
    }

    // TensorAttributesWrapper::dims()/strides()/dataType() throw on null
    // dims/strides; a malformed graph should make the engine decline, not throw
    // out of isApplicable, so treat any decode failure as "not applicable".
    try
    {
        const TensorAttributesWrapper x(xTensor);
        const TensorAttributesWrapper scale(scaleTensor);
        const TensorAttributesWrapper bias(biasTensor);
        const TensorAttributesWrapper y(yTensor);

        // 2D [M,N] activation only for the POC. Our kernel reduces exactly the
        // last dim, so we require Gamma = Beta = [1, N] (broadcast over rows,
        // per-column weight and bias).
        const auto xDims = x.dims();
        const auto scaleDims = scale.dims();
        const auto biasDims = bias.dims();
        const auto yDims = y.dims();
        if(xDims.size() != 2 || scaleDims.size() != 2 || biasDims.size() != 2 || yDims.size() != 2)
        {
            return std::nullopt;
        }

        const int64_t m = xDims[0];
        const int64_t n = xDims[1];
        if(yDims[0] != m || yDims[1] != n)
        {
            return std::nullopt; // output must match input shape
        }
        if(scaleDims[0] != 1 || scaleDims[1] != n || biasDims[0] != 1 || biasDims[1] != n)
        {
            return std::nullopt; // not a [1,N] per-column weight/bias over the last dim
        }

        // Require row-major X/Y ({N,1}) and contiguous last-dim weight/bias; the
        // kernel indexes X/Y/Gamma/Beta as packed rows of N elements.
        const auto xStrides = x.strides();
        const auto scaleStrides = scale.strides();
        const auto biasStrides = bias.strides();
        const auto yStrides = y.strides();
        const bool xRowMajor = xStrides.size() == 2 && xStrides[0] == n && xStrides[1] == 1;
        const bool yRowMajor = yStrides.size() == 2 && yStrides[0] == n && yStrides[1] == 1;
        const bool scaleContig = scaleStrides.size() == 2 && scaleStrides[1] == 1;
        const bool biasContig = biasStrides.size() == 2 && biasStrides[1] == 1;
        if(!xRowMajor || !yRowMajor || !scaleContig || !biasContig)
        {
            return std::nullopt;
        }

        // A single supported element dtype (f16 or bf16) across X/Gamma/Beta/Y.
        // providerDtype() already declines anything else; we only require the
        // four operands agree so the baked ABI dtype token is unambiguous.
        const auto dtype = providerDtype(x.dataType());
        if(!dtype.has_value() || scale.dataType() != x.dataType() || bias.dataType() != x.dataType()
           || y.dataType() != x.dataType())
        {
            return std::nullopt;
        }

        // Epsilon must be bakeable now (the LaunchAbi bakes scalars at
        // plan-build); decline pure runtime user-supplied epsilon.
        if(!bakedEpsilon(graph, attrs.epsilon_tensor_uid()).has_value())
        {
            return std::nullopt;
        }

        catalog::ProblemShape shape;
        shape.emplace("dtype", catalog::ShapeValue{*dtype});
        shape.emplace("M", catalog::ShapeValue{m});
        shape.emplace("N", catalog::ShapeValue{n});
        return shape;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "aot-catalog(layernorm): declining graph, decode failed: " << e.what());
        return std::nullopt;
    }
}

catalog::LaunchBindings LayerNormAdapter::buildBindings(const IGraph& graph,
                                                        const catalog::ProblemShape& problem,
                                                        const catalog::KernelEntry& kernel) const
{
    (void)kernel; // layernorm binds a fixed (X,Gamma,Beta,Y,M,N,eps) ABI regardless of kernel

    const auto& node = graph.getNodeWrapper(0);
    const auto& attrs = node.attributesAs<data_objects::LayernormAttributes>();

    const std::optional<float> eps = bakedEpsilon(graph, attrs.epsilon_tensor_uid());
    if(!eps.has_value())
    {
        // decode() already gated this; if we get here the graph changed under us.
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog(layernorm): epsilon is not bakeable at plan-build time");
    }

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("X", attrs.x_tensor_uid());
    bindings.pointerUids.emplace("Gamma", attrs.scale_tensor_uid());
    bindings.pointerUids.emplace("Beta", attrs.bias_tensor_uid());
    bindings.pointerUids.emplace("Y", attrs.y_tensor_uid());
    bindings.scalars.emplace("M", catalog::ScalarValue{problemInt(problem, "M")});
    bindings.scalars.emplace("N", catalog::ScalarValue{problemInt(problem, "N")});
    bindings.scalars.emplace("eps", catalog::ScalarValue{*eps});
    return bindings;
}

launch::SymbolTable LayerNormAdapter::gridSymbols(const catalog::ProblemShape& problem,
                                                  const catalog::KernelEntry& kernel) const
{
    (void)kernel;

    launch::SymbolTable symbols;
    symbols.emplace("M", problemInt(problem, "M"));
    symbols.emplace("N", problemInt(problem, "N"));
    return symbols;
}

} // namespace aot_catalog_engine::ops
