// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file LayerNormNode.hpp
 * @brief Graph node for layer normalization operations
 */

#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/detail/LayerNormPacker.hpp>
#include <hipdnn_frontend/detail/LayerNormUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
/**
 * @class LayerNormNode
 * @brief Graph node that performs layer normalization
 *
 * Validates input tensors, infers output shapes, and serializes the
 * layer normalization operation to a backend descriptor.
 *
 * @see LayernormAttributes, Graph::layernorm()
 */
class LayerNormNode : public BaseNode<LayerNormNode, NodeType::LAYER_NORM>
{
public:
    LayernormAttributes attributes;

    LayerNormNode(LayernormAttributes&& layernormAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(layernormAttrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        LayernormAttributes lnAttr;
        HIPDNN_CHECK_ERROR(detail::unpackLayernormOperation(opDesc, tensorMap, lnAttr));
        attributes = std::move(lnAttr);
        return {};
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // LAYER NORMALIZATION FORWARD VALIDATION
        // ====================================================================
        // Algorithm Overview:
        // LayerNorm computes statistics over the feature dimensions (last normalized_shape dims):
        //   For input shape [N, ..., D1, D2, ..., Dk] where last k dims are normalized:
        //   mean = (1/m) * sum x[..., i] over normalized dims, where m = D1*D2*...*Dk
        //   var  = (1/m) * sum (x[..., i] - mean)^2 over normalized dims
        //
        // Normalizes: xhat = (x - mean) / sqrt(var + epsilon)
        // Transforms: y = scale * xhat + bias (scale and bias have shape of normalized dims)
        //
        // Outputs:
        // - Y: normalized output (same shape as X)
        // - Mean: computed mean per sample (optional, shape: batch dims, then optional 1s)
        // - InvVariance: inverse standard deviation (optional, shape: as Mean)
        // ====================================================================

        // SECTION 1: Validate Forward Phase
        HIPDNN_RETURN_IF_FALSE(attributes.get_forward_phase() != NormFwdPhase::NOT_SET,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "Forward phase not set of layernorm node.");

        // SECTION 2: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing y for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_epsilon(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing epsilon for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_bias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing bias for pre-validation");

        // Get tensor references
        const auto x = attributes.get_x();
        const auto y = attributes.get_y();
        const auto scale = attributes.get_scale();
        const auto bias = attributes.get_bias();
        const auto epsilon = attributes.get_epsilon();

        // SECTION 3: Validate Required Parameter Dimensions
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 1, "Input tensor (x)"));

        // Epsilon provides numerical stability: xhat = (x - mean) / sqrt(var + epsilon)
        HIPDNN_CHECK_ERROR(detail::validateScalarParameter(epsilon, "Epsilon"));

        // SECTION 4: Validate Output Tensor Shape Consistency
        // LayerNorm preserves tensor shape - output has same shape as input
        HIPDNN_CHECK_ERROR(
            detail::validateTensorShapesMatchIfSet(x, y, "Input tensor (x)", "Output tensor (y)"));

        // SECTION 5: Validate Scale and Bias Tensors
        // Scale and bias are per-feature parameters matching the normalized dimensions.
        // For input shape [N, C, H, W] normalized over last 3 dims, scale/bias shape is [C, H, W]
        // Unset scale or bias dims are filled by infer_properties_node(); the checks below
        // validate the dims they receive there.
        const auto& xDims = x->get_dim();
        const bool scaleDimsSet = detail::areTensorDimensionsSet(scale);
        const bool biasDimsSet = detail::areTensorDimensionsSet(bias);
        std::vector<int64_t> inferredAffineDims;
        if(!scaleDimsSet || !biasDimsSet)
        {
            inferredAffineDims = inferAffineDims(xDims, attributes.get_normalized_dim_count());
        }
        const auto& scaleDims = scaleDimsSet ? scale->get_dim() : inferredAffineDims;

        // Scale and bias should have matching shapes
        if(scaleDimsSet && biasDimsSet)
        {
            HIPDNN_CHECK_ERROR(
                detail::validateTensorShapesMatch(scale, bias, "Scale tensor", "Bias tensor"));
        }
        else if(scaleDimsSet != biasDimsSet)
        {
            const auto& setTensor = scaleDimsSet ? scale : bias;
            HIPDNN_RETURN_IF_TRUE(setTensor->get_dim() != inferredAffineDims,
                                  ErrorCode::INVALID_VALUE,
                                  "LayerNormNode "
                                      + detail::getTensorNameForError(
                                          setTensor, scaleDimsSet ? "Scale tensor" : "Bias tensor")
                                      + " dims must equal the dims inferred from x for the unset "
                                      + (scaleDimsSet ? "bias" : "scale") + " tensor");
        }

        // SECTION 6: Validate Scale, Bias and Statistics Against the Normalized Dims of X
        // Scale and bias hold the last k dims of x, optionally preceded by 1s; mean and
        // inv_variance hold the first n - k (batch) dims of x, optionally followed by k 1s.
        int64_t normalizedDimCount = 0;
        HIPDNN_CHECK_ERROR(
            detail::deriveLayernormNormalizedDimCount(xDims,
                                                      scaleDims,
                                                      attributes.get_normalized_dim_count(),
                                                      "LayerNormNode",
                                                      normalizedDimCount));
        HIPDNN_CHECK_ERROR(
            detail::validateLayernormAffineIfSet(scale, xDims, normalizedDimCount, "Scale tensor"));
        HIPDNN_CHECK_ERROR(
            detail::validateLayernormAffineIfSet(bias, xDims, normalizedDimCount, "Bias tensor"));

        if(attributes.get_forward_phase() != NormFwdPhase::INFERENCE)
        {
            HIPDNN_CHECK_ERROR(detail::validateLayernormStatsIfSet(
                attributes.get_mean(), xDims, normalizedDimCount, "Mean tensor"));
            HIPDNN_CHECK_ERROR(detail::validateLayernormStatsIfSet(attributes.get_inv_variance(),
                                                                   xDims,
                                                                   normalizedDimCount,
                                                                   "Inverse variance tensor"));
        }

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        const auto y = attributes.get_y();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "LayerNormNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "LayerNormNode missing y for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // Output Y has the same shape as input X
        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            if(!x->get_stride().empty())
            {
                y->set_stride(x->get_stride());
            }
            else
            {
                y->set_stride(hipdnn_data_sdk::utilities::generateStrides(y->get_dim()));
            }
        }

        // Infer unset scale and bias dims from x (see inferAffineDims) and their strides from
        // the stride order of x
        const auto scale = attributes.get_scale();
        const auto bias = attributes.get_bias();

        const auto inferAffineTensor = [&](const std::shared_ptr<TensorAttributes>& tensor) {
            if(!tensor || x->get_dim().empty())
            {
                return;
            }
            if(tensor->get_dim().empty())
            {
                tensor->set_dim(
                    inferAffineDims(x->get_dim(), attributes.get_normalized_dim_count()));
            }
            if(tensor->get_stride().empty())
            {
                const auto& dims = tensor->get_dim();
                if(!x->get_stride().empty() && x->get_stride().size() == dims.size())
                {
                    const auto strideOrder
                        = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                    tensor->set_stride(
                        hipdnn_data_sdk::utilities::generateStrides(dims, strideOrder));
                }
                else
                {
                    tensor->set_stride(hipdnn_data_sdk::utilities::generateStrides(dims));
                }
            }
        };

        inferAffineTensor(scale);
        inferAffineTensor(bias);

        // Normalized dim count: the explicit attribute if set, else derived from x and scale
        const bool shapesSet = scale && !scale->get_dim().empty() && !x->get_dim().empty();
        int64_t normalizedDimCount = 0;
        if(shapesSet)
        {
            HIPDNN_CHECK_ERROR(
                detail::deriveLayernormNormalizedDimCount(x->get_dim(),
                                                          scale->get_dim(),
                                                          attributes.get_normalized_dim_count(),
                                                          "LayerNormNode",
                                                          normalizedDimCount));
            attributes.set_normalized_dim_count(normalizedDimCount);
        }

        // Infer mean and inv_variance shapes (training phase outputs)
        // Mean and inv_variance hold the batch dims of x followed by 1s for the normalized dims
        auto mean = attributes.get_mean();
        auto invVariance = attributes.get_inv_variance();

        const auto inferStatsTensor = [&](std::shared_ptr<TensorAttributes>& tensorToInfer) {
            if(tensorToInfer && tensorToInfer->get_dim().empty() && shapesSet)
            {
                tensorToInfer->set_dim(
                    detail::inferLayernormStatsDims(x->get_dim(), normalizedDimCount));
            }

            if(tensorToInfer && tensorToInfer->get_stride().empty())
            {
                if(!x->get_stride().empty()
                   && x->get_stride().size() == tensorToInfer->get_dim().size())
                {
                    const auto strideOrder
                        = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                    tensorToInfer->set_stride(hipdnn_data_sdk::utilities::generateStrides(
                        tensorToInfer->get_dim(), strideOrder));
                }
                else
                {
                    tensorToInfer->set_stride(
                        hipdnn_data_sdk::utilities::generateStrides(tensorToInfer->get_dim()));
                }
            }
        };

        if(attributes.get_forward_phase() != NormFwdPhase::INFERENCE)
        {
            if(mean)
            {
                inferStatsTensor(mean);
            }

            if(invVariance)
            {
                inferStatsTensor(invVariance);
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createLayernormOperation(attributes, tensorDescs, operations);
    }

private:
    // Dims infer_properties_node() gives an unset scale or bias. Without an explicit
    // normalized_dim_count this is x with the batch dim set to 1, as cuDNN infers
    // ([N, C, H, W] gives [1, C, H, W]). With an explicit count k it is 1s followed by the
    // last k dims of x ([N, C, H, W] and k = 2 give [1, 1, H, W]), so the dims satisfy k.
    // An out-of-range count falls back to the first form; validation rejects the count.
    static std::vector<int64_t> inferAffineDims(const std::vector<int64_t>& xDims,
                                                int64_t explicitCount)
    {
        auto dims = xDims;
        const auto rank = static_cast<int64_t>(xDims.size());
        const int64_t leadingOnes
            = explicitCount > 0 && explicitCount <= rank ? rank - explicitCount : 1;
        std::fill(dims.begin(),
                  dims.begin() + static_cast<std::ptrdiff_t>(std::min(leadingOnes, rank)),
                  1);
        return dims;
    }
};
} // namespace hipdnn_frontend::graph
