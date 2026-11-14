// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenConvApplicabilityChecks.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenUtils.hpp"

namespace miopen_plugin::conv_applicability
{

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;

[[noreturn]] void decline(const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, message);
}

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>* values)
{
    if(values == nullptr)
    {
        return {};
    }
    return {values->begin(), values->end()};
}

void checkBiasIsPerChannel(const TensorAttributes& bias, const TensorAttributes& convOutput)
{
    const auto biasDims = toVector(bias.dims());
    const auto biasStrides = toVector(bias.strides());
    const auto outputDims = toVector(convOutput.dims());

    if(outputDims.size() < 2 || biasDims.size() != outputDims.size()
       || biasStrides.size() != biasDims.size())
    {
        decline("Bias tensor must have the same rank as the convolution output");
    }

    for(size_t i = 0; i < biasDims.size(); ++i)
    {
        const int64_t expected = (i == 1) ? outputDims[1] : 1;
        if(biasDims[i] != expected)
        {
            decline("Bias tensor must be a per-channel tensor with dims [1, C, 1, ...] where C is "
                    "the convolution output channel count");
        }
    }

    if(biasDims[1] > 1 && biasStrides[1] != 1)
    {
        decline("Bias tensor channels must be contiguous");
    }
}

// Packed strides for `outputDims` laid out in the dimension order MIOpen derives for the input
// tensor (TensorDescriptor::find_permutation): dimensions sorted by (stride, length)
// descending, stable on ties. MIOpen's fused execute compares whole tensor descriptors, so the
// caller compares the strides of unit dimensions as well.
std::vector<int64_t> packedStridesInLayoutOrder(const std::vector<int64_t>& inputDims,
                                                const std::vector<int64_t>& inputStrides,
                                                const std::vector<int64_t>& outputDims)
{
    std::vector<size_t> order(outputDims.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::tie(inputStrides[a], inputDims[a]) > std::tie(inputStrides[b], inputDims[b]);
    });

    std::vector<int64_t> strides(outputDims.size(), 0);
    int64_t running = 1;
    for(auto it = order.rbegin(); it != order.rend(); ++it)
    {
        strides[*it] = running;
        running *= outputDims[*it];
    }
    return strides;
}

void checkOutputMatchesFusedDescriptor(const TensorAttributes& input,
                                       const TensorAttributes& convOutput,
                                       const TensorAttributes& output)
{
    if(output.data_type() != input.data_type())
    {
        decline("Fused convolution output must have the same data type as the input");
    }

    const auto outputDims = toVector(output.dims());
    if(outputDims != toVector(convOutput.dims()))
    {
        decline("Fused convolution output dims must equal the convolution output dims");
    }

    const auto inputDims = toVector(input.dims());
    const auto inputStrides = toVector(input.strides());
    const auto outputStrides = toVector(output.strides());
    if(inputDims.size() != outputDims.size() || inputStrides.size() != outputDims.size()
       || outputStrides.size() != outputDims.size())
    {
        decline("Fused convolution input and output must have the same rank");
    }

    const auto expected = packedStridesInLayoutOrder(inputDims, inputStrides, outputDims);
    for(size_t i = 0; i < outputDims.size(); ++i)
    {
        if(outputStrides[i] != expected[i])
        {
            decline("Fused convolution output must be packed in the same layout as the input");
        }
    }
}

} // namespace

bool hasPassByValueTensor(const TensorMap& tensorMap)
{
    return std::any_of(tensorMap.begin(), tensorMap.end(), [](const auto& entry) {
        return hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(entry.second);
    });
}

void checkConvFwdBiasActivLayouts(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr,
    const TensorMap& tensorMap)
{
    const auto& convOutput = miopen_utils::findTensorAttributes(tensorMap, convAttr.y_tensor_uid());

    if(biasAttr != nullptr)
    {
        const auto biasUid = biasAttr->in_0_tensor_uid() == convAttr.y_tensor_uid()
                                 ? biasAttr->in_1_tensor_uid().value()
                                 : biasAttr->in_0_tensor_uid();
        checkBiasIsPerChannel(miopen_utils::findTensorAttributes(tensorMap, biasUid), convOutput);
    }

    checkOutputMatchesFusedDescriptor(
        miopen_utils::findTensorAttributes(tensorMap, convAttr.x_tensor_uid()),
        convOutput,
        miopen_utils::findTensorAttributes(tensorMap, activAttr.out_0_tensor_uid()));
}

} // namespace miopen_plugin::conv_applicability
