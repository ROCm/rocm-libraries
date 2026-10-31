// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/ResampleBackwardInputs.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleFwd.hpp>

namespace hipdnn_integration_tests
{
namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::ShallowTensor;
using hipdnn_data_sdk::utilities::Tensor;

const fb::TensorAttributes* findTensor(const fb::Graph& graph, int64_t uid)
{
    if(graph.tensors() == nullptr)
    {
        return nullptr;
    }
    for(const auto* tensor : *graph.tensors())
    {
        if(tensor->uid() == uid)
        {
            return tensor;
        }
    }
    return nullptr;
}

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>* values)
{
    if(values == nullptr)
    {
        return {};
    }
    return {values->begin(), values->end()};
}

} // anonymous namespace

std::optional<std::string> deriveResampleBackwardIndex(const fb::Graph& graph,
                                                       const fb::ResampleBwdAttributes& attributes,
                                                       InputTensorMap& inputs,
                                                       const std::vector<int64_t>& ownedUids,
                                                       unsigned int seed)
{
    if(attributes.resample_mode() != fb::ResampleMode::MAXPOOL
       || !attributes.index_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    const int64_t indexUid = *attributes.index_tensor_uid();
    if(std::find(ownedUids.begin(), ownedUids.end(), indexUid) == ownedUids.end())
    {
        return std::nullopt;
    }

    const auto* indexAttributes = findTensor(graph, indexUid);
    if(indexAttributes == nullptr)
    {
        return "the index (uid " + std::to_string(indexUid) + ") is not in the graph";
    }
    if(indexAttributes->data_type() != fb::DataType::INT32)
    {
        return std::string("the references do not take a ")
               + fb::EnumNameDataType(indexAttributes->data_type()) + " index";
    }
    const auto* dxAttributes = findTensor(graph, attributes.dx_tensor_uid());
    if(dxAttributes == nullptr)
    {
        return "dx (uid " + std::to_string(attributes.dx_tensor_uid()) + ") is not in the graph";
    }

    auto& index = *inputs.at(indexUid);

    // The reference's own checks decide which shapes and parameters can be derived. It
    // validates before writing anything, so a rejected graph keeps its filled index for
    // the backward reference to fail.
    try
    {
        Tensor<float> x(toVector(dxAttributes->dims()), toVector(dxAttributes->strides()));
        x.fillWithRandomValues(-1.0f, 1.0f, seed);
        Tensor<float> y(index.dims(), index.strides());
        ShallowTensor<int32_t> indexView(index.rawHostData(), index.dims(), index.strides());

        hipdnn_test_sdk::utilities::CpuFpReferenceResampleFwd::
            forward<float, float, float, int32_t>(x,
                                                  y,
                                                  toVector(attributes.pre_padding()),
                                                  toVector(attributes.stride()),
                                                  toVector(attributes.window()),
                                                  fb::ResampleMode::MAXPOOL,
                                                  attributes.padding_mode(),
                                                  &indexView);
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }

    // The view writes the host copy; a device-resident tensor uploads it only once told
    // the host copy is the current one.
    index.markHostModified();
    return std::nullopt;
}

} // namespace hipdnn_integration_tests
