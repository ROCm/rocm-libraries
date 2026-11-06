// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/MatmulProblem.hpp"

#include "common/GemmFormulation.hpp"

#include <limits>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

namespace hipblaslt_plugin
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// A validated operand: rank 2 or more, every extent positive, every stride non-negative.
struct Operand
{
    int64_t uid = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

std::optional<Operand>
    deviceOperand(const std::unordered_map<int64_t, const data_objects::TensorAttributes*>& tensors,
                  int64_t uid)
{
    const auto found = tensors.find(uid);
    if(found == tensors.end() || found->second == nullptr)
    {
        return std::nullopt;
    }
    const auto& tensor = *found->second;
    if(tensor.virtual_() || tensor.is_runtime_pass_by_value()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor)
       || tensor.ragged_offset_tensor_uid().has_value() || tensor.ragged_offset_multiplier() != 1)
    {
        return std::nullopt;
    }

    const auto dataType = tensor.data_type();
    if(dataType != data_objects::DataType::FLOAT && dataType != data_objects::DataType::HALF
       && dataType != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }

    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != strides->size() || dims->size() < 2)
    {
        return std::nullopt;
    }

    Operand operand{
        uid, dataType, {dims->begin(), dims->end()}, {strides->begin(), strides->end()}};
    for(size_t axis = 0; axis < operand.dims.size(); ++axis)
    {
        if(operand.dims[axis] < 1 || operand.strides[axis] < 0)
        {
            return std::nullopt;
        }
    }
    return operand;
}

/// The one stride between consecutive matrices of @p operand's batch, or nullopt when its
/// batch axes above extent 1 do not nest into a single stride. 0 when every batch axis is 1.
std::optional<int64_t> batchStride(const Operand& operand)
{
    const size_t batchAxes = operand.dims.size() - 2;
    std::optional<int64_t> stride;
    int64_t expected = 0;
    for(size_t axis = batchAxes; axis-- > 0;)
    {
        if(operand.dims[axis] == 1)
        {
            continue;
        }
        if(stride.has_value() && operand.strides[axis] != expected)
        {
            return std::nullopt;
        }
        if(!stride.has_value())
        {
            stride = operand.strides[axis];
        }
        expected = operand.strides[axis] * operand.dims[axis];
    }
    return stride.value_or(0);
}

bool allBatchAxesAreOne(const Operand& operand)
{
    for(size_t axis = 0; axis + 2 < operand.dims.size(); ++axis)
    {
        if(operand.dims[axis] != 1)
        {
            return false;
        }
    }
    return true;
}

/// The batch stride @p operand contributes to a batch shaped like @p output's: its own when its
/// batch axes match, 0 when it broadcasts one matrix. nullopt otherwise.
std::optional<int64_t> operandBatchStride(const Operand& operand, const Operand& output)
{
    if(allBatchAxesAreOne(operand))
    {
        return 0;
    }
    for(size_t axis = 0; axis + 2 < operand.dims.size(); ++axis)
    {
        if(operand.dims[axis] != output.dims[axis])
        {
            return std::nullopt;
        }
    }
    return batchStride(operand);
}

/// The matrices of @p operand's last two axes, batched by @p stride.
MatrixView matrixView(const Operand& operand, int64_t stride)
{
    const size_t rank = operand.dims.size();
    return MatrixView{operand.uid,
                      operand.dataType,
                      operand.dims[rank - 2],
                      operand.dims[rank - 1],
                      operand.strides[rank - 2],
                      operand.strides[rank - 1],
                      stride,
                      0};
}

} // namespace

std::optional<GemmCall>
    gemmForMatmul(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    // One node, and shapes fixed at plan build.
    if(graph.nodeCount() != 1 || graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }
    const auto& node = graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::MatmulAttributes
       || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& attributes = node.attributesAs<data_objects::MatmulAttributes>();

    // The output must not alias an input.
    if(attributes.c_tensor_uid() == attributes.a_tensor_uid()
       || attributes.c_tensor_uid() == attributes.b_tensor_uid())
    {
        return std::nullopt;
    }

    const auto& tensors = graph.getTensorMap();
    const auto a = deviceOperand(tensors, attributes.a_tensor_uid());
    const auto b = deviceOperand(tensors, attributes.b_tensor_uid());
    const auto c = deviceOperand(tensors, attributes.c_tensor_uid());
    if(!a.has_value() || !b.has_value() || !c.has_value())
    {
        return std::nullopt;
    }

    // One input element type; the output is the same or FLOAT.
    if(a->dataType != b->dataType
       || (c->dataType != a->dataType && c->dataType != data_objects::DataType::FLOAT))
    {
        return std::nullopt;
    }

    const size_t rank = a->dims.size();
    if(b->dims.size() != rank || c->dims.size() != rank)
    {
        return std::nullopt;
    }
    const int64_t m = a->dims[rank - 2];
    const int64_t k = a->dims[rank - 1];
    const int64_t n = b->dims[rank - 1];
    if(b->dims[rank - 2] != k || c->dims[rank - 2] != m || c->dims[rank - 1] != n)
    {
        return std::nullopt;
    }

    const auto aStride = operandBatchStride(*a, *c);
    const auto bStride = operandBatchStride(*b, *c);
    const auto cStride = batchStride(*c);
    if(!aStride.has_value() || !bStride.has_value() || !cStride.has_value())
    {
        return std::nullopt;
    }

    int64_t batchCount = 1;
    for(size_t axis = 0; axis + 2 < rank; ++axis)
    {
        batchCount *= c->dims[axis];
        if(batchCount > std::numeric_limits<int32_t>::max())
        {
            return std::nullopt;
        }
    }

    return gemmForViews(
        matrixView(*a, *aStride), matrixView(*b, *bStride), matrixView(*c, *cStride), batchCount);
}

} // namespace hipblaslt_plugin
