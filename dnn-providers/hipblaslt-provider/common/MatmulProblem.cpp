// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/MatmulProblem.hpp"

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

constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();

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

/// How a rows x cols matrix is stored, as BLAS sees it.
struct MatrixLayout
{
    /// The column-major view of the storage is the matrix itself (true) or its transpose.
    bool columnMajor = false;
    int64_t leadingDimension = 0;
};

/// The layout of the last two axes of @p operand, or nullopt when neither is unit-stride.
std::optional<MatrixLayout> matrixLayout(const Operand& operand)
{
    const size_t rank = operand.dims.size();
    const int64_t rows = operand.dims[rank - 2];
    const int64_t cols = operand.dims[rank - 1];
    const int64_t rowStride = operand.strides[rank - 2];
    const int64_t colStride = operand.strides[rank - 1];

    // Row-major: the column-major view is the cols x rows transpose, columns rowStride apart.
    // A single row touches one column of that view, so its leading dimension is free.
    if(colStride == 1 && (rows == 1 || rowStride >= cols))
    {
        return MatrixLayout{false, rows == 1 ? cols : rowStride};
    }
    // Column-major: the view is the matrix, columns colStride apart.
    if(rowStride == 1 && (cols == 1 || colStride >= rows))
    {
        return MatrixLayout{true, cols == 1 ? rows : colStride};
    }
    return std::nullopt;
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

GemmOperand gemmOperand(const Operand& operand, const MatrixLayout& layout, int64_t stride)
{
    return GemmOperand{operand.uid, operand.dataType, layout.leadingDimension, stride};
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

    const auto aLayout = matrixLayout(*a);
    const auto bLayout = matrixLayout(*b);
    const auto cLayout = matrixLayout(*c);
    const auto aStride = operandBatchStride(*a, *c);
    const auto bStride = operandBatchStride(*b, *c);
    const auto cStride = batchStride(*c);
    if(!aLayout.has_value() || !bLayout.has_value() || !cLayout.has_value() || !aStride.has_value()
       || !bStride.has_value() || !cStride.has_value())
    {
        return std::nullopt;
    }

    int64_t batchCount = 1;
    for(size_t axis = 0; axis + 2 < rank; ++axis)
    {
        batchCount *= c->dims[axis];
        if(batchCount > INT32_LIMIT)
        {
            return std::nullopt;
        }
    }

    GemmCall call;
    call.batchCount = batchCount;
    call.k = k;
    if(cLayout->columnMajor)
    {
        // c = a b: a's column-major view is a when a is column-major, a^T otherwise.
        call.m = m;
        call.n = n;
        call.transposeA = !aLayout->columnMajor;
        call.transposeB = !bLayout->columnMajor;
        call.a = gemmOperand(*a, *aLayout, *aStride);
        call.b = gemmOperand(*b, *bLayout, *bStride);
    }
    else
    {
        // c^T = b^T a^T: a row-major operand's column-major view is already its transpose.
        call.m = n;
        call.n = m;
        call.transposeA = bLayout->columnMajor;
        call.transposeB = aLayout->columnMajor;
        call.a = gemmOperand(*b, *bLayout, *bStride);
        call.b = gemmOperand(*a, *aLayout, *aStride);
    }
    call.d = gemmOperand(*c, *cLayout, *cStride);

    // A written batch whose matrices overlap is a race.
    const int64_t outputSpan = call.d.leadingDimension * call.n;
    if(batchCount > 1 && call.d.batchStride < outputSpan)
    {
        return std::nullopt;
    }

    for(const int64_t value : {call.m,
                               call.n,
                               call.k,
                               call.a.leadingDimension,
                               call.b.leadingDimension,
                               call.d.leadingDimension})
    {
        if(value > INT32_LIMIT)
        {
            return std::nullopt;
        }
    }
    return call;
}

} // namespace hipblaslt_plugin
