// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipblaslt_plugin::test
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

inline constexpr int64_t A_UID = 1;
inline constexpr int64_t B_UID = 2;
inline constexpr int64_t C_UID = 3;

/// One matmul operand, field for field as a TensorAttributes table carries it.
struct MatmulTensorSpec
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dataType = data_objects::DataType::FLOAT;
    bool isVirtual = false;
    bool runtimePassByValue = false;
};

/// A single-node matmul graph, field for field.
struct MatmulGraphSpec
{
    MatmulTensorSpec a;
    MatmulTensorSpec b;
    MatmulTensorSpec c;
    int64_t cUid = C_UID;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    bool overrideShapeEnabled = false;
};

/// Packed strides for a [..., rows, cols] tensor, its matrix row-major or column-major.
inline std::vector<int64_t> matrixStrides(const std::vector<int64_t>& dims, bool columnMajor)
{
    const size_t rank = dims.size();
    std::vector<int64_t> strides(rank, 0);
    const int64_t rows = dims[rank - 2];
    const int64_t cols = dims[rank - 1];
    strides[rank - 1] = columnMajor ? rows : 1;
    strides[rank - 2] = columnMajor ? 1 : cols;
    int64_t running = rows * cols;
    for(size_t axis = rank - 2; axis-- > 0;)
    {
        strides[axis] = running;
        running *= dims[axis];
    }
    return strides;
}

/// A consistent spec: a[batch, m, k] x b[batch, k, n] = c[batch, m, n], each matrix in the
/// given storage order. An empty @p aBatch or @p bBatch broadcasts that operand.
inline MatmulGraphSpec matmulSpec(int64_t m,
                                  int64_t n,
                                  int64_t k,
                                  const std::vector<int64_t>& batch,
                                  bool aColumnMajor,
                                  bool bColumnMajor,
                                  bool cColumnMajor,
                                  bool broadcastA = false,
                                  bool broadcastB = false)
{
    const auto shape = [&batch](bool broadcast, int64_t rows, int64_t cols) {
        std::vector<int64_t> dims;
        for(const auto extent : batch)
        {
            dims.push_back(broadcast ? 1 : extent);
        }
        dims.push_back(rows);
        dims.push_back(cols);
        return dims;
    };

    MatmulGraphSpec spec;
    spec.a.dims = shape(broadcastA, m, k);
    spec.b.dims = shape(broadcastB, k, n);
    spec.c.dims = shape(false, m, n);
    spec.a.strides = matrixStrides(spec.a.dims, aColumnMajor);
    spec.b.strides = matrixStrides(spec.b.dims, bColumnMajor);
    spec.c.strides = matrixStrides(spec.c.dims, cColumnMajor);
    return spec;
}

inline flatbuffers::Offset<data_objects::TensorAttributes> serializeMatmulTensor(
    flatbuffers::FlatBufferBuilder& builder, int64_t uid, const MatmulTensorSpec& tensor)
{
    return data_objects::CreateTensorAttributesDirect(builder,
                                                      uid,
                                                      nullptr,
                                                      tensor.dataType,
                                                      &tensor.strides,
                                                      &tensor.dims,
                                                      tensor.isVirtual,
                                                      data_objects::TensorValue::NONE,
                                                      0,
                                                      tensor.runtimePassByValue);
}

inline flatbuffers::FlatBufferBuilder buildMatmulGraph(const MatmulGraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(serializeMatmulTensor(builder, A_UID, spec.a));
    tensors.push_back(serializeMatmulTensor(builder, B_UID, spec.b));
    if(spec.cUid != A_UID && spec.cUid != B_UID)
    {
        tensors.push_back(serializeMatmulTensor(builder, spec.cUid, spec.c));
    }

    const auto attributes = data_objects::CreateMatmulAttributes(builder, A_UID, B_UID, spec.cUid);
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "matmul",
                                                   spec.computeType,
                                                   data_objects::NodeAttributes::MatmulAttributes,
                                                   attributes.Union()));

    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "rocblas_matmul_test",
                                                   data_objects::DataType::UNSET,
                                                   data_objects::DataType::UNSET,
                                                   data_objects::DataType::UNSET,
                                                   &tensors,
                                                   &nodes,
                                                   flatbuffers::nullopt,
                                                   spec.overrideShapeEnabled));
    return builder;
}

/// Owns a built graph and reads it the way the provider does.
class MatmulGraph
{
public:
    explicit MatmulGraph(const MatmulGraphSpec& spec)
        : _builder(buildMatmulGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph() const
    {
        return _graph;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
};

} // namespace hipblaslt_plugin::test
