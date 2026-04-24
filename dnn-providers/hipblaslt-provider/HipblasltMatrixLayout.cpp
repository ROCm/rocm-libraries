// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltMatrixLayout.hpp"
#include "HipblasltUtils.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hipblaslt_plugin
{

HipblasltMatrixLayout::HipblasltMatrixLayout(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tensor)
    : _uid(tensor.uid())
{
    const auto& dims = tensor.dims();
    const auto& strides = tensor.strides();
    const auto rank = dims.size();

    PLUGIN_THROW_IF_TRUE(rank < 2,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor rank must be at least 2 for matrix layout. UID: "
                             + std::to_string(_uid));

    // Different types of matrices require different initialization of layout:
    // - If this is column-major matrix (strides[rank-2] == 1):
    //   we need to use the dimensions and the strides as-is since hipBLASLT expects column-major layout.
    // - If this is row-major matrix (strides[rank-1] == 1):
    //   we need to swap the dimensions and the strides.
    uint64_t rows, cols;
    int64_t ld;
    if(strides[rank - 1] == 1) // row-major matrix
    {
        rows = static_cast<uint64_t>(dims[rank - 1]);
        cols = static_cast<uint64_t>(dims[rank - 2]);
        ld = strides[rank - 2]; // row stride
    }
    else if(strides[strides.size() - 2] == 1) // column-major matrix
    {
        rows = static_cast<uint64_t>(dims[rank - 2]);
        cols = static_cast<uint64_t>(dims[rank - 1]);
        ld = strides[rank - 1]; // column stride
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported matrix: only column major and row major matrices are supported");
    }

    THROW_ON_HIPBLASLT_FAILURE(
        hipblasLtMatrixLayoutCreate(&_matrix_layout,
                                    hipblaslt_utils::tensorDataTypeToHipDataType(tensor.dataType()),
                                    rows,
                                    cols,
                                    ld));
}

HipblasltMatrixLayout::HipblasltMatrixLayout(HipblasltMatrixLayout&& other) noexcept
    : _uid(other._uid)
    , _matrix_layout(other._matrix_layout)
{
    other._matrix_layout = nullptr;
}

HipblasltMatrixLayout& HipblasltMatrixLayout::operator=(HipblasltMatrixLayout&& other) noexcept
{
    if(this != &other)
    {
        if(_matrix_layout != nullptr)
        {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutDestroy(_matrix_layout));
        }

        _uid = other._uid;
        _matrix_layout = other._matrix_layout;

        other._matrix_layout = nullptr;
    }
    return *this;
}

HipblasltMatrixLayout::~HipblasltMatrixLayout()
{
    if(_matrix_layout != nullptr)
    {
        LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutDestroy(_matrix_layout));
    }
}

int64_t HipblasltMatrixLayout::uid() const
{
    return _uid;
}

hipblasLtMatrixLayout_t HipblasltMatrixLayout::matrixLayout() const
{
    return _matrix_layout;
}

void HipblasltMatrixLayout::setBatchCount(int64_t count)
{
    PLUGIN_THROW_IF_NULL(_matrix_layout,
                         HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "Failed to set batch count: matrixLayout is nullptr");
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutSetAttribute(
        _matrix_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
}

void HipblasltMatrixLayout::setStridedBatchOffset(int64_t stride)
{
    PLUGIN_THROW_IF_NULL(_matrix_layout,
                         HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "Failed to set strided batch offset: matrixLayout is nullptr");
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutSetAttribute(
        _matrix_layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
}

} // namespace hipblaslt_plugin
