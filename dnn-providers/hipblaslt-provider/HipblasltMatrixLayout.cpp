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

    // Different types of matrices require different initialization of layout
    // (orientation per hipblaslt_utils::matrixOrder, which ignores the stride of
    // a size-1 dim):
    // - Column-major matrix: use the dimensions and the strides as-is since
    //   hipBLASLt expects column-major layout.
    // - Row-major matrix: swap the dimensions and the strides.
    // The leading dimension comes from hipblaslt_utils::leadingDimension.
    uint64_t rows;
    uint64_t cols;
    const auto order = hipblaslt_utils::matrixOrder(dims, strides);
    if(order == hipblaslt_utils::MatrixOrder::ROW_MAJOR)
    {
        rows = static_cast<uint64_t>(dims[rank - 1]);
        cols = static_cast<uint64_t>(dims[rank - 2]);
    }
    else if(order == hipblaslt_utils::MatrixOrder::COLUMN_MAJOR)
    {
        rows = static_cast<uint64_t>(dims[rank - 2]);
        cols = static_cast<uint64_t>(dims[rank - 1]);
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported matrix: only column major and row major matrices are supported");
    }

    // hipBLASLt does not reject every negative ld (-1 passes) and reads out of bounds with it.
    // A zero ld is computed correctly for an operand whose rows or columns repeat; the output
    // additionally rejects it in hipblaslt_utils::requireRowMajorOutput. Only the ld this layout
    // actually uses is checked; the stride of a size-1 dim is ignored.
    const auto ld = hipblaslt_utils::leadingDimension(tensor);
    PLUGIN_THROW_IF_TRUE(ld < 0,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Leading dimension must not be negative, got " + std::to_string(ld)
                             + ". UID: " + std::to_string(_uid));

    THROW_ON_HIPBLASLT_FAILURE(
        hipblasLtMatrixLayoutCreate(&_matrixLayout,
                                    hipblaslt_utils::tensorDataTypeToHipDataType(tensor.dataType()),
                                    rows,
                                    cols,
                                    ld));
}

HipblasltMatrixLayout::HipblasltMatrixLayout(hipDataType dataType,
                                             uint64_t rows,
                                             uint64_t cols,
                                             int64_t ld)
{
    THROW_ON_HIPBLASLT_FAILURE(
        hipblasLtMatrixLayoutCreate(&_matrixLayout, dataType, rows, cols, ld));
}

HipblasltMatrixLayout::HipblasltMatrixLayout(HipblasltMatrixLayout&& other) noexcept
    : _uid(other._uid)
    , _matrixLayout(other._matrixLayout)
{
    other._matrixLayout = nullptr;
}

HipblasltMatrixLayout& HipblasltMatrixLayout::operator=(HipblasltMatrixLayout&& other) noexcept
{
    if(this != &other)
    {
        if(_matrixLayout != nullptr)
        {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutDestroy(_matrixLayout));
        }

        _uid = other._uid;
        _matrixLayout = other._matrixLayout;

        other._matrixLayout = nullptr;
    }
    return *this;
}

HipblasltMatrixLayout::~HipblasltMatrixLayout()
{
    if(_matrixLayout != nullptr)
    {
        LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutDestroy(_matrixLayout));
    }
}

int64_t HipblasltMatrixLayout::uid() const
{
    return _uid;
}

hipblasLtMatrixLayout_t HipblasltMatrixLayout::matrixLayout() const
{
    return _matrixLayout;
}

void HipblasltMatrixLayout::setBatchCount(int64_t count)
{
    PLUGIN_THROW_IF_NULL(_matrixLayout,
                         HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "Failed to set batch count: matrixLayout is nullptr");
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutSetAttribute(
        _matrixLayout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
}

void HipblasltMatrixLayout::setStridedBatchOffset(int64_t stride)
{
    PLUGIN_THROW_IF_NULL(_matrixLayout,
                         HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "Failed to set strided batch offset: matrixLayout is nullptr");
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutSetAttribute(
        _matrixLayout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
}

} // namespace hipblaslt_plugin
