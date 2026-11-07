// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/GemmFormulation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace hipblaslt_plugin
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();

/// How a view's matrices are stored, as BLAS sees them.
struct MatrixLayout
{
    /// The column-major reading of the storage is the matrix itself (true) or its transpose.
    bool columnMajor = false;
    int64_t leadingDimension = 0;
};

/// The layout of @p view, or nullopt when neither of its axes is unit-stride.
std::optional<MatrixLayout> matrixLayout(const MatrixView& view)
{
    if(view.rows < 1 || view.cols < 1 || view.rowStride < 0 || view.colStride < 0)
    {
        return std::nullopt;
    }
    // Row-major: the column-major reading is the cols x rows transpose, columns rowStride
    // apart. A single row touches one column of it, so its leading dimension is free.
    if(view.colStride == 1 && (view.rows == 1 || view.rowStride >= view.cols))
    {
        return MatrixLayout{false, view.rows == 1 ? view.cols : view.rowStride};
    }
    // Column-major: the reading is the matrix, columns colStride apart.
    if(view.rowStride == 1 && (view.cols == 1 || view.colStride >= view.rows))
    {
        return MatrixLayout{true, view.cols == 1 ? view.rows : view.colStride};
    }
    return std::nullopt;
}

GemmOperand gemmOperand(const MatrixView& view, const MatrixLayout& layout)
{
    return GemmOperand{
        view.uid, view.dataType, layout.leadingDimension, view.batchStride, view.offset};
}

/// True when no two elements of the batched view share an address: its (batch, row, column)
/// axes, sorted by stride, each step at least past everything inside it. Batches may
/// interleave, as grouped channels-last convolutions do, as long as they stay disjoint.
bool hasDistinctElements(const MatrixView& view, int64_t batchCount)
{
    std::array<std::pair<int64_t, int64_t>, 3> axes{std::make_pair(batchCount, view.batchStride),
                                                    std::make_pair(view.rows, view.rowStride),
                                                    std::make_pair(view.cols, view.colStride)};
    std::sort(axes.begin(), axes.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
    });
    int64_t span = 1;
    for(const auto& [extent, stride] : axes)
    {
        if(extent <= 1)
        {
            continue;
        }
        if(stride < span || __builtin_mul_overflow(stride, extent, &span))
        {
            return false;
        }
    }
    return true;
}

bool isGemmElementType(data_objects::DataType dataType)
{
    return dataType == data_objects::DataType::FLOAT || dataType == data_objects::DataType::HALF
           || dataType == data_objects::DataType::BFLOAT16;
}

} // namespace

std::optional<GemmCall>
    gemmForViews(const MatrixView& a, const MatrixView& b, const MatrixView& c, int64_t batchCount)
{
    if(a.rows != c.rows || b.cols != c.cols || a.cols != b.rows || batchCount < 1
       || batchCount > INT32_LIMIT)
    {
        return std::nullopt;
    }
    if(!isGemmElementType(a.dataType) || a.dataType != b.dataType
       || (c.dataType != a.dataType && c.dataType != data_objects::DataType::FLOAT))
    {
        return std::nullopt;
    }
    if(a.batchStride < 0 || b.batchStride < 0 || c.batchStride < 0 || a.offset < 0 || b.offset < 0
       || c.offset < 0)
    {
        return std::nullopt;
    }

    const auto aLayout = matrixLayout(a);
    const auto bLayout = matrixLayout(b);
    const auto cLayout = matrixLayout(c);
    if(!aLayout.has_value() || !bLayout.has_value() || !cLayout.has_value())
    {
        return std::nullopt;
    }

    GemmCall call;
    call.batchCount = batchCount;
    call.k = a.cols;
    if(cLayout->columnMajor)
    {
        // c = a b: an operand's column-major reading is itself when column-major, its transpose
        // otherwise.
        call.m = c.rows;
        call.n = c.cols;
        call.transposeA = !aLayout->columnMajor;
        call.transposeB = !bLayout->columnMajor;
        call.a = gemmOperand(a, *aLayout);
        call.b = gemmOperand(b, *bLayout);
    }
    else
    {
        // c^T = b^T a^T: a row-major operand's column-major reading is already its transpose.
        call.m = c.cols;
        call.n = c.rows;
        call.transposeA = bLayout->columnMajor;
        call.transposeB = aLayout->columnMajor;
        call.a = gemmOperand(b, *bLayout);
        call.b = gemmOperand(a, *aLayout);
    }
    call.d = gemmOperand(c, *cLayout);

    // Every element of every output matrix is its own memory, or the GEMM races with itself.
    if(!hasDistinctElements(c, batchCount))
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

std::vector<GemmCall> splitWideBatches(std::vector<GemmCall> calls)
{
    std::vector<GemmCall> split;
    split.reserve(calls.size());
    for(auto& call : calls)
    {
        const bool wide = call.batchCount > 1
                          && std::max({call.a.batchStride, call.b.batchStride, call.d.batchStride})
                                 > MAX_BATCH_STRIDE;
        if(!wide)
        {
            split.push_back(std::move(call));
            continue;
        }
        for(int64_t matrix = 0; matrix < call.batchCount; ++matrix)
        {
            GemmCall one = call;
            one.batchCount = 1;
            for(auto* operand : {&one.a, &one.b, &one.d})
            {
                operand->offset += matrix * operand->batchStride;
                operand->batchStride = 0;
            }
            split.push_back(one);
        }
    }
    return split;
}

} // namespace hipblaslt_plugin
