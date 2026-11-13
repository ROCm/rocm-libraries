// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/GemmFormulation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string_view>
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

    // Leading dimensions go to the libraries as they are. Sizes past 32 bits are split by
    // fitLibraryLimits, except a reduction into a narrower output, whose chunks would round
    // between them.
    for(const int64_t value :
        {call.a.leadingDimension, call.b.leadingDimension, call.d.leadingDimension})
    {
        if(value > INT32_LIMIT)
        {
            return std::nullopt;
        }
    }
    if(call.k > INT32_LIMIT && call.d.dataType != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    return call;
}

namespace
{

/// The axes fitLibraryLimits splits a call along. Never m: a column-major d's leading dimension
/// is at least m, so an m past 32 bits already fails gemmForViews.
enum class SplitAxis
{
    BATCH,
    N,
    K,
};

int64_t& extentOf(GemmCall& call, SplitAxis axis)
{
    return axis == SplitAxis::N ? call.n : axis == SplitAxis::K ? call.k : call.batchCount;
}

/// @p call's operands advanced by @p count matrices, columns (n) or reduction elements (k).
void advance(GemmCall& call, SplitAxis axis, int64_t count)
{
    switch(axis)
    {
    case SplitAxis::BATCH:
        for(auto* operand : {&call.a, &call.b, &call.d})
        {
            operand->offset += count * operand->batchStride;
        }
        break;
    case SplitAxis::N:
        // op(b) is k x n: stored k x n or, transposed, n x k (rows unit stride).
        call.b.offset += call.transposeB ? count : count * call.b.leadingDimension;
        call.d.offset += count * call.d.leadingDimension;
        break;
    case SplitAxis::K:
    default:
        // op(a) is m x k: stored m x k (rows unit stride) or, transposed, k x m.
        call.a.offset += call.transposeA ? count : count * call.a.leadingDimension;
        call.b.offset += call.transposeB ? count * call.b.leadingDimension : count;
        break;
    }
}

/// @p calls with each split along @p axis into chunks of at most @p limit.
std::vector<GemmCall> splitAlong(std::vector<GemmCall> calls, SplitAxis axis, int64_t limit)
{
    std::vector<GemmCall> split;
    split.reserve(calls.size());
    for(auto& call : calls)
    {
        const int64_t whole = extentOf(call, axis);
        if(whole <= limit)
        {
            split.push_back(std::move(call));
            continue;
        }
        for(int64_t first = 0; first < whole; first += limit)
        {
            GemmCall chunk = call;
            advance(chunk, axis, first);
            extentOf(chunk, axis) = std::min(limit, whole - first);
            if(axis == SplitAxis::BATCH)
            {
                chunk.a.batchStride = chunk.b.batchStride = chunk.d.batchStride = 0;
            }
            // Later reduction chunks add to the first (fp32 only: gemmForViews).
            chunk.accumulate = call.accumulate || (axis == SplitAxis::K && first > 0);
            split.push_back(chunk);
        }
    }
    return split;
}

} // namespace

bool rocblasRoundsReductionChunks(const GemmCall& call)
{
    if(call.d.dataType == data_objects::DataType::FLOAT)
    {
        return false;
    }
    // rocBLAS's rule (Tensile gemm): k walks lda when a is not transposed, ldb when b is.
    const int64_t stride = std::max(call.transposeA ? int64_t{1} : call.a.leadingDimension,
                                    call.transposeB ? call.b.leadingDimension : int64_t{1});
    int64_t strideBytes = 0;
    if(__builtin_mul_overflow(stride, elementBytes(call.a.dataType), &strideBytes))
    {
        return call.k > 1;
    }
    const int64_t chunk
        = strideBytes > 0 ? std::max(int64_t{1}, ROCBLAS_REDUCTION_BYTES / strideBytes) : call.k;
    return call.k > chunk;
}

bool rocblasGemvRoundsBf16Products(const GemmCall& call, std::string_view gcnArchName)
{
    if(call.a.dataType != data_objects::DataType::BFLOAT16 || call.k == 0
       || (call.m != 1 && call.n != 1))
    {
        return false;
    }
    // rocblas_use_gemv_in_gemm: gfx11 takes every GEMV-shaped GEMM but a 1 x 1 output;
    // gfx906/908/90a and gfx10 apply the bf16 rules.
    const auto arch = gcnArchName.substr(0, gcnArchName.find(':'));
    const bool gfx11 = arch.rfind("gfx11", 0) == 0;
    const bool gfx10 = arch.rfind("gfx10", 0) == 0;
    const bool cdnaGemv = arch == "gfx906" || arch == "gfx908" || arch == "gfx90a";
    const bool nn = !call.transposeA && !call.transposeB;
    const bool nt = !call.transposeA && call.transposeB;
    const bool tn = call.transposeA && !call.transposeB;
    bool gemv = false;
    if(gfx11 && !(call.m == 1 && call.n == 1))
    {
        gemv = true;
    }
    else if(gfx11 || gfx10 || cdnaGemv)
    {
        gemv = !((call.n == 1 && (nn || tn))
                 || (call.n == 1 && nt && (arch == "gfx908" || arch == "gfx90a"))
                 || (call.n == 1 && call.b.leadingDimension > 8) || (call.m == 1 && (nn || tn)));
    }
    if(!gemv)
    {
        return false;
    }
    // rocblas_gemv_in_gemm_launcher: n == 1 runs a GEMV transposed like a; m == 1 one
    // transposed when b is not. A transposed GEMV of k <= 64 rows over more than 8 batch
    // entries is rocblas_gemvtsm_kernel.
    const bool transposedGemv = call.n == 1 ? call.transposeA : !call.transposeB;
    return transposedGemv && call.k <= 64 && call.batchCount > 8;
}

std::vector<GemmCall> fitLibraryLimits(std::vector<GemmCall> calls)
{
    std::vector<GemmCall> fitted;
    fitted.reserve(calls.size());
    for(auto& call : calls)
    {
        // One call per matrix when a batch stride is too wide (a batch of 1 passes none), then
        // n and k in chunks the libraries' 32-bit sizes hold, in order, so a later chunk
        // that adds to an earlier one runs after it.
        const bool wideBatch
            = call.batchCount > 1
              && std::max({call.a.batchStride, call.b.batchStride, call.d.batchStride})
                     > MAX_BATCH_STRIDE;
        std::vector<GemmCall> pieces{std::move(call)};
        if(wideBatch)
        {
            pieces = splitAlong(std::move(pieces), SplitAxis::BATCH, 1);
        }
        for(const auto axis : {SplitAxis::N, SplitAxis::K})
        {
            pieces = splitAlong(std::move(pieces), axis, MAX_GEMM_EXTENT);
        }
        for(auto& piece : pieces)
        {
            fitted.push_back(std::move(piece));
        }
    }
    return fitted;
}

} // namespace hipblaslt_plugin
