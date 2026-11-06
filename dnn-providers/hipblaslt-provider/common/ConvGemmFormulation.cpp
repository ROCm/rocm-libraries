// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/ConvGemmFormulation.hpp"

#include <utility>

#include "common/GemmFormulation.hpp"

namespace hipblaslt_plugin
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

std::vector<size_t> spatialAxes(const ConvOperand& operand)
{
    std::vector<size_t> axes;
    for(size_t axis = 2; axis < operand.dims.size(); ++axis)
    {
        axes.push_back(axis);
    }
    return axes;
}

MatrixView view(const ConvProblem& problem,
                const ConvOperand& operand,
                int64_t rows,
                int64_t cols,
                int64_t rowStride,
                int64_t colStride,
                int64_t batchStride,
                int64_t offset = 0)
{
    return MatrixView{operand.uid,
                      operand.dataType.value_or(problem.dataType),
                      rows,
                      cols,
                      rowStride,
                      colStride,
                      batchStride,
                      operand.offset + offset};
}

std::optional<std::vector<GemmCall>> single(const std::optional<GemmCall>& call)
{
    if(!call.has_value())
    {
        return std::nullopt;
    }
    return std::vector<GemmCall>{*call};
}

/// ONE_BY_ONE with pixels as GEMM rows: one call, batched over groups. Reads channels-last
/// tensors, whose (n, spatial) axes nest into one pixel stride.
std::optional<std::vector<GemmCall>> oneByOnePixelRows(const ConvProblem& problem)
{
    std::vector<size_t> pixelAxes{0};
    for(const auto axis : spatialAxes(problem.x))
    {
        pixelAxes.push_back(axis);
    }
    const auto xPixels = collapseAxes(problem.x, pixelAxes);
    const auto yPixels = collapseAxes(problem.y, pixelAxes);
    if(!xPixels.has_value() || !yPixels.has_value())
    {
        return std::nullopt;
    }
    const int64_t pixels = xPixels->first;
    const int64_t px = xPixels->second;
    const int64_t py = yPixels->second;
    const int64_t xc = problem.x.strides[1];
    const int64_t yk = problem.y.strides[1];
    const int64_t wk = problem.w.strides[0];
    const int64_t wc = problem.w.strides[1];
    const int64_t cg = problem.cPerGroup();
    const int64_t kg = problem.kPerGroup();

    switch(problem.direction)
    {
    case ConvDirection::FORWARD: // y (pixels x kg) = x (pixels x cg) w^T (cg x kg)
        return single(gemmForViews(view(problem, problem.x, pixels, cg, px, xc, cg * xc),
                                   view(problem, problem.w, cg, kg, wc, wk, kg * wk),
                                   view(problem, problem.y, pixels, kg, py, yk, kg * yk),
                                   problem.groups));
    case ConvDirection::BACKWARD_DATA: // dx (pixels x cg) = dy (pixels x kg) w (kg x cg)
        return single(gemmForViews(view(problem, problem.y, pixels, kg, py, yk, kg * yk),
                                   view(problem, problem.w, kg, cg, wk, wc, kg * wk),
                                   view(problem, problem.x, pixels, cg, px, xc, cg * xc),
                                   problem.groups));
    case ConvDirection::BACKWARD_WEIGHTS: // dw (kg x cg) = dy^T (kg x pixels) x (pixels x cg)
        return single(gemmForViews(view(problem, problem.y, kg, pixels, yk, py, kg * yk),
                                   view(problem, problem.x, pixels, cg, px, xc, cg * xc),
                                   view(problem, problem.w, kg, cg, wk, wc, kg * wk),
                                   problem.groups));
    default:
        break;
    }
    return std::nullopt;
}

/// ONE_BY_ONE with channels as GEMM rows, one image at a time. Reads channels-first tensors,
/// whose spatial axes nest into one stride. Without groups the images batch into one call; with
/// them each image is a call batched over groups. Backward weights reduces over images, so only
/// a single image is served here.
std::optional<std::vector<GemmCall>> oneByOneChannelRows(const ConvProblem& problem)
{
    const auto axes = spatialAxes(problem.x);
    const auto xSpatial = collapseAxes(problem.x, axes);
    const auto ySpatial = collapseAxes(problem.y, axes);
    if(!xSpatial.has_value() || !ySpatial.has_value())
    {
        return std::nullopt;
    }
    const int64_t pixels = xSpatial->first;
    const int64_t sx = xSpatial->second;
    const int64_t sy = ySpatial->second;
    const int64_t xn = problem.x.strides[0];
    const int64_t yn = problem.y.strides[0];
    const int64_t xc = problem.x.strides[1];
    const int64_t yk = problem.y.strides[1];
    const int64_t wk = problem.w.strides[0];
    const int64_t wc = problem.w.strides[1];
    const int64_t cg = problem.cPerGroup();
    const int64_t kg = problem.kPerGroup();

    if(problem.direction == ConvDirection::BACKWARD_WEIGHTS)
    {
        if(problem.n != 1)
        {
            return std::nullopt;
        }
        // dw (kg x cg) = dy (kg x pixels) x^T (pixels x cg)
        return single(gemmForViews(view(problem, problem.y, kg, pixels, yk, sy, kg * yk),
                                   view(problem, problem.x, pixels, cg, sx, xc, cg * xc),
                                   view(problem, problem.w, kg, cg, wk, wc, kg * wk),
                                   problem.groups));
    }

    const bool forward = problem.direction == ConvDirection::FORWARD;
    // forward:  y_n (kg x pixels) = w (kg x cg) x_n (cg x pixels)
    // backward: dx_n (cg x pixels) = w^T (cg x kg) dy_n (kg x pixels)
    const auto call = [&](int64_t image, bool batchImages) {
        const int64_t xBatch = batchImages ? xn : cg * xc;
        const int64_t yBatch = batchImages ? yn : kg * yk;
        const int64_t wBatch = batchImages ? 0 : kg * wk;
        const int64_t batch = batchImages ? problem.n : problem.groups;
        const auto xView = view(problem, problem.x, cg, pixels, xc, sx, xBatch, image * xn);
        const auto yView = view(problem, problem.y, kg, pixels, yk, sy, yBatch, image * yn);
        return forward ? gemmForViews(
                             view(problem, problem.w, kg, cg, wk, wc, wBatch), xView, yView, batch)
                       : gemmForViews(
                             view(problem, problem.w, cg, kg, wc, wk, wBatch), yView, xView, batch);
    };

    if(problem.groups == 1)
    {
        return single(call(0, true));
    }
    std::vector<GemmCall> calls;
    for(int64_t image = 0; image < problem.n; ++image)
    {
        const auto next = call(image, false);
        if(!next.has_value())
        {
            return std::nullopt;
        }
        calls.push_back(*next);
    }
    return calls;
}

/// POINT_OUTPUT: images are GEMM rows, and (channel, spatial) the reduction, read in an order
/// both x and w nest into one stride. Channels outermost (channels-first) keeps a group's
/// columns contiguous, so it batches over groups; channels innermost serves one group only.
std::optional<std::vector<GemmCall>> pointOutput(const ConvProblem& problem)
{
    const auto axes = spatialAxes(problem.x);
    std::vector<size_t> channelsOuter{1};
    std::vector<size_t> channelsInner;
    for(const auto axis : axes)
    {
        channelsOuter.push_back(axis);
        channelsInner.push_back(axis);
    }
    channelsInner.push_back(1);

    const int64_t xn = problem.x.strides[0];
    const int64_t yn = problem.y.strides[0];
    const int64_t xc = problem.x.strides[1];
    const int64_t yk = problem.y.strides[1];
    const int64_t wk = problem.w.strides[0];
    const int64_t cg = problem.cPerGroup();
    const int64_t kg = problem.kPerGroup();
    int64_t spatial = 1;
    for(const auto extent : problem.input)
    {
        spatial *= extent;
    }
    const int64_t columns = cg * spatial;

    for(const auto* order : {&channelsOuter, &channelsInner})
    {
        if(order == &channelsInner && problem.groups != 1)
        {
            continue;
        }
        const auto xColumns = collapseAxes(problem.x, *order);
        const auto wColumns = collapseAxes(problem.w, *order);
        if(!xColumns.has_value() || !wColumns.has_value())
        {
            continue;
        }
        const int64_t ux = xColumns->second;
        const int64_t uw = wColumns->second;
        const int64_t xBatch = cg * xc;
        std::optional<GemmCall> call;
        switch(problem.direction)
        {
        case ConvDirection::FORWARD: // y (n x kg) = x (n x columns) w^T (columns x kg)
            call = gemmForViews(view(problem, problem.x, problem.n, columns, xn, ux, xBatch),
                                view(problem, problem.w, columns, kg, uw, wk, kg * wk),
                                view(problem, problem.y, problem.n, kg, yn, yk, kg * yk),
                                problem.groups);
            break;
        case ConvDirection::BACKWARD_DATA: // dx (n x columns) = dy (n x kg) w (kg x columns)
            call = gemmForViews(view(problem, problem.y, problem.n, kg, yn, yk, kg * yk),
                                view(problem, problem.w, kg, columns, wk, uw, kg * wk),
                                view(problem, problem.x, problem.n, columns, xn, ux, xBatch),
                                problem.groups);
            break;
        case ConvDirection::BACKWARD_WEIGHTS: // dw (kg x columns) = dy^T (kg x n) x (n x columns)
            call = gemmForViews(view(problem, problem.y, kg, problem.n, yk, yn, kg * yk),
                                view(problem, problem.x, problem.n, columns, xn, ux, xBatch),
                                view(problem, problem.w, kg, columns, wk, uw, kg * wk),
                                problem.groups);
            break;
        default:
            break;
        }
        if(call.has_value())
        {
            return std::vector<GemmCall>{*call};
        }
    }
    return std::nullopt;
}

bool allEqual(const std::vector<int64_t>& values, int64_t expected)
{
    for(const auto value : values)
    {
        if(value != expected)
        {
            return false;
        }
    }
    return true;
}

} // namespace

std::optional<std::pair<int64_t, int64_t>> collapseAxes(const ConvOperand& operand,
                                                        const std::vector<size_t>& axes)
{
    int64_t extent = 1;
    std::optional<int64_t> innerStride;
    int64_t expected = 0;
    for(size_t i = axes.size(); i-- > 0;)
    {
        const auto axis = axes[i];
        const int64_t dim = operand.dims[axis];
        if(dim == 1)
        {
            continue;
        }
        if(innerStride.has_value() && operand.strides[axis] != expected)
        {
            return std::nullopt;
        }
        if(!innerStride.has_value())
        {
            innerStride = operand.strides[axis];
        }
        expected = operand.strides[axis] * dim;
        extent *= dim;
    }
    return std::make_pair(extent, innerStride.value_or(1));
}

std::string_view toString(ConvGemmRow row)
{
    switch(row)
    {
    case ConvGemmRow::ONE_BY_ONE:
        return "one_by_one";
    case ConvGemmRow::POINT_OUTPUT:
        return "point_output";
    case ConvGemmRow::IM2COL:
        return "im2col";
    default:
        break;
    }
    return "";
}

bool isOneByOne(const ConvProblem& problem)
{
    return allEqual(problem.filter, 1) && allEqual(problem.stride, 1)
           && allEqual(problem.prePadding, 0) && allEqual(problem.postPadding, 0);
}

bool isPointOutput(const ConvProblem& problem)
{
    return problem.filter == problem.input && allEqual(problem.prePadding, 0)
           && allEqual(problem.postPadding, 0) && allEqual(problem.dilation, 1)
           && allEqual(problem.output, 1);
}

std::optional<ConvGemmPlanCalls> directGemmCalls(const ConvProblem& problem)
{
    if(isOneByOne(problem))
    {
        // A 1x1 filter reflects onto itself, so both convolution modes are the same GEMM.
        auto calls = oneByOnePixelRows(problem);
        if(!calls.has_value())
        {
            calls = oneByOneChannelRows(problem);
        }
        if(calls.has_value())
        {
            ConvGemmPlanCalls plan;
            plan.row = ConvGemmRow::ONE_BY_ONE;
            plan.stages.push_back(ConvGemmStage{{}, std::move(*calls), {}});
            return plan;
        }
    }

    // A reflected filter is not a plain reading of w, so true convolution is served only when
    // the filter is one element, which ONE_BY_ONE already covers.
    if(isPointOutput(problem) && problem.mode == data_objects::ConvMode::CROSS_CORRELATION)
    {
        auto calls = pointOutput(problem);
        if(calls.has_value())
        {
            ConvGemmPlanCalls plan;
            plan.row = ConvGemmRow::POINT_OUTPUT;
            plan.stages.push_back(ConvGemmStage{{}, std::move(*calls), {}});
            return plan;
        }
    }
    return std::nullopt;
}

} // namespace hipblaslt_plugin
