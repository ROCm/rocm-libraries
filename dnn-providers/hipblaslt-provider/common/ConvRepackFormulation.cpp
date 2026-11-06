// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/ConvRepackFormulation.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace hipblaslt_plugin
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Every workspace tensor starts on this boundary, enough for any vector access.
constexpr size_t WORKSPACE_ALIGNMENT = 256;

std::optional<int64_t> product(const std::vector<int64_t>& values)
{
    int64_t result = 1;
    for(const auto value : values)
    {
        if(__builtin_mul_overflow(result, value, &result))
        {
            return std::nullopt;
        }
    }
    return result;
}

/// Packed strides for (n, c, spatial...) @p dims with the channel axis innermost.
std::vector<int64_t> channelsLastStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 0);
    int64_t running = 1;
    strides[1] = running;
    running *= dims[1];
    for(size_t axis = dims.size(); axis-- > 2;)
    {
        strides[axis] = running;
        running *= dims[axis];
    }
    strides[0] = running;
    return strides;
}

/// Hands out workspace tensors, each aligned, each with a uid the graph does not use.
class WorkspaceLayout
{
public:
    explicit WorkspaceLayout(int64_t firstUid)
        : _nextUid(firstUid)
    {
    }

    /// A tensor of @p elements of @p dataType, or nullopt when the total would overflow.
    std::optional<int64_t> add(int64_t elements, data_objects::DataType dataType)
    {
        int64_t bytes = 0;
        if(__builtin_mul_overflow(elements, elementBytes(dataType), &bytes))
        {
            return std::nullopt;
        }
        const size_t offset
            = (_bytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
        if(static_cast<uint64_t>(bytes) > std::numeric_limits<size_t>::max() - offset)
        {
            return std::nullopt;
        }
        _tensors.push_back(WorkspaceTensor{_nextUid, offset, static_cast<size_t>(bytes)});
        _bytes = offset + static_cast<size_t>(bytes);
        return _nextUid++;
    }

    size_t bytes() const
    {
        return _bytes;
    }

    const std::vector<WorkspaceTensor>& tensors() const
    {
        return _tensors;
    }

private:
    int64_t _nextUid;
    size_t _bytes = 0;
    std::vector<WorkspaceTensor> _tensors;
};

/// @p problem described as 3-D for the repack kernels: missing leading spatial axes are
/// extent-1 axes that read nothing.
ConvColumnGeometry geometryFor(const ConvProblem& problem, bool channelsInner)
{
    ConvColumnGeometry geometry{};
    geometry.n = problem.n;
    geometry.groups = problem.groups;
    geometry.cPerGroup = problem.cPerGroup();
    geometry.xStrides[0] = problem.x.strides[0];
    geometry.xStrides[1] = problem.x.strides[1];
    const size_t spatialRank = problem.input.size();
    const size_t lead = CONV_REPACK_SPATIAL_DIMS - spatialRank;
    for(size_t axis = 0; axis < CONV_REPACK_SPATIAL_DIMS; ++axis)
    {
        if(axis < lead)
        {
            geometry.input[axis] = 1;
            geometry.output[axis] = 1;
            geometry.filter[axis] = 1;
            geometry.prePadding[axis] = 0;
            geometry.stride[axis] = 1;
            geometry.dilation[axis] = 1;
            geometry.xStrides[2 + axis] = 0;
            continue;
        }
        const size_t from = axis - lead;
        geometry.input[axis] = problem.input[from];
        geometry.output[axis] = problem.output[from];
        geometry.filter[axis] = problem.filter[from];
        geometry.prePadding[axis] = problem.prePadding[from];
        geometry.stride[axis] = problem.stride[from];
        geometry.dilation[axis] = problem.dilation[from];
        geometry.xStrides[2 + axis] = problem.x.strides[2 + from];
    }
    geometry.channelsInner = channelsInner ? 1 : 0;
    geometry.flipFilter = problem.mode == data_objects::ConvMode::CONVOLUTION ? 1 : 0;
    return geometry;
}

/// A copy of every element of @p from to the same index of @p to; both share dims.
RepackStep copyStep(const ConvOperand& from, const ConvOperand& to, data_objects::DataType dataType)
{
    StridedCopyArgs args{};
    args.rank = static_cast<int>(from.dims.size());
    args.total = 1;
    for(size_t axis = 0; axis < from.dims.size(); ++axis)
    {
        args.dims[axis] = from.dims[axis];
        args.srcStrides[axis] = from.strides[axis];
        args.dstStrides[axis] = to.strides[axis];
        args.total *= from.dims[axis];
    }
    return RepackStep{args, dataType, from.uid, to.uid};
}

} // namespace

std::optional<ConvGemmPlanCalls> im2colGemmCalls(const ConvProblem& problem)
{
    // The column order follows the filter's layout, so the GEMM reads w in place: channels
    // outermost for KCRS, innermost for KRSC.
    std::vector<size_t> channelsOuter{1};
    std::vector<size_t> channelsInner;
    for(size_t axis = 2; axis < problem.w.dims.size(); ++axis)
    {
        channelsOuter.push_back(axis);
        channelsInner.push_back(axis);
    }
    channelsInner.push_back(1);
    bool inner = false;
    auto filterColumns = collapseAxes(problem.w, channelsOuter);
    if(!filterColumns.has_value())
    {
        inner = true;
        filterColumns = collapseAxes(problem.w, channelsInner);
    }
    if(!filterColumns.has_value())
    {
        return std::nullopt;
    }

    const auto taps = product(problem.filter);
    const auto outputs = product(problem.output);
    const auto inputs = product(problem.input);
    int64_t columns = 0;
    int64_t rows = 0;
    int64_t colElements = 0;
    if(!taps.has_value() || !outputs.has_value() || !inputs.has_value()
       || __builtin_mul_overflow(problem.c, *taps, &columns)
       || __builtin_mul_overflow(problem.n, *outputs, &rows)
       || __builtin_mul_overflow(rows, columns, &colElements))
    {
        return std::nullopt;
    }

    const int64_t lastUid = std::max({problem.x.uid, problem.w.uid, problem.y.uid});
    if(lastUid > std::numeric_limits<int64_t>::max() - 2)
    {
        return std::nullopt;
    }
    WorkspaceLayout workspace(lastUid + 1);

    // Backward data's GEMM writes fp32 so col2im sums before it rounds.
    const bool backwardData = problem.direction == ConvDirection::BACKWARD_DATA;
    const auto colType = backwardData ? data_objects::DataType::FLOAT : problem.dataType;
    const auto colUid = workspace.add(colElements, colType);
    if(!colUid.has_value())
    {
        return std::nullopt;
    }

    // The column matrix as a channels-last tensor of C * filterSize channels over the output
    // positions, and the filter as a 1x1 one over those channels.
    std::vector<int64_t> colDims{problem.n, columns};
    colDims.insert(colDims.end(), problem.output.begin(), problem.output.end());
    ConvOperand col{*colUid, colDims, channelsLastStrides(colDims), std::nullopt};
    if(colType != problem.dataType)
    {
        col.dataType = colType;
    }
    ConvOperand filter{problem.w.uid,
                       {problem.k, problem.cPerGroup() * *taps},
                       {problem.w.strides[0], filterColumns->second},
                       std::nullopt};
    const size_t spatialRank = problem.input.size();
    filter.dims.resize(2 + spatialRank, 1);
    filter.strides.resize(2 + spatialRank, 1);

    ConvProblem columnProblem = problem;
    columnProblem.mode = data_objects::ConvMode::CROSS_CORRELATION;
    columnProblem.c = columns;
    columnProblem.input = problem.output;
    columnProblem.filter.assign(spatialRank, 1);
    columnProblem.prePadding.assign(spatialRank, 0);
    columnProblem.postPadding.assign(spatialRank, 0);
    columnProblem.stride.assign(spatialRank, 1);
    columnProblem.dilation.assign(spatialRank, 1);
    columnProblem.x = col;
    columnProblem.w = filter;

    ConvGemmPlanCalls plan;
    plan.row = ConvGemmRow::IM2COL;
    const auto geometry = geometryFor(problem, inner);
    if(backwardData)
    {
        Col2ImArgs args{};
        args.geometry = geometry;
        args.total = problem.n * problem.c * *inputs;
        plan.after.push_back(RepackStep{args, problem.dataType, *colUid, problem.x.uid});
    }
    else
    {
        Im2ColArgs args{};
        args.geometry = geometry;
        args.total = colElements;
        plan.before.push_back(RepackStep{args, problem.dataType, problem.x.uid, *colUid});
    }

    auto calls = directGemmCalls(columnProblem);
    if(!calls.has_value())
    {
        // Stage y (dy) channels-last, which the pixel-rows GEMM reads in every direction.
        const auto yElements = product(problem.y.dims);
        const auto stagedUid
            = yElements.has_value() ? workspace.add(*yElements, problem.dataType) : std::nullopt;
        if(!stagedUid.has_value())
        {
            return std::nullopt;
        }
        const ConvOperand staged{
            *stagedUid, problem.y.dims, channelsLastStrides(problem.y.dims), std::nullopt};
        if(problem.direction == ConvDirection::FORWARD)
        {
            plan.after.push_back(copyStep(staged, problem.y, problem.dataType));
        }
        else
        {
            plan.before.push_back(copyStep(problem.y, staged, problem.dataType));
        }
        columnProblem.y = staged;
        calls = directGemmCalls(columnProblem);
        if(!calls.has_value())
        {
            return std::nullopt;
        }
    }

    if(workspace.bytes() > MAX_REPACK_WORKSPACE_BYTES)
    {
        return std::nullopt;
    }
    plan.calls = std::move(calls->calls);
    plan.workspace = workspace.tensors();
    plan.workspaceBytes = workspace.bytes();
    return plan;
}

std::optional<ConvGemmPlanCalls> convGemmCalls(const ConvProblem& problem)
{
    auto direct = directGemmCalls(problem);
    if(direct.has_value())
    {
        return direct;
    }
    return im2colGemmCalls(problem);
}

} // namespace hipblaslt_plugin
