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

/// Workspace tensors, each aligned, laid out in the order they are added.
class WorkspaceLayout
{
public:
    /// Adds tensor @p uid of @p elements of @p dataType; false when the total would overflow.
    bool add(int64_t uid, int64_t elements, data_objects::DataType dataType)
    {
        int64_t bytes = 0;
        if(__builtin_mul_overflow(elements, elementBytes(dataType), &bytes))
        {
            return false;
        }
        const size_t offset
            = (_bytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
        if(static_cast<uint64_t>(bytes) > std::numeric_limits<size_t>::max() - offset)
        {
            return false;
        }
        _tensors.push_back(WorkspaceTensor{uid, offset, static_cast<size_t>(bytes)});
        _bytes = offset + static_cast<size_t>(bytes);
        return true;
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

/// The strides arguments of a copy or cast of every element of @p dims from @p srcStrides to
/// @p dstStrides.
template <typename Args>
Args stridedArgs(const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& srcStrides,
                 const std::vector<int64_t>& dstStrides)
{
    Args args{};
    args.rank = static_cast<int>(dims.size());
    args.total = 1;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        args.dims[axis] = dims[axis];
        args.srcStrides[axis] = srcStrides[axis];
        args.dstStrides[axis] = dstStrides[axis];
        args.total *= dims[axis];
    }
    return args;
}

/// Packed row-major strides for @p dims, last axis innermost.
std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t axis = dims.size(); axis-- > 1;)
    {
        strides[axis - 1] = strides[axis] * dims[axis];
    }
    return strides;
}

/// True when @p operand's channel axis is outside every spatial axis it actually walks: a
/// channels-first layout.
bool channelsOutermost(const ConvOperand& operand)
{
    for(size_t axis = 2; axis < operand.dims.size(); ++axis)
    {
        if(operand.dims[axis] > 1 && operand.strides[axis] > operand.strides[1])
        {
            return false;
        }
    }
    return operand.dims[1] > 1;
}

/// Everything about a problem's IM2COL plan that does not depend on how its images are sliced.
struct ColumnForm
{
    bool channelsInner = false;
    /// The column matrix is laid out like a channels-first tensor (positions innermost).
    bool columnsFirst = false;
    int64_t filterUnitStride = 0;
    int64_t taps = 0;
    int64_t outputs = 0;
    int64_t inputs = 0;
    int64_t columns = 0;
    data_objects::DataType colType = data_objects::DataType::UNSET;
    int64_t colUid = 0;
    int64_t stagedUid = 0;
    int64_t accumulatorUid = 0;
};

/// The stage that computes images [first, first + count) of @p problem through a column matrix
/// of that many images, or nullopt when the GEMM cannot read its operands. @p stageY routes y
/// (dy) through a channels-last copy in the workspace; @p wTarget is the filter the GEMM reads or
/// writes; @p accumulate adds a backward-weights slice to what earlier slices wrote.
std::optional<ConvGemmStage> sliceStage(const ConvProblem& problem,
                                        const ColumnForm& form,
                                        int64_t first,
                                        int64_t count,
                                        bool stageY,
                                        const ConvOperand& wTarget,
                                        bool accumulate)
{
    const auto bytes = elementBytes(problem.dataType);
    const size_t spatialRank = problem.input.size();

    // The column matrix as a channels-last tensor of C * filterSize channels over the output
    // positions of `count` images.
    std::vector<int64_t> colDims{count, form.columns};
    colDims.insert(colDims.end(), problem.output.begin(), problem.output.end());
    ConvOperand col{form.colUid,
                    colDims,
                    form.columnsFirst ? packedStrides(colDims) : channelsLastStrides(colDims),
                    std::nullopt,
                    0};
    if(form.colType != problem.dataType)
    {
        col.dataType = form.colType;
    }

    ConvOperand ySlice = problem.y;
    ySlice.dims[0] = count;
    ySlice.offset = first * problem.y.strides[0];

    ConvProblem columnProblem = problem;
    columnProblem.n = count;
    columnProblem.mode = data_objects::ConvMode::CROSS_CORRELATION;
    columnProblem.c = form.columns;
    columnProblem.input = problem.output;
    columnProblem.filter.assign(spatialRank, 1);
    columnProblem.prePadding.assign(spatialRank, 0);
    columnProblem.postPadding.assign(spatialRank, 0);
    columnProblem.stride.assign(spatialRank, 1);
    columnProblem.dilation.assign(spatialRank, 1);
    columnProblem.x = col;
    columnProblem.w = wTarget;
    columnProblem.y = ySlice;

    ConvGemmStage stage;
    auto geometry = geometryFor(problem, form.channelsInner);
    geometry.n = count;
    geometry.columnsFirst = form.columnsFirst ? 1 : 0;
    const int64_t xByteOffset = first * problem.x.strides[0] * bytes;
    if(problem.direction == ConvDirection::BACKWARD_DATA)
    {
        Col2ImArgs args{};
        args.geometry = geometry;
        args.total = count * problem.c * form.inputs;
        stage.after.push_back(
            RepackStep{args, problem.dataType, form.colUid, problem.x.uid, 0, xByteOffset});
    }
    else
    {
        Im2ColArgs args{};
        args.geometry = geometry;
        args.total = count * form.outputs * form.columns;
        stage.before.push_back(
            RepackStep{args, problem.dataType, problem.x.uid, form.colUid, xByteOffset, 0});
    }

    if(stageY)
    {
        const ConvOperand staged{
            form.stagedUid, ySlice.dims, channelsLastStrides(ySlice.dims), std::nullopt, 0};
        const int64_t yByteOffset = first * problem.y.strides[0] * bytes;
        if(problem.direction == ConvDirection::FORWARD)
        {
            stage.after.push_back(RepackStep{
                stridedArgs<StridedCopyArgs>(ySlice.dims, staged.strides, problem.y.strides),
                problem.dataType,
                staged.uid,
                problem.y.uid,
                0,
                yByteOffset});
        }
        else
        {
            stage.before.push_back(RepackStep{
                stridedArgs<StridedCopyArgs>(ySlice.dims, problem.y.strides, staged.strides),
                problem.dataType,
                problem.y.uid,
                staged.uid,
                yByteOffset,
                0});
        }
        columnProblem.y = staged;
    }

    // Channels-first backward weights reduces over images, which no single channels-first GEMM
    // reads: one GEMM per image, each adding to the one before.
    const bool perImage = problem.direction == ConvDirection::BACKWARD_WEIGHTS && form.columnsFirst
                          && count > 1 && !stageY;
    for(int64_t image = 0; image < (perImage ? count : 1); ++image)
    {
        ConvProblem imageProblem = columnProblem;
        if(perImage)
        {
            imageProblem.n = 1;
            imageProblem.x.dims[0] = 1;
            imageProblem.x.offset = image * col.strides[0];
            imageProblem.y.dims[0] = 1;
            imageProblem.y.offset = ySlice.offset + image * problem.y.strides[0];
        }
        auto calls = directGemmCalls(imageProblem);
        if(!calls.has_value() || calls->stages.size() != 1)
        {
            return std::nullopt;
        }
        for(auto& call : calls->stages[0].calls)
        {
            call.accumulate = accumulate || image > 0;
            stage.calls.push_back(call);
        }
    }
    return stage;
}

/// Bytes of @p elements of @p dataType.
std::optional<int64_t> tensorBytes(int64_t elements, data_objects::DataType dataType)
{
    int64_t bytes = 0;
    if(__builtin_mul_overflow(elements, elementBytes(dataType), &bytes))
    {
        return std::nullopt;
    }
    return bytes;
}

} // namespace

std::optional<ConvGemmPlanCalls>
    im2colGemmCalls(const ConvProblem& problem, size_t maxWorkspaceBytes, ColumnLayout layout)
{
    ColumnForm form;
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
    auto filterColumns = collapseAxes(problem.w, channelsOuter);
    if(!filterColumns.has_value())
    {
        form.channelsInner = true;
        filterColumns = collapseAxes(problem.w, channelsInner);
    }
    if(!filterColumns.has_value())
    {
        return std::nullopt;
    }
    form.filterUnitStride = filterColumns->second;

    const auto taps = product(problem.filter);
    const auto outputs = product(problem.output);
    const auto inputs = product(problem.input);
    int64_t imageColumnElements = 0;
    if(!taps.has_value() || !outputs.has_value() || !inputs.has_value()
       || __builtin_mul_overflow(problem.c, *taps, &form.columns)
       || __builtin_mul_overflow(*outputs, form.columns, &imageColumnElements))
    {
        return std::nullopt;
    }
    form.taps = *taps;
    form.outputs = *outputs;
    form.inputs = *inputs;

    const int64_t lastUid = std::max({problem.x.uid, problem.w.uid, problem.y.uid});
    if(lastUid > std::numeric_limits<int64_t>::max() - 3)
    {
        return std::nullopt;
    }
    form.colUid = lastUid + 1;
    form.stagedUid = lastUid + 2;
    form.accumulatorUid = lastUid + 3;

    // Backward data's GEMM writes fp32 so col2im sums before it rounds.
    const bool backwardWeights = problem.direction == ConvDirection::BACKWARD_WEIGHTS;
    form.colType = problem.direction == ConvDirection::BACKWARD_DATA ? data_objects::DataType::FLOAT
                                                                     : problem.dataType;

    // The filter as a 1x1 one over the column matrix's channels.
    const size_t spatialRank = problem.input.size();
    ConvOperand filter{problem.w.uid,
                       {problem.k, problem.cPerGroup() * form.taps},
                       {problem.w.strides[0], form.filterUnitStride},
                       std::nullopt,
                       0};
    filter.dims.resize(2 + spatialRank, 1);
    filter.strides.resize(2 + spatialRank, 1);

    // Per-image workspace, and the images one slice can hold within the cap. Every tensor after
    // the first may start up to one alignment past the end of the one before it.
    const auto colImageBytes = tensorBytes(imageColumnElements, form.colType);
    const auto yImageElements
        = product(std::vector<int64_t>(problem.y.dims.begin() + 1, problem.y.dims.end()));
    const auto yImageBytes = yImageElements.has_value()
                                 ? tensorBytes(*yImageElements, problem.dataType)
                                 : std::nullopt;
    const auto filterElements = product(problem.w.dims);
    const auto accumulatorBytes = filterElements.has_value()
                                      ? tensorBytes(*filterElements, data_objects::DataType::FLOAT)
                                      : std::nullopt;
    if(!colImageBytes.has_value() || !yImageBytes.has_value() || !accumulatorBytes.has_value())
    {
        return std::nullopt;
    }
    const auto cap = static_cast<int64_t>(
        std::min(maxWorkspaceBytes, static_cast<size_t>(std::numeric_limits<int64_t>::max())));
    const auto pad = static_cast<int64_t>(WORKSPACE_ALIGNMENT);
    const auto imagesWithin = [&](int64_t perImage, int64_t fixed) {
        return fixed >= cap || perImage <= 0 ? int64_t{0}
                                             : std::min(problem.n, (cap - fixed) / perImage);
    };

    // The column matrix takes y's layout, so the GEMM reads both along the same axis: for
    // channels-first y, positions innermost, which keeps a long reduction over positions
    // contiguous in both operands. Whether that layout, or y as given, can be read does not
    // depend on the slice size except through the number of images, so probe at the largest
    // slice the cap allows; fall back to a channels-last column matrix, then to staging y.
    bool stageY = false;
    int64_t slice = imagesWithin(*colImageBytes, 0);
    if(slice == 0)
    {
        return std::nullopt;
    }
    form.columnsFirst = layout == ColumnLayout::FOLLOW_Y && channelsOutermost(problem.y);
    if(form.columnsFirst && !sliceStage(problem, form, 0, slice, false, filter, false).has_value())
    {
        form.columnsFirst = false;
    }
    if(!sliceStage(problem, form, 0, slice, false, filter, false).has_value())
    {
        stageY = true;
        slice = imagesWithin(*colImageBytes + *yImageBytes, pad);
    }
    // Backward weights over several slices sums them: in w itself for fp32, otherwise in an fp32
    // accumulator rounded once at the end.
    // So does channels-first backward weights over several images, one GEMM per image: any sum
    // of several GEMMs is kept in fp32 and rounded into a narrower w once, never per GEMM.
    const bool sliced = slice < problem.n;
    const bool perImage = form.columnsFirst && !stageY && problem.n > 1;
    const bool accumulator = backwardWeights && (sliced || perImage)
                             && problem.dataType != data_objects::DataType::FLOAT;
    if(accumulator)
    {
        slice = imagesWithin(*colImageBytes + (stageY ? *yImageBytes : 0),
                             *accumulatorBytes + (stageY ? 2 : 1) * pad);
    }
    if(slice == 0)
    {
        return std::nullopt;
    }

    ConvOperand wTarget = filter;
    std::vector<int64_t> accumulatorStrides;
    if(accumulator)
    {
        // The GEMM's view of w, packed: rows k, columns (c, taps) in the filter's own order.
        const int64_t columnsPerGroup = problem.cPerGroup() * form.taps;
        wTarget = ConvOperand{form.accumulatorUid,
                              filter.dims,
                              {columnsPerGroup, 1},
                              data_objects::DataType::FLOAT,
                              0};
        wTarget.strides.resize(2 + spatialRank, 1);
        const std::vector<int64_t>& filterDims = problem.w.dims;
        if(form.channelsInner)
        {
            // (k, spatial..., c) packed, read back through (k, c, spatial...) indices.
            std::vector<int64_t> krsc{filterDims[0]};
            krsc.insert(krsc.end(), filterDims.begin() + 2, filterDims.end());
            krsc.push_back(filterDims[1]);
            const auto packed = packedStrides(krsc);
            accumulatorStrides.assign(filterDims.size(), 0);
            accumulatorStrides[0] = packed[0];
            accumulatorStrides[1] = packed.back();
            for(size_t axis = 2; axis < filterDims.size(); ++axis)
            {
                accumulatorStrides[axis] = packed[axis - 1];
            }
        }
        else
        {
            accumulatorStrides = packedStrides(filterDims);
        }
    }

    WorkspaceLayout workspace;
    const int64_t sliceColumnElements = slice * imageColumnElements;
    if(!workspace.add(form.colUid, sliceColumnElements, form.colType)
       || (stageY && !workspace.add(form.stagedUid, slice * *yImageElements, problem.dataType))
       || (accumulator
           && !workspace.add(form.accumulatorUid, *filterElements, data_objects::DataType::FLOAT)))
    {
        return std::nullopt;
    }
    if(workspace.bytes() > maxWorkspaceBytes)
    {
        return std::nullopt;
    }

    ConvGemmPlanCalls plan;
    plan.row = ConvGemmRow::IM2COL;
    for(int64_t first = 0; first < problem.n; first += slice)
    {
        const int64_t count = std::min(slice, problem.n - first);
        auto stage = sliceStage(
            problem, form, first, count, stageY, wTarget, backwardWeights && first > 0);
        if(!stage.has_value())
        {
            return std::nullopt;
        }
        plan.stages.push_back(std::move(*stage));
    }
    if(accumulator)
    {
        ConvGemmStage round;
        round.before.push_back(RepackStep{
            stridedArgs<StridedCastArgs>(problem.w.dims, accumulatorStrides, problem.w.strides),
            problem.dataType,
            form.accumulatorUid,
            problem.w.uid,
            0,
            0});
        plan.stages.push_back(std::move(round));
    }
    plan.workspace = workspace.tensors();
    plan.workspaceBytes = workspace.bytes();
    plan.columnsFirst = form.columnsFirst;
    return plan;
}

std::optional<ConvGemmPlanCalls>
    convGemmCalls(const ConvProblem& problem, size_t maxWorkspaceBytes, ColumnLayout layout)
{
    auto direct = directGemmCalls(problem);
    if(direct.has_value())
    {
        return direct;
    }
    return im2colGemmCalls(problem, maxWorkspaceBytes, layout);
}

} // namespace hipblaslt_plugin
