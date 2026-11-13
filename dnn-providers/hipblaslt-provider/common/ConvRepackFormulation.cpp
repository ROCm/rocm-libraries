// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/ConvRepackFormulation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "common/GemmFormulation.hpp"

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

/// A band of one image along its outermost spatial axis, for an image whose column matrix
/// exceeds the workspace: output rows [outFirst, outFirst + outRows) and, for backward data, the
/// input rows [inFirst, inFirst + inRows) col2im writes from them. Without one, a stage covers
/// whole images.
struct ImageBand
{
    int64_t outFirst = 0;
    int64_t outRows = 0;
    int64_t inFirst = 0;
    int64_t inRows = 0;
};

/// The stage that computes images [first, first + count) of @p problem, or @p band of image
/// @p first, through a column matrix of that many images (rows), or nullopt when the GEMM cannot
/// read its operands. @p stageY routes y (dy) through a channels-last copy in the workspace;
/// @p wTarget is the filter the GEMM reads or writes; @p accumulate adds a backward-weights
/// slice to what earlier slices wrote.
std::optional<ConvGemmStage> sliceStage(const ConvProblem& problem,
                                        const ColumnForm& form,
                                        int64_t first,
                                        int64_t count,
                                        bool stageY,
                                        const ConvOperand& wTarget,
                                        bool accumulate,
                                        const std::optional<ImageBand>& band = std::nullopt)
{
    const auto bytes = elementBytes(problem.dataType);
    const size_t spatialRank = problem.input.size();
    std::vector<int64_t> output = problem.output;
    std::vector<int64_t> input = problem.input;
    if(band.has_value())
    {
        output[0] = band->outRows;
        if(problem.direction == ConvDirection::BACKWARD_DATA)
        {
            input[0] = band->inRows;
        }
    }
    int64_t outputs = 1;
    int64_t inputs = 1;
    for(size_t axis = 0; axis < spatialRank; ++axis)
    {
        outputs *= output[axis];
        inputs *= input[axis];
    }

    // The column matrix as a channels-last tensor of C * filterSize channels over the output
    // positions of `count` images.
    std::vector<int64_t> colDims{count, form.columns};
    colDims.insert(colDims.end(), output.begin(), output.end());
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
    if(band.has_value())
    {
        ySlice.dims[2] = band->outRows;
        ySlice.offset += band->outFirst * problem.y.strides[2];
    }

    ConvProblem columnProblem = problem;
    columnProblem.n = count;
    columnProblem.mode = data_objects::ConvMode::CROSS_CORRELATION;
    columnProblem.c = form.columns;
    columnProblem.input = output;
    columnProblem.output = output;
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
    int64_t xByteOffset = first * problem.x.strides[0] * bytes;
    if(band.has_value())
    {
        // A band is the image's convolution with its first output row at outFirst: padding
        // shifted by the rows before it, so the repack kernels' bounds checks still see the
        // whole input (or, for col2im, only the band's input rows).
        const size_t axis = CONV_REPACK_SPATIAL_DIMS - spatialRank;
        geometry.output[axis] = band->outRows;
        geometry.prePadding[axis] -= band->outFirst * problem.stride[0];
        if(problem.direction == ConvDirection::BACKWARD_DATA)
        {
            geometry.input[axis] = band->inRows;
            geometry.prePadding[axis] += band->inFirst;
            xByteOffset += band->inFirst * problem.x.strides[2] * bytes;
        }
    }
    if(problem.direction == ConvDirection::BACKWARD_DATA)
    {
        Col2ImArgs args{};
        args.geometry = geometry;
        args.total = count * problem.c * inputs;
        stage.after.push_back(
            RepackStep{args, problem.dataType, form.colUid, problem.x.uid, 0, xByteOffset});
    }
    else
    {
        Im2ColArgs args{};
        args.geometry = geometry;
        args.total = count * outputs * form.columns;
        stage.before.push_back(
            RepackStep{args, problem.dataType, problem.x.uid, form.colUid, xByteOffset, 0});
    }

    if(stageY)
    {
        const ConvOperand staged{
            form.stagedUid, ySlice.dims, channelsLastStrides(ySlice.dims), std::nullopt, 0};
        const int64_t yByteOffset = ySlice.offset * bytes;
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

/// The band of output rows [first, first + rows) along @p problem's outermost spatial axis.
ImageBand bandOf(const ConvProblem& problem, int64_t first, int64_t rows)
{
    ImageBand band;
    band.outFirst = first;
    band.outRows = rows;
    band.inFirst = 0;
    band.inRows = problem.input[0];
    return band;
}

/// The band that writes input rows [first, first + rows) of backward data: the output rows any
/// of them reads from, at least one so the stage is well formed (a row no output reads gets 0).
ImageBand inputBandOf(const ConvProblem& problem, int64_t first, int64_t rows)
{
    const int64_t stride = problem.stride[0];
    const int64_t reach = (problem.filter[0] - 1) * problem.dilation[0];
    const int64_t pad = problem.prePadding[0];
    const int64_t lowest = first + pad - reach;
    const int64_t highest = first + rows - 1 + pad;
    int64_t outFirst = lowest <= 0 ? 0 : (lowest + stride - 1) / stride;
    int64_t outLast = highest < 0 ? -1 : std::min(problem.output[0] - 1, highest / stride);
    if(outLast < outFirst)
    {
        outFirst = outLast = std::clamp<int64_t>(outFirst, 0, problem.output[0] - 1);
    }
    ImageBand band;
    band.outFirst = outFirst;
    band.outRows = outLast - outFirst + 1;
    band.inFirst = first;
    band.inRows = rows;
    return band;
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

    const bool backwardWeights = problem.direction == ConvDirection::BACKWARD_WEIGHTS;
    // Backward data's GEMM writes fp32 so col2im sums before it rounds; a filter whose taps never
    // overlap (convColumnsDisjoint, a single tap included) gives every dx element at most one
    // term, so its columns stay in dx's type (half the bytes, a copying col2im). The summing
    // col2im kernels read fp32 columns only: keeping 16-bit columns for a few overlapping taps
    // (the libraries' tuned 16-bit-output GEMMs run 4-5x faster than the fp32-output ones on
    // RDNA3 for a vocoder upsampler) needs those kernels templated on the column type.
    bool disjoint = true;
    for(size_t axis = 0; axis < problem.filter.size(); ++axis)
    {
        disjoint
            = disjoint
              && problem.stride[axis] >= (problem.filter[axis] - 1) * problem.dilation[axis] + 1;
    }
    form.colType = problem.direction == ConvDirection::BACKWARD_DATA && !disjoint
                       ? data_objects::DataType::FLOAT
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
    // An image whose column matrix alone exceeds the cap is built in bands of output rows along
    // its outermost spatial axis, one image at a time; probe with a two-row band.
    bool stageY = false;
    int64_t slice = imagesWithin(*colImageBytes, 0);
    const bool banded = slice == 0;
    const int64_t outRows = problem.output[0];
    std::optional<ImageBand> probe;
    if(banded)
    {
        slice = 1;
        probe = bandOf(problem, 0, std::min<int64_t>(outRows, 2));
    }
    const bool channelsLast
        = layout == ColumnLayout::CHANNELS_LAST || layout == ColumnLayout::STAGED_CHANNELS_LAST;
    form.columnsFirst = !channelsLast && channelsOutermost(problem.y);
    if(form.columnsFirst
       && !sliceStage(problem, form, 0, slice, false, filter, false, probe).has_value())
    {
        form.columnsFirst = false;
    }
    const bool stageAllImages = layout == ColumnLayout::STAGED_CHANNELS_LAST && !banded
                                && problem.n > 1 && channelsOutermost(problem.y);
    if(stageAllImages
       || !sliceStage(problem, form, 0, slice, false, filter, false, probe).has_value())
    {
        stageY = true;
        if(!banded)
        {
            slice = imagesWithin(*colImageBytes + *yImageBytes, pad);
        }
    }
    // Backward weights over several slices sums them: in w itself for fp32, otherwise in an fp32
    // accumulator rounded once at the end.
    // So does channels-first backward weights over several images, one GEMM per image: any sum
    // of several GEMMs is kept in fp32 and rounded into a narrower w once, never per GEMM.
    const bool sliced = banded || slice < problem.n;
    const bool perImage = form.columnsFirst && !stageY && problem.n > 1;
    const bool accumulator = backwardWeights && (sliced || perImage)
                             && problem.dataType != data_objects::DataType::FLOAT;
    if(accumulator && !banded)
    {
        slice = imagesWithin(*colImageBytes + (stageY ? *yImageBytes : 0),
                             *accumulatorBytes + (stageY ? 2 : 1) * pad);
    }
    if(slice == 0)
    {
        return std::nullopt;
    }

    // Bands: the output rows the cap holds after the fixed tensors; backward data turns them
    // into the input rows whose every contributing output row fits one band.
    int64_t bandRows = outRows;
    int64_t bandInputRows = problem.input[0];
    if(banded)
    {
        const int64_t rowColumnBytes = *colImageBytes / outRows;
        const int64_t rowYBytes = stageY ? *yImageBytes / outRows : 0;
        const int64_t fixed = (accumulator ? *accumulatorBytes + pad : 0) + (stageY ? pad : 0);
        bandRows
            = fixed >= cap ? 0 : std::min(outRows, (cap - fixed) / (rowColumnBytes + rowYBytes));
        if(problem.direction == ConvDirection::BACKWARD_DATA)
        {
            bandInputRows = (bandRows - 1) * problem.stride[0]
                            - (problem.filter[0] - 1) * problem.dilation[0] + 1;
            bandInputRows = std::min(bandInputRows, problem.input[0]);
        }
        if(bandRows < 1 || bandInputRows < 1)
        {
            return std::nullopt;
        }
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
    const int64_t sliceColumnElements = slice * imageColumnElements / outRows * bandRows;
    if(!workspace.add(form.colUid, sliceColumnElements, form.colType)
       || (stageY
           && !workspace.add(
               form.stagedUid, slice * *yImageElements / outRows * bandRows, problem.dataType))
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
        if(!banded)
        {
            auto stage = sliceStage(
                problem, form, first, count, stageY, wTarget, backwardWeights && first > 0);
            if(!stage.has_value())
            {
                return std::nullopt;
            }
            plan.stages.push_back(std::move(*stage));
            continue;
        }
        const bool backwardData = problem.direction == ConvDirection::BACKWARD_DATA;
        const int64_t rows = backwardData ? problem.input[0] : outRows;
        const int64_t step = backwardData ? bandInputRows : bandRows;
        for(int64_t row = 0; row < rows; row += step)
        {
            const auto band = backwardData ? inputBandOf(problem, row, std::min(step, rows - row))
                                           : bandOf(problem, row, std::min(step, rows - row));
            auto stage = sliceStage(problem,
                                    form,
                                    first,
                                    1,
                                    stageY,
                                    wTarget,
                                    backwardWeights && !plan.stages.empty(),
                                    band);
            if(!stage.has_value())
            {
                return std::nullopt;
            }
            plan.stages.push_back(std::move(*stage));
        }
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

namespace
{

/// Pixels each batched backward-weights GEMM reduces over at least before its image is split
/// into chunks: a long reduction behind a small output (3 x 32 over 2 M pixels) leaves the
/// GPU idle, so the chunks become more batch entries.
constexpr int64_t CHUNK_PIXELS = 4096;
/// Most chunks one image is split into.
constexpr int64_t MAX_CHUNKS = 1024;

/// A 1x1, unit-stride, unpadded backward weights of several channels-first images, whose
/// reduction over (image, pixel) no single GEMM reads; its (x, dy) spatial strides, or nullopt.
struct OneByOneWeights
{
    int64_t pixels = 0;
    int64_t xPixelStride = 0;
    int64_t yPixelStride = 0;
};

std::optional<OneByOneWeights> oneByOneWeightsOf(const ConvProblem& problem)
{
    if(problem.direction != ConvDirection::BACKWARD_WEIGHTS || problem.n < 2)
    {
        return std::nullopt;
    }
    for(size_t axis = 0; axis < problem.filter.size(); ++axis)
    {
        if(problem.filter[axis] != 1 || problem.stride[axis] != 1 || problem.prePadding[axis] != 0
           || problem.postPadding[axis] != 0)
        {
            return std::nullopt;
        }
    }
    std::vector<size_t> spatial;
    for(size_t axis = 2; axis < problem.x.dims.size(); ++axis)
    {
        spatial.push_back(axis);
    }
    const auto xSpatial = collapseAxes(problem.x, spatial);
    const auto ySpatial = collapseAxes(problem.y, spatial);
    if(!xSpatial.has_value() || !ySpatial.has_value())
    {
        return std::nullopt;
    }
    return OneByOneWeights{xSpatial->first, xSpatial->second, ySpatial->second};
}

/// The chunks each image's pixels split into: the largest divisor of @p pixels that keeps
/// every chunk at least CHUNK_PIXELS long, 1 for a short image.
int64_t chunksFor(int64_t pixels)
{
    for(int64_t chunks = std::min(MAX_CHUNKS, pixels / CHUNK_PIXELS); chunks > 1; --chunks)
    {
        if(pixels % chunks == 0)
        {
            return chunks;
        }
    }
    return 1;
}

/// Batched partials: per image (and chunk of its pixels) dw as an fp32 partial in the workspace,
/// one strided-batched GEMM per group and per image or chunk, whichever is fewer; then a cast
/// adds the partials in order and rounds once into w. nullopt when the partials exceed
/// @p maxWorkspaceBytes or a GEMM cannot read the operands.
std::optional<ConvGemmPlanCalls> batchedOneByOneWeights(const ConvProblem& problem,
                                                        const OneByOneWeights& shape,
                                                        size_t maxWorkspaceBytes)
{
    const int64_t kg = problem.kPerGroup();
    const int64_t cg = problem.cPerGroup();
    const int64_t slab = problem.k * cg;
    const int64_t chunks = chunksFor(shape.pixels);
    const int64_t chunk = shape.pixels / chunks;
    int64_t partialElements = 0;
    if(__builtin_mul_overflow(problem.n * chunks, slab, &partialElements)
       || static_cast<size_t>(partialElements) > maxWorkspaceBytes / sizeof(float)
       || problem.n * chunks > std::numeric_limits<int>::max())
    {
        return std::nullopt;
    }
    const int64_t lastUid = std::max({problem.x.uid, problem.w.uid, problem.y.uid});
    if(lastUid > std::numeric_limits<int64_t>::max() - 5)
    {
        return std::nullopt;
    }
    const int64_t partialUid = lastUid + 5;

    // Partials (outer, inner, k, cg): the loop runs over the outer of images and chunks, each
    // call batched over the inner.
    const bool loopChunks = chunks <= problem.n;
    const int64_t outer = loopChunks ? chunks : problem.n;
    const int64_t inner = loopChunks ? problem.n : chunks;
    const int64_t xInner = loopChunks ? problem.x.strides[0] : chunk * shape.xPixelStride;
    const int64_t yInner = loopChunks ? problem.y.strides[0] : chunk * shape.yPixelStride;
    const int64_t xOuter = loopChunks ? chunk * shape.xPixelStride : problem.x.strides[0];
    const int64_t yOuter = loopChunks ? chunk * shape.yPixelStride : problem.y.strides[0];
    ConvGemmStage stage;
    for(int64_t o = 0; o < outer; ++o)
    {
        for(int64_t group = 0; group < problem.groups; ++group)
        {
            // partial (kg x cg, rows of cg) = dy (kg x chunk) x^T (chunk x cg).
            const MatrixView dy{problem.y.uid,
                                problem.dataType,
                                kg,
                                chunk,
                                problem.y.strides[1],
                                shape.yPixelStride,
                                yInner,
                                problem.y.offset + o * yOuter + group * kg * problem.y.strides[1]};
            const MatrixView x{problem.x.uid,
                               problem.dataType,
                               chunk,
                               cg,
                               shape.xPixelStride,
                               problem.x.strides[1],
                               xInner,
                               problem.x.offset + o * xOuter + group * cg * problem.x.strides[1]};
            const MatrixView partial{partialUid,
                                     data_objects::DataType::FLOAT,
                                     kg,
                                     cg,
                                     cg,
                                     1,
                                     slab,
                                     o * inner * slab + group * kg * cg};
            const auto call = gemmForViews(dy, x, partial, inner);
            if(!call.has_value())
            {
                return std::nullopt;
            }
            stage.calls.push_back(*call);
        }
    }
    // The partials as w's (k, c, spatial...) dims, packed (k, c), summed over every slab.
    std::vector<int64_t> partialStrides(problem.w.dims.size(), 0);
    partialStrides[0] = cg;
    partialStrides[1] = 1;
    auto sum = stridedArgs<StridedCastArgs>(problem.w.dims, partialStrides, problem.w.strides);
    sum.batches = static_cast<int>(problem.n * chunks);
    sum.batchStride = slab;
    stage.after.push_back(RepackStep{sum, problem.dataType, partialUid, problem.w.uid, 0, 0});

    ConvGemmPlanCalls plan;
    plan.row = ConvGemmRow::ONE_BY_ONE;
    plan.stages.push_back(std::move(stage));
    plan.workspace.push_back(
        WorkspaceTensor{partialUid, 0, static_cast<size_t>(partialElements) * sizeof(float)});
    plan.workspaceBytes = static_cast<size_t>(partialElements) * sizeof(float);
    return plan;
}

/// Staged channels-last: x and dy copied channels-last into the workspace, where (image, pixel)
/// nest into one stride, and the channels-last direct row's one GEMM reduces over all of them,
/// writing w. nullopt when the copies exceed @p maxWorkspaceBytes or no direct row reads them.
std::optional<ConvGemmPlanCalls> stagedOneByOneWeights(const ConvProblem& problem,
                                                       size_t maxWorkspaceBytes)
{
    const int64_t lastUid = std::max({problem.x.uid, problem.w.uid, problem.y.uid});
    if(lastUid > std::numeric_limits<int64_t>::max() - 7)
    {
        return std::nullopt;
    }
    const auto xElements = product(problem.x.dims);
    const auto yElements = product(problem.y.dims);
    WorkspaceLayout workspace;
    if(!xElements.has_value() || !yElements.has_value()
       || !workspace.add(lastUid + 6, *xElements, problem.dataType)
       || !workspace.add(lastUid + 7, *yElements, problem.dataType)
       || workspace.bytes() > maxWorkspaceBytes)
    {
        return std::nullopt;
    }
    ConvProblem staged = problem;
    staged.x = ConvOperand{
        lastUid + 6, problem.x.dims, channelsLastStrides(problem.x.dims), std::nullopt, 0};
    staged.y = ConvOperand{
        lastUid + 7, problem.y.dims, channelsLastStrides(problem.y.dims), std::nullopt, 0};
    auto plan = directGemmCalls(staged);
    if(!plan.has_value() || plan->stages.size() != 1)
    {
        return std::nullopt;
    }
    const int64_t bytes = elementBytes(problem.dataType);
    for(const auto* operand : {&problem.x, &problem.y})
    {
        const auto& target = operand == &problem.x ? staged.x : staged.y;
        plan->stages[0].before.push_back(RepackStep{
            stridedArgs<StridedCopyArgs>(operand->dims, operand->strides, target.strides),
            problem.dataType,
            operand->uid,
            target.uid,
            operand->offset * bytes,
            0});
    }
    plan->workspace = workspace.tensors();
    plan->workspaceBytes = workspace.bytes();
    return plan;
}

/// The 1x1 channels-first backward weights of several images in @p layout: batched partials
/// (CHANNELS_FIRST), staged channels-last (CHANNELS_LAST, STAGED_CHANNELS_LAST), or whichever moves fewer workspace
/// bytes (FOLLOW_Y), each falling back to the other when it does not apply.
std::optional<ConvGemmPlanCalls>
    oneByOneWeights(const ConvProblem& problem, size_t maxWorkspaceBytes, ColumnLayout layout)
{
    const auto shape = oneByOneWeightsOf(problem);
    if(!shape.has_value())
    {
        return std::nullopt;
    }
    auto batched = batchedOneByOneWeights(problem, *shape, maxWorkspaceBytes);
    auto staged = stagedOneByOneWeights(problem, maxWorkspaceBytes);
    if(layout == ColumnLayout::CHANNELS_FIRST)
    {
        return batched.has_value() ? batched : staged;
    }
    if(layout == ColumnLayout::CHANNELS_LAST || layout == ColumnLayout::STAGED_CHANNELS_LAST)
    {
        return staged.has_value() ? staged : batched;
    }
    // Partials are written and read once; staged copies are read and written once each.
    if(batched.has_value() && staged.has_value())
    {
        return batched->workspaceBytes <= staged->workspaceBytes ? batched : staged;
    }
    return batched.has_value() ? batched : staged;
}

/// Output elements past which a backward-weights GEMM chain is left as it is: its own tiles
/// already fill the GPU, and partials per image would cost more than they save.
constexpr int64_t MAX_BATCHED_OUTPUT = int64_t{1} << 20;
/// Output elements up to which a long reduction is also cut into chunks: few enough tiles that
/// the libraries' own split of k leaves the GPU idle (3 x 32, 32 x 451).
constexpr int64_t MAX_CHUNKED_OUTPUT = int64_t{1} << 16;
/// Shortest reduction a forward or backward data GEMM into a 16-bit output runs as fp32
/// partials (narrowPartials). hipBLASLt on gfx1100 rounds its own split's parts to the output
/// at 524288 terms (ratio 11 against the once-rounded tolerance) and stays within it up to
/// 65536 (ratio <= 1.6, as rocBLAS without a split); the fp32-output kernels the partials need
/// cost 1.1-4x there below it (14x14 c1024 3x3: 42 -> 168 us).
constexpr int64_t MIN_NARROW_PARTIALS_K = int64_t{1} << 16;

/// @p call's a and b advanced @p count elements along the reduction (k).
void advanceReduction(GemmCall& call, int64_t count)
{
    call.a.offset += call.transposeA ? count : count * call.a.leadingDimension;
    call.b.offset += call.transposeB ? count * call.b.leadingDimension : count;
}

/// One axis a batched backward-weights GEMM may run along: its extent and the element steps of
/// a, b and the fp32 partial between entries.
struct ReductionAxis
{
    int64_t count;
    int64_t aStep;
    int64_t bStep;
    int64_t partialStep;
};

/// @p plan with a backward-weights GEMM stage that is a chain of accumulating GEMMs into one small
/// output -- one per image, as channels-first im2col runs them, each batched over the groups --
/// or one GEMM over a long reduction, rewritten as batched GEMMs over the groups, images and
/// chunks of the reduction, each an fp32 partial in the workspace, and a cast adding the partials
/// in order into the output. A small output behind a long reduction leaves the GPU idle; the
/// images and chunks are parallel work. Each call batches the longest of the three axes and the
/// calls loop the other two. Plans with one GEMM stage only; @p plan unchanged when the chain does
/// not qualify or the partials do not fit @p maxWorkspaceBytes.
/// Forward and backward data qualify, with @p narrowPartials, with one GEMM straight into a
/// 16-bit y or dx.
/// The chunks a split reduction cuts a reduction of @p k into: enough that @p tiles output tiles
/// per chunk reach SPLIT_TARGET_TILES, each chunk at least CHUNK_PIXELS long and the count
/// dividing k; 1 when no such count above 1 exists.
int64_t splitChunksFor(int64_t k, int64_t tiles)
{
    const int64_t wanted = (SPLIT_TARGET_TILES + tiles - 1) / std::max<int64_t>(tiles, 1);
    for(int64_t chunks = std::min({wanted, MAX_CHUNKS, k / CHUNK_PIXELS}); chunks > 1; --chunks)
    {
        if(k % chunks == 0)
        {
            return chunks;
        }
    }
    return 1;
}

ConvGemmPlanCalls batchReductionChains(const ConvProblem& problem,
                                       ConvGemmPlanCalls plan,
                                       size_t maxWorkspaceBytes,
                                       bool splitReduction,
                                       bool narrowPartials)
{
    ConvGemmStage* gemmStage = nullptr;
    for(auto& stage : plan.stages)
    {
        if(stage.calls.empty())
        {
            continue;
        }
        if(gemmStage != nullptr)
        {
            return plan;
        }
        gemmStage = &stage;
    }
    if(gemmStage == nullptr)
    {
        return plan;
    }
    const auto& calls = gemmStage->calls;
    const GemmCall& first = calls.front();
    // Forward and backward data: a single GEMM straight into a 16-bit y or dx only. A library
    // may split its long reduction and round the parts to the output type between them
    // (hipBLASLt on gfx11 does), so the partials here are fp32, summed in order and rounded
    // once. An fp32 output, or an fp32 column matrix, is left as it is.
    if(problem.direction != ConvDirection::BACKWARD_WEIGHTS
       && (!narrowPartials || calls.size() != 1 || first.k < MIN_NARROW_PARTIALS_K
           || first.d.dataType == data_objects::DataType::FLOAT
           || first.d.uid
                  != (problem.direction == ConvDirection::FORWARD ? problem.y.uid : problem.x.uid)))
    {
        return plan;
    }
    const auto images = static_cast<int64_t>(calls.size());
    const int64_t aStep = images > 1 ? calls[1].a.offset - first.a.offset : 0;
    const int64_t bStep = images > 1 ? calls[1].b.offset - first.b.offset : 0;
    for(int64_t i = 0; i < images; ++i)
    {
        const GemmCall& call = calls[static_cast<size_t>(i)];
        if(call.batchCount != first.batchCount || call.m != first.m || call.n != first.n
           || call.k != first.k || call.transposeA != first.transposeA
           || call.transposeB != first.transposeB || call.a.uid != first.a.uid
           || call.b.uid != first.b.uid || call.a.leadingDimension != first.a.leadingDimension
           || call.b.leadingDimension != first.b.leadingDimension
           || call.a.batchStride != first.a.batchStride || call.b.batchStride != first.b.batchStride
           || call.d.uid != first.d.uid || call.d.offset != first.d.offset
           || call.d.leadingDimension != first.d.leadingDimension
           || call.d.batchStride != first.d.batchStride
           || call.a.offset != first.a.offset + i * aStep
           || call.b.offset != first.b.offset + i * bStep || call.accumulate != (i > 0))
        {
            return plan;
        }
    }
    const int64_t groups = first.batchCount;
    const int64_t outputElements = first.m * first.n;
    int64_t groupedElements = 0;
    if(__builtin_mul_overflow(groups, outputElements, &groupedElements)
       || groupedElements > MAX_BATCHED_OUTPUT)
    {
        return plan;
    }
    // Split: chunks by the output's tiles (fewer than SPLIT_TARGET_TILES), whatever its size.
    const int64_t tiles = groups * ((first.m + 127) / 128) * ((first.n + 127) / 128);
    const int64_t chunks = splitReduction
                               ? (tiles < SPLIT_TARGET_TILES ? splitChunksFor(first.k, tiles) : 1)
                               : (groupedElements <= MAX_CHUNKED_OUTPUT ? chunksFor(first.k) : 1);
    if(images < 2 && chunks < 2)
    {
        return plan;
    }
    int64_t partialElements = 0;
    if(__builtin_mul_overflow(images * chunks, groupedElements, &partialElements)
       || images * chunks > std::numeric_limits<int>::max())
    {
        return plan;
    }
    const size_t offset = (plan.workspaceBytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT
                          * WORKSPACE_ALIGNMENT;
    const size_t bytes = static_cast<size_t>(partialElements) * sizeof(float);
    const int64_t lastUid = std::max({problem.x.uid, problem.w.uid, problem.y.uid});
    if(offset + bytes > maxWorkspaceBytes || lastUid > std::numeric_limits<int64_t>::max() - 8)
    {
        return plan;
    }
    const int64_t partialUid = lastUid + 8;
    const int64_t chunk = first.k / chunks;

    // Partials (image, chunk, group, n, m), column-major m x n each.
    GemmCall chunkStep = first;
    chunkStep.a.offset = 0;
    chunkStep.b.offset = 0;
    advanceReduction(chunkStep, chunk);
    std::array<ReductionAxis, 3> axes{
        ReductionAxis{groups, first.a.batchStride, first.b.batchStride, outputElements},
        ReductionAxis{images, aStep, bStep, chunks * groupedElements},
        ReductionAxis{chunks, chunkStep.a.offset, chunkStep.b.offset, groupedElements}};
    std::stable_sort(axes.begin(), axes.end(), [](const ReductionAxis& l, const ReductionAxis& r) {
        return l.count > r.count;
    });
    const ReductionAxis& batch = axes[0];
    std::vector<GemmCall> batched;
    for(int64_t outer = 0; outer < axes[1].count; ++outer)
    {
        for(int64_t inner = 0; inner < axes[2].count; ++inner)
        {
            GemmCall call = first;
            call.k = chunk;
            call.accumulate = false;
            call.batchCount = batch.count;
            call.a.offset += outer * axes[1].aStep + inner * axes[2].aStep;
            call.b.offset += outer * axes[1].bStep + inner * axes[2].bStep;
            call.a.batchStride = batch.aStep;
            call.b.batchStride = batch.bStep;
            call.d = GemmOperand{partialUid,
                                 data_objects::DataType::FLOAT,
                                 first.m,
                                 batch.partialStep,
                                 outer * axes[1].partialStep + inner * axes[2].partialStep};
            batched.push_back(call);
        }
    }
    // The output as (group, n, m) with d's strides, the partials packed; summed over the images
    // and chunks.
    StridedCastArgs sum{};
    sum.rank = 3;
    sum.dims[0] = groups;
    sum.dims[1] = first.n;
    sum.dims[2] = first.m;
    sum.srcStrides[0] = outputElements;
    sum.srcStrides[1] = first.m;
    sum.srcStrides[2] = 1;
    sum.dstStrides[0] = first.d.batchStride;
    sum.dstStrides[1] = first.d.leadingDimension;
    sum.dstStrides[2] = 1;
    sum.total = groupedElements;
    sum.batches = static_cast<int>(images * chunks);
    sum.batchStride = groupedElements;
    gemmStage->calls = std::move(batched);
    gemmStage->after.insert(gemmStage->after.begin(),
                            RepackStep{sum,
                                       first.d.dataType,
                                       partialUid,
                                       first.d.uid,
                                       0,
                                       first.d.offset * elementBytes(first.d.dataType)});
    plan.workspace.push_back(WorkspaceTensor{partialUid, offset, bytes});
    plan.workspaceBytes = offset + bytes;
    return plan;
}

/// @p plan with a backward-weights GEMM whose reduction a backend could split (rocBLAS does,
/// past 2^31 bytes of the operand k walks) summed into an fp32 copy of w in the workspace and
/// rounded into w once, as the sliced plans already are; nullopt when the copy does not fit.
std::optional<ConvGemmPlanCalls> sumLongReductionsInFp32(const ConvProblem& problem,
                                                         ConvGemmPlanCalls plan,
                                                         size_t maxWorkspaceBytes)
{
    bool needed = false;
    for(const auto& stage : plan.stages)
    {
        for(const auto& call : stage.calls)
        {
            needed = needed || (call.d.uid == problem.w.uid && rocblasRoundsReductionChunks(call));
        }
    }
    if(problem.direction != ConvDirection::BACKWARD_WEIGHTS || !needed)
    {
        return plan;
    }
    // The copy takes w's own strides, so every GEMM's view of w addresses it unchanged.
    int64_t span = 1;
    for(size_t axis = 0; axis < problem.w.dims.size(); ++axis)
    {
        span += (problem.w.dims[axis] - 1) * problem.w.strides[axis];
    }
    const int64_t lastUid = std::max({problem.x.uid, problem.w.uid, problem.y.uid});
    if(lastUid > std::numeric_limits<int64_t>::max() - 4)
    {
        return std::nullopt;
    }
    const int64_t uid = lastUid + 4;
    const size_t offset = (plan.workspaceBytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT
                          * WORKSPACE_ALIGNMENT;
    const size_t bytes = static_cast<size_t>(span) * sizeof(float);
    if(offset + bytes > maxWorkspaceBytes)
    {
        return std::nullopt;
    }
    for(auto& stage : plan.stages)
    {
        for(auto& call : stage.calls)
        {
            if(call.d.uid == problem.w.uid)
            {
                call.d.uid = uid;
                call.d.dataType = data_objects::DataType::FLOAT;
            }
        }
    }
    ConvGemmStage round;
    round.before.push_back(RepackStep{
        stridedArgs<StridedCastArgs>(problem.w.dims, problem.w.strides, problem.w.strides),
        problem.dataType,
        uid,
        problem.w.uid,
        0,
        0});
    plan.stages.push_back(std::move(round));
    plan.workspace.push_back(WorkspaceTensor{uid, offset, bytes});
    plan.workspaceBytes = offset + bytes;
    return plan;
}

} // namespace

std::optional<ConvGemmPlanCalls> convGemmCalls(const ConvProblem& problem,
                                               size_t maxWorkspaceBytes,
                                               ColumnLayout layout,
                                               bool splitReduction,
                                               bool narrowPartials)
{
    auto plan = directGemmCalls(problem);
    if(!plan.has_value())
    {
        plan = oneByOneWeights(problem, maxWorkspaceBytes, layout);
    }
    if(!plan.has_value())
    {
        plan = im2colGemmCalls(problem, maxWorkspaceBytes, layout);
    }
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    return sumLongReductionsInFp32(
        problem,
        batchReductionChains(
            problem, std::move(*plan), maxWorkspaceBytes, splitReduction, narrowPartials),
        maxWorkspaceBytes);
}

std::optional<ConvGemmPlanCalls> reducedPrecisionGemmCalls(const ConvProblem& problem)
{
    if(problem.dataType != data_objects::DataType::HALF)
    {
        return std::nullopt;
    }
    const auto shape = oneByOneWeightsOf(problem);
    // One GEMM already reduces over every image exactly: nothing for a reduced form to gain.
    if(!shape.has_value() || directGemmCalls(problem).has_value())
    {
        return std::nullopt;
    }
    const int64_t kg = problem.kPerGroup();
    const int64_t cg = problem.cPerGroup();
    ConvGemmStage stage;
    for(int64_t image = 0; image < problem.n; ++image)
    {
        // dw_g (kg x cg) += dy_n,g (kg x pixels) x_n,g^T (pixels x cg), batched over the groups.
        const MatrixView dy{problem.y.uid,
                            problem.dataType,
                            kg,
                            shape->pixels,
                            problem.y.strides[1],
                            shape->yPixelStride,
                            kg * problem.y.strides[1],
                            problem.y.offset + image * problem.y.strides[0]};
        const MatrixView x{problem.x.uid,
                           problem.dataType,
                           shape->pixels,
                           cg,
                           shape->xPixelStride,
                           problem.x.strides[1],
                           cg * problem.x.strides[1],
                           problem.x.offset + image * problem.x.strides[0]};
        const MatrixView w{problem.w.uid,
                           problem.dataType,
                           kg,
                           cg,
                           problem.w.strides[0],
                           problem.w.strides[1],
                           kg * problem.w.strides[0],
                           problem.w.offset};
        auto call = gemmForViews(dy, x, w, problem.groups);
        if(!call.has_value())
        {
            return std::nullopt;
        }
        call->accumulate = image > 0;
        stage.calls.push_back(*call);
    }
    ConvGemmPlanCalls plan;
    plan.row = ConvGemmRow::ONE_BY_ONE;
    plan.stages.push_back(std::move(stage));
    return plan;
}

} // namespace hipblaslt_plugin
