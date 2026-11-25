// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenConvApplicabilityChecks.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenUtils.hpp"

namespace miopen_plugin::conv_applicability
{

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;

// Layout class bits of one tensor. A tensor has no bit when it is neither channels-first nor
// channels-last. Both bits are set only for a packed tensor whose two canonical stride vectors
// coincide (C == 1 and every spatial extent 1).
constexpr unsigned LAYOUT_CHANNELS_FIRST = 1U;
constexpr unsigned LAYOUT_CHANNELS_LAST = 2U;

// N, C and at least one spatial dimension.
constexpr size_t MIN_RANK = 3;

// The operation that prefixes every reason.
constexpr const char* FWD_OPERATION = "Convolution forward";
constexpr const char* BWD_DATA_OPERATION = "Convolution backward data";
constexpr const char* WRW_OPERATION = "Convolution backward weights";

// Backward data declines a dy or dx with more elements than this (the largest 32-bit signed count).
constexpr int64_t MAX_BWD_DATA_ELEMENTS = std::numeric_limits<int32_t>::max();

struct TensorInfo
{
    const char* name = "";
    DataType dataType = DataType::UNSET;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    bool packed = false;
    unsigned layoutMask = 0U;
};

[[noreturn]] void decline(const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, message);
}

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>* values)
{
    if(values == nullptr)
    {
        return {};
    }
    return {values->begin(), values->end()};
}

// Dimension indices in the order MIOpen derives for a tensor (TensorDescriptor::find_permutation):
// sorted by stride descending, then extent descending, then index ascending. A strict total
// order, so the result is deterministic; std::stable_sort is avoided because clang-tidy rejects
// it through std::get_temporary_buffer. [0, 1, .., n-1] is the channels-first order,
// [0, 2, .., n-1, 1] the channels-last one.
std::vector<size_t> layoutOrder(const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& strides)
{
    std::vector<size_t> order(dims.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&dims, &strides](size_t a, size_t b) {
        if(strides[a] != strides[b])
        {
            return strides[a] > strides[b];
        }
        if(dims[a] != dims[b])
        {
            return dims[a] > dims[b];
        }
        return a < b;
    });
    return order;
}

// Left-pads `values` with `fill` up to `rank` (numpy-style right alignment).
std::vector<int64_t> leftPad(const std::vector<int64_t>& values, size_t rank, int64_t fill)
{
    std::vector<int64_t> padded(rank - values.size(), fill);
    padded.insert(padded.end(), values.begin(), values.end());
    return padded;
}

void checkBiasIsPerChannel(const TensorAttributes& bias, const TensorAttributes& convOutput)
{
    const auto outputDims = toVector(convOutput.dims());
    if(outputDims.size() < 2)
    {
        decline("Convolution output must have at least a batch and a channel dimension");
    }

    auto biasDims = toVector(bias.dims());
    auto biasStrides = toVector(bias.strides());
    if(biasDims.size() > outputDims.size() || biasStrides.size() != biasDims.size())
    {
        decline("Bias tensor rank must not exceed the convolution output rank");
    }

    // A lower-rank bias is right-aligned against the output, so its missing leading
    // dimensions are units.
    biasDims = leftPad(biasDims, outputDims.size(), 1);
    biasStrides = leftPad(biasStrides, outputDims.size(), 1);

    for(size_t i = 0; i < biasDims.size(); ++i)
    {
        const int64_t expected = (i == 1) ? outputDims[1] : 1;
        if(biasDims[i] != expected)
        {
            decline("Bias tensor must be a per-channel tensor with dims [1, C, 1, ...] where C is "
                    "the convolution output channel count");
        }
    }

    if(biasDims[1] > 1 && biasStrides[1] != 1)
    {
        decline("Bias tensor channels must be contiguous");
    }
}

// Bias data type. The fused MIOpen solver never validates the bias data type and returns garbage
// when it differs from the io data type. The node checks of the fused builder let an fp32 bias
// through on an fp16 or bf16 convolution because the convolution intermediate is float and equals
// the bias type. Workaround tracked by ALMIOPEN-3016; remove it when the MIOpen defect
// ALMIOPEN-3011 is fixed.
void checkBiasDataTypeMatchesInput(const TensorAttributes& bias, const TensorAttributes& input)
{
    if(bias.data_type() != input.data_type())
    {
        decline(std::string("Bias tensor data type ")
                + hipdnn_flatbuffers_sdk::data_objects::toString(bias.data_type())
                + " must equal the convolution input data type "
                + hipdnn_flatbuffers_sdk::data_objects::toString(input.data_type()));
    }
}

// Packed strides for `outputDims` laid out in the dimension order MIOpen derives for the input
// tensor (see layoutOrder: dimensions sorted by stride, then length, descending, ties by index;
// TensorDescriptor::find_permutation, which backs GetLayout / GetLayout_str in
// projects/miopen/src/tensor.cpp; ConvolutionDescriptor::GetForwardOutputTensor[WithLayout] in
// projects/miopen/src/convolution.cpp builds the output descriptor from that layout name).
// MIOpen's fused execute compares whole tensor descriptors, unit dimensions included
// (FusionPlanDescriptor::Execute, "The output descriptors dont match.", projects/miopen/src/
// fusion.cpp), so the caller compares the strides of unit dimensions as well. No public API
// returns the fused output strides, so they are reproduced here; MIOpen's rejection of a layout
// it does not name is left to the device check.
std::vector<int64_t> packedStridesInLayoutOrder(const std::vector<int64_t>& inputDims,
                                                const std::vector<int64_t>& inputStrides,
                                                const std::vector<int64_t>& outputDims)
{
    const auto order = layoutOrder(inputDims, inputStrides);

    std::vector<int64_t> strides(outputDims.size(), 0);
    int64_t running = 1;
    for(auto it = order.rbegin(); it != order.rend(); ++it)
    {
        strides[*it] = running;
        running *= outputDims[*it];
    }
    return strides;
}

void checkOutputMatchesFusedDescriptor(const TensorAttributes& input,
                                       const TensorAttributes& convOutput,
                                       const TensorAttributes& output)
{
    if(output.data_type() != input.data_type())
    {
        decline("Fused convolution output must have the same data type as the input");
    }

    const auto outputDims = toVector(output.dims());
    if(outputDims != toVector(convOutput.dims()))
    {
        decline("Fused convolution output dims must equal the convolution output dims");
    }

    const auto inputDims = toVector(input.dims());
    const auto inputStrides = toVector(input.strides());
    const auto outputStrides = toVector(output.strides());
    if(inputDims.size() != outputDims.size() || inputStrides.size() != outputDims.size()
       || outputStrides.size() != outputDims.size())
    {
        decline("Fused convolution input and output must have the same rank");
    }

    const auto expected = packedStridesInLayoutOrder(inputDims, inputStrides, outputDims);
    for(size_t i = 0; i < outputDims.size(); ++i)
    {
        if(outputStrides[i] != expected[i])
        {
            decline("Fused convolution output must be packed in the same layout as the input");
        }
    }
}

// Layout class mask of a tensor, computed on the graph tensor (unpadded rank 3, 4 or 5).
//
// A packed tensor is channels-first when its strides equal the canonical channels-first strides of
// its dims and channels-last when they equal the canonical channels-last strides, compared over
// ALL dimensions including those of extent 1. The strides of an extent-1 dimension never change
// which memory is addressed, but they are part of what the declared layout is: MIOpen's
// HeuristicUpdateLayouts falls back to ignoring degenerate strides when the three tensors disagree,
// which labels a mixed-layout problem as uniform and hands the descriptors to a solver that is
// wrong on them. Judging those strides here keeps such a problem declined, while a uniform graph
// with degenerate shapes (K == 1, C == 1 with an Rx1 filter, a 1x1 output) keeps its canonical
// strides and is accepted.
//
// A non-packed tensor has no canonical strides, so it is classified by its layout order
// (layoutOrder).
unsigned classifyLayout(const std::vector<int64_t>& dims,
                        const std::vector<int64_t>& strides,
                        bool packed)
{
    unsigned mask = 0U;

    if(packed)
    {
        if(strides == hipdnn_data_sdk::utilities::generateStrides(dims))
        {
            mask |= LAYOUT_CHANNELS_FIRST;
        }
        if(strides
           == hipdnn_data_sdk::utilities::generateStrides(
               dims, hipdnn_data_sdk::utilities::strideOrderNhwc(dims.size())))
        {
            mask |= LAYOUT_CHANNELS_LAST;
        }
        return mask;
    }

    const size_t rank = dims.size();
    const auto order = layoutOrder(dims, strides);

    std::vector<size_t> channelsFirstOrder(rank);
    std::iota(channelsFirstOrder.begin(), channelsFirstOrder.end(), size_t{0});

    std::vector<size_t> channelsLastOrder;
    channelsLastOrder.reserve(rank);
    channelsLastOrder.push_back(0);
    for(size_t i = 2; i < rank; ++i)
    {
        channelsLastOrder.push_back(i);
    }
    channelsLastOrder.push_back(1);

    if(order == channelsFirstOrder)
    {
        mask |= LAYOUT_CHANNELS_FIRST;
    }
    else if(order == channelsLastOrder)
    {
        mask |= LAYOUT_CHANNELS_LAST;
    }
    return mask;
}

std::string layoutName(unsigned mask)
{
    const bool channelsFirst = (mask & LAYOUT_CHANNELS_FIRST) != 0U;
    const bool channelsLast = (mask & LAYOUT_CHANNELS_LAST) != 0U;

    if(channelsFirst && channelsLast)
    {
        return "channels-first and channels-last";
    }
    if(channelsFirst)
    {
        return "channels-first";
    }
    if(channelsLast)
    {
        return "channels-last";
    }
    return "other";
}

// Product of the dims of a tensor with each dim raised to at least `smallestFactor`, saturating at
// INT64_MAX so that absurd dims cannot overflow. Rejects null and negative dims, which have no
// element count. Every dim is read, so the result does not depend on the dim order: a zero factor
// (possible only with `smallestFactor` 0) gives 0 whatever the other dims are, otherwise a product
// that does not fit gives INT64_MAX.
int64_t saturatingProduct(const char* operation,
                          const char* name,
                          const TensorAttributes& attr,
                          int64_t smallestFactor)
{
    const auto* dims = attr.dims();
    const std::string prefix = std::string(operation) + ": tensor " + name;

    if(dims == nullptr)
    {
        decline(prefix + " has null dims");
    }

    constexpr int64_t MAX_COUNT = std::numeric_limits<int64_t>::max();
    int64_t count = 1;
    bool zero = false;
    bool saturated = false;
    for(const int64_t dim : *dims)
    {
        if(dim < 0)
        {
            decline(prefix + " has a negative dim");
        }
        const int64_t factor = std::max(dim, smallestFactor);
        if(factor == 0)
        {
            zero = true;
        }
        else if(!saturated)
        {
            if(count > MAX_COUNT / factor)
            {
                saturated = true;
            }
            else
            {
                count *= factor;
            }
        }
    }
    if(zero)
    {
        return 0;
    }
    return saturated ? MAX_COUNT : count;
}

// Number of elements (product of the dims) of a tensor, saturating at INT64_MAX. A zero dim makes
// the count zero, whatever the other dims are and wherever the zero stands.
int64_t
    saturatingElementCount(const char* operation, const char* name, const TensorAttributes& attr)
{
    return saturatingProduct(operation, name, attr, 0);
}

// Product of the dims with every zero dim counted as 1, saturating at INT64_MAX. It bounds every
// partial product that strides and packing computations form over the dims, in any order (a zero
// dim placed first would otherwise hide dims that overflow after it).
int64_t saturatingExtent(const char* operation, const char* name, const TensorAttributes& attr)
{
    return saturatingProduct(operation, name, attr, 1);
}

// True when a * b is representable in int64_t.
bool productFits(int64_t a, int64_t b)
{
    constexpr int64_t MAX_VALUE = std::numeric_limits<int64_t>::max();
    constexpr int64_t MIN_VALUE = std::numeric_limits<int64_t>::min();

    if(a == 0 || b == 0)
    {
        return true;
    }
    // -1 is excluded below so that MIN_VALUE / -1 is never formed.
    if(a == -1)
    {
        return b != MIN_VALUE;
    }
    if(b == -1)
    {
        return a != MIN_VALUE;
    }
    if(a > 0)
    {
        return b > 0 ? a <= MAX_VALUE / b : b >= MIN_VALUE / a;
    }
    return b > 0 ? a >= MIN_VALUE / b : a >= MAX_VALUE / b;
}

// True when a + b is representable in int64_t.
bool sumFits(int64_t a, int64_t b)
{
    return b >= 0 ? a <= std::numeric_limits<int64_t>::max() - b
                  : a >= std::numeric_limits<int64_t>::min() - b;
}

// True when hipdnn_data_sdk::utilities::isTensorPacked evaluates without signed overflow for
// `dims` (non-negative, with a product that fits, see saturatingExtent) and any `strides`: it
// sums the terms (dim - 1) * stride in dim order, starting from 0, and then adds 1 to the sum. The
// check replays exactly that arithmetic in that order, so it declines only strides for which a
// term, a partial sum or the final sum plus 1 overflows (a partial sum counts even when the full
// sum fits) and accepts every other stride, negative ones included. It is an overflow guard only:
// the packed verdict is isTensorPacked's own (see readTensor), and the guard is conservative
// where isTensorPacked returns early on a broadcast axis. DeclinesStridesWhoseArithmeticOverflows
// and StrideOverflowFollowsThePackingSumOrder pin its boundary and its order;
// PackingVerdictAgreesWithIsTensorPacked pins that it does not decline non-overflowing strides.
bool packingArithmeticFits(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    int64_t space = 0;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        const int64_t extentMinusOne = dims[i] - 1;
        if(!productFits(extentMinusOne, strides[i]))
        {
            return false;
        }
        const int64_t term = extentMinusOne * strides[i];
        if(!sumFits(space, term))
        {
            return false;
        }
        space += term;
    }
    return sumFits(space, 1);
}

// Reads one tensor and rejects shapes the checks cannot judge. `operation` prefixes every reason
// ("Convolution forward", ...) and `name` is the tensor's role in that operation ("x", "dy", ...).
TensorInfo readTensor(const char* operation, const char* name, const TensorAttributes& attr)
{
    const auto* dims = attr.dims();
    const auto* strides = attr.strides();
    const std::string prefix = std::string(operation) + ": tensor " + name;

    if(dims == nullptr || strides == nullptr)
    {
        decline(prefix + " has null dims or strides");
    }
    if(dims->size() != strides->size())
    {
        decline(prefix + " has different dims and strides sizes");
    }
    if(dims->size() < MIN_RANK)
    {
        decline(prefix + " has rank below 3");
    }

    // The packing and layout checks below multiply dims and strides in int64_t, in dim order and
    // in stride order; dims whose zero-as-one product does not fit would overflow there (signed
    // overflow is undefined behaviour), even when a zero dim makes the element count zero. No real
    // tensor has that many elements, so decline before computing anything. A zero dim does not
    // decline here, but that is not acceptance of every empty tensor: generateStrides gives stride
    // 0 to every dim that is outer to a zero dim in the layout's memory order. For channels-first
    // those are the dims before the zero dim. For channels-last they are the batch dim, plus the
    // spatial dims when the channel dim is zero, or the earlier spatial dims when a spatial dim
    // is zero (the channel dim keeps its stride). A dim of extent above 1 at stride 0 is a
    // broadcast axis: the tensor is not packed (isTensorPacked) and its stride order matches
    // neither layout (classifyLayout), so the layout check below declines it. Without such a dim
    // the tensor is packed with canonical strides and is accepted, as is an empty leading dim.
    // By hand: channels-first {1, 0, 8, 8} has strides {0, 64, 8, 1} and is accepted;
    // channels-last {1, 0, 8, 8} has {0, 1, 0, 0} and is declined; channels-last {1, 4, 0, 8} has
    // {0, 1, 32, 4} and is accepted; channels-first {1, 4, 0, 8} has {0, 0, 8, 1} and is declined.
    if(saturatingExtent(operation, name, attr) == std::numeric_limits<int64_t>::max())
    {
        decline(prefix + " has dims whose element count reaches the 64-bit limit");
    }

    TensorInfo info;
    info.name = name;
    info.dataType = attr.data_type();
    info.dims = toVector(dims);
    info.strides = toVector(strides);

    // isTensorPacked then forms (dim - 1) * stride per dim and sums the terms in int64_t, with
    // strides straight from the graph (the backend does not bound them); decline strides for which
    // that arithmetic overflows. The dims are non-negative and bounded by the check above.
    if(!packingArithmeticFits(info.dims, info.strides))
    {
        decline(prefix + " has strides whose packing arithmetic overflows a 64-bit integer");
    }

    info.packed = hipdnn_data_sdk::utilities::isTensorPacked(info.dims, info.strides);
    info.layoutMask = classifyLayout(info.dims, info.strides, info.packed);
    return info;
}

bool isSupportedDataType(DataType dataType)
{
    return dataType == DataType::FLOAT || dataType == DataType::HALF
           || dataType == DataType::BFLOAT16;
}

// "a, b and c" for the three tensors of one operation.
std::string tensorNames(const TensorInfo& a, const TensorInfo& b, const TensorInfo& c)
{
    return std::string(a.name) + ", " + b.name + " and " + c.name;
}

// Data type. MIOpen reports a solution for mixed data types across the three tensors and then fails
// at execution, returns garbage or, for 3D, faults the GPU; only some types are supported at all.
// Workaround tracked by ALMIOPEN-3015 (forward) and ALMIOPEN-3303 (backward data and backward
// weights); remove it when the MIOpen defects ALMIOPEN-3010 and ALMIOPEN-3302 are fixed.
void checkDataType(const char* operation,
                   const TensorInfo& a,
                   const TensorInfo& b,
                   const TensorInfo& c)
{
    const bool sameDataType = a.dataType == b.dataType && b.dataType == c.dataType;
    if(!sameDataType)
    {
        decline(std::string(operation) + ": " + tensorNames(a, b, c) + " must share one data type ("
                + a.name + "=" + hipdnn_flatbuffers_sdk::data_objects::toString(a.dataType) + ", "
                + b.name + "=" + hipdnn_flatbuffers_sdk::data_objects::toString(b.dataType) + ", "
                + c.name + "=" + hipdnn_flatbuffers_sdk::data_objects::toString(c.dataType) + ")");
    }
    if(!isSupportedDataType(a.dataType))
    {
        decline(std::string(operation) + ": data type "
                + hipdnn_flatbuffers_sdk::data_objects::toString(a.dataType) + " is not supported; "
                + tensorNames(a, b, c) + " must be FLOAT, HALF or BFLOAT16");
    }
}

// Layout. With the three tensors in different layouts, Find can pick a naive solver that returns
// wrong output for them (forward: ConvDirectNaiveConvFwd). Workaround tracked by ALMIOPEN-3017
// (forward) and ALMIOPEN-3299 (backward data and backward weights); remove it when the MIOpen
// defects ALMIOPEN-3012 and ALMIOPEN-3298 are fixed.
void checkLayout(const char* operation,
                 const TensorInfo& a,
                 const TensorInfo& b,
                 const TensorInfo& c)
{
    if((a.layoutMask & b.layoutMask & c.layoutMask) == 0U)
    {
        decline(std::string(operation) + ": " + tensorNames(a, b, c)
                + " must all be channels-first or all channels-last (" + a.name + " is "
                + layoutName(a.layoutMask) + ", " + b.name + " is " + layoutName(b.layoutMask)
                + ", " + c.name + " is " + layoutName(c.layoutMask) + ")");
    }
}

// Backward data and backward weights share every rule; they differ only in the roles of the three
// tensors, which `operation` and the tensor names carry into the reasons.
void checkBackwardTensors(const char* operation,
                          const TensorInfo& a,
                          const TensorInfo& b,
                          const TensorInfo& c)
{
    checkDataType(operation, a, b, c);
    checkLayout(operation, a, b, c);

    // Packing. For channels-last tensors with padded (non-packed) strides MIOpen returns wrong
    // output (values up to 1e38, infinities, NaN). Channels-first non-packed tensors are computed
    // correctly, so unlike forward this declines only a tensor that is both non-packed and
    // channels-last. A non-packed tensor is classified by stride order and so carries at most one
    // layout bit; a packed tensor never triggers this rule. Workaround tracked by ALMIOPEN-3301;
    // remove it when the MIOpen defect ALMIOPEN-3300 is fixed.
    for(const TensorInfo* tensor : {&a, &b, &c})
    {
        if(!tensor->packed && (tensor->layoutMask & LAYOUT_CHANNELS_LAST) != 0U)
        {
            decline(std::string(operation) + ": tensor " + tensor->name
                    + " is channels-last and not packed; " + tensorNames(a, b, c)
                    + " must not be channels-last with padded strides");
        }
    }
}

void checkBwdDataElementCount(const char* name, const TensorAttributes& attr)
{
    if(saturatingElementCount(BWD_DATA_OPERATION, name, attr) > MAX_BWD_DATA_ELEMENTS)
    {
        decline(std::string(BWD_DATA_OPERATION) + ": tensor " + name + " has more than "
                + std::to_string(MAX_BWD_DATA_ELEMENTS) + " elements");
    }
}

// MIOpen's TensorDescriptor::IsPossibleLayout in StrictDecreasingStrides mode
// (projects/miopen/src/tensor.cpp) on the descriptor the plugin hands MIOpen: the strides, read in
// the order of the layout (N, C, spatial.. or N, spatial.., C), never increase, extent-1
// dimensions included. A rank-3 tensor is first padded to rank 4 as createPaddedTensor does
// (MiopenUtils.cpp): a trailing extent-1 dimension of stride C when the channel stride is below
// the length stride, else of stride 1.
bool isStrictlyInLayout(const TensorInfo& tensor, bool channelsLast)
{
    auto dims = tensor.dims;
    auto strides = tensor.strides;
    if(dims.size() == MIN_RANK)
    {
        const bool lengthOutside = strides[1] < strides[2];
        dims.push_back(1);
        strides.push_back(lengthOutside ? dims[1] : 1);
    }

    std::vector<size_t> order;
    order.reserve(dims.size());
    order.push_back(0);
    if(!channelsLast)
    {
        order.push_back(1);
    }
    for(size_t i = 2; i < dims.size(); ++i)
    {
        order.push_back(i);
    }
    if(channelsLast)
    {
        order.push_back(1);
    }

    for(size_t k = 1; k < order.size(); ++k)
    {
        if(strides[order[k - 1]] < strides[order[k]])
        {
            return false;
        }
    }
    return true;
}

// Layout, forward. The layout classes judge the strides of extent-1 dimensions (classifyLayout),
// so a graph whose tensors differ only there is mixed for checkLayout. MIOpen labels a problem
// in ProblemDescription::HeuristicUpdateLayouts (projects/miopen/src/conv/problem_description.cpp)
// and tries the strict mode first: when the three packed tensors are all strictly in one layout,
// MIOpen labels them with that layout without ignoring any stride, and it computed every such
// graph tested correctly. Graphs that reach a label only by ignoring extent-1 strides stay
// declined, which keeps the failing degenerate shapes of ALMIOPEN-3017 declined. Common shapes
// this accepts: a 1x1 convolution on a [N, C, 1, 1] input whose output is channels-last (what
// PyTorch allocates for such an input), and a batch-1 channel slice whose batch stride is that of
// its parent. Only packed graphs of rank 5 or below are accepted this way: no non-packed one was
// tested, and MIOpen names no layout above rank 5. A packed tensor can still have elements that
// share an address (see hasDisjointElements); checkForwardPacking declines it.
void checkForwardLayout(const TensorInfo& x, const TensorInfo& w, const TensorInfo& y)
{
    if(x.packed && w.packed && y.packed && x.dims.size() <= 5)
    {
        for(const bool channelsLast : {false, true})
        {
            if(isStrictlyInLayout(x, channelsLast) && isStrictlyInLayout(w, channelsLast)
               && isStrictlyInLayout(y, channelsLast))
            {
                return;
            }
        }
    }
    checkLayout(FWD_OPERATION, x, w, y);
}

// True when the elements of the tensor provably do not share an address: the dimensions of extent
// above 1, taken by increasing stride, each start at or beyond the span of the ones inside them,
// starting from stride 1. A broadcast axis (stride 0), a negative stride and strides that
// interleave two dimensions fail it. The check is conservative, not exact: it also fails some
// strides whose elements are all distinct, for example [2, 4, 2, 2] with strides [16, 4, 3, 2]
// (addresses 0, 2, 3 and 5 in each plane). A tensor with a zero extent has no elements and
// passes. The strides passed packingArithmeticFits, so only the span of the last dimension can
// overflow, and it needs no room for another dimension.
bool hasDisjointElements(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    std::vector<size_t> axes;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(dims[i] == 0)
        {
            return true;
        }
        if(dims[i] > 1)
        {
            axes.push_back(i);
        }
    }
    std::sort(axes.begin(), axes.end(), [&strides](size_t a, size_t b) {
        return strides[a] < strides[b];
    });

    int64_t span = 1;
    for(size_t k = 0; k < axes.size(); ++k)
    {
        const size_t axis = axes[k];
        if(strides[axis] < span)
        {
            return false;
        }
        if(!productFits(strides[axis], dims[axis]))
        {
            return k + 1 == axes.size();
        }
        span = strides[axis] * dims[axis];
    }
    return true;
}

// Packing, forward. A non-packed tensor is accepted when the convolution is ungrouped, not 3D
// channels-last, and hasDisjointElements holds for the tensor: MIOpen computed every such graph
// tested correctly (padded rows, channel and batch slices, an output written into a wider buffer),
// also next to a packed problem of the same shape in either order. The declined cases:
// - grouped: ConvDirectNaiveConvFwd and ConvHipImplicitGemm3DGroupFwdXdlops return wrong output
//   and write outside the logical output tensor. Workaround tracked by ALMIOPEN-3018; remove it
//   when the MIOpen defect ALMIOPEN-3013 is fixed.
// - 3D channels-last: a plane-padded problem is wrong on its own, and any padded problem reuses
//   the invoker of a same-shape packed problem run before it in the process, and the packed one
//   reuses its invoker when it runs first: MIOpen's invoker cache key leaves out the strides
//   (ALMIOPEN-3356). Remove it when that is fixed.
// - elements that may share an address (a broadcast axis, interleaved strides), packed or not:
//   isTensorPacked only compares the span of the strides with the element count, so overlapping
//   strides such as [256, 64, 5, 4] on [2, 4, 8, 8] count as packed (and MIOpen counts them as
//   packed too), and the strict layout path of checkForwardLayout accepts them. Not tested on a
//   GPU; an output of that kind cannot be written correctly at all.
void checkForwardPacking(const TensorInfo& x, const TensorInfo& w, const TensorInfo& y)
{
    const bool grouped = x.dims[1] != w.dims[1];
    const bool channelsLast3d
        = x.dims.size() == 5
          && (x.layoutMask & w.layoutMask & y.layoutMask & LAYOUT_CHANNELS_LAST) != 0U;

    for(const TensorInfo* tensor : {&x, &w, &y})
    {
        const std::string prefix = std::string(FWD_OPERATION) + ": tensor " + tensor->name;
        if(!tensor->packed && grouped)
        {
            decline(prefix + " is not packed; a grouped convolution needs x, w and y packed");
        }
        if(!tensor->packed && channelsLast3d)
        {
            decline(prefix
                    + " is not packed; a 3D channels-last convolution needs x, w and y packed");
        }
        if(!hasDisjointElements(tensor->dims, tensor->strides))
        {
            decline(prefix
                    + " may have elements that share an address (a zero, negative or "
                      "overlapping stride)");
        }
    }
}

} // namespace

bool hasPassByValueTensor(const TensorMap& tensorMap)
{
    return std::any_of(tensorMap.begin(), tensorMap.end(), [](const auto& entry) {
        return hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(entry.second);
    });
}

void checkConvFwdBiasActivLayouts(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr,
    const TensorMap& tensorMap)
{
    const auto& input = miopen_utils::findTensorAttributes(tensorMap, convAttr.x_tensor_uid());
    const auto& convOutput = miopen_utils::findTensorAttributes(tensorMap, convAttr.y_tensor_uid());

    if(biasAttr != nullptr)
    {
        const auto biasUid = biasAttr->in_0_tensor_uid() == convAttr.y_tensor_uid()
                                 ? biasAttr->in_1_tensor_uid().value()
                                 : biasAttr->in_0_tensor_uid();
        const auto& bias = miopen_utils::findTensorAttributes(tensorMap, biasUid);
        checkBiasIsPerChannel(bias, convOutput);
        checkBiasDataTypeMatchesInput(bias, input);
    }

    checkOutputMatchesFusedDescriptor(
        input,
        convOutput,
        miopen_utils::findTensorAttributes(tensorMap, activAttr.out_0_tensor_uid()));
}

void validateConvFwdIoTensors(const TensorAttributes& x,
                              const TensorAttributes& w,
                              const TensorAttributes& y)
{
    const TensorInfo xInfo = readTensor(FWD_OPERATION, "x", x);
    const TensorInfo wInfo = readTensor(FWD_OPERATION, "w", w);
    const TensorInfo yInfo = readTensor(FWD_OPERATION, "y", y);

    checkDataType(FWD_OPERATION, xInfo, wInfo, yInfo);
    checkForwardLayout(xInfo, wInfo, yInfo);
    checkForwardPacking(xInfo, wInfo, yInfo);
}

void validateConvBwdDataIoTensors(const TensorAttributes& dy,
                                  const TensorAttributes& w,
                                  const TensorAttributes& dx)
{
    const TensorInfo dyInfo = readTensor(BWD_DATA_OPERATION, "dy", dy);
    const TensorInfo wInfo = readTensor(BWD_DATA_OPERATION, "w", w);
    const TensorInfo dxInfo = readTensor(BWD_DATA_OPERATION, "dx", dx);

    checkBackwardTensors(BWD_DATA_OPERATION, dyInfo, wInfo, dxInfo);
}

void validateConvWrwIoTensors(const TensorAttributes& x,
                              const TensorAttributes& dy,
                              const TensorAttributes& dw)
{
    const TensorInfo xInfo = readTensor(WRW_OPERATION, "x", x);
    const TensorInfo dyInfo = readTensor(WRW_OPERATION, "dy", dy);
    const TensorInfo dwInfo = readTensor(WRW_OPERATION, "dw", dw);

    checkBackwardTensors(WRW_OPERATION, xInfo, dyInfo, dwInfo);
}

void validateConvBwdDataSize(const TensorAttributes& dy, const TensorAttributes& dx)
{
    // Size. For tensors of more than 2^31 elements MIOpen offers
    // ConvHipImplicitGemmGroupBwdXdlops, which fails during Find: a launch error and
    // miopenStatusUnknownError on one architecture, an illegal memory access that poisons the
    // process on another. Backward data only; backward weights and forward are not covered by the
    // ticket. Workaround tracked by ALMIOPEN-3306; remove it when the MIOpen defect ALMIOPEN-3305
    // is fixed. Only the dims are read, so this is safe to call before anything that computes
    // strides.
    checkBwdDataElementCount("dy", dy);
    checkBwdDataElementCount("dx", dx);
}

} // namespace miopen_plugin::conv_applicability
