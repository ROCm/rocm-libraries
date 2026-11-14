// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenConvTensorChecks.hpp"

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
#include <hipdnn_plugin_sdk/PluginException.hpp>

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

[[noreturn]] void decline(const std::string& reason)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, reason);
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
// A non-packed tensor has no canonical strides, so it is classified by its stride order: dimension
// indices sorted by stride descending, then extent descending, then index ascending (a
// deterministic total order; std::stable_sort is avoided because clang-tidy rejects it through
// std::get_temporary_buffer). [0, 1, .., n-1] is channels-first, [0, 2, .., n-1, 1] channels-last.
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

    std::vector<size_t> order(rank);
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

    TensorInfo info;
    info.name = name;
    info.dataType = attr.data_type();
    info.dims.assign(dims->begin(), dims->end());
    info.strides.assign(strides->begin(), strides->end());
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

// Number of elements (product of the dims) of a tensor, saturating at INT64_MAX so that absurd dims
// cannot overflow. Rejects null and negative dims, which have no element count.
int64_t
    saturatingElementCount(const char* operation, const char* name, const TensorAttributes& attr)
{
    const auto* dims = attr.dims();
    const std::string prefix = std::string(operation) + ": tensor " + name;

    if(dims == nullptr)
    {
        decline(prefix + " has null dims");
    }

    constexpr int64_t MAX_COUNT = std::numeric_limits<int64_t>::max();
    int64_t count = 1;
    for(const int64_t dim : *dims)
    {
        if(dim < 0)
        {
            decline(prefix + " has a negative dim");
        }
        if(dim > 0 && count > MAX_COUNT / dim)
        {
            return MAX_COUNT;
        }
        count *= dim;
    }
    return count;
}

void checkBwdDataElementCount(const char* name, const TensorAttributes& attr)
{
    if(saturatingElementCount(BWD_DATA_OPERATION, name, attr) > MAX_BWD_DATA_ELEMENTS)
    {
        decline(std::string(BWD_DATA_OPERATION) + ": tensor " + name + " has more than "
                + std::to_string(MAX_BWD_DATA_ELEMENTS) + " elements");
    }
}

} // namespace

void validateConvFwdIoTensors(const TensorAttributes& x,
                              const TensorAttributes& w,
                              const TensorAttributes& y)
{
    const TensorInfo xInfo = readTensor(FWD_OPERATION, "x", x);
    const TensorInfo wInfo = readTensor(FWD_OPERATION, "w", w);
    const TensorInfo yInfo = readTensor(FWD_OPERATION, "y", y);

    checkDataType(FWD_OPERATION, xInfo, wInfo, yInfo);
    checkLayout(FWD_OPERATION, xInfo, wInfo, yInfo);

    // Packing. Padded (non-packed) strides make ConvDirectNaiveConvFwd and
    // ConvHipImplicitGemm3DGroupFwdXdlops return wrong output and write outside the logical
    // output tensor. Workaround tracked by ALMIOPEN-3018; remove it when the MIOpen defect
    // ALMIOPEN-3013 is fixed.
    for(const TensorInfo* tensor : {&xInfo, &wInfo, &yInfo})
    {
        if(!tensor->packed)
        {
            decline(std::string(FWD_OPERATION) + ": tensor " + tensor->name
                    + " is not packed; x, w and y must all be packed");
        }
    }
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
