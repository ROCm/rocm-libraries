// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenConvTensorChecks.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

// Reads one tensor and rejects shapes the checks cannot judge.
TensorInfo readTensor(const char* name, const TensorAttributes& attr)
{
    const auto* dims = attr.dims();
    const auto* strides = attr.strides();
    const std::string prefix = std::string("Convolution forward: tensor ") + name;

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

} // namespace

void validateConvFwdIoTensors(const TensorAttributes& x,
                              const TensorAttributes& w,
                              const TensorAttributes& y)
{
    const TensorInfo xInfo = readTensor("x", x);
    const TensorInfo wInfo = readTensor("w", w);
    const TensorInfo yInfo = readTensor("y", y);

    // Data type. MIOpen reports a solution for mixed x/w/y types (GemmFwdRest) and then fails at
    // execution, returns garbage or, for 3D, faults the GPU. Workaround tracked by ALMIOPEN-3015;
    // remove it when the MIOpen defect ALMIOPEN-3010 is fixed.
    const bool sameDataType = xInfo.dataType == wInfo.dataType && wInfo.dataType == yInfo.dataType;
    if(!sameDataType)
    {
        decline(std::string("Convolution forward: x, w and y must share one data type (x=")
                + hipdnn_flatbuffers_sdk::data_objects::toString(xInfo.dataType)
                + ", w=" + hipdnn_flatbuffers_sdk::data_objects::toString(wInfo.dataType)
                + ", y=" + hipdnn_flatbuffers_sdk::data_objects::toString(yInfo.dataType) + ")");
    }
    if(!isSupportedDataType(xInfo.dataType))
    {
        decline(std::string("Convolution forward: data type ")
                + hipdnn_flatbuffers_sdk::data_objects::toString(xInfo.dataType)
                + " is not supported; x, w and y must be FLOAT, HALF or BFLOAT16");
    }

    // Layout. With x, w and y in different layouts, Find can pick ConvDirectNaiveConvFwd, which
    // returns wrong output for them. Workaround tracked by ALMIOPEN-3017; remove it when the
    // MIOpen defect ALMIOPEN-3012 is fixed.
    if((xInfo.layoutMask & wInfo.layoutMask & yInfo.layoutMask) == 0U)
    {
        decline("Convolution forward: x, w and y must all be channels-first or all channels-last "
                "(x is "
                + layoutName(xInfo.layoutMask) + ", w is " + layoutName(wInfo.layoutMask)
                + ", y is " + layoutName(yInfo.layoutMask) + ")");
    }

    // Packing. Padded (non-packed) strides make ConvDirectNaiveConvFwd and
    // ConvHipImplicitGemm3DGroupFwdXdlops return wrong output and write outside the logical
    // output tensor. Workaround tracked by ALMIOPEN-3018; remove it when the MIOpen defect
    // ALMIOPEN-3013 is fixed.
    for(const TensorInfo* tensor : {&xInfo, &wInfo, &yInfo})
    {
        if(!tensor->packed)
        {
            decline(std::string("Convolution forward: tensor ") + tensor->name
                    + " is not packed; x, w and y must all be packed");
        }
    }
}

} // namespace miopen_plugin::conv_applicability
