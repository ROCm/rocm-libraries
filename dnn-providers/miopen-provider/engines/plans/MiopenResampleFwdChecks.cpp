// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <limits>
#include <set>
#include <string>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "Workarounds.hpp"
#include "engines/plans/MiopenResampleFwdChecks.hpp"

namespace miopen_plugin::resample_fwd_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PaddingMode;
using hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes;
using hipdnn_flatbuffers_sdk::data_objects::ResampleMode;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;

namespace
{

// The prefix isSupported() logs. Kept out of every throw below so it never doubles.
constexpr auto LOG_PREFIX = "Resample forward plan builder: ";

[[noreturn]] void decline(const std::string& reason)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, reason);
}

enum class Layout
{
    CHANNELS_FIRST, // NCHW / NCDHW
    CHANNELS_LAST, // NHWC / NDHWC
};

struct TensorInfo
{
    const TensorAttributes* attr;
    int64_t uid;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

TensorInfo describeTensor(const IGraph& opGraph, int64_t uid, const std::string& name)
{
    const auto& tensorMap = opGraph.getTensorMap();
    const auto it = tensorMap.find(uid);
    if(it == tensorMap.end() || it->second == nullptr)
    {
        decline(name + " tensor (uid " + std::to_string(uid)
                + ") does not resolve in the graph's tensor map");
    }

    const auto* attr = it->second;
    if(attr->virtual_())
    {
        decline(name + " tensor must not be virtual");
    }
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(attr))
    {
        decline(name + " tensor must not be pass-by-value");
    }
    if(attr->ragged_offset_tensor_uid().has_value())
    {
        decline(name + " tensor must not be ragged");
    }
    if(attr->dims() == nullptr || attr->strides() == nullptr
       || attr->dims()->size() != attr->strides()->size())
    {
        decline(name + " tensor must have one stride per dimension");
    }

    TensorInfo info{attr,
                    uid,
                    std::vector<int64_t>(attr->dims()->begin(), attr->dims()->end()),
                    std::vector<int64_t>(attr->strides()->begin(), attr->strides()->end())};

    // MIOpen's pooling kernels address tensors with 32-bit offsets.
    int64_t elementCount = 1;
    for(const auto dim : info.dims)
    {
        if(dim <= 0)
        {
            decline(name + " tensor has a non-positive dimension");
        }
        if(elementCount > std::numeric_limits<int32_t>::max() / dim)
        {
            decline(name + " tensor element count exceeds INT32_MAX");
        }
        elementCount *= dim;
    }

    return info;
}

std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims, Layout layout)
{
    using namespace hipdnn_data_sdk::utilities;
    return layout == Layout::CHANNELS_FIRST ? generateStrides(dims)
                                            : generateStrides(dims, strideOrderNhwc(dims.size()));
}

// True when the tensor puts every element where a packed tensor of `layout` would. Length-1
// axes are never stepped along, so their strides do not matter.
bool addressesLike(const TensorInfo& tensor, Layout layout)
{
    const auto expected = packedStrides(tensor.dims, layout);
    for(size_t i = 0; i < tensor.dims.size(); ++i)
    {
        if(tensor.dims[i] > 1 && tensor.strides[i] != expected[i])
        {
            return false;
        }
    }
    return true;
}

std::vector<int> toSpatialVector(const flatbuffers::Vector<int64_t>* values,
                                 size_t spatialRank,
                                 const std::string& name,
                                 int64_t minValue)
{
    if(values == nullptr || values->size() != spatialRank)
    {
        decline(name + " must have one entry per spatial dimension (" + std::to_string(spatialRank)
                + ")");
    }

    std::vector<int> result;
    result.reserve(spatialRank);
    for(const auto value : *values)
    {
        if(value < minValue || value > std::numeric_limits<int>::max())
        {
            decline(name + " values must be in [" + std::to_string(minValue) + ", INT_MAX], got "
                    + std::to_string(value));
        }
        result.push_back(static_cast<int>(value));
    }
    return result;
}

miopenPoolingMode_t toMiopenPoolingMode(ResampleMode mode)
{
    switch(mode)
    {
    case ResampleMode::MAXPOOL:
        return miopenPoolingMax;
    case ResampleMode::AVGPOOL_EXCLUDE_PADDING:
        return miopenPoolingAverage;
    case ResampleMode::AVGPOOL_INCLUDE_PADDING:
        return miopenPoolingAverageInclusive;
    default:
        decline("unsupported resample mode "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameResampleMode(mode)));
    }
}

miopenIndexType_t toMiopenIndexType(DataType dataType)
{
    // hipDNN's index is signed, MIOpen's unsigned; the range check in resolve() keeps every
    // index MIOpen can write representable in the signed type.
    switch(dataType)
    {
    case DataType::INT8:
        return miopenIndexUint8;
    case DataType::INT32:
        return miopenIndexUint32;
    default:
        decline("index tensor data type must be INT8 or INT32, got "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType)));
    }
}

// MIOpen pools channels-last tensors only by transposing them around its tiled NCHW kernels
// (TransposedPoolingFwd2d/Nd); unlike channels-first there is no naive fallback. A tiled kernel is
// applicable only while its per-lane tile fits in scratch, sized as sizeof_private_memory() in
// MIOpen's src/solver/pooling/forward2d.cpp and forwardNd.cpp; past that the forward call fails
// with "No solver found". The limit assumes wave64, the smaller of MIOpen's two.
bool channelsLastTileFits(const ResampleFwdConfig& config)
{
    constexpr uint64_t K_MAX_WAVE_SCRATCH_SIZE = uint64_t{1024} * 8191;
    constexpr uint64_t K_LANE_SCRATCH_LIMIT = K_MAX_WAVE_SCRATCH_SIZE / 64;

    auto roundUpTo16 = [](uint64_t bytes) { return (bytes + 15) / 16 * 16; };

    // Every factor is at least 1, so stopping as soon as the tile outgrows the limit also keeps
    // the product from overflowing.
    uint64_t tileBytes = config.dataType == DataType::FLOAT ? 4 : 2;
    auto grow = [&tileBytes](uint64_t factor) {
        if(factor > K_LANE_SCRATCH_LIMIT || tileBytes * factor > K_LANE_SCRATCH_LIMIT)
        {
            return false;
        }
        tileBytes *= factor;
        return true;
    };

    const auto inputsSpanned = [&config](size_t dim, uint64_t outputsPerLane) {
        return (outputsPerLane - 1) * static_cast<uint64_t>(config.stride[dim])
               + static_cast<uint64_t>(config.window[dim]);
    };

    if(config.window.size() == 2)
    {
        // 16 output rows of one output column per lane, plus their 16 fp32 accumulators.
        return grow(inputsSpanned(0, 16)) && grow(inputsSpanned(1, 1))
               && roundUpTo16(tileBytes) + roundUpTo16(sizeof(float) * 16) <= K_LANE_SCRATCH_LIMIT;
    }

    // A 4 x 4 x 4 block of outputs per lane.
    return grow(inputsSpanned(0, 4)) && grow(inputsSpanned(1, 4)) && grow(inputsSpanned(2, 4))
           && roundUpTo16(tileBytes) <= K_LANE_SCRATCH_LIMIT;
}

} // namespace

ResampleFwdConfig resolve(const IGraph& opGraph)
{
    if(opGraph.nodeCount() != 1)
    {
        decline("applicable only for single-node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    if(!opGraph.hasOnlySupportedAttributes(
           std::set<NodeAttributes>{NodeAttributes::ResampleFwdAttributes}))
    {
        decline("graph contains unsupported node attributes");
    }

    if(opGraph.getNode(0).compute_data_type() != DataType::FLOAT)
    {
        decline("only supports nodes with an fp32 compute_data_type");
    }

    const auto& attrs = opGraph.getNodeWrapper(0).attributesAs<ResampleFwdAttributes>();

    ResampleFwdConfig config;
    config.mode = toMiopenPoolingMode(attrs.resample_mode());
    const bool isMaxPool = attrs.resample_mode() == ResampleMode::MAXPOOL;

    if(attrs.x_tensor_uid() == attrs.y_tensor_uid())
    {
        decline("does not support in-place execution; X and Y must be distinct tensors");
    }

    const auto x = describeTensor(opGraph, attrs.x_tensor_uid(), "X");
    const auto y = describeTensor(opGraph, attrs.y_tensor_uid(), "Y");

    const auto rank = x.dims.size();
    if(rank != 4 && rank != 5)
    {
        decline("X must be 4D or 5D, got rank " + std::to_string(rank)
                + "; MIOpen pooling has no 1D path");
    }
    if(y.dims.size() != rank)
    {
        decline("X and Y must have the same rank");
    }

    const auto dataType = x.attr->data_type();
    if(dataType != DataType::FLOAT && dataType != DataType::HALF && dataType != DataType::BFLOAT16)
    {
        decline("X data type must be FLOAT, HALF or BFLOAT16, got "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType)));
    }
    if(y.attr->data_type() != dataType)
    {
        decline("Y data type must match X");
    }

    const auto spatialRank = rank - 2;
    config.window = toSpatialVector(attrs.window(), spatialRank, "window", 1);
    config.stride = toSpatialVector(attrs.stride(), spatialRank, "stride", 1);
    config.padding = toSpatialVector(attrs.pre_padding(), spatialRank, "pre_padding", 0);

    if(toSpatialVector(attrs.post_padding(), spatialRank, "post_padding", 0) != config.padding)
    {
        decline("asymmetric padding is not supported; MIOpen pools with one symmetric pad per "
                "spatial dimension, so pre_padding must equal post_padding");
    }

    for(size_t i = 0; i < spatialRank; ++i)
    {
        if(config.padding[i] >= config.window[i])
        {
            // A window that lies entirely in the padding is where hipDNN and MIOpen disagree:
            // hipDNN's index sentinel is -1, MIOpen writes 0.
            decline("padding must be smaller than the window in every spatial dimension, got "
                    "padding "
                    + std::to_string(config.padding[i]) + " and window "
                    + std::to_string(config.window[i]) + " at spatial dimension "
                    + std::to_string(i));
        }
    }

    // For FP32 and FP16, MIOpen's max pooling never selects a padded position over an in-bounds
    // one, which matches NEG_INF_PAD, and its average pooling counts padded positions as zero,
    // which matches ZERO_PAD. Any other padding_mode, PADDING_NOT_SET included, only agrees with
    // that when there are no padded positions. BFLOAT16 max pooling can select a padded position
    // (ALMIOPEN-3295; declined below when an index is requested). A max pooling window with no
    // finite in-bounds input differs from hipDNN's references whatever the padding: MIOpen writes
    // the lowest finite value (FP32, FP16) or -inf (BFLOAT16) and index 0. That depends on the
    // data, so no graph check can decline it.
    const auto implicitPaddingMode = isMaxPool ? PaddingMode::NEG_INF_PAD : PaddingMode::ZERO_PAD;
    const bool hasPadding
        = std::any_of(config.padding.begin(), config.padding.end(), [](int p) { return p > 0; });
    if(attrs.padding_mode() != implicitPaddingMode && hasPadding)
    {
        decline("padding_mode "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::EnumNamePaddingMode(attrs.padding_mode()))
                + " with resample mode "
                + hipdnn_flatbuffers_sdk::data_objects::EnumNameResampleMode(attrs.resample_mode())
                + " requires all padding to be zero; MIOpen pads this mode as "
                + hipdnn_flatbuffers_sdk::data_objects::EnumNamePaddingMode(implicitPaddingMode));
    }

    if(x.dims[0] != y.dims[0] || x.dims[1] != y.dims[1])
    {
        decline("X and Y must have the same batch and channel dimensions");
    }

    for(size_t i = 0; i < spatialRank; ++i)
    {
        const auto paddedInput = x.dims[i + 2] + 2 * static_cast<int64_t>(config.padding[i]);
        if(paddedInput < config.window[i])
        {
            decline("the window must fit inside the padded input at spatial dimension "
                    + std::to_string(i));
        }

        // MIOpen does not check Y against its own (floor mode) output size formula.
        const auto expected = (paddedInput - config.window[i]) / config.stride[i] + 1;
        if(y.dims[i + 2] != expected)
        {
            decline("Y spatial dimension " + std::to_string(i) + " is "
                    + std::to_string(y.dims[i + 2]) + ", expected " + std::to_string(expected)
                    + " = floor((x + pre_padding + post_padding - window) / stride) + 1");
        }

        // With a single output the stride never steps, so the smallest stride that still gives
        // one output changes nothing. MIOpen's estimates do depend on it: V2 sizes channels-last
        // scratch for an input of y * stride - 2 * pad + window - 1, and the tiled kernels size
        // their tile from the stride. Global pooling would otherwise overshoot both.
        if(expected == 1)
        {
            config.stride[i] = static_cast<int>(paddedInput - config.window[i] + 1);
        }
    }

    Layout layout = Layout::CHANNELS_FIRST;
    if(addressesLike(x, Layout::CHANNELS_FIRST) && addressesLike(y, Layout::CHANNELS_FIRST))
    {
        layout = Layout::CHANNELS_FIRST;
    }
    else if(addressesLike(x, Layout::CHANNELS_LAST) && addressesLike(y, Layout::CHANNELS_LAST))
    {
        layout = Layout::CHANNELS_LAST;
    }
    else
    {
        decline("X and Y must both be packed channels-first (NCHW/NCDHW) or both packed "
                "channels-last (NHWC/NDHWC)");
    }

    config.dataType = dataType;
    if(layout == Layout::CHANNELS_LAST && !channelsLastTileFits(config))
    {
        decline("the window and stride are too large for MIOpen's channels-last (NHWC/NDHWC) "
                "pooling kernels");
    }

    config.xUid = x.uid;
    config.xDims = x.dims;
    config.xStrides = packedStrides(x.dims, layout);
    config.yUid = y.uid;
    config.yDims = y.dims;
    config.yStrides = packedStrides(y.dims, layout);

    const bool generateIndex = attrs.generate_index().value_or(false);
    const auto indexUid = attrs.index_tensor_uid();
    // Average pooling produces no index, so for it generate_index without an index tensor asks
    // for nothing and is ignored; the frontend builds such graphs.
    if(generateIndex && !indexUid.has_value() && isMaxPool)
    {
        decline("generate_index is set but the node has no index tensor");
    }
    if(!generateIndex && indexUid.has_value())
    {
        decline("the node has an index tensor but generate_index is not set");
    }

    if(indexUid.has_value())
    {
        if(!isMaxPool)
        {
            decline("only max pooling produces indices");
        }

        // MIOpen's channels-last path transposes X and Y around a channels-first kernel but
        // leaves the indices in packed NC(D)HW order.
        if(layout != Layout::CHANNELS_FIRST)
        {
            decline("max pooling with generate_index requires channels-first (NCHW/NCDHW) X and "
                    "Y");
        }

        // Workaround for ALMIOPEN-3295 (see Workarounds.hpp): MIOpen's BFLOAT16 max pooling
        // can select a padded position and write an index outside the window.
        if(WORKAROUND_ALMIOPEN_3295_DECLINES_INDEX(dataType, hasPadding))
        {
            decline("max pooling with generate_index on BFLOAT16 tensors requires zero padding "
                    "until MIOpen issue ALMIOPEN-3295 is fixed; MIOpen can select a padded "
                    "position and write an index outside the window");
        }

        if(*indexUid == x.uid || *indexUid == y.uid)
        {
            decline("the index tensor must be distinct from X and Y");
        }

        const auto index = describeTensor(opGraph, *indexUid, "index");
        config.indexType = toMiopenIndexType(index.attr->data_type());

        if(index.dims != y.dims)
        {
            decline("index tensor dimensions must equal Y dimensions");
        }
        if(!addressesLike(index, Layout::CHANNELS_FIRST))
        {
            decline("index tensor must be packed channels-first, the order MIOpen writes indices "
                    "in");
        }

        // Image-mode indices are flattened spatial offsets into X, from 0 to the X spatial size
        // minus one. X's element count is already known to fit in INT32, so only an INT8 index
        // can overflow.
        int64_t spatialSize = 1;
        for(size_t i = 2; i < rank; ++i)
        {
            spatialSize *= x.dims[i];
        }
        if(index.attr->data_type() == DataType::INT8
           && spatialSize - 1 > std::numeric_limits<int8_t>::max())
        {
            decline("X spatial size " + std::to_string(spatialSize)
                    + " exceeds the range of an INT8 index");
        }

        config.indexUid = indexUid;
    }

    return config;
}

bool isSupported(const IGraph& opGraph)
{
    try
    {
        resolve(opGraph);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(LOG_PREFIX << e.what());
        return false;
    }
}

} // namespace miopen_plugin::resample_fwd_applicability
